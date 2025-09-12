import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import time

from src.models.hrm import HRMTradingAgent, HRMCarry
from src.models.hrm_trading_environment import HRMTradingEnvironment, HRMTradingWrapper
from src.models.parallel_env_manager import ParallelEnvironmentManager
from src.utils.data_loader import DataLoader
from src.env.trading_mode import TradingMode
from src.utils.config_loader import ConfigLoader

# Import new modular components
from src.training.loss_functions import HRMLossFunction
from src.training.training_modes import TrainingModeManager, get_instruments_for_mode
from src.training.gpu_optimization import GPUOptimizer, OfflineRLPreprocessor

logger = logging.getLogger(__name__)


# HRMLossFunction moved to src.training.loss_functions


class HRMTrainer:
    """Trainer for HRM Trading Agent"""
    
    def __init__(self, 
                 config_path: str = "config/hrm_config.yaml",
                 data_path: str = "data/final",
                 device: str = None,  # Auto-detect if None
                 debug_mode: bool = None):  # Auto-detect from config if None
        
        # Initialize device using automatic TPU/GPU/CPU detection
        from src.utils.device_manager import get_device_manager
        self.device_manager = get_device_manager()
        
        if device is None:
            # Use automatic device selection
            self.device = self.device_manager.get_device()
            self.device_manager.print_device_summary()
        else:
            # Use user-specified device
            self.device = torch.device(device)
        self.config_path = config_path
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load training mode configuration from settings.yaml
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        settings_config = config_loader.get_config()
        self.training_mode_config = settings_config.get('training_mode', {})
        self.gpu_config = settings_config.get('gpu_optimization', {})
        
        # Determine training mode (debug vs final)
        if debug_mode is None:
            # Auto-detect from configuration
            current_mode = self.training_mode_config.get('mode', 'debug')
            self.debug_mode = (current_mode == 'debug')
        else:
            # Use explicit parameter
            self.debug_mode = debug_mode
        
        # Get mode-specific configuration
        mode_key = 'debug' if self.debug_mode else 'final'
        self.mode_config = self.training_mode_config.get(mode_key, {})
        
        # Set GPU optimization based on mode and configuration
        self.enable_gpu_optimization = (not self.debug_mode and 
                                      self.mode_config.get('gpu_optimization', False) and
                                      self.device.type == 'cuda')
        
        # Initialize training mode manager
        self.training_mode_manager = TrainingModeManager(
            self.training_mode_config, self.gpu_config
        )
        
        # Initialize GPU optimizer
        self.gpu_optimizer = GPUOptimizer(self.device, self.gpu_config)
        
        # Initialize offline RL preprocessor
        self.offline_rl_preprocessor = OfflineRLPreprocessor(self.device, self.gpu_config)
        
        # Initialize components
        self.data_loader = DataLoader(final_data_dir=data_path)
        self.loss_function = HRMLossFunction(self.config)
        
        # Training state
        self.current_episode = 0
        self.best_performance = float('-inf')
        self.training_history = []
        
        # Progress tracking
        self.epoch_start_time = None
        self.total_epochs = 0
        
        # Advanced GPU training settings
        self.gradient_accumulation_steps = self.config.get('training', {}).get('gradient_accumulation_steps', 1)
        self.effective_batch_size = self.gradient_accumulation_steps  # Will be multiplied by parallel envs
        self.accumulated_loss = 0.0
        self.accumulation_count = 0
        
        # Model and optimizer will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_training_complete(self, available_instruments: List[str] = None):
        """Complete training setup for multi-data training - no resets, no hanging"""
        
        # Load configuration to get initial capital and episode length
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        initial_capital = config.get('environment', {}).get('initial_capital', 100000.0)
        episode_length = config.get('environment', {}).get('episode_length', 1500)
        parallel_envs = config.get('environment', {}).get('parallel_environments', 10)
        
        # Store available instruments based on training mode
        all_instruments = available_instruments or ["Bank_Nifty_5"]
        self.available_instruments = get_instruments_for_mode(
            all_instruments, self.training_mode_manager
        )
        self.current_instrument_idx = 0
        self.episode_length = episode_length
        
        # Set parallel environments based on training mode
        self.parallel_envs = self.training_mode_manager.get_max_instruments_parallel()
        
        # Check if we should use parallel environments (only on GPU/TPU)
        self.use_parallel_envs = self._should_use_parallel_envs()
        
        if self.use_parallel_envs:
            # Create parallel environment manager
            logger.info(f"Initializing parallel environment manager with {parallel_envs} environments")
            self.parallel_env_manager = ParallelEnvironmentManager(
                data_loader=self.data_loader,
                symbols=self.available_instruments,
                config_path=self.config_path,
                device=str(self.device)
                # max_parallel_envs will be determined automatically based on device type and config
            )
            
            # Log which instruments are being used in parallel
            parallel_symbols = self.parallel_env_manager.get_environment_symbols()
            logger.info(f"ParallelGroup Training: {len(parallel_symbols)} instruments - {', '.join(parallel_symbols)}")
            
            # Initialize with first environment's data for compatibility
            first_symbol = self.parallel_env_manager.get_environment_symbols()[0]
            self.current_symbol = first_symbol
            self.full_data = self.data_loader.load_final_data_for_symbol(first_symbol)
            self.total_data_length = len(self.full_data)
            self.current_episode_start = 0
            
            # Create a dummy environment for compatibility with existing methods
            self.env = HRMTradingEnvironment(
                data_loader=self.data_loader,
                symbol=first_symbol,
                initial_capital=initial_capital,
                mode=TradingMode.TRAINING,
                hrm_config_path=self.config_path,
                device=self.device,
                episode_length=episode_length
            )
            
            # Set up first episode data segment
            self._setup_current_episode()
            
            # Initialize observation space using the dummy environment
            feature_columns = self.env.observation_handler.initialize_observation_space(self.env.data, config)
            self.env.observation_space = self.env.observation_handler.observation_space
            
            # Initialize HRM agent using the dummy environment
            self.env._initialize_hrm_agent()
            self.model = self.env.hrm_agent
            
            # Initialize all components manually
            self.env.engine.reset()
            self.env.reward_calculator.reset(self.env.initial_capital)
            self.env.observation_handler.reset()
            self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
            
        else:
            # Fall back to single environment training (CPU mode without parallel environments)
            logger.info("Using single environment training (CPU mode)")
            
            # Initialize with first instrument
            first_symbol = self.available_instruments[0]
            self.current_symbol = first_symbol
            
            # 1. Load full data for first instrument
            self.full_data = self.data_loader.load_final_data_for_symbol(first_symbol)
            
            # 2. Initialize episode tracking
            self.current_episode_start = 0
            self.total_data_length = len(self.full_data)
            
            # 3. Create environment with episode length config
            self.env = HRMTradingEnvironment(
                data_loader=self.data_loader,
                symbol=first_symbol,
                initial_capital=initial_capital,
                mode=TradingMode.TRAINING,
                hrm_config_path=self.config_path,
                device=self.device,
                episode_length=episode_length
            )
            
            # 4. Set up first episode data segment
            self._setup_current_episode()
            
            # 5. Initialize observation space
            feature_columns = self.env.observation_handler.initialize_observation_space(self.env.data, config)
            self.env.observation_space = self.env.observation_handler.observation_space
            
            # Initialize all components manually
            self.env.engine.reset()
            self.env.reward_calculator.reset(self.env.initial_capital)
            self.env.observation_handler.reset()
            self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
            
            # 4. Initialize HRM agent
            self.env._initialize_hrm_agent()
            self.model = self.env.hrm_agent
            
            # 5. Initialize HRM carry state
            self.env.current_carry = self.model.create_initial_carry(batch_size=1)
        
        # 6. Setup optimizer (same for both modes)
        self._setup_optimizer()
        
        # 7. Apply GPU optimizations to model
        if self.enable_gpu_optimization:
            self.model = self.gpu_optimizer.optimize_model(self.model)
            logger.info("GPU optimizations applied to model")
        
        # Log training mode information
        self.training_mode_manager.log_training_mode_info()
        
        logger.info(f"Training setup completed for {len(self.available_instruments)} instruments")
        logger.info(f"Parallel environments: {'ENABLED' if self.use_parallel_envs else 'DISABLED'}")
        logger.info(f"GPU optimization: {'ENABLED' if self.enable_gpu_optimization else 'DISABLED'}")
        logger.info("Memory-efficient mode: Data files loaded one at a time during training")
    
    def _should_use_parallel_envs(self) -> bool:
        """Check if parallel environments should be used based on device type and configuration"""
        # Use parallel environments on GPU, TPU, and CPU
        device_type = self.device_manager.get_device_type()
        if device_type not in ['gpu', 'tpu', 'cpu']:
            logger.info("Parallel environments disabled: Unsupported device type")
            return False
        
        # Check if parallel environments are configured
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # Get parallel environments setting based on device type
        parallel_envs_config = config.get('environment', {}).get('parallel_environments', {})
        if isinstance(parallel_envs_config, dict):
            # New format with separate settings for CPU and GPU/TPU
            if device_type == 'cpu':
                parallel_envs = parallel_envs_config.get('cpu', 5)
            else:  # gpu or tpu - MAXIMIZE GPU UTILIZATION
                parallel_envs = parallel_envs_config.get('gpu_tpu', 25)
        else:
            # Old format - fallback to GPU-optimized values
            parallel_envs = 5 if device_type == 'cpu' else 25
            logger.warning("Using legacy parallel environments configuration. Please update settings.yaml")
        
        # Ensure parallel_envs is an integer
        if not isinstance(parallel_envs, int):
            logger.warning(f"parallel_envs value {parallel_envs} is not an integer, defaulting to 5")
            parallel_envs = 5
        
        # Enable aggressive parallelization for GPU
        if device_type == 'gpu':
            # Force high GPU utilization
            parallel_envs = max(parallel_envs, 25)
            logger.info(f"GPU detected: Forcing high parallelization with {parallel_envs} environments for maximum VRAM utilization")
        
        # In debug mode, use parallel envs but modify logging behavior
        if self.debug_mode and parallel_envs > 1:
            logger.info(f"Debug mode with {parallel_envs} parallel environments: Detailed logs will show parallel instrument names")
            # Don't disable parallel envs in debug mode, just change logging behavior
        
        if parallel_envs <= 1:
            logger.info("Parallel environments disabled: Configuration set to 1 or less")
            return False
        
        logger.info(f"Parallel environments enabled: {parallel_envs} environments on {device_type.upper()}")
        return True
    
    def _setup_current_episode(self):
        """Set up data for current episode - single environment feeding"""
        
        # Calculate episode boundaries
        episode_end = min(self.current_episode_start + self.episode_length, self.total_data_length)
        
        # Get sequential data segment
        self.env.data = self.full_data.iloc[self.current_episode_start:episode_end].copy()
        
        # Set environment positions
        self.env.current_step = self.env.lookback_window - 1
        self.env.episode_end_step = len(self.env.data) - 1
        
        logger.info(f"Episode data: rows {self.current_episode_start} to {episode_end-1} "
                   f"({len(self.env.data)} rows)")
        if hasattr(self.env.data.index, 'strftime'):
            start_time = self.env.data.index[0].strftime('%Y-%m-%d %H:%M:%S')
            end_time = self.env.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"Time range: {start_time} to {end_time}")
    
    def _advance_to_next_episode(self):
        """Move to next episode within current data file (single environment mode)"""
        
        # Move to next episode segment within current data file
        self.current_episode_start += self.episode_length
        
        # Check if we've reached end of current data file
        if self.current_episode_start >= self.total_data_length - self.episode_length:
            # Move to next data file
            return self._advance_to_next_data_file()
        
        # Set up next episode data within same file
        self._setup_current_episode()
        
        # Reset environment state for new episode (but keep model parameters)
        self.env.engine.reset()
        self.env.reward_calculator.reset(self.env.initial_capital)
        self.env.observation_handler.reset()
        self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
        
        # Reset HRM carry state for new episode
        if hasattr(self.env, 'current_carry') and self.model:
            self.env.current_carry = self.model.create_initial_carry(batch_size=1)
        
        return False  # Not end of epoch
    
    def _advance_to_next_data_file(self):
        """Move to next data file in the training sequence"""
        
        # Move to next instrument
        self.current_instrument_idx += 1
        
        # Check if we've finished all instruments (end of epoch)
        if self.current_instrument_idx >= len(self.available_instruments):
            # Reset to first instrument for next epoch
            self.current_instrument_idx = 0
            logger.info("End of epoch reached - processed all data files")
            return True  # End of epoch
        
        # Load next data file
        next_symbol = self.available_instruments[self.current_instrument_idx]
        self.current_symbol = next_symbol
        
        logger.info(f"Advancing to next data file: {next_symbol} ({self.current_instrument_idx + 1}/{len(self.available_instruments)})")
        
        # Note: Data will be loaded in main training loop to avoid memory bloating
        # Just reset episode tracking here
        self.current_episode_start = 0
        
        # Update environment symbol (data will be loaded in main loop)
        self.env.symbol = next_symbol
        self.env._resolve_market_context()
        
        # Set up first episode of new data file
        # Note: _setup_current_episode will be called after data is loaded in main loop
        self.current_episode_start = 0
        
        # Reset environment for new data file (but keep model parameters)
        self.env.engine.reset()
        self.env.reward_calculator.reset(self.env.initial_capital)
        self.env.observation_handler.reset()
        self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
        
        # Reset HRM carry state for new data file
        if hasattr(self.env, 'current_carry') and self.model:
            self.env.current_carry = self.model.create_initial_carry(batch_size=1)
        
        return False  # Not end of epoch
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        training_config = self.config['training']
        
        # Different learning rates for different components
        param_groups = []
        
        # Strategic parameters (H-module) - slower learning
        strategic_params = list(self.model.high_level_module.parameters())
        param_groups.append({
            'params': strategic_params,
            'lr': training_config['learning_rates']['strategic_lr']
        })
        
        # Tactical parameters (L-module) - faster learning  
        tactical_params = list(self.model.low_level_module.parameters())
        param_groups.append({
            'params': tactical_params,
            'lr': training_config['learning_rates']['tactical_lr']
        })
        
        # ACT parameters - fast adaptation
        act_params = list(self.model.act_module.parameters())
        param_groups.append({
            'params': act_params,
            'lr': training_config['learning_rates']['act_lr']
        })
        
        # Deep supervision parameters
        ds_params = list(self.model.deep_supervision.parameters())
        param_groups.append({
            'params': ds_params,
            'lr': training_config['learning_rates']['base_lr']
        })
        
        # Create optimizer
        if training_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=training_config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(param_groups)
        
        # Learning rate scheduler
        if training_config['lr_schedule'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('total_episodes', 1000),
                eta_min=training_config['learning_rates']['base_lr'] * training_config['min_lr_ratio']
            )
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode"""
        
        # Print debug header for parallel training only
        if self.debug_mode and self.use_parallel_envs:
            # For parallel training, show instrument names being trained
            parallel_symbols = self.parallel_env_manager.get_environment_symbols()
            instrument_list = ', '.join([sym.split('_')[0] for sym in parallel_symbols])
            print(f"\nEPOCH {getattr(self, 'current_episode', 0) + 1} PARALLEL TRAINING: {len(parallel_symbols)} instruments")
            print(f"Instruments: {instrument_list}")
            print("-" * 100)
        
        # Environment is already initialized in setup_training_complete, no reset needed
        
        episode_metrics = {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'steps': 0,
            'hierarchical_metrics': {}
        }
        
        done = False
        step_count = 0
        
        if self.use_parallel_envs:
            # Parallel environment training
            return self._train_episode_parallel()
        else:
            # Single environment training (original method)
            while not done and step_count < 1000:  # Max steps per episode
                
                # Environment step (HRM handles the reasoning internally)
                observation, reward, done, info = self.env.step()
                
                episode_metrics['total_reward'] += reward
                step_count += 1
                
                # Single environment debug logging removed as requested
                
                # Extract loss information if available in info
                if 'segment_rewards' in info:
                    # Collect losses for batched processing
                    if not hasattr(self, '_loss_buffer'):
                        self._loss_buffer = []
                    
                    # Create self-supervised targets from model outputs
                    # This enables the model to learn from its own behavior and outcomes
                    if hasattr(self.env, 'hrm_agent') and hasattr(self.env.hrm_agent, '_last_outputs'):
                        outputs = getattr(self.env.hrm_agent, '_last_outputs', {})
                        
                        if outputs:
                            targets = self._create_training_targets(info, outputs)
                            
                            loss, loss_components = self.loss_function.calculate_loss(
                                outputs, 
                                targets, 
                                info['segment_rewards']
                            )
                            
                            self._loss_buffer.append(loss)
            
            # Batch process accumulated losses
            if hasattr(self, '_loss_buffer') and self._loss_buffer:
                # Single backward pass for all collected losses
                self.optimizer.zero_grad()
                total_loss = torch.stack(self._loss_buffer).mean()
                total_loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
                
                episode_metrics['total_loss'] += total_loss.item()
                self._loss_buffer = []  # Clear buffer
            
            episode_metrics['steps'] = step_count
            episode_metrics['avg_reward'] = episode_metrics['total_reward'] / max(step_count, 1)
            episode_metrics['avg_loss'] = episode_metrics['total_loss'] / max(step_count, 1)
            
            # Get hierarchical performance metrics
            episode_metrics['hierarchical_metrics'] = self.env.get_hierarchical_performance_metrics()
            
            return episode_metrics
    
    def _train_episode_parallel(self) -> Dict[str, float]:
        """Train for one episode using parallel environments"""
        
        episode_metrics = {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'steps': 0,
            'hierarchical_metrics': {}
        }
        
        done = False
        step_count = 0
        max_steps = 1000
        
        # Reset all parallel environments
        batch_obs = self.parallel_env_manager.reset_all()
        
        while not done and step_count < max_steps:
            # Step all environments in parallel
            batch_obs, batch_rewards, batch_dones, batch_infos = self.parallel_env_manager.step_all()
            
            # Log step-by-step instrument data if in debug mode
            if self.debug_mode:
                self._log_parallel_step_debug(step_count, batch_rewards, batch_infos)
            
            # Process batch rewards and infos
            episode_metrics['total_reward'] += batch_rewards.mean().item()
            step_count += 1
            
            # Batch process all environment losses together for better GPU utilization - VECTORIZED
            batch_losses = []
            batch_targets = []
            batch_outputs = []
            
            # Collect all losses from environments for vectorized processing
            for i, info in enumerate(batch_infos):
                if 'segment_rewards' in info and hasattr(self.env, 'hrm_agent') and hasattr(self.env.hrm_agent, '_last_outputs'):
                    outputs = getattr(self.env.hrm_agent, '_last_outputs', {})
                    
                    if outputs:
                        targets = self._create_training_targets(info, outputs)
                        batch_outputs.append(outputs)
                        batch_targets.append(targets)
            
            # Vectorized loss computation for all environments at once
            if batch_outputs:
                total_loss = self._compute_vectorized_batch_loss(batch_outputs, batch_targets, 
                                                               [info['segment_rewards'] for info in batch_infos 
                                                                if 'segment_rewards' in info])
            
            # Advanced GPU training with gradient accumulation and mixed precision
            if batch_outputs:
                # Scale loss for gradient accumulation
                scaled_loss = total_loss / self.gradient_accumulation_steps
                self.accumulated_loss += scaled_loss.item()
                self.accumulation_count += 1
                
                # Use mixed precision if available for better GPU utilization
                if self.scaler is not None and torch.cuda.is_available():
                    # Mixed precision backward pass with gradient accumulation
                    self.scaler.scale(scaled_loss).backward()
                else:
                    # Standard precision backward pass with gradient accumulation
                    scaled_loss.backward()
                
                # Perform optimizer step when gradient accumulation is complete
                if self.accumulation_count >= self.gradient_accumulation_steps:
                    if self.scaler is not None and torch.cuda.is_available():
                        # Mixed precision optimizer step
                        if self.config['training']['gradient_clip'] > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Standard precision optimizer step
                        if self.config['training']['gradient_clip'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        
                        self.optimizer.step()
                    
                    # Reset for next accumulation cycle
                    self.optimizer.zero_grad()
                    episode_metrics['total_loss'] += self.accumulated_loss
                    self.accumulated_loss = 0.0
                    self.accumulation_count = 0
            
            # Check if all environments are done
            done = batch_dones.all().item()
        
        episode_metrics['steps'] = step_count
        episode_metrics['avg_reward'] = episode_metrics['total_reward'] / max(step_count, 1)
        episode_metrics['avg_loss'] = episode_metrics['total_loss'] / max(step_count, 1)
        
        # For parallel environments, we'll return metrics from the first environment
        # In a more advanced implementation, we could aggregate metrics from all environments
        episode_metrics['hierarchical_metrics'] = self.env.get_hierarchical_performance_metrics()
        
        return episode_metrics
    
    def _log_parallel_step_debug(self, step_num: int, batch_rewards: torch.Tensor, batch_infos: List[Dict]):
        """Log minimal step-by-step info for parallel training showing actual PnL in rupees instead of percentage"""
        env_symbols = self.parallel_env_manager.get_environment_symbols()
        
        # Show header every 20 steps for readability
        if step_num == 1 or step_num % 20 == 1:
            print(f"\n{'='*140}")
            print(f"PARALLEL TRAINING - Instruments: {', '.join([sym.split('_')[0] for sym in env_symbols])}")
            print(f"{'='*140}")
            print(f"{'Step':<5} {'Instrument':<12} {'TF':<4} {'Initial':<10} {'Current':<10} {'P&L(Rs)':<10} {'Win%':<5} {'Reward':<8}")
            print("-" * 140)
        
        # Show minimal info for each instrument every step
        for i, (symbol, info) in enumerate(zip(env_symbols, batch_infos)):
            # Extract instrument and timeframe
            parts = symbol.split('_')
            if len(parts) >= 2:
                instrument = '_'.join(parts[:-1])[:11]  # Truncate for display
                timeframe = parts[-1]
            else:
                instrument = symbol[:11]
                timeframe = "?"
            
            # Extract performance info
            initial_cap = info.get('initial_capital', 100000.0)
            account_state = info.get('account_state', {})
            current_cap = account_state.get('capital', initial_cap)
            
            # Calculate actual PnL in rupees instead of percentage
            pnl_rupees = current_cap - initial_cap
            
            # Get win rate
            win_rate = info.get('win_rate', 0.0) * 100
            
            # Get reward
            reward = batch_rewards[i].item() if i < len(batch_rewards) else 0.0
            
            print(f"{step_num:<5} {instrument:<12} {timeframe:<4} {initial_cap:<10,.0f} {current_cap:<10,.0f} ₹{pnl_rupees:<+9,.0f} {win_rate:<5.1f} {reward:<8.3f}")
        
        # Show average after all instruments
        avg_reward = batch_rewards.mean().item()
        avg_pnl = sum((info.get('account_state', {}).get('capital', 100000.0) - info.get('initial_capital', 100000.0)) for info in batch_infos) / max(1, len(batch_infos))
        print(f"{'':>5} {'AVG':<12} {'':>4} {'':>10} {'':>10} ₹{avg_pnl:<+9,.0f} {'':>5} {avg_reward:<8.3f}")
    
    def _log_step_debug(self, step_num: int, reward: float, info: Dict):
        """Log detailed step information using environment as single source of truth.
        Only logs for single instrument training to avoid conflicts during parallel training."""
        
        # Skip detailed logging for parallel training to avoid conflicts
        if self.use_parallel_envs:
            return
        
        # Extract all information from environment's info dict
        instrument_name = getattr(self.env, 'symbol', 'Unknown')
        timeframe = 'Unknown'
        if hasattr(self.env, 'symbol'):
            parts = self.env.symbol.split('_')
            if len(parts) >= 2:
                instrument_name = '_'.join(parts[:-1])
                timeframe = f"{parts[-1]}min"
        
        # Get datetime from environment info
        datetime_readable = 'N/A'
        if 'datetime' in info and hasattr(info['datetime'], 'strftime'):
            datetime_readable = info['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        
        # Get all data from environment info
        action_taken = info.get('action', 'HOLD')
        quantity = info.get('quantity', 0.0)
        account_state = info.get('account_state', {})
        initial_capital = info.get('initial_capital', 100000.0)
        current_capital = account_state.get('capital', 100000.0)
        is_position_open = account_state.get('is_position_open', False)
        entry_price = account_state.get('current_position_entry_price', 0.0)
        current_price = info.get('current_price', 0.0)  # Get current price from info
        
        # Get win rate from environment calculation
        win_rate = info.get('win_rate', 0.0) * 100  # Convert to percentage
        
        # Get P&L information from environment
        if is_position_open:
            # Show unrealized P&L when position is open
            profit_loss = info.get('unrealized_pnl', 0.0)
        elif info.get('position_closed', False):
            # Position was just closed - show the realized P&L of this trade
            engine_decision = info.get('engine_decision', {})
            profit_loss = engine_decision.get('realized_pnl', 0.0)
        else:
            # No position and no recent closure - show no P&L
            profit_loss = 0.0
        
        # Show quantity only when position was actually opened this step
        if info.get('position_opened', False):
            # Position was actually opened this step, show quantity
            action_str = f"{action_taken}-{quantity:.1f}"
        else:
            # No position opened (invalid action, hold, rejected, etc.) - don't show quantity
            action_str = action_taken
        
        # Format price displays using environment data
        if is_position_open and entry_price > 0:
            entry_display = f"Rs{entry_price:7.2f}"
            current_price_display = f"Rs{current_price:7.2f}"  # Format current price
            pnl_display = f"Rs{profit_loss:+8.0f}"
            
            # Get risk management prices from environment info
            target_price = info.get('target_price', 0.0)
            sl_price = info.get('sl_price', 0.0)
            
            if target_price > 0 and sl_price > 0:
                # Calculate actual points difference from entry
                target_points = target_price - entry_price
                sl_points = entry_price - sl_price
                target_display = f"Rs{target_price:7.2f}({target_points:+4.0f})"
                sl_display = f"Rs{sl_price:7.2f}({-sl_points:+4.0f})"
            else:
                target_display = "        -       "
                sl_display = "        -       "
        else:
            # No position - show dashes or total P&L if we just closed
            entry_display = "     -     "
            current_price_display = f"Rs{current_price:7.2f}"  # Show current price even when no position
            if profit_loss != 0:
                pnl_display = f"Rs{profit_loss:+8.0f}"
            else:
                pnl_display = "Rs      -"
            target_display = "        -       "
            sl_display = "        -       "
        
        # Get reason based on actual position changes, not engine decisions
        reason = "-"
        
        # Check if position actually changed this step
        if info.get('position_opened', False):
            reason = "ENTRY"
        elif info.get('position_closed', False):
            # Check if it was an automated exit (SL/Trail) or model decision
            engine_decision = info.get('engine_decision', {})
            exit_reason = engine_decision.get('exit_reason', '')
            if 'STOP_LOSS' in str(exit_reason):
                reason = "SL_HIT"
            elif 'TRAILING' in str(exit_reason):
                reason = "TRAIL_HIT"
            else:
                reason = "MODEL_EXIT"
        # For all other cases (no position change, holds, rejected actions), reason stays "-"
        
        # Format single line log with current price added before entry price
        log_line = (f"{step_num:4d} | {instrument_name:14s} | {timeframe:9s} | {datetime_readable} | "
                   f"Rs{initial_capital:,.0f} | Rs{current_capital:,.0f} | "
                   f"{action_str:8s} | {win_rate:5.1f}% | "
                   f"{pnl_display} | {current_price_display} | {entry_display} | "
                   f"{target_display} | {sl_display} | {reward:+6.3f} | {reason}")
        
        print(log_line)
    
    def _create_training_targets(self, info: Dict, model_outputs: Dict) -> Dict[str, torch.Tensor]:
        """
        Create training targets for HRM loss function
        Uses expert targets from technical indicators when available, falls back to self-supervised
        """
        from src.utils.target_generator import create_training_targets
        
        # Try to get current data row from environment for expert targets
        data_row = None
        try:
            if (hasattr(self, 'env') and 
                hasattr(self.env, 'data') and 
                hasattr(self.env, 'current_step') and
                self.env.data is not None and
                len(self.env.data) > 0):
                safe_step = max(0, min(self.env.current_step, len(self.env.data) - 1))
                if safe_step >= 0 and safe_step < len(self.env.data):
                    data_row = self.env.data.iloc[safe_step]
                    # Check if data_row is valid
                    if data_row is None or data_row.empty:
                        data_row = None
        except Exception as e:
            # If we can't get the data row, fall back to self-supervised targets
            data_row = None
        
        targets = create_training_targets(info, model_outputs, data_row)
        
        # Ensure targets are on the correct device
        device = next(iter(model_outputs.values())).device if model_outputs else torch.device('cpu')
        for key, value in targets.items():
            if isinstance(value, torch.Tensor):
                targets[key] = value.to(device)
        
        return targets
    
    def _compute_vectorized_batch_loss(self, batch_outputs: List[Dict], batch_targets: List[Dict], 
                                     batch_segment_rewards: List[List[float]]) -> torch.Tensor:
        """
        Compute loss for multiple environments using vectorized GPU operations with mixed precision
        This significantly improves GPU utilization and reduces CPU-GPU transfer overhead
        """
        if not batch_outputs:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Mixed precision context for better GPU performance
        if self.scaler is not None and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                return self._compute_batch_loss_internal(batch_outputs, batch_targets, batch_segment_rewards)
        else:
            return self._compute_batch_loss_internal(batch_outputs, batch_targets, batch_segment_rewards)
    
    def _compute_batch_loss_internal(self, batch_outputs: List[Dict], batch_targets: List[Dict], 
                                   batch_segment_rewards: List[List[float]]) -> torch.Tensor:
        """Internal loss computation with proper GPU optimization"""
        # Stack all outputs and targets into batch tensors for vectorized computation
        batch_size = len(batch_outputs)
        total_batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Process each component loss type across all environments simultaneously
        loss_components = {}
        
        # Vectorized market understanding loss
        if all('market_understanding_outputs' in outputs for outputs in batch_outputs):
            market_outputs = torch.stack([outputs['market_understanding_outputs'] for outputs in batch_outputs])
            market_targets = torch.stack([targets.get('market_understanding', torch.zeros_like(outputs['market_understanding_outputs'])) 
                                        for outputs, targets in zip(batch_outputs, batch_targets)])
            
            if market_targets.numel() > 0:
                understanding_loss = nn.MSELoss()(market_outputs, market_targets)
                loss_components['market_understanding_loss'] = understanding_loss
                total_batch_loss = total_batch_loss + self.loss_function.loss_weights.get('tactical_loss', 1.0) * understanding_loss
        
        # Vectorized ACT loss computation
        if all('halt_logits' in outputs and 'continue_logits' in outputs for outputs in batch_outputs):
            halt_logits_batch = torch.stack([outputs['halt_logits'] for outputs in batch_outputs])
            continue_logits_batch = torch.stack([outputs['continue_logits'] for outputs in batch_outputs])
            halt_targets_batch = torch.stack([outputs['q_halt_target'].detach() for outputs in batch_outputs])
            continue_targets_batch = torch.stack([outputs['q_continue_target'].detach() for outputs in batch_outputs])
            
            # Ensure tensor compatibility
            if halt_logits_batch.shape != halt_targets_batch.shape:
                halt_targets_batch = halt_targets_batch.view_as(halt_logits_batch)
            if continue_logits_batch.shape != continue_targets_batch.shape:
                continue_targets_batch = continue_targets_batch.view_as(continue_logits_batch)
            
            halt_loss = nn.MSELoss()(halt_logits_batch, halt_targets_batch)
            continue_loss = nn.MSELoss()(continue_logits_batch, continue_targets_batch)
            
            act_loss = halt_loss + continue_loss
            loss_components['act_loss'] = act_loss
            total_batch_loss = total_batch_loss + self.loss_function.loss_weights['act_loss'] * act_loss
        
        # Vectorized entropy loss
        if all('action_probabilities' in outputs for outputs in batch_outputs):
            action_probs_batch = torch.stack([outputs['action_probabilities'] for outputs in batch_outputs])
            # Compute entropy across batch dimension for better GPU utilization
            entropy_batch = -torch.sum(action_probs_batch * torch.log(action_probs_batch + 1e-8), dim=-1)
            entropy_loss = -torch.mean(entropy_batch)  # Mean across batch
            loss_components['entropy_loss'] = entropy_loss
            total_batch_loss = total_batch_loss + self.loss_function.loss_weights.get('entropy_loss', 0.01) * entropy_loss
        
        # Vectorized performance loss - GPU optimized
        if batch_segment_rewards:
            # Flatten all segment rewards and convert to GPU tensor in one operation
            all_rewards = [reward for rewards_list in batch_segment_rewards for reward in rewards_list]
            if all_rewards:
                rewards_tensor = torch.tensor(all_rewards, device=self.device, dtype=torch.float32)
                performance_loss = -torch.mean(rewards_tensor)
                loss_components['performance_loss'] = performance_loss
                total_batch_loss = total_batch_loss + self.loss_function.loss_weights['performance_loss'] * performance_loss
        
        # Return averaged loss across batch
        return total_batch_loss / max(1, batch_size)
    
    def _create_batch_episode_data(self, all_datasets: Dict[str, pd.DataFrame]) -> List[List[Dict]]:
        """
        Create batched episode data for true simultaneous processing
        Returns list of batches, each containing episode data from multiple datasets
        """
        batch_episode_data = []
        max_episodes_per_dataset = {}
        
        # Calculate how many episodes each dataset can provide
        for symbol, dataset in all_datasets.items():
            episodes_count = max(1, (len(dataset) - 1) // self.episode_length)
            max_episodes_per_dataset[symbol] = episodes_count
        
        total_episodes = sum(max_episodes_per_dataset.values())
        # Ensure parallel_envs is an integer
        parallel_envs = self.parallel_envs if isinstance(self.parallel_envs, int) else 4
        batch_size = min(len(all_datasets), parallel_envs)  # Use available parallel environments
        
        logger.info(f"Creating batched episodes: {total_episodes} total episodes across {len(all_datasets)} datasets")
        
        # Create episode batches that mix data from different datasets
        current_episode_indices = {symbol: 0 for symbol in all_datasets.keys()}
        
        while any(current_episode_indices[symbol] < max_episodes_per_dataset[symbol] 
                 for symbol in all_datasets.keys()):
            
            batch_data = []
            
            # Create a batch by taking one episode from each available dataset
            for symbol in all_datasets.keys():
                if current_episode_indices[symbol] < max_episodes_per_dataset[symbol]:
                    episode_start = current_episode_indices[symbol] * self.episode_length
                    episode_end = min(episode_start + self.episode_length, len(all_datasets[symbol]))
                    
                    episode_data = {
                        'symbol': symbol,
                        'data': all_datasets[symbol].iloc[episode_start:episode_end].copy(),
                        'episode_idx': current_episode_indices[symbol]
                    }
                    batch_data.append(episode_data)
                    current_episode_indices[symbol] += 1
                    
                    # Break if we've filled the batch
                    if len(batch_data) >= batch_size:
                        break
            
            if batch_data:
                batch_episode_data.append(batch_data)
        
        logger.info(f"Created {len(batch_episode_data)} batches with avg {sum(len(b) for b in batch_episode_data)/len(batch_episode_data):.1f} episodes per batch")
        return batch_episode_data
    
    def _train_batch_episodes(self, batch_data: List[Dict]) -> List[Dict]:
        """
        Train on a batch of episodes from different datasets simultaneously
        This is the core method for true batch processing
        """
        batch_metrics = []
        
        # Setup parallel environments for this batch
        batch_envs = []
        
        for i, episode_data in enumerate(batch_data):
            # Create environment for this episode
            try:
                env = HRMTradingEnvironment(
                    data_loader=self.data_loader,
                    symbol=episode_data['symbol'],
                    initial_capital=self.config.get('environment', {}).get('initial_capital', 100000.0),
                    mode=TradingMode.TRAINING,
                    hrm_config_path=self.config_path,
                    device=str(self.device),
                    episode_length=self.episode_length
                )
            except Exception as e:
                logger.error(f"Environment creation failed: {e}")
                continue
            
            # Set the episode data
            env.data = episode_data['data']
            env.current_step = env.lookback_window - 1
            env.episode_end_step = len(env.data) - 1
            
            # Initialize environment components
            try:
                env.engine.reset()
                env.reward_calculator.reset(env.initial_capital)
                env.observation_handler.reset()
                env.termination_manager.reset(env.initial_capital, env.episode_end_step)
            except Exception as e:
                logger.error(f"Component initialization failed: {e}")
                continue
            
            # Reset the environment to initialize observation space
            try:
                env.reset()
            except Exception as e:
                logger.error(f"Environment reset failed: {e}")
                continue
            
            # Initialize HRM agent if needed (after reset to ensure observation space is set)
            try:
                if not hasattr(env, 'hrm_agent') or env.hrm_agent is None:
                    env._initialize_hrm_agent()
            except Exception as e:
                logger.error(f"HRM agent initialization failed: {e}")
                continue
            
            batch_envs.append(env)
        
        if not batch_envs:
            logger.warning(f"No environments created successfully, skipping batch")
            return []
        
        # Run batch training episode
        episode_metrics = {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'steps': 0,
            'hierarchical_metrics': {}
        }
        
        done_mask = [False] * len(batch_envs)
        step_count = 0
        max_steps = 1000
        
        # Train all environments in batch simultaneously
        while not all(done_mask) and step_count < max_steps:
            # Collect observations from all environments
            batch_observations = []
            batch_rewards = []
            batch_dones = []
            batch_infos = []
            
            for i, env in enumerate(batch_envs):
                if not done_mask[i]:
                    try:
                        # Generate action using HRM agent
                        current_obs = env.observation_handler.get_observation(env.data, env.current_step, env.engine)
                        
                        # Use HRM agent to select action
                        if hasattr(env, 'hrm_agent') and env.hrm_agent is not None:
                            try:
                                with torch.no_grad():
                                    obs_tensor = torch.tensor(current_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                                    
                                    # Create initial carry state if not exists
                                    if not hasattr(env, '_hrm_carry'):
                                        from src.models.hrm.hrm_agent import HRMCarry, HRMTradingState
                                        initial_state = HRMTradingState(
                                            z_H=torch.zeros(1, env.hrm_agent.hidden_dim, device=self.device),
                                            z_L=torch.zeros(1, env.hrm_agent.hidden_dim, device=self.device),
                                            step_count=0,
                                            segment_count=0
                                        )
                                        env._hrm_carry = HRMCarry(
                                            inner_state=initial_state,
                                            halted=torch.zeros(1, dtype=torch.bool, device=self.device),
                                            performance_history=[]
                                        )
                                    
                                    # Get instrument and timeframe IDs
                                    instrument_id = torch.tensor([0], device=self.device)  # Default to 0
                                    timeframe_id = torch.tensor([0], device=self.device)   # Default to 0
                                    
                                    # Forward pass
                                    new_carry, outputs = env.hrm_agent.forward(
                                        carry=env._hrm_carry,
                                        observation=obs_tensor,
                                        training=False,
                                        instrument_id=instrument_id,
                                        timeframe_id=timeframe_id
                                    )
                                    
                                    # Update carry state
                                    env._hrm_carry = new_carry
                                    
                                    # Extract trading decision
                                    account_state = env.engine.get_account_state()
                                    decision = env.hrm_agent.extract_trading_decision(
                                        outputs,
                                        current_position=account_state.get('position_quantity', 0.0),
                                        available_capital=account_state.get('capital', 100000.0)
                                    )
                                    
                                    # Convert decision to action index using settings
                                    action_name = decision.get('action', 'HOLD')
                                    from src.config.settings import get_settings
                                    settings = get_settings()
                                    action_mapping = settings['actions']['action_types']
                                    action = action_mapping.get(action_name, action_mapping.get('HOLD', 4))  # Default to HOLD
                                    
                            except Exception as e:
                                logger.warning(f"HRM agent error: {e}, using random action")
                                action = env.action_space.sample()
                        else:
                            # Fallback to random action
                            action = env.action_space.sample()
                        
                        # Execute step with action
                        obs, reward, done, info = env.step(action)
                        batch_observations.append(obs)
                        batch_rewards.append(reward)
                        batch_dones.append(done)
                        batch_infos.append(info)
                        done_mask[i] = done
                            
                    except Exception as e:
                        logger.warning(f"Error in batch environment {i}: {e}")
                        done_mask[i] = True
                        batch_observations.append(np.zeros(100))  # Default obs
                        batch_rewards.append(0.0)
                        batch_dones.append(True)
                        batch_infos.append({})
            
            if batch_observations:
                # Convert to tensors for batch processing
                batch_obs_tensor = torch.tensor(np.array(batch_observations), dtype=torch.float32, device=self.device)
                batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
                
                # Process batch loss
                self._process_batch_training_step(batch_obs_tensor, batch_rewards_tensor, batch_infos)
                
                # Update metrics
                episode_metrics['total_reward'] += batch_rewards_tensor.mean().item()
                step_count += 1
        
        # Create metrics for each environment in the batch
        for i, env in enumerate(batch_envs):
            metrics = {
                'total_reward': episode_metrics['total_reward'] / len(batch_envs),
                'total_loss': episode_metrics['total_loss'] / len(batch_envs),
                'steps': step_count,
                'avg_reward': (episode_metrics['total_reward'] / len(batch_envs)) / max(step_count, 1),
                'avg_loss': (episode_metrics['total_loss'] / len(batch_envs)) / max(step_count, 1),
                'hierarchical_metrics': env.get_hierarchical_performance_metrics() if hasattr(env, 'get_hierarchical_performance_metrics') else {}
            }
            batch_metrics.append(metrics)
        
        return batch_metrics
    
    def _process_batch_training_step(self, batch_obs: torch.Tensor, batch_rewards: torch.Tensor, batch_infos: List[Dict]):
        """Process a single training step for the entire batch"""
        # This method handles the loss computation and optimization for the batch
        # Similar to the parallel environment training but optimized for true batching
        
        if hasattr(self.env, 'hrm_agent') and hasattr(self.env.hrm_agent, '_last_outputs'):
            outputs = getattr(self.env.hrm_agent, '_last_outputs', {})
            
            if outputs and batch_infos:
                # Create batch targets
                batch_targets = []
                batch_segment_rewards = []
                
                for info in batch_infos:
                    if 'segment_rewards' in info:
                        targets = self._create_training_targets(info, outputs)
                        batch_targets.append(targets)
                        batch_segment_rewards.append(info['segment_rewards'])
                
                if batch_targets:
                    # Compute vectorized batch loss
                    batch_outputs = [outputs] * len(batch_targets)  # Same outputs for all in batch
                    total_loss = self._compute_vectorized_batch_loss(batch_outputs, batch_targets, batch_segment_rewards)
                    
                    # Advanced GPU training with gradient accumulation and mixed precision
                    if total_loss.requires_grad:
                        # Scale loss for gradient accumulation
                        scaled_loss = total_loss / self.gradient_accumulation_steps
                        self.accumulated_loss += scaled_loss.item()
                        self.accumulation_count += 1
                        
                        # Use mixed precision if available
                        if self.scaler is not None and torch.cuda.is_available():
                            self.scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                        
                        # Perform optimizer step when gradient accumulation is complete
                        if self.accumulation_count >= self.gradient_accumulation_steps:
                            if self.scaler is not None and torch.cuda.is_available():
                                if self.config['training']['gradient_clip'] > 0:
                                    self.scaler.unscale_(self.optimizer)
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                if self.config['training']['gradient_clip'] > 0:
                                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                                self.optimizer.step()
                            
                            self.optimizer.zero_grad()
                            self.accumulated_loss = 0.0
                            self.accumulation_count = 0
    
    def train(self, 
              epochs: int = 100, 
              available_instruments: List[str] = None,
              save_frequency: int = 25,
              log_frequency: int = 5):
        """Main training loop with multi-data per epoch architecture"""
        
        # Setup training only if not already done
        if self.model is None:
            if available_instruments is None:
                raise ValueError("Available instruments must be provided for initial setup")
            self.setup_training_complete(available_instruments)
        
        self.total_epochs = epochs
        logger.info(f"Starting HRM training for {epochs} epochs")
        logger.info(f"Training on {len(self.available_instruments)} instruments per epoch")
        logger.info(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        logger.info(f"Device: {self.device}")
        
        # Calculate total steps for progress tracking (memory-efficient)
        total_episodes = 0
        logger.info("Calculating total episodes (loading data headers only)...")
        for symbol in self.available_instruments:
            # Load only to get length, then immediately release memory
            data = self.data_loader.load_final_data_for_symbol(symbol)
            episodes_per_file = max(1, (len(data) - 1) // self.episode_length)
            total_episodes += episodes_per_file
            del data  # Immediately free memory
        
        total_steps = epochs * total_episodes
        logger.info(f"Total training steps: {epochs} epochs × {total_episodes} episodes = {total_steps} steps")
        logger.info("Memory-efficient training: Loading only one data file at a time")
        
        # Initialize metrics tracking
        running_metrics = {
            'reward': [],
            'loss': [],
            'best_reward': float('-inf'),
            'total_time': 0,
            'current_epoch': 0,
            'current_file': 0,
            'current_episode_in_file': 0
        }
        
        # Create progress bar
        if not self.debug_mode:
            progress_bar = tqdm(
                total=total_steps,
                desc="Training Progress",
                unit="step",
                ncols=150,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        # Enable advanced GPU optimizations
        if self.device_manager.get_device_type() == 'gpu':
            # Advanced GPU memory management
            torch.cuda.empty_cache()
            
            # Enable optimized attention and memory efficient attention if available
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Enable TensorFloat-32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable optimized pooling
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Enable advanced GPU memory management
            if hasattr(torch.cuda, 'memory_pool_stats') and hasattr(torch.cuda, 'set_memory_fraction'):
                # Reserve GPU memory for stable performance
                torch.cuda.set_memory_fraction(0.95)  # Use 95% of GPU memory
            
            logger.info("Advanced GPU optimizations enabled:")
            logger.info("  - TensorFloat-32 (TF32) for Ampere GPUs")
            logger.info("  - Optimized SDPA backends")
            logger.info("  - CuDNN benchmarking")
            logger.info("  - 95% GPU memory allocation")
        
        start_time = time.time()
        global_step = 0
        
        for epoch in range(epochs):
            running_metrics['current_epoch'] = epoch + 1
            epoch_start_time = time.time()
            epoch_metrics = {'reward': [], 'loss': []}
            
            logger.info(f"Starting Epoch {epoch + 1}/{epochs}")
            
            # Reset to first instrument for each epoch
            self.current_instrument_idx = 0
            
            # TRUE BATCH PROCESSING - All datasets loaded and processed simultaneously
            if self.use_parallel_envs:
                logger.info("🚀 TRUE BATCH MODE: Loading and processing ALL datasets simultaneously for maximum parallelization")
                
                # Load ALL datasets at once for true batch processing
                all_datasets = {}
                total_memory_usage = 0
                
                # Determine how many datasets we can load based on available memory
                if self.device_manager.get_device_type() == 'gpu':
                    # GPU: Be more aggressive with memory usage
                    target_memory_gb = self.device_manager.get_device_info().get('memory_total_gb', 15) * 0.85  # Use 85% of VRAM
                    max_datasets = len(self.available_instruments)  # Try to load all datasets
                else:
                    # CPU: Be more conservative
                    target_memory_gb = 8.0  # Assume 8GB available for datasets
                    max_datasets = min(len(self.available_instruments), 4)  # Limit to 4 datasets on CPU
                
                logger.info(f"Loading up to {max_datasets} datasets simultaneously (Target: {target_memory_gb:.1f}GB)")
                
                # Load all datasets simultaneously
                loaded_count = 0
                for symbol in self.available_instruments[:max_datasets]:
                    try:
                        dataset = self.data_loader.load_final_data_for_symbol(symbol)
                        dataset_memory = dataset.memory_usage(deep=True).sum() / 1024**3  # GB
                        
                        if total_memory_usage + dataset_memory < target_memory_gb:
                            all_datasets[symbol] = dataset
                            total_memory_usage += dataset_memory
                            loaded_count += 1
                            logger.info(f"✓ Batch loaded {symbol}: {dataset_memory:.2f}GB (Total: {total_memory_usage:.2f}GB)")
                        else:
                            logger.info(f"⚠️ Skipping {symbol}: Would exceed memory limit")
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load {symbol}: {e}")
                
                # Setup parallel environment manager with all loaded datasets
                if all_datasets:
                    # Update parallel environment manager to use all loaded datasets
                    logger.info(f"🔄 Configuring parallel environments for {loaded_count} datasets")
                    
                    # Create batched data loaders for simultaneous processing
                    batch_episode_data = self._create_batch_episode_data(all_datasets)
                    
                    # Process all datasets in true batch mode
                    for batch_idx, batch_data in enumerate(batch_episode_data):
                        logger.info(f"Processing batch {batch_idx + 1}/{len(batch_episode_data)} with {len(batch_data)} episodes")
                        
                        # Train on entire batch simultaneously
                        batch_metrics = self._train_batch_episodes(batch_data)
                        
                        # Aggregate metrics from all episodes in batch
                        for metrics in batch_metrics:
                            self.training_history.append(metrics)
                            epoch_metrics['reward'].append(metrics['avg_reward'])
                            epoch_metrics['loss'].append(metrics['avg_loss'])
                            
                            running_metrics['reward'].append(metrics['avg_reward'])
                            running_metrics['loss'].append(metrics['avg_loss'])
                            if metrics['avg_reward'] > running_metrics['best_reward']:
                                running_metrics['best_reward'] = metrics['avg_reward']
                            
                            global_step += 1
                        
                        # Update progress bar
                        if not self.debug_mode:
                            avg_reward = torch.tensor([m['avg_reward'] for m in batch_metrics]).mean().item()
                            progress_bar.set_postfix({
                                'Epoch': f"{epoch+1}/{epochs}",
                                'Batch': f"{batch_idx+1}/{len(batch_episode_data)}",
                                'Datasets': loaded_count,
                                'Avg_Reward': f"{avg_reward:.4f}",
                                'Best': f"{running_metrics['best_reward']:.4f}",
                                'Memory': f"{total_memory_usage:.1f}GB"
                            })
                            progress_bar.update(len(batch_metrics))
                    
                    logger.info(f"✅ Completed true batch processing of {loaded_count} datasets")
                    break
                else:
                    logger.error("No datasets could be loaded for batch processing!")
                    # Fall back to sequential processing
                    logger.info("Falling back to sequential processing...")
            
            else:
                # FALLBACK MODE: Sequential processing (original approach)
                while True:
                    # Load current data file (memory-efficient: only one at a time)
                    current_symbol = self.available_instruments[self.current_instrument_idx]
                    self.current_symbol = current_symbol
                    
                    logger.info(f"Loading data file: {current_symbol}")
                    
                    # Free previous data from memory if exists
                    if hasattr(self, 'full_data'):
                        del self.full_data
                    
                    # Load only current data file
                    self.full_data = self.data_loader.load_final_data_for_symbol(current_symbol)
                    self.total_data_length = len(self.full_data)
                    episodes_in_current_file = max(1, (self.total_data_length - 1) // self.episode_length)
                    
                    logger.info(f"Processing {current_symbol}: {episodes_in_current_file} episodes ({len(self.full_data)} rows)")
                    logger.info(f"Memory usage: One data file loaded ({self.full_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")
                
                # Update environment for current data file
                self.env.symbol = current_symbol
                self.env._resolve_market_context()
                
                # Process all episodes in current data file
                self.current_episode_start = 0
                
                for episode_in_file in range(episodes_in_current_file):
                    # Setup current episode
                    self._setup_current_episode()
                    
                    # Reset environment for new episode
                    self.env.engine.reset()
                    self.env.reward_calculator.reset(self.env.initial_capital)
                    self.env.observation_handler.reset()
                    self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
                    
                    # Reset HRM carry state for new episode
                    if hasattr(self.env, 'current_carry') and self.model:
                        self.env.current_carry = self.model.create_initial_carry(batch_size=1)
                    
                    # Train episode
                    metrics = self.train_episode()
                    self.training_history.append(metrics)
                    
                    # Track metrics
                    epoch_metrics['reward'].append(metrics['avg_reward'])
                    epoch_metrics['loss'].append(metrics['avg_loss'])
                    
                    # Update running metrics
                    running_metrics['reward'].append(metrics['avg_reward'])
                    running_metrics['loss'].append(metrics['avg_loss'])
                    if metrics['avg_reward'] > running_metrics['best_reward']:
                        running_metrics['best_reward'] = metrics['avg_reward']
                    
                    global_step += 1
                    
                    # Update progress bar
                    if not self.debug_mode:
                        progress_bar.set_postfix({
                            'Epoch': f"{epoch+1}/{epochs}",
                            'File': f"{self.current_instrument_idx+1}/{len(self.available_instruments)}",
                            'Symbol': current_symbol.split('_')[0] if '_' in current_symbol else current_symbol,
                            'Reward': f"{metrics['avg_reward']:.4f}",
                            'Best': f"{running_metrics['best_reward']:.4f}"
                        })
                        progress_bar.update(1)
                    
                    # Move to next episode within file
                    self.current_episode_start += self.episode_length
                
                # Finished current data file - free memory before moving to next
                logger.info(f"Completed processing {current_symbol}, freeing memory...")
                del self.full_data  # Free current data file from memory
                
                # Move to next data file
                end_of_epoch = self._advance_to_next_data_file()
                if end_of_epoch:
                    break
            
            # Epoch completed - calculate epoch metrics
            epoch_time = time.time() - epoch_start_time
            running_metrics['total_time'] += epoch_time
            
            # Use tensor operations for faster computation
            epoch_avg_reward = torch.tensor(epoch_metrics['reward']).mean().item() if epoch_metrics['reward'] else 0.0
            epoch_avg_loss = torch.tensor(epoch_metrics['loss']).mean().item() if epoch_metrics['loss'] else 0.0
            
            # Update learning rate after each epoch
            if self.scheduler:
                self.scheduler.step()
            
            # Log epoch summary
            if self.debug_mode or (epoch + 1) % log_frequency == 0:
                logger.info(f"Epoch {epoch + 1} completed: Avg Reward: {epoch_avg_reward:.4f}, "
                           f"Avg Loss: {epoch_avg_loss:.4f}, Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            if (epoch + 1) % save_frequency == 0:
                self._save_latest_checkpoint(epoch)
            
            # Save best model if improved
            if epoch_avg_reward > self.best_performance:
                self.best_performance = epoch_avg_reward
                self._save_best_model()
                if not self.debug_mode:
                    progress_bar.write(f"🎯 New best model saved! Reward: {self.best_performance:.4f}")
        
        # Close progress bar
        if not self.debug_mode:
            progress_bar.close()
        
        # Final summary
        total_time = time.time() - start_time
        self._log_training_summary(epochs, running_metrics, total_time)
        
        logger.info("Multi-data training completed")
        return self.training_history
    
    def _log_training_progress_debug(self, episode: int, metrics: Dict, epoch_time: float):
        """Enhanced epoch summary logging for debug mode"""
        
        # Calculate average win rate from environment if available
        total_trades = getattr(self.env, 'total_trades', 0)
        winning_trades = getattr(self.env, 'winning_trades', 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # Print separator and epoch summary
        print("\n" + "="*100)
        print(f"EPOCH {episode+1:4d} SUMMARY | "
              f"Total Reward: {metrics.get('total_reward', 0):+8.2f} | "
              f"Loss: {metrics.get('avg_loss', 0):6.4f} | "
              f"Win Rate: {win_rate:5.1f}% | "
              f"Steps: {metrics.get('steps', 0):3d} | "
              f"Time: {epoch_time:.2f}s")
        print("="*100 + "\n")
    
    def _log_epoch_summary(self, completed_episodes: int, running_metrics: Dict, total_episodes: int):
        """Log epoch summary with running statistics"""
        
        recent_rewards = running_metrics['reward'][-10:] if len(running_metrics['reward']) >= 10 else running_metrics['reward']
        recent_losses = running_metrics['loss'][-10:] if len(running_metrics['loss']) >= 10 else running_metrics['loss']
        
        # Use tensor operations for faster computation
        avg_reward = torch.tensor(recent_rewards).mean().item() if recent_rewards else 0.0
        avg_loss = torch.tensor(recent_losses).mean().item() if recent_losses else 0.0
        progress_pct = (completed_episodes / total_episodes) * 100
        
        logger.info(f"📊 Progress: {completed_episodes}/{total_episodes} ({progress_pct:.1f}%) | "
                   f"Avg Reward (last 10): {avg_reward:.4f} | "
                   f"Avg Loss (last 10): {avg_loss:.4f} | "
                   f"Best Reward: {running_metrics['best_reward']:.4f}")
    
    def _log_training_summary(self, episodes: int, running_metrics: Dict, total_time: float):
        """Log final training summary"""
        
        logger.info("=" * 80)
        logger.info("🎉 TRAINING COMPLETED - SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Episodes: {episodes}")
        logger.info(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        logger.info(f"Average Time per Episode: {total_time/episodes:.2f} seconds")
        
        if running_metrics['reward']:
            # Use tensor operations for faster computation
            final_avg_reward = torch.tensor(running_metrics['reward'][-10:]).mean().item()
            reward_trend = torch.tensor(running_metrics['reward'][-20:]).mean().item()
            logger.info(f"Final Average Reward: {final_avg_reward:.4f}")
            logger.info(f"Best Reward Achieved: {running_metrics['best_reward']:.4f}")
            logger.info(f"Reward Trend (last 20 episodes): {reward_trend:.4f}")
        
        if running_metrics['loss']:
            final_avg_loss = torch.tensor(running_metrics['loss'][-10:]).mean().item()
            logger.info(f"Final Average Loss: {final_avg_loss:.4f}")
        
        logger.info("=" * 80)
    
    def _save_latest_checkpoint(self, episode: int):
        """Save latest training checkpoint (overwrites previous)"""
        
        checkpoint_dir = Path("checkpoints/hrm")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_performance': self.best_performance,
            'config': self.config,
            'training_history': self.training_history[-100:]  # Keep last 100 episodes
        }
        
        # Use fixed filename to overwrite previous checkpoint
        filepath = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, filepath)
        logger.info(f"Latest checkpoint saved: {filepath}")
    
    def _save_best_model(self):
        """Save the best performing model"""
        
        model_dir = Path("models/hrm")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'performance': self.best_performance,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = model_dir / "hrm_best_model.pt"
        torch.save(model_data, filepath)
        logger.info(f"Best model saved: {filepath} (performance: {self.best_performance:.4f})")
    
    def evaluate(self, 
                episodes: int = 10, 
                symbol: str = "Bank_Nifty_5") -> Dict[str, float]:
        """Evaluate trained model"""
        
        self.model.eval()
        
        evaluation_results = []
        
        with torch.no_grad():
            for episode in range(episodes):
                observation = self.env.reset()
                
                episode_reward = 0.0
                step_count = 0
                done = False
                
                while not done and step_count < 1000:
                    observation, reward, done, info = self.env.step()
                    episode_reward += reward
                    step_count += 1
                
                evaluation_results.append({
                    'episode_reward': episode_reward,
                    'steps': step_count,
                    'avg_reward': episode_reward / max(step_count, 1)
                })
        
        # Calculate aggregate statistics using tensor operations
        total_rewards = [r['episode_reward'] for r in evaluation_results]
        avg_rewards = [r['avg_reward'] for r in evaluation_results]
        
        # Use tensor operations for faster computation
        total_rewards_tensor = torch.tensor(total_rewards)
        avg_rewards_tensor = torch.tensor(avg_rewards)
        
        results = {
            'mean_episode_reward': total_rewards_tensor.mean().item(),
            'std_episode_reward': total_rewards_tensor.std().item(),
            'mean_avg_reward': avg_rewards_tensor.mean().item(),
            'std_avg_reward': avg_rewards_tensor.std().item(),
            'best_episode': total_rewards_tensor.max().item(),
            'worst_episode': total_rewards_tensor.min().item()
        }
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint and resume training"""
        
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.model is None:
            raise ValueError("Model must be initialized before loading checkpoint")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.best_performance = checkpoint['best_performance']
        self.current_episode = checkpoint['episode']
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from episode {self.current_episode}")
        logger.info(f"Best performance so far: {self.best_performance:.4f}")
        
        return self.current_episode


def main():
    """Main function to run HRM training"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = HRMTrainer(
        config_path="config/hrm_config.yaml",
        data_path="data/final",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    try:
        # Train the model
        history = trainer.train(
            episodes=500,
            symbol="Bank_Nifty_5",
            save_frequency=50,
            log_frequency=5
        )
        
        # Evaluate the model
        results = trainer.evaluate(episodes=20)
        
        logger.info("HRM training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()