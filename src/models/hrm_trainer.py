import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
import time

from src.models.hrm import HRMTradingAgent, HRMCarry
from src.models.hrm_trading_environment import HRMTradingEnvironment, HRMTradingWrapper
from src.utils.data_loader import DataLoader
from src.env.trading_mode import TradingMode

logger = logging.getLogger(__name__)


class HRMLossFunction:
    """Multi-component loss function for HRM trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loss_weights = config['training']['loss_weights']
        
    def calculate_loss(self, 
                      outputs: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor],
                      segment_rewards: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate multi-component HRM loss with expert targets"""
        
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # ONLY market understanding guidance loss (help model interpret market context)
        if 'market_understanding_outputs' in outputs and 'market_understanding' in targets:
            if targets['market_understanding'].numel() > 0:
                model_output = outputs['market_understanding_outputs']
                target_tensor = targets['market_understanding']
                
                # Market understanding loss to guide market context interpretation
                understanding_loss = nn.MSELoss()(model_output, target_tensor)
                losses['market_understanding_loss'] = understanding_loss
                total_loss = total_loss + self.loss_weights.get('tactical_loss', 1.0) * understanding_loss
        
        # REMOVED: regime loss - let H-module learn regimes from 100 historical candles naturally  
        # REMOVED: risk context loss - let model learn risk management from environment rewards
        
        # ACT loss - use the Q-learning targets from the ACT module
        if 'halt_logits' in outputs and 'continue_logits' in outputs and 'q_halt_target' in outputs:
            # Ensure proper tensor shapes for MSE loss
            halt_logits = outputs['halt_logits']
            continue_logits = outputs['continue_logits']
            halt_target = outputs['q_halt_target'].detach()
            continue_target = outputs['q_continue_target'].detach()
            
            # Reshape tensors to ensure compatibility
            if halt_logits.dim() > halt_target.dim():
                halt_logits = halt_logits.view_as(halt_target)
            elif halt_target.dim() > halt_logits.dim():
                halt_target = halt_target.view_as(halt_logits)
                
            if continue_logits.dim() > continue_target.dim():
                continue_logits = continue_logits.view_as(continue_target)
            elif continue_target.dim() > continue_logits.dim():
                continue_target = continue_target.view_as(continue_logits)
            
            halt_loss = nn.MSELoss()(halt_logits, halt_target)
            continue_loss = nn.MSELoss()(continue_logits, continue_target)
            
            act_loss = halt_loss + continue_loss
            losses['act_loss'] = act_loss
            total_loss = total_loss + self.loss_weights['act_loss'] * act_loss
        
        # Performance-based loss (primary signal)
        if segment_rewards:
            # Convert rewards to loss (we want to maximize rewards, so minimize negative rewards)
            performance_loss = -torch.tensor(np.mean(segment_rewards))
            losses['performance_loss'] = performance_loss
            total_loss = total_loss + self.loss_weights['performance_loss'] * performance_loss
        
        # Convert loss values to float for logging
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_values['total_loss'] = total_loss.item()
        
        return total_loss, loss_values


class HRMTrainer:
    """Trainer for HRM Trading Agent"""
    
    def __init__(self, 
                 config_path: str = "config/hrm_config.yaml",
                 data_path: str = "data/final",
                 device: str = None,  # Auto-detect if None
                 debug_mode: bool = False):
        
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
        self.debug_mode = debug_mode
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        
        # Model and optimizer will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_training_complete(self, symbol: str = "Bank_Nifty_5"):
        """Complete training setup in one centralized place - no resets, no hanging"""
        
        # Load configuration to get initial capital and episode length
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        initial_capital = config.get('environment', {}).get('initial_capital', 100000.0)
        episode_length = config.get('environment', {}).get('episode_length', 1500)
        
        # 1. Load full data first
        self.full_data = self.data_loader.load_final_data_for_symbol(symbol)
        
        # 2. Initialize episode tracking
        self.current_episode_start = 0
        self.episode_length = episode_length
        self.total_data_length = len(self.full_data)
        
        # 3. Create environment with episode length config
        self.env = HRMTradingEnvironment(
            data_loader=self.data_loader,
            symbol=symbol,
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
        
        # 6. Setup optimizer
        self._setup_optimizer()
        
        logger.info("Training setup completed successfully")
    
    def _setup_current_episode(self):
        """Set up data for current episode - sequential feeding"""
        
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
        """Move to next sequential episode"""
        
        # Move to next episode segment
        self.current_episode_start += self.episode_length
        
        # Check if we've reached end of data
        if self.current_episode_start >= self.total_data_length - self.episode_length:
            # Reset to beginning for next epoch
            self.current_episode_start = 0
            logger.info("End of data reached, resetting to beginning for next epoch")
        
        # Set up next episode data
        self._setup_current_episode()
        
        # Reset environment state for new episode
        self.env.engine.reset()
        self.env.reward_calculator.reset(self.env.initial_capital)
        self.env.observation_handler.reset()
        self.env.termination_manager.reset(self.env.initial_capital, self.env.episode_end_step)
        
        # Reset HRM carry state for new episode
        if hasattr(self.env, 'current_carry') and self.model:
            self.env.current_carry = self.model.create_initial_carry(batch_size=1)
    
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
        
        # Print step debug header if in debug mode
        if self.debug_mode:
            # Get instrument and timeframe info
            instrument_name = getattr(self.env, 'symbol', 'Unknown')
            timeframe = 'Unknown'
            if hasattr(self.env, 'symbol'):
                parts = self.env.symbol.split('_')
                if len(parts) >= 2:
                    instrument_name = '_'.join(parts[:-1])
                    timeframe = f"{parts[-1]}min"
            
            print(f"\nEPOCH {getattr(self, 'current_episode', 0) + 1} STEP-BY-STEP: {instrument_name} ({timeframe})")
            print("Step | Instrument     | Timeframe | DateTime           | Initial Cap | Current Cap | Action   | Win%  | P&L      | Current Price | Entry   | Target Price(Pts) | SL Price(Pts)   | Reward | Reason")
            print("-" * 175)
        
        # Environment is already initialized in setup_training_complete, no reset needed
        
        episode_metrics = {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'steps': 0,
            'hierarchical_metrics': {}
        }
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Max steps per episode
            
            # Environment step (HRM handles the reasoning internally)
            observation, reward, done, info = self.env.step()
            
            episode_metrics['total_reward'] += reward
            step_count += 1
            
            # Debug step logging if enabled
            if self.debug_mode:
                self._log_step_debug(step_count, reward, info)
            
            # Extract loss information if available in info
            if 'segment_rewards' in info:
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
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        if self.config['training']['gradient_clip'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        
                        self.optimizer.step()
                        
                        episode_metrics['total_loss'] += loss.item()
        
        episode_metrics['steps'] = step_count
        episode_metrics['avg_reward'] = episode_metrics['total_reward'] / max(step_count, 1)
        episode_metrics['avg_loss'] = episode_metrics['total_loss'] / max(step_count, 1)
        
        # Get hierarchical performance metrics
        episode_metrics['hierarchical_metrics'] = self.env.get_hierarchical_performance_metrics()
        
        return episode_metrics
    
    def _log_step_debug(self, step_num: int, reward: float, info: Dict):
        """Log detailed step information using environment as single source of truth."""
        
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
    
    def train(self, 
              episodes: int = 1000, 
              symbol: str = None,
              save_frequency: int = 100,
              log_frequency: int = 10):
        """Main training loop with enhanced progress tracking"""
        
        # Setup training only if not already done
        if self.model is None:
            if symbol is None:
                raise ValueError("Symbol must be provided for initial setup")
            self.setup_training_complete(symbol)
        
        self.total_epochs = episodes
        logger.info(f"Starting HRM training for {episodes} episodes")
        logger.info(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Progress tracking: {'Step-by-step' if self.debug_mode else 'Epoch-level progress bar'}")
        
        # Initialize metrics tracking for progress bar
        running_metrics = {
            'reward': [],
            'loss': [],
            'best_reward': float('-inf'),
            'total_time': 0
        }
        
        # Create progress bar
        if not self.debug_mode:
            progress_bar = tqdm(
                total=episodes,
                desc="Training Progress",
                unit="epoch",
                ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
        
        start_time = time.time()
        
        for episode in range(episodes):
            self.current_episode = episode
            epoch_start_time = time.time()
            
            # Train episode
            metrics = self.train_episode()
            self.training_history.append(metrics)
            
            # Advance to next sequential episode after training
            self._advance_to_next_episode()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Track running metrics
            running_metrics['reward'].append(metrics['avg_reward'])
            running_metrics['loss'].append(metrics['avg_loss'])
            if metrics['avg_reward'] > running_metrics['best_reward']:
                running_metrics['best_reward'] = metrics['avg_reward']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            running_metrics['total_time'] += epoch_time
            
            if self.debug_mode:
                # Detailed logging for debug mode
                self._log_training_progress_debug(episode, metrics, epoch_time)
            else:
                # Update progress bar
                if episode >= 5:  # Calculate running averages after first 5 episodes
                    avg_reward = np.mean(running_metrics['reward'][-5:])
                    avg_loss = np.mean(running_metrics['loss'][-5:])
                else:
                    avg_reward = metrics['avg_reward']
                    avg_loss = metrics['avg_loss']
                
                # Update progress bar postfix
                progress_bar.set_postfix({
                    'Reward': f"{avg_reward:.4f}",
                    'Loss': f"{avg_loss:.4f}",
                    'Best': f"{running_metrics['best_reward']:.4f}",
                    'Time': f"{epoch_time:.1f}s"
                })
                progress_bar.update(1)
                
                # Log summary every log_frequency episodes
                if (episode + 1) % log_frequency == 0 or episode == 0:
                    self._log_epoch_summary(episode + 1, running_metrics, episodes)
            
            # Save latest checkpoint (overwrites previous)
            if episode % save_frequency == 0:
                self._save_latest_checkpoint(episode)
            
            # Check for improvement and save best model
            if metrics['avg_reward'] > self.best_performance:
                self.best_performance = metrics['avg_reward']
                self._save_best_model()
                if not self.debug_mode:
                    progress_bar.write(f"🎯 New best model saved! Reward: {self.best_performance:.4f}")
        
        # Close progress bar
        if not self.debug_mode:
            progress_bar.close()
        
        # Final summary
        total_time = time.time() - start_time
        self._log_training_summary(episodes, running_metrics, total_time)
        
        logger.info("Training completed")
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
        
        avg_reward = np.mean(recent_rewards)
        avg_loss = np.mean(recent_losses)
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
            logger.info(f"Final Average Reward: {np.mean(running_metrics['reward'][-10:]):.4f}")
            logger.info(f"Best Reward Achieved: {running_metrics['best_reward']:.4f}")
            logger.info(f"Reward Trend (last 20 episodes): {np.mean(running_metrics['reward'][-20:]):.4f}")
        
        if running_metrics['loss']:
            logger.info(f"Final Average Loss: {np.mean(running_metrics['loss'][-10:]):.4f}")
        
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
        
        # Calculate aggregate statistics
        total_rewards = [r['episode_reward'] for r in evaluation_results]
        avg_rewards = [r['avg_reward'] for r in evaluation_results]
        
        results = {
            'mean_episode_reward': np.mean(total_rewards),
            'std_episode_reward': np.std(total_rewards),
            'mean_avg_reward': np.mean(avg_rewards),
            'std_avg_reward': np.std(avg_rewards),
            'best_episode': max(total_rewards),
            'worst_episode': min(total_rewards)
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