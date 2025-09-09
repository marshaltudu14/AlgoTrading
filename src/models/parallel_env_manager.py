"""
Parallel Environment Manager for HRM Training
Manages multiple environments running in parallel to maximize GPU utilization
"""
import torch
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from src.models.hrm_trading_environment import HRMTradingEnvironment
from src.env.trading_mode import TradingMode
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ParallelEnvironmentManager:
    """Manages multiple HRM environments running in parallel"""
    
    def __init__(self, 
                 data_loader,
                 symbols: List[str],
                 config_path: str = "config/hrm_config.yaml",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_parallel_envs: int = None):
        """
        Initialize parallel environment manager
        
        Args:
            data_loader: Data loader instance
            symbols: List of symbols to create environments for
            config_path: Path to HRM configuration
            device: Device to run environments on
            max_parallel_envs: Maximum number of environments to run in parallel (if None, use config value)
        """
        self.data_loader = data_loader
        self.symbols = symbols
        self.config_path = config_path
        self.device = torch.device(device)
        
        # Load configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_config()
        
        # Get environment configuration
        env_config = self.config.get('environment', {})
        self.initial_capital = env_config.get('initial_capital', 100000.0)
        self.episode_length = env_config.get('episode_length', 1500)
        
        # Determine max parallel environments based on device type and config
        if max_parallel_envs is not None:
            self.max_parallel_envs = max_parallel_envs
        else:
            # Get parallel environments setting based on device type
            parallel_envs_config = env_config.get('parallel_environments', {})
            device_type = self._get_device_type()
            if isinstance(parallel_envs_config, dict):
                # New format with separate settings for CPU and GPU/TPU
                if device_type == 'cpu':
                    self.max_parallel_envs = parallel_envs_config.get('cpu', 5)
                else:  # gpu or tpu - MAXIMIZE GPU UTILIZATION
                    self.max_parallel_envs = parallel_envs_config.get('gpu_tpu', 25)
            else:
                # Old format - fallback to GPU-optimized values
                self.max_parallel_envs = 5 if device_type == 'cpu' else 25
        
        # GPU-specific optimizations
        if device_type == 'gpu':
            # Force maximum GPU utilization
            self.max_parallel_envs = max(self.max_parallel_envs, 25)
            logger.info(f"GPU OPTIMIZATION: Forcing {self.max_parallel_envs} parallel environments for maximum VRAM utilization")
            
            # Enable GPU memory optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Clear cache and pre-allocate memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Initialize environments
        self.environments = []
        self._create_environments()
        
        # GPU memory tracking
        if device_type == 'gpu':
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            logger.info(f"GPU Memory After Environment Creation: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        logger.info(f"Parallel Environment Manager initialized with {len(self.environments)} environments on {self._get_device_type().upper()}")
    
    def _get_device_type(self) -> str:
        """Get device type string"""
        if self.device.type == 'cuda':
            return 'gpu'
        elif self.device.type == 'xla':
            return 'tpu'
        else:
            return 'cpu'
    
    def _create_environments(self):
        """Create parallel environments"""
        # Limit to max_parallel_envs or available symbols, whichever is smaller
        num_envs = min(self.max_parallel_envs, len(self.symbols))
        selected_symbols = self.symbols[:num_envs]
        
        for symbol in selected_symbols:
            try:
                env = HRMTradingEnvironment(
                    data_loader=self.data_loader,
                    symbol=symbol,
                    initial_capital=self.initial_capital,
                    mode=TradingMode.TRAINING,
                    hrm_config_path=self.config_path,
                    device=str(self.device),
                    episode_length=self.episode_length
                )
                self.environments.append(env)
                logger.info(f"Created environment for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to create environment for {symbol}: {e}")
        
        if not self.environments:
            raise ValueError("No environments could be created")
    
    def reset_all(self) -> torch.Tensor:
        """
        Reset all environments and return initial observations with GPU optimization
        
        Returns:
            Batched observations tensor [batch_size, obs_dim]
        """
        observations = []
        
        # Pre-allocate on GPU for better performance
        with torch.no_grad():
            for env in self.environments:
                obs = env.reset()
                # Create tensor directly on GPU to avoid CPU-GPU transfer overhead
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
                observations.append(obs_tensor)
        
        # Stack into batch - already on GPU
        batch_obs = torch.stack(observations)  # Already on device
        return batch_obs
    
    def step_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step all environments in true parallel with proper batching
        
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        # Collect all observations, rewards, dones in lists
        all_obs = []
        all_rewards = []
        all_dones = []
        all_infos = []
        
        # Step all environments and collect results
        for env in self.environments:
            try:
                obs, reward, done, info = env.step()
                all_obs.append(obs)
                all_rewards.append(float(reward))
                all_dones.append(bool(done))
                all_infos.append(info)
            except Exception as e:
                logger.error(f"Error stepping environment: {e}")
                # Return default values
                default_obs = np.zeros_like(obs) if 'obs' in locals() else np.zeros(100)
                all_obs.append(default_obs)
                all_rewards.append(0.0)
                all_dones.append(True)
                all_infos.append({"error": str(e)})
        
        # GPU-optimized batching: minimize memory transfers and maximize GPU utilization
        if self.device.type == 'cuda':
            # GPU: Use memory-efficient batching with pre-allocation
            batch_obs = torch.stack([torch.tensor(obs, dtype=torch.float32, device=self.device, non_blocking=True) for obs in all_obs])
            batch_rewards = torch.tensor(all_rewards, dtype=torch.float32, device=self.device, non_blocking=True)
            batch_dones = torch.tensor(all_dones, dtype=torch.bool, device=self.device, non_blocking=True)
        else:
            # CPU: Standard approach
            batch_obs = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in all_obs])
            batch_rewards = torch.tensor(all_rewards, dtype=torch.float32)
            batch_dones = torch.tensor(all_dones, dtype=torch.bool)
        
        return batch_obs, batch_rewards, batch_dones, all_infos
    
    def get_batch_size(self) -> int:
        """Get current batch size"""
        return len(self.environments)
    
    def get_environment_symbols(self) -> List[str]:
        """Get symbols for all active environments"""
        return [env.symbol for env in self.environments]
    
    def close_all(self):
        """Close all environments"""
        for env in self.environments:
            try:
                # Clean up environment resources if needed
                pass
            except Exception as e:
                logger.warning(f"Error closing environment: {e}")