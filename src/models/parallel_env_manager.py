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
                 max_parallel_envs: int = 10):
        """
        Initialize parallel environment manager
        
        Args:
            data_loader: Data loader instance
            symbols: List of symbols to create environments for
            config_path: Path to HRM configuration
            device: Device to run environments on
            max_parallel_envs: Maximum number of environments to run in parallel
        """
        self.data_loader = data_loader
        self.symbols = symbols
        self.config_path = config_path
        self.device = torch.device(device)
        self.max_parallel_envs = max_parallel_envs
        
        # Load configuration
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.get_config()
        
        # Get environment configuration
        env_config = self.config.get('environment', {})
        self.initial_capital = env_config.get('initial_capital', 100000.0)
        self.episode_length = env_config.get('episode_length', 1500)
        
        # Initialize environments
        self.environments = []
        self._create_environments()
        
        logger.info(f"Parallel Environment Manager initialized with {len(self.environments)} environments")
    
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
        Reset all environments and return initial observations
        
        Returns:
            Batched observations tensor [batch_size, obs_dim]
        """
        observations = []
        for env in self.environments:
            obs = env.reset()
            observations.append(torch.tensor(obs, dtype=torch.float32))
        
        # Stack into batch
        batch_obs = torch.stack(observations).to(self.device)
        return batch_obs
    
    def step_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Step all environments in parallel
        
        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for env in self.environments:
            try:
                obs, reward, done, info = env.step()
                observations.append(torch.tensor(obs, dtype=torch.float32))
                rewards.append(float(reward))
                dones.append(bool(done))
                infos.append(info)
            except Exception as e:
                logger.error(f"Error stepping environment: {e}")
                # Return default values for failed environment
                observations.append(torch.zeros_like(torch.tensor(obs, dtype=torch.float32)))
                rewards.append(0.0)
                dones.append(True)
                infos.append({"error": str(e)})
        
        # Stack into batches
        batch_obs = torch.stack(observations).to(self.device)
        batch_rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        batch_dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        return batch_obs, batch_rewards, batch_dones, infos
    
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