"""
Configuration utilities for parallel training with Ray RLlib.
"""

from typing import Dict, Any, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

class ParallelTrainingConfig:
    """
    Configuration builder for parallel training setup.
    """
    
    @staticmethod
    def create_default_config(
        symbols: List[str],
        data_dir: str = "data/raw",
        num_workers: int = 4,
        algorithm: str = "PPO"
    ) -> Dict[str, Any]:
        """
        Create default configuration for parallel training.
        
        Args:
            symbols: List of trading symbols to train on
            data_dir: Directory containing trading data
            num_workers: Number of parallel worker processes
            algorithm: RL algorithm to use ("PPO" or "IMPALA")
            
        Returns:
            Complete configuration dictionary
        """
        config = {
            # Ray configuration
            "num_cpus": None,  # Use all available CPUs
            "num_gpus": 1 if _has_gpu() else 0,
            "local_mode": False,  # Set to True for debugging
            
            # Environment configuration
            "env_config": {
                "data_loader": {
                    "final_data_dir": "data/final",
                    "raw_data_dir": data_dir,
                    "chunk_size": 10000,
                    "use_parquet": True
                },
                "symbol": symbols[0] if symbols else "DEFAULT_SYMBOL",
                "initial_capital": 100000.0,
                "lookback_window": 50,
                "trailing_stop_percentage": 0.02,
                "reward_function": "pnl",
                "episode_length": 1000,
                "use_streaming": True
            },
            
            # Training configuration
            "training_config": {
                "algorithm": algorithm,
                "num_workers": num_workers,
                "learning_rate": 3e-4,
                "train_batch_size": 4000,
                "sgd_minibatch_size": 128,
                "num_sgd_iter": 10,
                "gamma": 0.99,
                "lambda": 0.95,
                "clip_param": 0.2,
                "entropy_coeff": 0.01,
                "vf_loss_coeff": 0.5,
                "rollout_fragment_length": 200,
                "num_gpus": 1 if _has_gpu() else 0,
                "num_cpus_per_worker": 1,
                "model_config": {
                    "hidden_size": 256,
                    "num_layers": 2
                }
            },
            
            # Multi-symbol training configuration
            "multi_symbol_config": {
                "symbols": symbols,
                "training_mode": "sequential",  # "sequential" or "parallel"
                "symbol_rotation_freq": 10,  # Switch symbols every N iterations
            },
            
            # Checkpointing and evaluation
            "checkpoint_config": {
                "checkpoint_freq": 10,
                "checkpoint_dir": "checkpoints/parallel_training",
                "keep_checkpoints_num": 5,
                "evaluation_freq": 20,
                "evaluation_episodes": 10
            }
        }
        
        return config
    
    @staticmethod
    def create_distributed_config(
        symbols: List[str],
        num_nodes: int = 2,
        cpus_per_node: int = 8,
        gpus_per_node: int = 1
    ) -> Dict[str, Any]:
        """
        Create configuration for distributed training across multiple nodes.
        
        Args:
            symbols: List of trading symbols
            num_nodes: Number of compute nodes
            cpus_per_node: CPUs per node
            gpus_per_node: GPUs per node
            
        Returns:
            Distributed training configuration
        """
        base_config = ParallelTrainingConfig.create_default_config(symbols)
        
        # Update for distributed setup
        base_config.update({
            "num_cpus": cpus_per_node * num_nodes,
            "num_gpus": gpus_per_node * num_nodes,
            "local_mode": False
        })
        
        # Increase parallelism for distributed setup
        base_config["training_config"].update({
            "num_workers": min(16, cpus_per_node * num_nodes - 2),  # Leave some CPUs for learner
            "train_batch_size": 8000,
            "sgd_minibatch_size": 256,
            "rollout_fragment_length": 100
        })
        
        return base_config
    
    @staticmethod
    def create_development_config(symbols: List[str]) -> Dict[str, Any]:
        """
        Create lightweight configuration for development and testing.
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Development configuration
        """
        config = ParallelTrainingConfig.create_default_config(symbols)
        
        # Reduce resource usage for development
        config.update({
            "local_mode": True,  # Run in single process for debugging
            "num_gpus": 0
        })
        
        config["training_config"].update({
            "num_workers": 2,
            "train_batch_size": 1000,
            "sgd_minibatch_size": 64,
            "rollout_fragment_length": 50,
            "model_config": {
                "hidden_size": 128,
                "num_layers": 2
            }
        })
        
        config["env_config"].update({
            "episode_length": 200,  # Shorter episodes for faster testing
        })
        
        config["checkpoint_config"].update({
            "checkpoint_freq": 5,
            "evaluation_freq": 10
        })
        
        return config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        # Check required sections
        required_sections = ["env_config", "training_config", "checkpoint_config"]
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate environment config
        if "env_config" in config:
            env_config = config["env_config"]
            if "symbol" not in env_config:
                issues.append("env_config missing 'symbol'")
            if "initial_capital" not in env_config or env_config["initial_capital"] <= 0:
                issues.append("env_config 'initial_capital' must be positive")
        
        # Validate training config
        if "training_config" in config:
            training_config = config["training_config"]
            if "num_workers" not in training_config or training_config["num_workers"] < 1:
                issues.append("training_config 'num_workers' must be >= 1")
            if "learning_rate" not in training_config or training_config["learning_rate"] <= 0:
                issues.append("training_config 'learning_rate' must be positive")
        
        # Validate resource allocation
        num_workers = config.get("training_config", {}).get("num_workers", 1)
        num_cpus = config.get("num_cpus")
        if num_cpus and num_workers >= num_cpus:
            issues.append(f"num_workers ({num_workers}) should be less than num_cpus ({num_cpus})")
        
        return issues

def _has_gpu() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def create_multi_symbol_configs(
    symbols: List[str], 
    base_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Create separate configurations for each symbol.
    
    Args:
        symbols: List of trading symbols
        base_config: Base configuration to copy
        
    Returns:
        List of symbol-specific configurations
    """
    configs = []
    
    for symbol in symbols:
        symbol_config = base_config.copy()
        symbol_config["env_config"]["symbol"] = symbol
        symbol_config["checkpoint_config"]["checkpoint_dir"] = f"checkpoints/{symbol}"
        configs.append(symbol_config)
    
    return configs

def get_recommended_config(
    symbols: List[str],
    mode: str = "production"
) -> Dict[str, Any]:
    """
    Get recommended configuration based on mode.
    
    Args:
        symbols: List of trading symbols
        mode: Configuration mode ("development", "production", "distributed")
        
    Returns:
        Recommended configuration
    """
    if mode == "development":
        return ParallelTrainingConfig.create_development_config(symbols)
    elif mode == "distributed":
        return ParallelTrainingConfig.create_distributed_config(symbols)
    else:  # production
        return ParallelTrainingConfig.create_default_config(symbols)
