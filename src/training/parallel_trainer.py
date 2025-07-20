import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Any, List, Tuple
import os

from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.agents.base_agent import BaseAgent
from src.agents.moe_agent import MoEAgent
from src.utils.hardware_optimizer import get_hardware_optimizer

torch, nn = try_import_torch()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingEnvWrapper(TradingEnv):
    """
    Ray RLlib compatible wrapper for TradingEnv.
    """
    def __init__(self, config: EnvContext):
        # Extract configuration from Ray's EnvContext
        data_loader_config = config.get("data_loader", {})
        symbol = config.get("symbol", "DEFAULT_SYMBOL")
        initial_capital = config.get("initial_capital", 100000.0)
        lookback_window = config.get("lookback_window", 50)
        trailing_stop_percentage = config.get("trailing_stop_percentage", 0.02)
        reward_function = config.get("reward_function", "pnl")
        episode_length = config.get("episode_length", 1000)
        use_streaming = config.get("use_streaming", True)
        
        # Initialize DataLoader
        data_loader = DataLoader(
            final_data_dir=data_loader_config.get("final_data_dir", "data/final"),
            raw_data_dir=data_loader_config.get("raw_data_dir", "data/raw"),
            chunk_size=data_loader_config.get("chunk_size", 10000),
            use_parquet=data_loader_config.get("use_parquet", True)
        )
        
        # Initialize parent TradingEnv
        super().__init__(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=lookback_window,
            trailing_stop_percentage=trailing_stop_percentage,
            reward_function=reward_function,
            episode_length=episode_length,
            use_streaming=use_streaming
        )
        
        logger.info(f"TradingEnvWrapper initialized for symbol: {symbol}")

class TradingPolicyModel(TorchModelV2, nn.Module):
    """
    Custom PyTorch model for trading policy compatible with Ray RLlib.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.obs_size = obs_space.shape[0]
        self.action_size = action_space.n if hasattr(action_space, 'n') else action_space.shape[0]
        
        # Extract model configuration
        hidden_size = model_config.get("custom_model_config", {}).get("hidden_size", 256)
        num_layers = model_config.get("custom_model_config", {}).get("num_layers", 2)
        
        # Build neural network layers
        layers = []
        input_size = self.obs_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, num_outputs)
        
        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"TradingPolicyModel initialized: obs_size={self.obs_size}, action_size={self.action_size}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network."""
        obs = input_dict["obs"].float()
        
        # Shared feature extraction
        shared_features = self.shared_layers(obs)
        
        # Policy logits
        policy_logits = self.policy_head(shared_features)
        
        # Store value for value_function() call
        self._value_out = self.value_head(shared_features)
        
        return policy_logits, state
    
    def value_function(self):
        """Return the value function output."""
        return torch.reshape(self._value_out, [-1])

class ParallelTrainer:
    """
    Parallel training implementation using Ray RLlib with Actor-Learner architecture.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.algorithm = None
        self.is_initialized = False

        # Initialize hardware optimization
        self.hardware_optimizer = get_hardware_optimizer()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.get("num_cpus", None),
                num_gpus=config.get("num_gpus", 0),
                local_mode=config.get("local_mode", False)
            )
            logger.info("Ray initialized for parallel training")

        # Register custom environment and model
        self._register_components()
    
    def _register_components(self):
        """Register custom environment and model with Ray RLlib."""
        # Register custom environment
        tune.register_env("trading_env", lambda config: TradingEnvWrapper(config))
        
        # Register custom model
        ModelCatalog.register_custom_model("trading_policy_model", TradingPolicyModel)
        
        logger.info("Custom components registered with Ray RLlib")
    
    def setup_algorithm(self, algorithm_type: str = "PPO") -> None:
        """
        Set up the Ray RLlib algorithm for parallel training.
        
        Args:
            algorithm_type: Type of algorithm ("PPO" or "IMPALA")
        """
        env_config = self.config.get("env_config", {})
        training_config = self.config.get("training_config", {})
        
        if algorithm_type.upper() == "PPO":
            config = (PPOConfig()
                     .environment(env="trading_env", env_config=env_config)
                     .framework("torch")
                     .training(
                         model={
                             "custom_model": "trading_policy_model",
                             "custom_model_config": training_config.get("model_config", {})
                         },
                         lr=training_config.get("learning_rate", 3e-4),
                         train_batch_size=training_config.get("train_batch_size", 4000),
                         sgd_minibatch_size=training_config.get("sgd_minibatch_size", 128),
                         num_sgd_iter=training_config.get("num_sgd_iter", 10),
                         gamma=training_config.get("gamma", 0.99),
                         lambda_=training_config.get("lambda", 0.95),
                         clip_param=training_config.get("clip_param", 0.2),
                         entropy_coeff=training_config.get("entropy_coeff", 0.01),
                         vf_loss_coeff=training_config.get("vf_loss_coeff", 0.5)
                     )
                     .rollouts(
                         num_rollout_workers=training_config.get("num_workers", 4),
                         rollout_fragment_length=training_config.get("rollout_fragment_length", 200),
                         batch_mode="complete_episodes"
                     )
                     .resources(
                         num_gpus=training_config.get("num_gpus", 0),
                         num_cpus_per_worker=training_config.get("num_cpus_per_worker", 1)
                     )
                     .debugging(
                         log_level="INFO"
                     ))
        
        elif algorithm_type.upper() == "IMPALA":
            config = (IMPALAConfig()
                     .environment(env="trading_env", env_config=env_config)
                     .framework("torch")
                     .training(
                         model={
                             "custom_model": "trading_policy_model",
                             "custom_model_config": training_config.get("model_config", {})
                         },
                         lr=training_config.get("learning_rate", 3e-4),
                         train_batch_size=training_config.get("train_batch_size", 500),
                         gamma=training_config.get("gamma", 0.99),
                         entropy_coeff=training_config.get("entropy_coeff", 0.01),
                         vf_loss_coeff=training_config.get("vf_loss_coeff", 0.5)
                     )
                     .rollouts(
                         num_rollout_workers=training_config.get("num_workers", 4),
                         rollout_fragment_length=training_config.get("rollout_fragment_length", 50)
                     )
                     .resources(
                         num_gpus=training_config.get("num_gpus", 0),
                         num_cpus_per_worker=training_config.get("num_cpus_per_worker", 1)
                     ))
        
        else:
            raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
        
        # Build the algorithm
        self.algorithm = config.build()
        self.is_initialized = True
        
        logger.info(f"{algorithm_type} algorithm configured with {training_config.get('num_workers', 4)} workers")

    def train(self, num_iterations: int = 100, checkpoint_freq: int = 10,
              checkpoint_dir: str = "checkpoints") -> Dict[str, Any]:
        """
        Run parallel training for specified number of iterations.

        Args:
            num_iterations: Number of training iterations
            checkpoint_freq: Frequency of checkpointing (every N iterations)
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Training results and metrics
        """
        if not self.is_initialized:
            raise RuntimeError("Algorithm not initialized. Call setup_algorithm() first.")

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_results = []
        best_reward = float('-inf')

        logger.info(f"Starting parallel training for {num_iterations} iterations")

        for iteration in range(num_iterations):
            # Train one iteration
            result = self.algorithm.train()
            training_results.append(result)

            # Extract key metrics
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episode_len_mean = result.get("episode_len_mean", 0)
            episodes_this_iter = result.get("episodes_this_iter", 0)

            # Log progress
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                logger.info(
                    f"Iteration {iteration + 1}/{num_iterations}: "
                    f"Reward={episode_reward_mean:.2f}, "
                    f"Episode Length={episode_len_mean:.1f}, "
                    f"Episodes={episodes_this_iter}"
                )

            # Save checkpoint if this is the best model so far
            if episode_reward_mean > best_reward:
                best_reward = episode_reward_mean
                best_checkpoint_path = os.path.join(checkpoint_dir, f"best_model_iter_{iteration + 1}")
                self.algorithm.save(best_checkpoint_path)
                logger.info(f"New best model saved: {best_checkpoint_path} (reward: {best_reward:.2f})")

            # Regular checkpointing
            if (iteration + 1) % checkpoint_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration + 1}")
                self.algorithm.save(checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Final results summary
        final_results = {
            "total_iterations": num_iterations,
            "best_reward": best_reward,
            "final_reward": training_results[-1].get("episode_reward_mean", 0) if training_results else 0,
            "training_results": training_results
        }

        logger.info(f"Training completed. Best reward: {best_reward:.2f}")
        return final_results

    def evaluate(self, num_episodes: int = 10, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate the trained model.

        Args:
            num_episodes: Number of episodes for evaluation
            checkpoint_path: Path to checkpoint to load (if None, uses current model)

        Returns:
            Evaluation results
        """
        if not self.is_initialized:
            raise RuntimeError("Algorithm not initialized. Call setup_algorithm() first.")

        # Load checkpoint if specified
        if checkpoint_path:
            self.algorithm.restore(checkpoint_path)
            logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

        # Run evaluation episodes
        evaluation_rewards = []
        evaluation_lengths = []

        logger.info(f"Starting evaluation for {num_episodes} episodes")

        for episode in range(num_episodes):
            # Create evaluation environment
            env_config = self.config.get("env_config", {})
            env = TradingEnvWrapper(env_config)

            # Run episode
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Get action from policy
                action = self.algorithm.compute_single_action(obs)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            evaluation_rewards.append(episode_reward)
            evaluation_lengths.append(episode_length)

            logger.info(f"Evaluation episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}")

        # Calculate evaluation metrics
        eval_results = {
            "num_episodes": num_episodes,
            "mean_reward": np.mean(evaluation_rewards),
            "std_reward": np.std(evaluation_rewards),
            "min_reward": np.min(evaluation_rewards),
            "max_reward": np.max(evaluation_rewards),
            "mean_length": np.mean(evaluation_lengths),
            "episode_rewards": evaluation_rewards,
            "episode_lengths": evaluation_lengths
        }

        logger.info(
            f"Evaluation completed: "
            f"Mean reward={eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}, "
            f"Mean length={eval_results['mean_length']:.1f}"
        )

        return eval_results

    def get_policy_weights(self) -> Dict[str, Any]:
        """Get current policy weights for model synchronization."""
        if not self.is_initialized:
            raise RuntimeError("Algorithm not initialized.")

        return self.algorithm.get_policy().get_weights()

    def set_policy_weights(self, weights: Dict[str, Any]) -> None:
        """Set policy weights for model synchronization."""
        if not self.is_initialized:
            raise RuntimeError("Algorithm not initialized.")

        self.algorithm.get_policy().set_weights(weights)

    def cleanup(self):
        """Clean up resources."""
        if self.algorithm:
            self.algorithm.stop()

        if ray.is_initialized():
            ray.shutdown()

        logger.info("Parallel trainer cleanup completed")
