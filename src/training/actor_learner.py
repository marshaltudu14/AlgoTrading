"""
Explicit Actor-Learner implementation for parallel training.
"""

import ray
import torch
import numpy as np
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from collections import deque
import threading
import queue

from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent

logger = logging.getLogger(__name__)

@ray.remote
class ExperienceReplayBuffer:
    """
    Distributed experience replay buffer for parallel training.
    """
    def __init__(self, max_size: int = 100000, batch_size: int = 256):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
        logger.info(f"ExperienceReplayBuffer initialized: max_size={max_size}, batch_size={batch_size}")
    
    def add_experiences(self, experiences: List[Tuple]) -> None:
        """Add a batch of experiences to the buffer."""
        with self.lock:
            self.buffer.extend(experiences)
        
        logger.debug(f"Added {len(experiences)} experiences, buffer size: {len(self.buffer)}")
    
    def sample_batch(self) -> List[Tuple]:
        """Sample a batch of experiences for training."""
        with self.lock:
            if len(self.buffer) < self.batch_size:
                return list(self.buffer)  # Return all if not enough for full batch
            
            # Random sampling
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
        
        return batch
    
    def get_size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()

@ray.remote
class TradingActor:
    """
    Actor process that interacts with TradingEnv and collects experiences.
    """
    def __init__(self, actor_id: int, env_config: Dict[str, Any], agent_config: Dict[str, Any]):
        self.actor_id = actor_id
        self.env_config = env_config
        self.agent_config = agent_config
        
        # Initialize environment
        data_loader = DataLoader(
            final_data_dir=env_config.get("final_data_dir", "data/final"),
            raw_data_dir=env_config.get("raw_data_dir", "data/raw"),
            chunk_size=env_config.get("chunk_size", 10000),
            use_parquet=env_config.get("use_parquet", True)
        )
        
        self.env = TradingEnv(
            data_loader=data_loader,
            symbol=env_config.get("symbol", "DEFAULT_SYMBOL"),
            initial_capital=env_config.get("initial_capital", 100000.0),
            lookback_window=env_config.get("lookback_window", 50),
            trailing_stop_percentage=env_config.get("trailing_stop_percentage", 0.02),
            reward_function=env_config.get("reward_function", "pnl"),
            episode_length=env_config.get("episode_length", 1000),
            use_streaming=env_config.get("use_streaming", True)
        )
        
        # Initialize agent (local copy for action selection)
        self.agent = PPOAgent(
            observation_dim=agent_config.get("observation_dim", 10),
            action_dim_discrete=agent_config.get("action_dim_discrete", 2),
            action_dim_continuous=agent_config.get("action_dim_continuous", 1),
            hidden_dim=agent_config.get("hidden_dim", 64),
            lr_actor=agent_config.get("lr_actor", 0.001),
            lr_critic=agent_config.get("lr_critic", 0.001),
            gamma=agent_config.get("gamma", 0.99),
            epsilon_clip=agent_config.get("epsilon_clip", 0.2),
            k_epochs=agent_config.get("k_epochs", 3)
        )
        
        self.episode_count = 0
        self.total_steps = 0
        
        logger.info(f"TradingActor {actor_id} initialized for symbol: {env_config.get('symbol')}")
    
    def collect_experiences(self, num_episodes: int = 1) -> List[Tuple]:
        """
        Collect experiences by running episodes in the environment.
        
        Args:
            num_episodes: Number of episodes to run
            
        Returns:
            List of experiences (state, action, reward, next_state, done)
        """
        all_experiences = []
        
        for episode in range(num_episodes):
            experiences = []
            observation = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            
            while not done:
                # Select action using current policy
                action = self.agent.select_action(observation)
                
                # Take step in environment
                next_observation, reward, done, truncated, info = self.env.step(action)
                
                # Store experience
                experience = (observation, action, reward, next_observation, done)
                experiences.append(experience)
                
                observation = next_observation
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1
            
            all_experiences.extend(experiences)
            self.episode_count += 1
            
            logger.debug(
                f"Actor {self.actor_id} completed episode {self.episode_count}: "
                f"reward={episode_reward:.2f}, steps={episode_steps}"
            )
        
        logger.info(
            f"Actor {self.actor_id} collected {len(all_experiences)} experiences "
            f"from {num_episodes} episodes"
        )
        
        return all_experiences
    
    def update_policy(self, policy_weights: Dict[str, Any]) -> None:
        """Update the actor's policy with new weights from the learner."""
        try:
            # Update agent's policy networks
            self.agent.actor.load_state_dict(policy_weights.get("actor", {}))
            self.agent.policy_old.load_state_dict(policy_weights.get("policy_old", {}))
            
            logger.debug(f"Actor {self.actor_id} policy updated")
        except Exception as e:
            logger.error(f"Actor {self.actor_id} failed to update policy: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "actor_id": self.actor_id,
            "episode_count": self.episode_count,
            "total_steps": self.total_steps,
            "symbol": self.env_config.get("symbol")
        }

@ray.remote
class TradingLearner:
    """
    Learner process that updates the model based on experiences from actors.
    """
    def __init__(self, agent_config: Dict[str, Any], learning_config: Dict[str, Any]):
        self.agent_config = agent_config
        self.learning_config = learning_config
        
        # Initialize master agent
        self.agent = PPOAgent(
            observation_dim=agent_config.get("observation_dim", 10),
            action_dim_discrete=agent_config.get("action_dim_discrete", 2),
            action_dim_continuous=agent_config.get("action_dim_continuous", 1),
            hidden_dim=agent_config.get("hidden_dim", 64),
            lr_actor=agent_config.get("lr_actor", 0.001),
            lr_critic=agent_config.get("lr_critic", 0.001),
            gamma=agent_config.get("gamma", 0.99),
            epsilon_clip=agent_config.get("epsilon_clip", 0.2),
            k_epochs=agent_config.get("k_epochs", 3)
        )
        
        self.update_count = 0
        self.total_experiences_processed = 0
        
        logger.info("TradingLearner initialized")
    
    def update_model(self, experiences: List[Tuple]) -> Dict[str, Any]:
        """
        Update the model using collected experiences.
        
        Args:
            experiences: List of experiences from actors
            
        Returns:
            Training metrics
        """
        if not experiences:
            return {"error": "No experiences provided"}
        
        try:
            # Update the agent using the experiences
            self.agent.learn(experiences)
            
            self.update_count += 1
            self.total_experiences_processed += len(experiences)
            
            # Return updated policy weights and metrics
            policy_weights = {
                "actor": self.agent.actor.state_dict(),
                "critic": self.agent.critic.state_dict(),
                "policy_old": self.agent.policy_old.state_dict()
            }
            
            metrics = {
                "update_count": self.update_count,
                "experiences_processed": len(experiences),
                "total_experiences": self.total_experiences_processed,
                "policy_weights": policy_weights
            }
            
            logger.info(
                f"Learner update {self.update_count}: processed {len(experiences)} experiences"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Learner update failed: {e}")
            return {"error": str(e)}
    
    def get_policy_weights(self) -> Dict[str, Any]:
        """Get current policy weights for distribution to actors."""
        return {
            "actor": self.agent.actor.state_dict(),
            "critic": self.agent.critic.state_dict(),
            "policy_old": self.agent.policy_old.state_dict()
        }
    
    def save_model(self, path: str) -> None:
        """Save the current model."""
        self.agent.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            "update_count": self.update_count,
            "total_experiences_processed": self.total_experiences_processed
        }

class ActorLearnerCoordinator:
    """
    Coordinates the Actor-Learner parallel training system.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.actors = []
        self.learner = None
        self.replay_buffer = None
        self.is_running = False

        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(
                num_cpus=config.get("num_cpus"),
                num_gpus=config.get("num_gpus", 0),
                local_mode=config.get("local_mode", False)
            )
            logger.info("Ray initialized for Actor-Learner system")

    def setup(self) -> None:
        """Set up the Actor-Learner system."""
        # Create experience replay buffer
        buffer_config = self.config.get("buffer_config", {})
        self.replay_buffer = ExperienceReplayBuffer.remote(
            max_size=buffer_config.get("max_size", 100000),
            batch_size=buffer_config.get("batch_size", 256)
        )

        # Create learner
        agent_config = self.config.get("agent_config", {})
        learning_config = self.config.get("learning_config", {})
        self.learner = TradingLearner.remote(agent_config, learning_config)

        # Create actors
        num_actors = self.config.get("num_actors", 4)
        env_config = self.config.get("env_config", {})

        for actor_id in range(num_actors):
            actor = TradingActor.remote(actor_id, env_config, agent_config)
            self.actors.append(actor)

        logger.info(f"Actor-Learner system setup: {num_actors} actors, 1 learner, 1 replay buffer")

    def train(self, num_iterations: int = 100, sync_freq: int = 10) -> Dict[str, Any]:
        """
        Run the Actor-Learner training loop.

        Args:
            num_iterations: Number of training iterations
            sync_freq: Frequency of policy synchronization (every N iterations)

        Returns:
            Training results
        """
        if not self.actors or not self.learner or not self.replay_buffer:
            raise RuntimeError("System not set up. Call setup() first.")

        self.is_running = True
        training_metrics = []

        logger.info(f"Starting Actor-Learner training for {num_iterations} iterations")

        try:
            for iteration in range(num_iterations):
                # Phase 1: Actors collect experiences in parallel
                experience_futures = []
                for actor in self.actors:
                    future = actor.collect_experiences.remote(num_episodes=1)
                    experience_futures.append(future)

                # Wait for all actors to complete
                all_experiences = []
                for future in experience_futures:
                    experiences = ray.get(future)
                    all_experiences.extend(experiences)

                # Phase 2: Add experiences to replay buffer
                if all_experiences:
                    ray.get(self.replay_buffer.add_experiences.remote(all_experiences))

                # Phase 3: Learner updates model
                batch = ray.get(self.replay_buffer.sample_batch.remote())
                if batch:
                    update_result = ray.get(self.learner.update_model.remote(batch))
                    training_metrics.append(update_result)

                # Phase 4: Synchronize policy weights (every sync_freq iterations)
                if (iteration + 1) % sync_freq == 0:
                    policy_weights = ray.get(self.learner.get_policy_weights.remote())

                    # Update all actors with new policy
                    update_futures = []
                    for actor in self.actors:
                        future = actor.update_policy.remote(policy_weights)
                        update_futures.append(future)

                    # Wait for all updates to complete
                    ray.get(update_futures)

                    logger.info(f"Iteration {iteration + 1}: Policy synchronized across all actors")

                # Log progress
                if (iteration + 1) % 10 == 0:
                    buffer_size = ray.get(self.replay_buffer.get_size.remote())
                    learner_stats = ray.get(self.learner.get_stats.remote())

                    logger.info(
                        f"Iteration {iteration + 1}/{num_iterations}: "
                        f"Buffer size={buffer_size}, "
                        f"Learner updates={learner_stats['update_count']}"
                    )

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_running = False

        # Collect final statistics
        final_stats = self.get_system_stats()

        results = {
            "iterations_completed": len(training_metrics),
            "training_metrics": training_metrics,
            "final_stats": final_stats
        }

        logger.info("Actor-Learner training completed")
        return results

    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics from all system components."""
        if not self.actors or not self.learner or not self.replay_buffer:
            return {}

        # Get stats from all components
        actor_stats_futures = [actor.get_stats.remote() for actor in self.actors]
        actor_stats = ray.get(actor_stats_futures)

        learner_stats = ray.get(self.learner.get_stats.remote())
        buffer_size = ray.get(self.replay_buffer.get_size.remote())

        return {
            "actor_stats": actor_stats,
            "learner_stats": learner_stats,
            "buffer_size": buffer_size,
            "num_actors": len(self.actors)
        }

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.learner:
            ray.get(self.learner.save_model.remote(path))

    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False

        if ray.is_initialized():
            ray.shutdown()

        logger.info("Actor-Learner system cleanup completed")
