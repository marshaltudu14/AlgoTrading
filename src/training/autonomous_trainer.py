"""
Autonomous Training Module

This module orchestrates the generational training loop for autonomous agents,
integrating neural architecture search, self-modification, and evolutionary
optimization to create truly autonomous trading systems.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import yaml
from datetime import datetime
import pickle

from src.agents.autonomous_agent import AutonomousAgent
from src.nas.search_controller import NASController
from src.nas.search_space import SearchSpace
from src.reasoning.self_modification import (
    SelfModificationManager, 
    PerformanceMetrics, 
    ModificationConfig
)
from src.backtesting.environment import TradingEnv
from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader
from src.config.instrument import Instrument
from src.utils.instrument_loader import load_instruments
from src.config.config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)


@dataclass
class AutonomousTrainingConfig:
    """Configuration for autonomous training."""
    # Population parameters
    population_size: int = 20
    generations: int = 50
    elite_size: int = 5
    
    # Agent parameters
    observation_dim: int = 65
    action_dim: int = 5
    hidden_dim: int = 128
    memory_size: int = 1000
    memory_embedding_dim: int = 64
    
    # Training parameters
    episodes_per_evaluation: int = 10
    episode_length: int = 1000
    # initial_capital will be loaded from config.INITIAL_CAPITAL
    
    # Data parameters
    symbol: str = "NIFTY"
    lookback_window: int = 50
    
    # Evolution parameters
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    
    # Self-modification parameters
    enable_self_modification: bool = True
    modification_frequency: int = 5  # Every N generations
    
    # Saving parameters - only save final champion
    save_directory: str = "models/autonomous_agents"
    
    # Evaluation parameters
    fitness_metrics: List[str] = None
    
    def __post_init__(self):
        """Set default fitness metrics if not provided."""
        if self.fitness_metrics is None:
            self.fitness_metrics = ["sharpe_ratio", "profit_factor", "max_drawdown"]


class AutonomousTrainer:
    """
    Autonomous Trainer for evolutionary agent optimization.
    
    This class orchestrates the generational training loop, managing
    populations of autonomous agents, evaluating their performance,
    and evolving them using neural architecture search and self-modification.
    """
    
    def __init__(self, config: AutonomousTrainingConfig):
        """
        Initialize the Autonomous Trainer.
        
        Args:
            config: Configuration for autonomous training
        """
        self.config = config
        
        # Initialize components
        self.search_space = SearchSpace()
        self.nas_controller = NASController(
            search_space=self.search_space,
            population_size=config.population_size,
            elite_size=config.elite_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            max_generations=config.generations
        )
        
        self.self_modification_manager = SelfModificationManager(
            nas_controller=self.nas_controller
        )
        
        # Initialize data and environment
        self.data_loader = DataLoader()
        self.instruments = load_instruments("config/instruments.yaml")

        # Extract base symbol (remove timeframe suffix)
        base_symbol = self.data_loader.get_base_symbol(config.symbol)
        self.instrument = self.instruments.get(base_symbol)

        if self.instrument is None:
            raise ValueError(f"Instrument {base_symbol} not found (original symbol: {config.symbol})")
        
        # Population and training state
        self.population: List[AutonomousAgent] = []
        self.generation = 0
        self.best_agent: Optional[AutonomousAgent] = None
        self.best_fitness = float('-inf')
        
        # Training history
        self.fitness_history: List[List[float]] = []
        self.best_fitness_history: List[float] = []
        self.modification_history: List[Dict] = []
        
        # Create save directory
        os.makedirs(config.save_directory, exist_ok=True)
        
        logger.info(f"Initialized AutonomousTrainer with population size {config.population_size}")
    
    def initialize_population(self) -> None:
        """Initialize the population of autonomous agents."""
        logger.info("Initializing population...")
        
        # Initialize NAS controller population
        self.nas_controller.initialize_population(
            input_dim=self.config.observation_dim,
            output_dim=self.config.action_dim
        )
        
        # Create agents from NAS architectures and hyperparameters
        self.population = []
        for i, individual in enumerate(self.nas_controller.population):
            # Create autonomous agent with evolved hyperparameters
            agent = AutonomousAgent(
                observation_dim=self.config.observation_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
                memory_size=self.config.memory_size,
                memory_embedding_dim=self.config.memory_embedding_dim,
                hyperparameters=individual.hyperparameters
            )

            # Store architecture and individual reference for evolution
            agent._nas_architecture = individual.architecture
            agent._nas_individual = individual
            agent._agent_id = i

            self.population.append(agent)
        
        logger.info(f"Created population of {len(self.population)} agents")
    
    def evaluate_agent(self, agent: AutonomousAgent) -> PerformanceMetrics:
        """
        Evaluate a single agent's performance through backtesting.
        
        Args:
            agent: The agent to evaluate
            
        Returns:
            PerformanceMetrics object with evaluation results
        """
        # Create trading environment
        env = TradingEnv(
            data_loader=self.data_loader,
            symbol=self.config.symbol,
            initial_capital=INITIAL_CAPITAL,
            lookback_window=self.config.lookback_window,
            episode_length=self.config.episode_length
        )

        # Reset environment to get actual observation dimension
        obs = env.reset()
        actual_obs_dim = obs.shape[0] if hasattr(obs, 'shape') else len(obs)

        # Update agent's observation dimension if mismatch
        if actual_obs_dim != self.config.observation_dim:
            from src.utils.error_logger import log_warning
            log_warning(f"Observation dimension mismatch: expected {self.config.observation_dim}, got {actual_obs_dim}",
                       f"Agent evaluation for symbol {self.config.symbol}")
            # Update the agent's observation dimension instead of skipping
            agent.observation_dim = actual_obs_dim
            # Reinitialize the agent's world model with correct dimensions
            if hasattr(agent, 'world_model'):
                # Recreate world model with correct input dimension
                from src.models.world_model import TransformerWorldModel
                agent.world_model = TransformerWorldModel(
                    input_dim=actual_obs_dim,
                    action_dim=agent.action_dim,
                    hidden_dim=agent.hidden_dim,
                    prediction_horizon=agent.prediction_horizon
                )
        
        # Run multiple episodes for robust evaluation
        episode_results = []
        
        for episode in range(self.config.episodes_per_evaluation):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            step_count = 0
            
            while not done and step_count < self.config.episode_length:
                # Get action from agent
                action = agent.act(obs)
                
                # Take step in environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if truncated:
                    done = True
            
            # Collect episode results
            episode_results.append({
                'total_reward': episode_reward,
                'final_capital': env.backtesting_engine._capital,
                'num_trades': env.backtesting_engine._trade_count,
                'realized_pnl': env.backtesting_engine._total_realized_pnl
            })
        
        # Calculate performance metrics
        total_returns = [r['total_reward'] for r in episode_results]
        final_capitals = [r['final_capital'] for r in episode_results]
        
        # Calculate metrics
        avg_return = np.mean(total_returns)
        volatility = np.std(total_returns) if len(total_returns) > 1 else 0.0
        sharpe_ratio = avg_return / (volatility + 1e-8)
        
        # Calculate profit factor and drawdown
        avg_capital = np.mean(final_capitals)
        profit_factor = avg_capital / INITIAL_CAPITAL
        
        # Simple drawdown calculation
        returns_series = np.array(total_returns)
        cumulative_returns = np.cumsum(returns_series)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate calculation
        positive_episodes = sum(1 for r in total_returns if r > 0)
        win_rate = positive_episodes / len(total_returns) if total_returns else 0.0
        
        # Total trades
        total_trades = sum(r['num_trades'] for r in episode_results)
        
        return PerformanceMetrics(
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return=avg_return,
            volatility=volatility,
            num_trades=total_trades,
            recent_performance_trend=0.0  # Will be calculated later
        )
    
    def evaluate_population(self) -> List[float]:
        """
        Evaluate the entire population and return fitness scores.
        
        Returns:
            List of fitness scores for each agent
        """
        logger.info(f"Evaluating population for generation {self.generation}")
        
        fitness_scores = []
        performance_metrics = []
        
        for i, agent in enumerate(self.population):
            logger.info(f"Evaluating agent {i+1}/{len(self.population)}")
            
            try:
                metrics = self.evaluate_agent(agent)
                performance_metrics.append(metrics)
                
                # Calculate composite fitness score
                fitness = self._calculate_fitness(metrics)
                fitness_scores.append(fitness)
                
                logger.info(f"Agent {i+1}: Fitness={fitness:.4f}, "
                           f"Sharpe={metrics.sharpe_ratio:.3f}, "
                           f"PF={metrics.profit_factor:.3f}")
                
            except Exception as e:
                from src.utils.error_logger import log_error
                log_error(f"Error evaluating agent {i+1}: {e}", f"Generation: {self.generation}, Symbol: {self.config.symbol}")
                logger.error(f"Error evaluating agent {i+1}: {e}")
                fitness_scores.append(0.0)
                performance_metrics.append(PerformanceMetrics())
        
        # Store fitness history
        self.fitness_history.append(fitness_scores.copy())
        
        # Update best agent
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_agent = self.population[best_idx]
            logger.info(f"New best agent found with fitness {self.best_fitness:.4f}")
        
        self.best_fitness_history.append(max(fitness_scores))
        
        # Apply self-modification if enabled
        if (self.config.enable_self_modification and 
            self.generation % self.config.modification_frequency == 0):
            self._apply_self_modification(performance_metrics)
        
        return fitness_scores

    def _calculate_fitness(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate composite fitness score from performance metrics.

        Args:
            metrics: Performance metrics

        Returns:
            Composite fitness score
        """
        # Weighted combination of metrics
        weights = {
            'sharpe_ratio': 0.4,
            'profit_factor': 0.3,
            'max_drawdown': 0.2,  # Negative impact
            'win_rate': 0.1
        }

        # Normalize and combine metrics
        sharpe_component = max(0, metrics.sharpe_ratio) * weights['sharpe_ratio']
        profit_component = max(0, metrics.profit_factor - 1.0) * weights['profit_factor']
        drawdown_component = max(0, -metrics.max_drawdown) * weights['max_drawdown']  # Penalty for drawdown
        winrate_component = metrics.win_rate * weights['win_rate']

        fitness = sharpe_component + profit_component - drawdown_component + winrate_component

        return max(0.0, fitness)  # Ensure non-negative fitness

    def _apply_self_modification(self, performance_metrics: List[PerformanceMetrics]) -> None:
        """Apply self-modification to agents based on performance."""
        logger.info("Applying self-modification to population")

        modifications_applied = []

        for i, (agent, metrics) in enumerate(zip(self.population, performance_metrics)):
            try:
                modifications = self.self_modification_manager.check_performance_and_adapt(
                    agent=agent,
                    performance_metrics=metrics,
                    episode_number=self.generation
                )
                modifications_applied.append({
                    'agent_id': i,
                    'modifications': [mod.value for mod in modifications]
                })

            except Exception as e:
                logger.error(f"Self-modification failed for agent {i}: {e}")

        self.modification_history.append({
            'generation': self.generation,
            'modifications': modifications_applied
        })

    def evolve_population(self, fitness_scores: List[float]) -> None:
        """
        Evolve the population to the next generation.

        Args:
            fitness_scores: Fitness scores for current population
        """
        logger.info(f"Evolving population for generation {self.generation + 1}")

        # Evolve using NAS controller
        new_nas_population = self.nas_controller.evolve_population(fitness_scores)

        # Create new agent population
        new_population = []

        for i, individual in enumerate(new_nas_population):
            # Create new agent with evolved architecture and hyperparameters
            agent = AutonomousAgent(
                observation_dim=self.config.observation_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
                memory_size=self.config.memory_size,
                memory_embedding_dim=self.config.memory_embedding_dim,
                hyperparameters=individual.hyperparameters
            )

            # Store architecture and individual reference
            agent._nas_architecture = individual.architecture
            agent._nas_individual = individual
            agent._agent_id = i

            new_population.append(agent)

        self.population = new_population
        self.generation += 1

        logger.info(f"Evolved to generation {self.generation}")


    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.fitness_history:
            return {}

        current_fitness = self.fitness_history[-1]

        # Get best agent hyperparameters
        best_agent_hyperparams = {}
        if self.best_agent and hasattr(self.best_agent, 'get_hyperparameters'):
            best_agent_hyperparams = self.best_agent.get_hyperparameters()

        # Get population hyperparameter statistics
        population_hyperparams = {}
        if self.population and hasattr(self.population[0], 'get_hyperparameters'):
            all_hyperparams = [agent.get_hyperparameters() for agent in self.population]
            if all_hyperparams:
                for param_name in all_hyperparams[0].keys():
                    param_values = [hp[param_name] for hp in all_hyperparams]
                    population_hyperparams[param_name] = {
                        'mean': np.mean(param_values),
                        'std': np.std(param_values),
                        'min': np.min(param_values),
                        'max': np.max(param_values)
                    }

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': self.best_fitness,
            'current_avg_fitness': np.mean(current_fitness),
            'current_std_fitness': np.std(current_fitness),
            'fitness_improvement': (
                self.best_fitness_history[-1] - self.best_fitness_history[0]
                if len(self.best_fitness_history) > 1 else 0.0
            ),
            'total_modifications': len(self.modification_history),
            'best_agent_hyperparameters': best_agent_hyperparams,
            'population_hyperparameter_statistics': population_hyperparams,
            'nas_statistics': self.nas_controller.get_statistics()
        }

    def save_champion_agent(self, filepath: str) -> None:
        """
        Save the champion agent with complete state for backtesting and live trading.

        Args:
            filepath: Path to save the champion agent
        """
        if self.best_agent is None:
            logger.warning("No champion agent to save")
            return

        # Prepare champion data
        champion_data = {
            'agent_state_dict': self.best_agent.state_dict() if hasattr(self.best_agent, 'state_dict') else None,
            'agent_config': {
                'observation_dim': self.config.observation_dim,
                'action_dim': self.config.action_dim,
                'hidden_dim': self.config.hidden_dim,
                'memory_size': self.config.memory_size,
                'memory_embedding_dim': self.config.memory_embedding_dim
            },
            'hyperparameters': self.best_agent.get_hyperparameters(),
            'architecture': getattr(self.best_agent, '_nas_architecture', None),
            'fitness_score': self.best_fitness,
            'generation': self.generation,
            'training_config': self.config,
            'model_type': 'autonomous_agent',
            'version': '1.0'
        }

        # Save world model state if available
        if hasattr(self.best_agent, 'world_model'):
            champion_data['world_model_state_dict'] = self.best_agent.world_model.state_dict()

        # Save external memory if available
        if hasattr(self.best_agent, 'external_memory'):
            champion_data['external_memory_state'] = {
                'memories': getattr(self.best_agent.external_memory, 'memories', []),
                'config': getattr(self.best_agent.external_memory, 'config', {})
            }

        # Save pattern recognizer state if available
        if hasattr(self.best_agent, 'pattern_recognizer'):
            champion_data['pattern_recognizer_config'] = {
                'sequence_length': getattr(self.best_agent.pattern_recognizer, 'sequence_length', 50),
                'min_pattern_confidence': getattr(self.best_agent.pattern_recognizer, 'min_pattern_confidence', 0.6),
                'use_neural_network': getattr(self.best_agent.pattern_recognizer, 'use_neural_network', True)
            }

        # Save market classifier state if available
        if hasattr(self.best_agent, 'market_classifier'):
            champion_data['market_classifier_config'] = {
                'trend_period': getattr(self.best_agent.market_classifier, 'trend_period', 20),
                'volatility_period': getattr(self.best_agent.market_classifier, 'volatility_period', 14),
                'trend_threshold': getattr(self.best_agent.market_classifier, 'trend_threshold', 25.0)
            }

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save using torch
        torch.save(champion_data, filepath)

        logger.info(f"Champion agent saved to {filepath}")
        logger.info(f"Champion fitness: {self.best_fitness:.4f}")
        logger.info(f"Champion hyperparameters: {champion_data['hyperparameters']}")


def run_autonomous_stage(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main function to run the autonomous training stage.

    Args:
        config: Configuration dictionary for autonomous training

    Returns:
        Dictionary with training results and statistics
    """
    logger.info("Starting autonomous training stage")

    # Create training configuration with symbol and initial capital from config
    autonomous_config = config.get('autonomous', {})
    # Ensure initial_capital comes from the global config
    autonomous_config.pop('initial_capital', None)  # Remove if present
    # Add symbol from the main config
    autonomous_config['symbol'] = config.get('symbol', 'NIFTY')
    training_config = AutonomousTrainingConfig(**autonomous_config)

    # Initialize trainer
    trainer = AutonomousTrainer(training_config)

    # Initialize population
    trainer.initialize_population()

    # Main training loop
    for generation in range(training_config.generations):
        logger.info(f"=== Generation {generation + 1}/{training_config.generations} ===")

        # Evaluate population
        fitness_scores = trainer.evaluate_population()

        # No intermediate saving - only save final champion

        # Evolve to next generation (unless this is the last generation)
        if generation < training_config.generations - 1:
            trainer.evolve_population(fitness_scores)

        # Log progress
        stats = trainer.get_training_statistics()
        logger.info(f"Generation {generation + 1} complete. "
                   f"Best fitness: {stats['best_fitness']:.4f}, "
                   f"Avg fitness: {stats['current_avg_fitness']:.4f}")

    # Don't save individual champion agents - only the universal model is saved
    # The universal model is saved by the sequence manager
    logger.info("Autonomous training completed - universal model will be saved by sequence manager")

    # Return final statistics
    final_stats = trainer.get_training_statistics()
    final_stats['champion_path'] = None  # No individual champion saved

    logger.info("Autonomous training stage completed")
    return final_stats


def load_champion_agent(filepath: str) -> AutonomousAgent:
    """
    Load a champion agent from file.

    Args:
        filepath: Path to load the champion agent from

    Returns:
        Loaded AutonomousAgent
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Champion agent file not found: {filepath}")

    # Load champion data
    champion_data = torch.load(filepath, map_location='cpu')

    # Create agent with saved configuration
    agent_config = champion_data['agent_config']
    hyperparameters = champion_data['hyperparameters']

    agent = AutonomousAgent(
        observation_dim=agent_config['observation_dim'],
        action_dim=agent_config['action_dim'],
        hidden_dim=agent_config['hidden_dim'],
        memory_size=agent_config['memory_size'],
        memory_embedding_dim=agent_config['memory_embedding_dim'],
        hyperparameters=hyperparameters
    )

    # Load world model state if available
    if 'world_model_state_dict' in champion_data and champion_data['world_model_state_dict']:
        agent.world_model.load_state_dict(champion_data['world_model_state_dict'])

    # Load external memory if available
    if 'external_memory_state' in champion_data:
        memory_state = champion_data['external_memory_state']
        if 'memories' in memory_state:
            agent.external_memory.memories = memory_state['memories']

    # Store architecture reference
    if 'architecture' in champion_data:
        agent._nas_architecture = champion_data['architecture']

    logger.info(f"Champion agent loaded from {filepath}")
    logger.info(f"Champion fitness: {champion_data.get('fitness_score', 'unknown')}")

    return agent
