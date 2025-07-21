"""
Self-Modification Logic for Autonomous Trading Agent

This module provides the logic for self-modification, allowing the agent
to react to its own performance and trigger adaptive changes including
architecture evolution, risk adjustment, and neural plasticity.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import copy

logger = logging.getLogger(__name__)


class ModificationType(Enum):
    """Types of modifications the agent can perform."""
    ARCHITECTURE_SEARCH = "architecture_search"
    RISK_ADJUSTMENT = "risk_adjustment"
    SYNAPTIC_PRUNING = "synaptic_pruning"
    NEUROGENESIS = "neurogenesis"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    LEARNING_RATE_ADAPTATION = "learning_rate_adaptation"
    NO_MODIFICATION = "no_modification"


@dataclass
class PerformanceMetrics:
    """
    Container for agent performance metrics.
    """
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Additional metrics
    num_trades: int = 0
    avg_trade_duration: float = 0.0
    consecutive_losses: int = 0
    recent_performance_trend: float = 0.0  # Positive = improving, negative = declining
    
    def __post_init__(self):
        """Validate metrics after initialization."""
        if self.max_drawdown > 0:
            logger.warning("Max drawdown should typically be negative or zero")


@dataclass
class ModificationConfig:
    """
    Configuration for self-modification thresholds and parameters.
    """
    # Performance thresholds
    min_profit_factor: float = 1.2
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = -0.15  # -15%
    min_win_rate: float = 0.45
    max_consecutive_losses: int = 5
    
    # Architecture search triggers
    enable_architecture_search: bool = True
    architecture_search_cooldown: int = 10  # Episodes between searches
    min_episodes_before_search: int = 20
    
    # Risk adjustment parameters
    enable_risk_adjustment: bool = True
    risk_adjustment_factor: float = 0.1  # How much to adjust risk
    max_risk_reduction: float = 0.5  # Maximum risk reduction
    
    # Neural plasticity parameters
    enable_synaptic_pruning: bool = True
    pruning_threshold: float = 0.01  # Prune weights below this threshold
    max_pruning_ratio: float = 0.2  # Maximum percentage of weights to prune
    
    enable_neurogenesis: bool = True
    neurogenesis_threshold: float = 0.8  # Performance threshold to trigger growth
    max_growth_ratio: float = 0.1  # Maximum percentage of new neurons
    
    # Memory management
    enable_memory_consolidation: bool = True
    memory_consolidation_threshold: int = 1000  # Number of memories before consolidation
    
    # Learning rate adaptation
    enable_learning_rate_adaptation: bool = True
    learning_rate_adjustment_factor: float = 0.1
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 1e-2


class SelfModificationManager:
    """
    Manager for autonomous agent self-modification logic.
    
    This class implements the logic for performance-based adaptive changes,
    including architecture evolution, risk adjustment, and neural plasticity.
    """
    
    def __init__(
        self,
        config: Optional[ModificationConfig] = None,
        nas_controller: Optional[Any] = None
    ):
        """
        Initialize the Self-Modification Manager.
        
        Args:
            config: Configuration for modification thresholds and parameters
            nas_controller: Neural Architecture Search controller for architecture evolution
        """
        self.config = config or ModificationConfig()
        self.nas_controller = nas_controller
        
        # Track modification history
        self.modification_history: List[Dict[str, Any]] = []
        self.last_architecture_search_episode = -1
        self.performance_history: List[PerformanceMetrics] = []
        
        # Current agent state
        self.current_risk_factor = 1.0
        self.current_learning_rate = 1e-3
        
        logger.info("Initialized SelfModificationManager")
    
    def check_performance_and_adapt(
        self,
        agent: Any,
        performance_metrics: Union[PerformanceMetrics, Dict[str, float]],
        episode_number: int = 0
    ) -> List[ModificationType]:
        """
        Check agent performance and trigger adaptive changes.
        
        Args:
            agent: The autonomous agent to potentially modify
            performance_metrics: Current performance metrics
            episode_number: Current episode/evaluation number
            
        Returns:
            List of modifications that were applied
        """
        # Convert dict to PerformanceMetrics if needed
        if isinstance(performance_metrics, dict):
            performance_metrics = PerformanceMetrics(**performance_metrics)
        
        # Store performance history
        self.performance_history.append(performance_metrics)
        
        # Keep only recent history (last 50 episodes)
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]
        
        modifications_applied = []
        
        logger.info(f"Evaluating performance for episode {episode_number}")
        logger.info(f"Profit Factor: {performance_metrics.profit_factor:.3f}, "
                   f"Sharpe: {performance_metrics.sharpe_ratio:.3f}, "
                   f"Drawdown: {performance_metrics.max_drawdown:.3f}")
        
        # 1. Check for architecture search triggers
        if self._should_trigger_architecture_search(performance_metrics, episode_number):
            if self._trigger_architecture_search(agent, performance_metrics):
                modifications_applied.append(ModificationType.ARCHITECTURE_SEARCH)
        
        # 2. Check for risk adjustment
        if self._should_adjust_risk(performance_metrics):
            if self._adjust_risk_parameters(agent, performance_metrics):
                modifications_applied.append(ModificationType.RISK_ADJUSTMENT)
        
        # 3. Check for synaptic pruning
        if self._should_prune_synapses(performance_metrics):
            if self._perform_synaptic_pruning(agent, performance_metrics):
                modifications_applied.append(ModificationType.SYNAPTIC_PRUNING)
        
        # 4. Check for neurogenesis
        if self._should_trigger_neurogenesis(performance_metrics):
            if self._perform_neurogenesis(agent, performance_metrics):
                modifications_applied.append(ModificationType.NEUROGENESIS)
        
        # 5. Check for memory consolidation
        if self._should_consolidate_memory(agent, performance_metrics):
            if self._consolidate_memory(agent, performance_metrics):
                modifications_applied.append(ModificationType.MEMORY_CONSOLIDATION)
        
        # 6. Check for learning rate adaptation
        if self._should_adapt_learning_rate(performance_metrics):
            if self._adapt_learning_rate(agent, performance_metrics):
                modifications_applied.append(ModificationType.LEARNING_RATE_ADAPTATION)
        
        # Record modifications
        if modifications_applied:
            self.modification_history.append({
                'episode': episode_number,
                'modifications': [mod.value for mod in modifications_applied],
                'performance_before': performance_metrics,
                'timestamp': torch.tensor(episode_number, dtype=torch.float32)
            })
            
            logger.info(f"Applied modifications: {[mod.value for mod in modifications_applied]}")
        else:
            modifications_applied.append(ModificationType.NO_MODIFICATION)
        
        return modifications_applied
    
    def _should_trigger_architecture_search(
        self, 
        performance_metrics: PerformanceMetrics, 
        episode_number: int
    ) -> bool:
        """Determine if architecture search should be triggered."""
        if not self.config.enable_architecture_search or self.nas_controller is None:
            return False
        
        # Check cooldown period
        if (episode_number - self.last_architecture_search_episode) < self.config.architecture_search_cooldown:
            return False
        
        # Need minimum episodes before first search
        if episode_number < self.config.min_episodes_before_search:
            return False
        
        # Trigger conditions
        poor_performance = (
            performance_metrics.profit_factor < self.config.min_profit_factor or
            performance_metrics.sharpe_ratio < self.config.min_sharpe_ratio or
            performance_metrics.max_drawdown < self.config.max_drawdown_threshold
        )
        
        # Also trigger if performance is stagnating
        stagnating = self._is_performance_stagnating()
        
        return poor_performance or stagnating
    
    def _should_adjust_risk(self, performance_metrics: PerformanceMetrics) -> bool:
        """Determine if risk parameters should be adjusted."""
        if not self.config.enable_risk_adjustment:
            return False
        
        # Reduce risk if drawdown is high or consecutive losses
        high_risk_conditions = (
            performance_metrics.max_drawdown < self.config.max_drawdown_threshold or
            performance_metrics.consecutive_losses >= self.config.max_consecutive_losses or
            performance_metrics.volatility > 0.3  # High volatility
        )
        
        # Increase risk if performance is very good and stable
        low_risk_conditions = (
            performance_metrics.profit_factor > 2.0 and
            performance_metrics.sharpe_ratio > 1.5 and
            performance_metrics.max_drawdown > -0.05 and  # Low drawdown
            self.current_risk_factor < 1.0
        )
        
        return high_risk_conditions or low_risk_conditions
    
    def _should_prune_synapses(self, performance_metrics: PerformanceMetrics) -> bool:
        """Determine if synaptic pruning should be performed."""
        if not self.config.enable_synaptic_pruning:
            return False
        
        # Prune if performance is poor (might be overfitting)
        return (
            performance_metrics.profit_factor < 1.0 or
            performance_metrics.sharpe_ratio < 0.0 or
            len(self.performance_history) > 10 and self._is_performance_declining()
        )
    
    def _should_trigger_neurogenesis(self, performance_metrics: PerformanceMetrics) -> bool:
        """Determine if neurogenesis should be triggered."""
        if not self.config.enable_neurogenesis:
            return False
        
        # Trigger neurogenesis if performance is good but could be better
        return (
            performance_metrics.profit_factor > self.config.neurogenesis_threshold and
            performance_metrics.sharpe_ratio > 0.5 and
            performance_metrics.recent_performance_trend > 0
        )
    
    def _should_consolidate_memory(self, agent: Any, performance_metrics: PerformanceMetrics) -> bool:
        """Determine if memory consolidation should be performed."""
        if not self.config.enable_memory_consolidation:
            return False
        
        # Check if agent has external memory and it's getting full
        if hasattr(agent, 'external_memory') and hasattr(agent.external_memory, 'memories'):
            try:
                memory_size = len(agent.external_memory.memories)
                return memory_size >= self.config.memory_consolidation_threshold
            except (TypeError, AttributeError):
                return False
        
        return False
    
    def _should_adapt_learning_rate(self, performance_metrics: PerformanceMetrics) -> bool:
        """Determine if learning rate should be adapted."""
        if not self.config.enable_learning_rate_adaptation:
            return False
        
        # Adapt learning rate based on performance trends
        return len(self.performance_history) >= 5

    def _trigger_architecture_search(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Trigger neural architecture search."""
        try:
            logger.info("Triggering architecture search due to poor performance")

            # Get current architecture if available
            if hasattr(agent, 'world_model'):
                current_input_dim = agent.world_model.input_dim
                current_action_dim = agent.world_model.action_dim

                # Initialize population with current architecture as baseline
                self.nas_controller.initialize_population(current_input_dim, current_action_dim)

                # Run a few generations of evolution
                # Note: In practice, this would be integrated with the training loop
                dummy_fitness_scores = [0.5] * self.nas_controller.population_size
                for _ in range(3):  # Quick search
                    self.nas_controller.evolve_population(dummy_fitness_scores)

                # Get the best architecture
                best_architecture = self.nas_controller.get_best_architecture()

                if best_architecture:
                    logger.info("Found new architecture candidate")
                    # Note: Actual architecture replacement would happen in training loop
                    self.last_architecture_search_episode = len(self.performance_history)
                    return True

            return False

        except Exception as e:
            logger.error(f"Architecture search failed: {e}")
            return False

    def _adjust_risk_parameters(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Adjust agent's risk parameters."""
        try:
            # Determine risk adjustment direction
            if (performance_metrics.max_drawdown < self.config.max_drawdown_threshold or
                performance_metrics.consecutive_losses >= self.config.max_consecutive_losses):
                # Reduce risk
                adjustment = -self.config.risk_adjustment_factor
                new_risk_factor = max(
                    self.current_risk_factor * (1 + adjustment),
                    1.0 - self.config.max_risk_reduction
                )
                logger.info(f"Reducing risk factor from {self.current_risk_factor:.3f} to {new_risk_factor:.3f}")

            elif (performance_metrics.profit_factor > 2.0 and
                  performance_metrics.sharpe_ratio > 1.5 and
                  self.current_risk_factor < 1.0):
                # Increase risk
                adjustment = self.config.risk_adjustment_factor
                new_risk_factor = min(self.current_risk_factor * (1 + adjustment), 1.0)
                logger.info(f"Increasing risk factor from {self.current_risk_factor:.3f} to {new_risk_factor:.3f}")
            else:
                return False

            # Apply risk adjustment to agent
            self.current_risk_factor = new_risk_factor

            # If agent has risk-related attributes, update them
            if hasattr(agent, 'risk_aversion_factor'):
                agent.risk_aversion_factor = 1.0 / new_risk_factor

            if hasattr(agent, 'position_sizing_factor'):
                agent.position_sizing_factor = new_risk_factor

            return True

        except Exception as e:
            logger.error(f"Risk adjustment failed: {e}")
            return False

    def _perform_synaptic_pruning(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Perform synaptic pruning on agent's neural networks."""
        try:
            logger.info("Performing synaptic pruning")

            pruned_count = 0
            total_params = 0

            # Prune weights in the world model
            if hasattr(agent, 'world_model'):
                for name, param in agent.world_model.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        total_params += param.numel()

                        # Create mask for weights to keep
                        mask = torch.abs(param.data) > self.config.pruning_threshold

                        # Count pruned weights
                        pruned_count += torch.sum(~mask).item()

                        # Apply pruning (set small weights to zero)
                        param.data *= mask.float()

            # Prune weights in other components if they exist
            if hasattr(agent, 'state_embedder'):
                for name, param in agent.state_embedder.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        total_params += param.numel()
                        mask = torch.abs(param.data) > self.config.pruning_threshold
                        pruned_count += torch.sum(~mask).item()
                        param.data *= mask.float()

            pruning_ratio = pruned_count / total_params if total_params > 0 else 0

            # Only proceed if pruning ratio is within limits
            if pruning_ratio <= self.config.max_pruning_ratio:
                logger.info(f"Pruned {pruned_count}/{total_params} parameters ({pruning_ratio:.3f})")
                return True
            else:
                logger.warning(f"Pruning ratio {pruning_ratio:.3f} exceeds maximum {self.config.max_pruning_ratio}")
                return False

        except Exception as e:
            logger.error(f"Synaptic pruning failed: {e}")
            return False

    def _perform_neurogenesis(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Perform neurogenesis (add new neurons/connections)."""
        try:
            logger.info("Performing neurogenesis")

            # This is a simplified implementation
            # In practice, this would involve more sophisticated network growth

            if hasattr(agent, 'world_model') and hasattr(agent.world_model, 'transformer'):
                transformer = agent.world_model.transformer

                # Increase hidden dimension slightly
                current_hidden_dim = transformer.d_model
                growth_factor = 1.0 + min(self.config.max_growth_ratio, 0.05)  # Max 5% growth
                new_hidden_dim = int(current_hidden_dim * growth_factor)

                if new_hidden_dim > current_hidden_dim:
                    logger.info(f"Growing hidden dimension from {current_hidden_dim} to {new_hidden_dim}")

                    # Note: Actual implementation would require careful weight initialization
                    # and architecture modification. This is a placeholder.
                    return True

            return False

        except Exception as e:
            logger.error(f"Neurogenesis failed: {e}")
            return False

    def _consolidate_memory(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Consolidate agent's external memory."""
        try:
            if hasattr(agent, 'external_memory'):
                logger.info("Consolidating external memory")

                # Get current memories
                memories = agent.external_memory.memories

                if len(memories) > self.config.memory_consolidation_threshold:
                    # Keep only the most important memories
                    # Sort by importance (could be based on reward, recency, etc.)

                    # Simple consolidation: keep top 50% by reward
                    sorted_memories = sorted(
                        memories,
                        key=lambda x: x.get('reward', 0),
                        reverse=True
                    )

                    keep_count = len(sorted_memories) // 2
                    consolidated_memories = sorted_memories[:keep_count]

                    # Update memory
                    agent.external_memory.memories = consolidated_memories

                    logger.info(f"Consolidated memory from {len(memories)} to {len(consolidated_memories)} entries")
                    return True

            return False

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return False

    def _adapt_learning_rate(
        self,
        agent: Any,
        performance_metrics: PerformanceMetrics
    ) -> bool:
        """Adapt learning rate based on performance trends."""
        try:
            if len(self.performance_history) < 5:
                return False

            # Calculate performance trend
            recent_performance = [p.profit_factor for p in self.performance_history[-5:]]
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

            # Adjust learning rate based on trend
            if trend < -0.1:  # Declining performance
                # Reduce learning rate
                adjustment = -self.config.learning_rate_adjustment_factor
                new_lr = max(
                    self.current_learning_rate * (1 + adjustment),
                    self.config.min_learning_rate
                )
                logger.info(f"Reducing learning rate from {self.current_learning_rate:.6f} to {new_lr:.6f}")

            elif trend > 0.1:  # Improving performance
                # Increase learning rate slightly
                adjustment = self.config.learning_rate_adjustment_factor * 0.5
                new_lr = min(
                    self.current_learning_rate * (1 + adjustment),
                    self.config.max_learning_rate
                )
                logger.info(f"Increasing learning rate from {self.current_learning_rate:.6f} to {new_lr:.6f}")
            else:
                return False

            self.current_learning_rate = new_lr

            # Apply to agent's optimizers if accessible
            if hasattr(agent, 'optimizer'):
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_lr

            return True

        except Exception as e:
            logger.error(f"Learning rate adaptation failed: {e}")
            return False

    def _is_performance_stagnating(self) -> bool:
        """Check if performance is stagnating."""
        if len(self.performance_history) < 10:
            return False

        recent_performance = [p.profit_factor for p in self.performance_history[-10:]]

        # Check if variance is very low (stagnating)
        variance = np.var(recent_performance)
        return variance < 0.01  # Very low variance indicates stagnation

    def _is_performance_declining(self) -> bool:
        """Check if performance is declining."""
        if len(self.performance_history) < 5:
            return False

        recent_performance = [p.profit_factor for p in self.performance_history[-5:]]

        # Check if trend is negative
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        return trend < -0.05  # Declining trend

    def get_modification_statistics(self) -> Dict[str, Any]:
        """Get statistics about modifications performed."""
        if not self.modification_history:
            return {}

        modification_counts = {}
        for record in self.modification_history:
            for mod in record['modifications']:
                modification_counts[mod] = modification_counts.get(mod, 0) + 1

        return {
            'total_modifications': len(self.modification_history),
            'modification_counts': modification_counts,
            'current_risk_factor': self.current_risk_factor,
            'current_learning_rate': self.current_learning_rate,
            'last_architecture_search': self.last_architecture_search_episode,
            'performance_history_length': len(self.performance_history)
        }

    def reset_modification_state(self):
        """Reset modification state (useful for new training runs)."""
        self.modification_history.clear()
        self.performance_history.clear()
        self.last_architecture_search_episode = -1
        self.current_risk_factor = 1.0
        self.current_learning_rate = 1e-3

        logger.info("Reset modification state")


def check_performance_and_adapt(
    agent: Any,
    performance_metrics: Union[PerformanceMetrics, Dict[str, float]],
    config: Optional[ModificationConfig] = None,
    nas_controller: Optional[Any] = None,
    episode_number: int = 0
) -> List[ModificationType]:
    """
    Standalone function for performance checking and adaptation.

    This function provides a simple interface for the main training loop
    to trigger self-modification logic.

    Args:
        agent: The autonomous agent to potentially modify
        performance_metrics: Current performance metrics
        config: Configuration for modification thresholds
        nas_controller: Neural Architecture Search controller
        episode_number: Current episode/evaluation number

    Returns:
        List of modifications that were applied
    """
    # Create a temporary manager for this call
    manager = SelfModificationManager(config=config, nas_controller=nas_controller)

    return manager.check_performance_and_adapt(
        agent=agent,
        performance_metrics=performance_metrics,
        episode_number=episode_number
    )
