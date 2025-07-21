"""
Hyperparameter Search Space for Autonomous Agents

This module defines the search space for hyperparameters that can be
evolved alongside neural architectures in the autonomous training process.
"""

import numpy as np
import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HyperparameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    LOG_UNIFORM = "log_uniform"


@dataclass
class HyperparameterConfig:
    """
    Configuration for a single hyperparameter.
    """
    name: str
    param_type: HyperparameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    default_value: Any = None
    mutation_std: float = 0.1  # Standard deviation for mutations
    
    def __post_init__(self):
        """Validate hyperparameter configuration."""
        if self.param_type in [HyperparameterType.CONTINUOUS, HyperparameterType.LOG_UNIFORM]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Continuous/log_uniform parameter {self.name} requires min_value and max_value")
        elif self.param_type == HyperparameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Categorical parameter {self.name} requires choices")


class HyperparameterSearchSpace:
    """
    Defines the search space for hyperparameters in autonomous agents.
    """
    
    def __init__(self):
        """Initialize the hyperparameter search space."""
        self.hyperparameters = self._define_search_space()
        logger.info(f"Initialized hyperparameter search space with {len(self.hyperparameters)} parameters")
    
    def _define_search_space(self) -> Dict[str, HyperparameterConfig]:
        """Define the search space for all hyperparameters."""
        return {
            'learning_rate': HyperparameterConfig(
                name='learning_rate',
                param_type=HyperparameterType.LOG_UNIFORM,
                min_value=1e-5,
                max_value=1e-1,
                default_value=1e-3,
                mutation_std=0.2
            ),
            'batch_size': HyperparameterConfig(
                name='batch_size',
                param_type=HyperparameterType.DISCRETE,
                choices=[8, 16, 32, 64, 128, 256],
                default_value=32,
                mutation_std=0.3
            ),
            'discount_factor': HyperparameterConfig(
                name='discount_factor',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.9,
                max_value=0.999,
                default_value=0.99,
                mutation_std=0.05
            ),
            'entropy_coefficient': HyperparameterConfig(
                name='entropy_coefficient',
                param_type=HyperparameterType.LOG_UNIFORM,
                min_value=1e-4,
                max_value=1e-1,
                default_value=0.01,
                mutation_std=0.2
            ),
            'value_loss_coefficient': HyperparameterConfig(
                name='value_loss_coefficient',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.1,
                max_value=2.0,
                default_value=0.5,
                mutation_std=0.1
            ),
            'gradient_clip_norm': HyperparameterConfig(
                name='gradient_clip_norm',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.1,
                max_value=2.0,
                default_value=0.5,
                mutation_std=0.1
            ),
            'exploration_noise': HyperparameterConfig(
                name='exploration_noise',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.01,
                max_value=0.5,
                default_value=0.1,
                mutation_std=0.05
            ),
            'memory_consolidation_threshold': HyperparameterConfig(
                name='memory_consolidation_threshold',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.5,
                max_value=0.95,
                default_value=0.8,
                mutation_std=0.05
            ),
            'risk_tolerance': HyperparameterConfig(
                name='risk_tolerance',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.1,
                max_value=1.0,
                default_value=0.5,
                mutation_std=0.1
            ),
            'pattern_confidence_threshold': HyperparameterConfig(
                name='pattern_confidence_threshold',
                param_type=HyperparameterType.CONTINUOUS,
                min_value=0.3,
                max_value=0.9,
                default_value=0.6,
                mutation_std=0.05
            )
        }
    
    def sample_hyperparameters(self) -> Dict[str, float]:
        """
        Sample a random set of hyperparameters from the search space.
        
        Returns:
            Dictionary of sampled hyperparameters
        """
        sampled = {}
        
        for name, config in self.hyperparameters.items():
            sampled[name] = self._sample_single_hyperparameter(config)
        
        return sampled
    
    def _sample_single_hyperparameter(self, config: HyperparameterConfig) -> Any:
        """Sample a single hyperparameter value."""
        if config.param_type == HyperparameterType.CONTINUOUS:
            return random.uniform(config.min_value, config.max_value)
        
        elif config.param_type == HyperparameterType.LOG_UNIFORM:
            log_min = np.log(config.min_value)
            log_max = np.log(config.max_value)
            return np.exp(random.uniform(log_min, log_max))
        
        elif config.param_type == HyperparameterType.DISCRETE:
            return random.choice(config.choices)
        
        elif config.param_type == HyperparameterType.CATEGORICAL:
            return random.choice(config.choices)
        
        else:
            return config.default_value
    
    def mutate_hyperparameters(
        self, 
        hyperparameters: Dict[str, float], 
        mutation_rate: float = 0.3
    ) -> Dict[str, float]:
        """
        Mutate a set of hyperparameters.
        
        Args:
            hyperparameters: Current hyperparameters
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            Mutated hyperparameters
        """
        mutated = hyperparameters.copy()
        
        for name, config in self.hyperparameters.items():
            if random.random() < mutation_rate:
                mutated[name] = self._mutate_single_hyperparameter(
                    current_value=hyperparameters.get(name, config.default_value),
                    config=config
                )
        
        return mutated
    
    def _mutate_single_hyperparameter(self, current_value: Any, config: HyperparameterConfig) -> Any:
        """Mutate a single hyperparameter value."""
        if config.param_type == HyperparameterType.CONTINUOUS:
            # Gaussian mutation with clipping
            noise = np.random.normal(0, config.mutation_std * (config.max_value - config.min_value))
            new_value = current_value + noise
            return np.clip(new_value, config.min_value, config.max_value)
        
        elif config.param_type == HyperparameterType.LOG_UNIFORM:
            # Log-space Gaussian mutation
            log_current = np.log(current_value)
            log_min = np.log(config.min_value)
            log_max = np.log(config.max_value)
            noise = np.random.normal(0, config.mutation_std * (log_max - log_min))
            new_log_value = np.clip(log_current + noise, log_min, log_max)
            return np.exp(new_log_value)
        
        elif config.param_type in [HyperparameterType.DISCRETE, HyperparameterType.CATEGORICAL]:
            # Random choice from available options
            return random.choice(config.choices)
        
        else:
            return current_value
    
    def crossover_hyperparameters(
        self, 
        parent1: Dict[str, float], 
        parent2: Dict[str, float],
        crossover_rate: float = 0.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform crossover between two sets of hyperparameters.
        
        Args:
            parent1: First parent hyperparameters
            parent2: Second parent hyperparameters
            crossover_rate: Probability of crossover for each parameter
            
        Returns:
            Tuple of two offspring hyperparameter sets
        """
        offspring1 = {}
        offspring2 = {}
        
        for name in self.hyperparameters.keys():
            if random.random() < crossover_rate:
                # Crossover: swap values
                offspring1[name] = parent2.get(name, self.hyperparameters[name].default_value)
                offspring2[name] = parent1.get(name, self.hyperparameters[name].default_value)
            else:
                # No crossover: keep original values
                offspring1[name] = parent1.get(name, self.hyperparameters[name].default_value)
                offspring2[name] = parent2.get(name, self.hyperparameters[name].default_value)
        
        return offspring1, offspring2
    
    def validate_hyperparameters(self, hyperparameters: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and fix hyperparameters to ensure they're within bounds.
        
        Args:
            hyperparameters: Hyperparameters to validate
            
        Returns:
            Validated hyperparameters
        """
        validated = {}
        
        for name, config in self.hyperparameters.items():
            value = hyperparameters.get(name, config.default_value)
            
            if config.param_type == HyperparameterType.CONTINUOUS:
                validated[name] = np.clip(value, config.min_value, config.max_value)
            
            elif config.param_type == HyperparameterType.LOG_UNIFORM:
                validated[name] = np.clip(value, config.min_value, config.max_value)
            
            elif config.param_type in [HyperparameterType.DISCRETE, HyperparameterType.CATEGORICAL]:
                if value in config.choices:
                    validated[name] = value
                else:
                    validated[name] = config.default_value
            
            else:
                validated[name] = value
        
        return validated
    
    def get_hyperparameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Get bounds and constraints for all hyperparameters."""
        bounds = {}
        
        for name, config in self.hyperparameters.items():
            bounds[name] = {
                'type': config.param_type.value,
                'min_value': config.min_value,
                'max_value': config.max_value,
                'choices': config.choices,
                'default_value': config.default_value
            }
        
        return bounds
