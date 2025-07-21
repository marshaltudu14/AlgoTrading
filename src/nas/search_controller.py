"""
Neural Architecture Search Controller

This module implements the search algorithm for discovering optimal
neural network architectures using evolutionary algorithms.
"""

import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import copy
import logging

from .search_space import SearchSpace, LayerConfig, LayerType
from .hyperparameter_space import HyperparameterSearchSpace

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    Represents an individual in the evolutionary population.
    """
    architecture: List[LayerConfig]
    fitness: float = 0.0
    age: int = 0
    complexity: int = 0
    hyperparameters: Dict[str, float] = None
    
    def __post_init__(self):
        """Calculate complexity after initialization."""
        search_space = SearchSpace()
        self.complexity = search_space.get_architecture_complexity(self.architecture)

        # Initialize hyperparameters if not provided
        if self.hyperparameters is None:
            hyperparameter_space = HyperparameterSearchSpace()
            self.hyperparameters = hyperparameter_space.sample_hyperparameters()


class NASController:
    """
    Neural Architecture Search Controller using Evolutionary Algorithm.
    
    This controller manages a population of neural network architectures
    and evolves them over generations to find optimal designs.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 20,
        elite_size: int = 5,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        max_generations: int = 50,
        complexity_penalty: float = 0.00001
    ):
        """
        Initialize the NAS Controller.
        
        Args:
            search_space: SearchSpace defining available operations
            population_size: Number of individuals in population
            elite_size: Number of top individuals to preserve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            max_generations: Maximum number of generations
            complexity_penalty: Penalty factor for complex architectures
        """
        self.search_space = search_space
        self.hyperparameter_space = HyperparameterSearchSpace()
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.complexity_penalty = complexity_penalty
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[float] = []
        
        logger.info(f"Initialized NAS Controller with population size {population_size}")
    
    def initialize_population(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the population with random architectures.
        
        Args:
            input_dim: Input dimension for architectures
            output_dim: Output dimension for architectures
        """
        self.population = []
        
        for _ in range(self.population_size):
            architecture = self.search_space.sample_random_architecture(input_dim, output_dim)
            hyperparameters = self.hyperparameter_space.sample_hyperparameters()
            individual = Individual(architecture=architecture, hyperparameters=hyperparameters)
            self.population.append(individual)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def generate_architecture(self, input_dim: int, output_dim: int) -> List[LayerConfig]:
        """
        Generate a single random architecture from the search space.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            List of LayerConfig objects defining the architecture
        """
        return self.search_space.sample_random_architecture(input_dim, output_dim)
    
    def evaluate_population(self, fitness_scores: List[float]) -> None:
        """
        Assign fitness scores to the population.
        
        Args:
            fitness_scores: List of fitness scores for each individual
        """
        if len(fitness_scores) != len(self.population):
            raise ValueError(f"Number of fitness scores ({len(fitness_scores)}) "
                           f"must match population size ({len(self.population)})")
        
        for individual, fitness in zip(self.population, fitness_scores):
            # Apply complexity penalty
            complexity_penalty = self.complexity_penalty * individual.complexity
            individual.fitness = fitness - complexity_penalty
        
        # Update best individual
        best_individual = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or best_individual.fitness > self.best_individual.fitness:
            self.best_individual = copy.deepcopy(best_individual)
        
        # Record fitness history
        avg_fitness = np.mean([ind.fitness for ind in self.population])
        self.fitness_history.append(avg_fitness)
        
        logger.debug(f"Generation {self.generation}: Best fitness = {best_individual.fitness:.4f}, "
                    f"Avg fitness = {avg_fitness:.4f}")
    
    def selection(self) -> List[Individual]:
        """
        Select individuals for reproduction using tournament selection.
        
        Returns:
            List of selected individuals
        """
        selected = []
        
        # Always include elite individuals
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_population[:self.elite_size]
        selected.extend(copy.deepcopy(elite))
        
        # Tournament selection for remaining slots
        tournament_size = 3
        while len(selected) < self.population_size:
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(copy.deepcopy(winner))
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parent architectures.
        
        Args:
            parent1: First parent individual
            parent2: Second parent individual
            
        Returns:
            Tuple of two offspring individuals
        """
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        arch1 = parent1.architecture
        arch2 = parent2.architecture
        
        # Single-point crossover
        if len(arch1) > 1 and len(arch2) > 1:
            crossover_point1 = random.randint(1, len(arch1) - 1)
            crossover_point2 = random.randint(1, len(arch2) - 1)
            
            offspring1_arch = arch1[:crossover_point1] + arch2[crossover_point2:]
            offspring2_arch = arch2[:crossover_point2] + arch1[crossover_point1:]
        else:
            offspring1_arch = copy.deepcopy(arch1)
            offspring2_arch = copy.deepcopy(arch2)
        
        # Validate and fix architectures if needed
        if not self.search_space.validate_architecture(offspring1_arch):
            offspring1_arch = self._fix_architecture(offspring1_arch)
        
        if not self.search_space.validate_architecture(offspring2_arch):
            offspring2_arch = self._fix_architecture(offspring2_arch)
        
        # Crossover hyperparameters
        offspring1_hyperparams, offspring2_hyperparams = self.hyperparameter_space.crossover_hyperparameters(
            parent1.hyperparameters, parent2.hyperparameters, self.crossover_rate
        )

        offspring1 = Individual(architecture=offspring1_arch, hyperparameters=offspring1_hyperparams)
        offspring2 = Individual(architecture=offspring2_arch, hyperparameters=offspring2_hyperparams)

        return offspring1, offspring2
    
    def mutate(self, individual: Individual) -> Individual:
        """
        Mutate an individual's architecture.
        
        Args:
            individual: Individual to mutate
            
        Returns:
            Mutated individual
        """
        if random.random() > self.mutation_rate:
            return individual
        
        mutated_arch = copy.deepcopy(individual.architecture)
        
        # Choose mutation type
        mutation_types = ['add_layer', 'remove_layer', 'modify_layer', 'swap_layers']
        mutation_type = random.choice(mutation_types)
        
        if mutation_type == 'add_layer' and len(mutated_arch) < self.search_space.max_layers:
            self._add_random_layer(mutated_arch)
        
        elif mutation_type == 'remove_layer' and len(mutated_arch) > self.search_space.min_layers:
            self._remove_random_layer(mutated_arch)
        
        elif mutation_type == 'modify_layer':
            self._modify_random_layer(mutated_arch)
        
        elif mutation_type == 'swap_layers' and len(mutated_arch) > 1:
            self._swap_random_layers(mutated_arch)
        
        # Validate and fix if needed
        if not self.search_space.validate_architecture(mutated_arch):
            mutated_arch = self._fix_architecture(mutated_arch)

        # Mutate hyperparameters
        mutated_hyperparams = self.hyperparameter_space.mutate_hyperparameters(
            individual.hyperparameters, self.mutation_rate
        )

        return Individual(architecture=mutated_arch, hyperparameters=mutated_hyperparams)
    
    def evolve_population(self, fitness_scores: List[float]) -> List[Individual]:
        """
        Evolve the population for one generation.
        
        Args:
            fitness_scores: Fitness scores for current population
            
        Returns:
            New population after evolution
        """
        # Evaluate current population
        self.evaluate_population(fitness_scores)
        
        # Selection
        selected = self.selection()
        
        # Create new population through crossover and mutation
        new_population = []
        
        # Keep elite individuals
        elite = sorted(selected, key=lambda x: x.fitness, reverse=True)[:self.elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            
            offspring1, offspring2 = self.crossover(parent1, parent2)
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)
            
            new_population.extend([offspring1, offspring2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        
        # Update age
        for individual in new_population:
            individual.age += 1
        
        self.population = new_population
        self.generation += 1
        
        logger.info(f"Evolved to generation {self.generation}")
        
        return self.population
    
    def get_best_architecture(self) -> Optional[List[LayerConfig]]:
        """
        Get the best architecture found so far.
        
        Returns:
            Best architecture or None if no evolution has occurred
        """
        if self.best_individual is None:
            return None
        return self.best_individual.architecture
    
    def build_model_from_best(self) -> Optional[nn.Module]:
        """
        Build a PyTorch model from the best architecture.
        
        Returns:
            PyTorch model or None if no best architecture exists
        """
        best_arch = self.get_best_architecture()
        if best_arch is None:
            return None
        
        return self.search_space.build_model_from_architecture(best_arch)
    
    def _add_random_layer(self, architecture: List[LayerConfig]) -> None:
        """Add a random layer to the architecture."""
        if len(architecture) == 0:
            return
        
        # Choose insertion point
        insert_idx = random.randint(0, len(architecture))
        
        # Determine dimensions
        if insert_idx == 0:
            input_dim = architecture[0].input_dim if architecture else None
        else:
            input_dim = architecture[insert_idx - 1].output_dim
        
        if insert_idx == len(architecture):
            output_dim = input_dim  # Same dimension for non-dimensional layers
        else:
            output_dim = architecture[insert_idx].input_dim
        
        # Sample a new layer
        layer_type = random.choice(self.search_space.layer_types)
        new_layer = self.search_space.sample_layer_config(layer_type, input_dim, output_dim)
        
        architecture.insert(insert_idx, new_layer)
    
    def _remove_random_layer(self, architecture: List[LayerConfig]) -> None:
        """Remove a random layer from the architecture."""
        if len(architecture) <= self.search_space.min_layers:
            return
        
        remove_idx = random.randint(0, len(architecture) - 1)
        architecture.pop(remove_idx)
    
    def _modify_random_layer(self, architecture: List[LayerConfig]) -> None:
        """Modify a random layer in the architecture."""
        if len(architecture) == 0:
            return
        
        modify_idx = random.randint(0, len(architecture) - 1)
        old_layer = architecture[modify_idx]
        
        # Create a new layer of the same type with different parameters
        new_layer = self.search_space.sample_layer_config(
            old_layer.layer_type,
            old_layer.input_dim,
            old_layer.output_dim
        )
        
        architecture[modify_idx] = new_layer
    
    def _swap_random_layers(self, architecture: List[LayerConfig]) -> None:
        """Swap two random layers in the architecture."""
        if len(architecture) < 2:
            return
        
        idx1, idx2 = random.sample(range(len(architecture)), 2)
        architecture[idx1], architecture[idx2] = architecture[idx2], architecture[idx1]
    
    def _fix_architecture(self, architecture: List[LayerConfig]) -> List[LayerConfig]:
        """
        Fix an invalid architecture by adjusting dimensions.
        
        Args:
            architecture: Architecture to fix
            
        Returns:
            Fixed architecture
        """
        if len(architecture) == 0:
            return architecture
        
        # Fix dimension mismatches
        for i in range(len(architecture) - 1):
            current_layer = architecture[i]
            next_layer = architecture[i + 1]
            
            if (current_layer.output_dim is not None and 
                next_layer.input_dim is not None and
                current_layer.output_dim != next_layer.input_dim):
                
                # Adjust next layer's input dimension
                next_layer.input_dim = current_layer.output_dim
        
        return architecture
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search process."""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness for ind in self.population]
        complexities = [ind.complexity for ind in self.population]
        
        # Get hyperparameter statistics
        hyperparameter_stats = {}
        if self.population and self.population[0].hyperparameters:
            for param_name in self.population[0].hyperparameters.keys():
                param_values = [ind.hyperparameters[param_name] for ind in self.population]
                hyperparameter_stats[param_name] = {
                    'mean': np.mean(param_values),
                    'std': np.std(param_values),
                    'min': np.min(param_values),
                    'max': np.max(param_values)
                }

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'fitness_std': np.std(fitnesses) if fitnesses else 0,
            'avg_complexity': np.mean(complexities) if complexities else 0,
            'best_individual_complexity': self.best_individual.complexity if self.best_individual else 0,
            'best_individual_hyperparameters': self.best_individual.hyperparameters if self.best_individual else {},
            'hyperparameter_statistics': hyperparameter_stats,
            'fitness_history': self.fitness_history.copy()
        }
