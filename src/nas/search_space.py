"""
Neural Architecture Search Space Definition

This module defines the "Lego box" of possible layers and operations
that can be used to construct neural network architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Type, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import random

from src.models.core_transformer import CoreTransformer


class LayerType(Enum):
    """Enumeration of available layer types."""
    LINEAR = "linear"
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    DROPOUT = "dropout"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    CONV1D = "conv1d"
    RESIDUAL = "residual"
    SKIP_CONNECTION = "skip_connection"


@dataclass
class LayerConfig:
    """
    Configuration for a single layer in the architecture.
    """
    layer_type: LayerType
    params: Dict[str, Any] = field(default_factory=dict)
    input_dim: Optional[int] = None
    output_dim: Optional[int] = None
    
    def __post_init__(self):
        """Validate layer configuration after initialization."""
        if self.layer_type in [LayerType.LINEAR, LayerType.CONV1D]:
            if self.input_dim is None or self.output_dim is None:
                raise ValueError(f"{self.layer_type.value} requires input_dim and output_dim")


class SearchSpace:
    """
    Defines the search space for Neural Architecture Search.
    
    This class provides the "Lego box" of available components that can be
    combined to create neural network architectures. It includes both
    primitive operations and more complex blocks like transformers.
    """
    
    def __init__(self):
        """Initialize the search space with available operations."""
        self.layer_types = list(LayerType)
        self.activation_functions = [LayerType.RELU, LayerType.GELU, LayerType.TANH, LayerType.SIGMOID]
        self.normalization_layers = [LayerType.LAYER_NORM, LayerType.BATCH_NORM]
        self.regularization_layers = [LayerType.DROPOUT]
        
        # Define parameter ranges for each layer type
        self.parameter_ranges = {
            LayerType.LINEAR: {
                'bias': [True, False]
            },
            LayerType.DROPOUT: {
                'p': [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            LayerType.TRANSFORMER: {
                'num_heads': [2, 4, 6, 8, 12, 16],
                'num_layers': [1, 2, 3, 4, 6, 8],
                'dropout': [0.0, 0.1, 0.2, 0.3],
                'use_positional_encoding': [True, False]
            },
            LayerType.CONV1D: {
                'kernel_size': [1, 3, 5, 7],
                'stride': [1, 2],
                'padding': [0, 1, 2, 3],
                'bias': [True, False]
            },
            LayerType.ATTENTION: {
                'num_heads': [1, 2, 4, 8],
                'dropout': [0.0, 0.1, 0.2]
            }
        }
        
        # Define common dimension choices
        self.dimension_choices = [32, 64, 128, 256, 512, 1024]
        
        # Define architecture constraints
        self.max_layers = 20
        self.min_layers = 3
        self.max_depth = 10
    
    def get_available_layers(self) -> List[LayerType]:
        """Get list of all available layer types."""
        return self.layer_types.copy()
    
    def get_parameter_choices(self, layer_type: LayerType) -> Dict[str, List[Any]]:
        """
        Get available parameter choices for a given layer type.
        
        Args:
            layer_type: The type of layer
            
        Returns:
            Dictionary mapping parameter names to lists of possible values
        """
        return self.parameter_ranges.get(layer_type, {}).copy()
    
    def sample_layer_config(
        self, 
        layer_type: LayerType, 
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None
    ) -> LayerConfig:
        """
        Sample a random configuration for a given layer type.
        
        Args:
            layer_type: The type of layer to configure
            input_dim: Input dimension (if applicable)
            output_dim: Output dimension (if applicable)
            
        Returns:
            LayerConfig with randomly sampled parameters
        """
        params = {}
        
        # Sample parameters for this layer type
        param_choices = self.get_parameter_choices(layer_type)
        for param_name, choices in param_choices.items():
            params[param_name] = random.choice(choices)
        
        # Handle dimension sampling for layers that need it
        if layer_type == LayerType.LINEAR:
            if input_dim is None:
                input_dim = random.choice(self.dimension_choices)
            if output_dim is None:
                output_dim = random.choice(self.dimension_choices)
        
        elif layer_type == LayerType.CONV1D:
            if input_dim is None:
                input_dim = random.choice(self.dimension_choices)
            if output_dim is None:
                output_dim = random.choice(self.dimension_choices)
        
        return LayerConfig(
            layer_type=layer_type,
            params=params,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def sample_random_architecture(
        self, 
        input_dim: int, 
        output_dim: int,
        target_layers: Optional[int] = None
    ) -> List[LayerConfig]:
        """
        Sample a completely random architecture from the search space.
        
        Args:
            input_dim: Input dimension of the architecture
            output_dim: Output dimension of the architecture
            target_layers: Target number of layers (random if None)
            
        Returns:
            List of LayerConfig objects defining the architecture
        """
        if target_layers is None:
            num_layers = random.randint(self.min_layers, self.max_layers)
        else:
            num_layers = max(self.min_layers, min(target_layers, self.max_layers))
        
        architecture = []
        current_dim = input_dim
        
        for i in range(num_layers):
            # For the last layer, ensure output matches target
            if i == num_layers - 1:
                # Final layer should be linear to match output dimension
                layer_config = self.sample_layer_config(
                    LayerType.LINEAR,
                    input_dim=current_dim,
                    output_dim=output_dim
                )
            else:
                # Sample a random layer type
                layer_type = random.choice(self.layer_types)
                
                # Determine next dimension
                if layer_type in [LayerType.LINEAR, LayerType.CONV1D]:
                    next_dim = random.choice(self.dimension_choices)
                    layer_config = self.sample_layer_config(
                        layer_type,
                        input_dim=current_dim,
                        output_dim=next_dim
                    )
                    current_dim = next_dim
                else:
                    # Non-dimensional layers (activations, normalization, etc.)
                    layer_config = self.sample_layer_config(layer_type)
            
            architecture.append(layer_config)
        
        return architecture
    
    def create_layer_from_config(self, config: LayerConfig) -> nn.Module:
        """
        Create a PyTorch layer from a LayerConfig.
        
        Args:
            config: Layer configuration
            
        Returns:
            PyTorch module implementing the layer
        """
        if config.layer_type == LayerType.LINEAR:
            return nn.Linear(
                config.input_dim, 
                config.output_dim, 
                bias=config.params.get('bias', True)
            )
        
        elif config.layer_type == LayerType.RELU:
            return nn.ReLU()
        
        elif config.layer_type == LayerType.GELU:
            return nn.GELU()
        
        elif config.layer_type == LayerType.TANH:
            return nn.Tanh()
        
        elif config.layer_type == LayerType.SIGMOID:
            return nn.Sigmoid()
        
        elif config.layer_type == LayerType.DROPOUT:
            return nn.Dropout(p=config.params.get('p', 0.1))
        
        elif config.layer_type == LayerType.LAYER_NORM:
            if config.input_dim is None:
                # Return identity if no input dimension specified
                return nn.Identity()
            return nn.LayerNorm(config.input_dim)

        elif config.layer_type == LayerType.BATCH_NORM:
            if config.input_dim is None:
                # Return identity if no input dimension specified
                return nn.Identity()
            return nn.BatchNorm1d(config.input_dim)
        
        elif config.layer_type == LayerType.TRANSFORMER:
            if config.input_dim is None or config.output_dim is None:
                # Return identity if dimensions not specified
                return nn.Identity()

            ff_dim = config.params.get('ff_dim', config.input_dim * 4)
            return CoreTransformer(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                num_heads=config.params.get('num_heads', 8),
                num_layers=config.params.get('num_layers', 4),
                ff_dim=ff_dim,
                dropout=config.params.get('dropout', 0.1),
                use_positional_encoding=config.params.get('use_positional_encoding', True)
            )
        
        elif config.layer_type == LayerType.CONV1D:
            return nn.Conv1d(
                in_channels=config.input_dim,
                out_channels=config.output_dim,
                kernel_size=config.params.get('kernel_size', 3),
                stride=config.params.get('stride', 1),
                padding=config.params.get('padding', 1),
                bias=config.params.get('bias', True)
            )

        elif config.layer_type == LayerType.RESIDUAL:
            # Residual connection - return identity for now
            return nn.Identity()

        elif config.layer_type == LayerType.SKIP_CONNECTION:
            # Skip connection - return identity for now
            return nn.Identity()

        elif config.layer_type == LayerType.ATTENTION:
            # Simple attention layer - return identity for now
            # In a full implementation, this would be a proper attention mechanism
            return nn.Identity()

        else:
            raise ValueError(f"Unknown layer type: {config.layer_type}")
    
    def build_model_from_architecture(self, architecture: List[LayerConfig]) -> nn.Module:
        """
        Build a complete PyTorch model from an architecture specification.
        
        Args:
            architecture: List of LayerConfig objects
            
        Returns:
            PyTorch Sequential model
        """
        layers = []
        
        for config in architecture:
            layer = self.create_layer_from_config(config)
            layers.append(layer)
        
        return nn.Sequential(*layers)
    
    def validate_architecture(self, architecture: List[LayerConfig]) -> bool:
        """
        Validate that an architecture is feasible.
        
        Args:
            architecture: List of LayerConfig objects
            
        Returns:
            True if architecture is valid, False otherwise
        """
        if len(architecture) < self.min_layers or len(architecture) > self.max_layers:
            return False
        
        # Check dimension compatibility
        for i in range(len(architecture) - 1):
            current_layer = architecture[i]
            next_layer = architecture[i + 1]
            
            # Skip validation for layers without dimensions
            if (current_layer.output_dim is None or 
                next_layer.input_dim is None):
                continue
            
            if current_layer.output_dim != next_layer.input_dim:
                return False
        
        return True
    
    def get_architecture_complexity(self, architecture: List[LayerConfig]) -> int:
        """
        Estimate the complexity of an architecture.
        
        Args:
            architecture: List of LayerConfig objects
            
        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 0
        
        for config in architecture:
            if config.layer_type == LayerType.LINEAR:
                if config.input_dim is not None and config.output_dim is not None:
                    complexity += config.input_dim * config.output_dim
                else:
                    complexity += 1
            elif config.layer_type == LayerType.TRANSFORMER:
                # Transformer complexity is roughly O(n^2 * d)
                seq_len = 100  # Assume typical sequence length
                d_model = config.input_dim or 64  # Default if None
                num_layers = config.params.get('num_layers', 4)
                complexity += seq_len * seq_len * d_model * num_layers
            elif config.layer_type == LayerType.CONV1D:
                if config.input_dim is not None and config.output_dim is not None:
                    kernel_size = config.params.get('kernel_size', 3)
                    complexity += config.input_dim * config.output_dim * kernel_size
                else:
                    complexity += 1
            else:
                complexity += 1  # Minimal complexity for activations, etc.
        
        return complexity
