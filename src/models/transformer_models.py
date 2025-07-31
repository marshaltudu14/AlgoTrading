"""
Transformer-based models for trading agents.

This module provides Transformer-based Actor and Critic models that replace
the previous LSTM-based models. These models use the CoreTransformer as their
backbone for improved sequence processing and memory capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.models.core_transformer import CoreTransformer



class TransformerModel(nn.Module):
    """
    Base Transformer model for trading agents.
    
    This replaces the previous LSTMModel and provides a Transformer-based
    architecture for both actor and critic networks.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Initialize the TransformerModel.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension (used as ff_dim in transformer)
            output_dim: Dimension of output
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Core transformer backbone
        self.transformer = CoreTransformer(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_positional_encoding=True
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Handle single timestep input by adding sequence dimension
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        return self.transformer(x, mask=mask)
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get attention weights for interpretability."""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return self.transformer.get_attention_weights(x, mask=mask)


class MultiHeadTransformerModel(nn.Module):
    """
    Multi-head Transformer model for more complex architectures.
    
    This model can output multiple values simultaneously, useful for
    agents that need to predict multiple quantities (e.g., value, advantage, etc.)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_heads: dict,  # e.g., {'value': 1, 'advantage': action_dim}
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        """
        Initialize multi-head transformer model.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for transformer
            output_heads: Dictionary mapping head names to output dimensions
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super(MultiHeadTransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_heads = output_heads
        
        # Shared transformer backbone
        self.transformer = CoreTransformer(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=hidden_dim,
            output_dim=hidden_dim,  # Output to hidden_dim for shared representation
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_positional_encoding=True
        )
        
        # Create output heads
        self.heads = nn.ModuleDict()
        for head_name, head_dim in output_heads.items():
            self.heads[head_name] = nn.Linear(hidden_dim, head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass through multi-head model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dictionary mapping head names to their outputs
        """
        # Handle single timestep input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Get shared representation from transformer
        shared_repr = self.transformer(x, mask=mask)

        # Apply each head to the shared representation
        outputs = {}
        for head_name, head_layer in self.heads.items():
            outputs[head_name] = head_layer(shared_repr)

        return outputs
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Get attention weights for interpretability."""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return self.transformer.get_attention_weights(x, mask=mask)


class ActorTransformerModel(MultiHeadTransformerModel):
    """
    Actor model using Transformer architecture with multi-head output.

    This model outputs both discrete action probabilities and a continuous quantity.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        action_dim_discrete: int,
        action_dim_continuous: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_heads={'action_type': action_dim_discrete, 'quantity': action_dim_continuous},
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass with proper activations for actor outputs.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask

        Returns:
            Dictionary with 'action_type' (probabilities) and 'quantity' (positive values)
        """
        # Get raw outputs from parent
        outputs = super().forward(x, mask)

        # Apply softmax to action_type logits for valid probabilities
        # Add small epsilon for numerical stability and prevent NaN
        action_logits = outputs['action_type']

        # Check for NaN/Inf in logits and replace with zeros if found
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            action_logits = torch.zeros_like(action_logits)

        # Clamp to prevent extreme values that could cause NaN in softmax
        action_logits = torch.clamp(action_logits, min=-10, max=10)

        # Apply softmax with numerical stability
        action_probs = F.softmax(action_logits, dim=-1)

        # Final check for NaN in probabilities and use uniform distribution as fallback
        if torch.isnan(action_probs).any():
            batch_size, action_dim = action_probs.shape
            action_probs = torch.ones_like(action_probs) / action_dim

        outputs['action_type'] = action_probs

        # UNLIMITED QUANTITY PREDICTION - LET MODEL CHOOSE
        # Allow model to predict any quantity >= 1, no upper limit
        quantity_raw = outputs['quantity']

        # Check for NaN/Inf in quantity predictions and replace with default
        if torch.isnan(quantity_raw).any() or torch.isinf(quantity_raw).any():
            quantity_raw = torch.ones_like(quantity_raw)  # Default to 1.0

        # Apply ReLU to ensure positive values, then add 1 to ensure minimum of 1
        # No upper limit - let the model choose and capital-aware system will adjust
        final_quantity = torch.relu(quantity_raw) + 1.0

        # Round to nearest integer for clean quantities
        final_quantity = torch.round(final_quantity)

        # Return as tensor with same shape as original quantity_raw
        outputs['quantity'] = final_quantity.to(dtype=torch.float32)

        return outputs




class CriticTransformerModel(TransformerModel):
    """
    Critic model using Transformer architecture.
    
    This replaces the previous LSTMModel when used as a critic and outputs
    state values for reinforcement learning.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        # Critic always outputs a single value
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass returning state values.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            State values of shape (batch_size, 1)
        """
        return super().forward(x, mask)


# Backward compatibility aliases
# These maintain the same interface as the old LSTM models
LSTMModel = TransformerModel
ActorLSTMModel = ActorTransformerModel
