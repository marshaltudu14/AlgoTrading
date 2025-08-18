"""
Core Transformer Module for Trading Agents

This module provides a generic, reusable Transformer-based neural network architecture
that can be used across all trading agents (HRM, MoE, MAML, and Autonomous).
The module is designed with configurable parameters to support future Neural Architecture Search (NAS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input sequences.
    Adds positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class CoreTransformer(nn.Module):
    """
    Core Transformer module for trading agents.
    
    This is a generic, reusable Transformer-based architecture that can be
    integrated into all existing and future trading agents. It provides
    configurable parameters for flexibility and future NAS compatibility.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 2048,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        use_positional_encoding: bool = True
    ):
        """
        Initialize the CoreTransformer.
        
        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            ff_dim: Dimension of feed-forward network
            output_dim: Dimension of output
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
            use_positional_encoding: Whether to use positional encoding
        """
        super(CoreTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.use_positional_encoding = use_positional_encoding
        
        # Input projection layer to match transformer dimension
        self.input_projection = nn.Linear(input_dim, ff_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(ff_dim, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=ff_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Use batch_first=True for easier handling
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(ff_dim, output_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(ff_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization with smaller scale for numerical stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use very small initialization scale for numerical stability and to prevent NaN
                nn.init.xavier_uniform_(module.weight, gain=0.01)  # Even smaller gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights with small values
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=0.01)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=0.01)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask of shape (seq_len, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
            If return_attention=True, returns tuple (output, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to transformer dimension
        x = self.input_projection(x)  # (batch_size, seq_len, ff_dim)

        # Check for NaN/Inf after input projection and replace with zeros
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)

        # Add positional encoding if enabled
        if self.use_positional_encoding:
            # Convert to (seq_len, batch_size, ff_dim) for positional encoding
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            # Convert back to (batch_size, seq_len, ff_dim)
            x = x.transpose(0, 1)

        # Apply layer normalization
        x = self.layer_norm(x)

        # Check for NaN/Inf after layer norm
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)
        
        # Pass through transformer encoder
        transformer_output = self.transformer_encoder(x, mask=mask)

        # Check for NaN/Inf after transformer encoder
        if torch.isnan(transformer_output).any() or torch.isinf(transformer_output).any():
            transformer_output = torch.zeros_like(transformer_output)

        # Global average pooling over sequence dimension
        # This aggregates information from all time steps
        pooled_output = torch.mean(transformer_output, dim=1)  # (batch_size, ff_dim)

        # Check for NaN/Inf after pooling
        if torch.isnan(pooled_output).any() or torch.isinf(pooled_output).any():
            pooled_output = torch.zeros_like(pooled_output)

        # Project to output dimension
        output = self.output_projection(pooled_output)  # (batch_size, output_dim)

        # Final check for NaN/Inf in output
        if torch.isnan(output).any() or torch.isinf(output).any():
            output = torch.zeros_like(output)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Get attention weights from the transformer layers.
        Useful for interpretability and debugging.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            
        Returns:
            List of attention weight tensors from each layer
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to transformer dimension
        x = self.input_projection(x)
        
        # Add positional encoding if enabled
        if self.use_positional_encoding:
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)
            x = x.transpose(0, 1)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Manually pass through each transformer layer to collect attention weights
        attention_weights = []
        for layer in self.transformer_encoder.layers:
            # Get attention weights from this layer
            attn_output, attn_weights = layer.self_attn(x, x, x, attn_mask=mask, need_weights=True)
            attention_weights.append(attn_weights)
            
            # Continue with the layer's forward pass
            x = layer.norm1(x + layer.dropout1(attn_output))
            ff_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_output))
        
        return attention_weights
    
    def get_config(self) -> dict:
        """
        Get the configuration of this transformer.
        Useful for saving/loading and NAS.
        
        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            'input_dim': self.input_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'use_positional_encoding': self.use_positional_encoding
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'CoreTransformer':
        """
        Create a CoreTransformer from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            CoreTransformer instance
        """
        return cls(**config)
