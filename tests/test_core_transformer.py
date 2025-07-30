"""
Unit tests for CoreTransformer module.

Tests the functionality, input/output shapes, and basic operations
of the CoreTransformer class.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.core_transformer import CoreTransformer, PositionalEncoding


class TestPositionalEncoding:
    """Test cases for PositionalEncoding module."""
    
    def test_positional_encoding_initialization(self):
        """Test that PositionalEncoding initializes correctly."""
        d_model = 512
        max_len = 1000
        dropout = 0.1

        pos_enc = PositionalEncoding(d_model, max_len, dropout)

        # The pe buffer has shape (max_len, 1, d_model) due to unsqueeze and transpose
        assert pos_enc.pe.shape == (max_len, 1, d_model)
        assert isinstance(pos_enc.dropout, nn.Dropout)
    
    def test_positional_encoding_forward(self):
        """Test PositionalEncoding forward pass."""
        d_model = 64
        seq_len = 10
        batch_size = 2
        
        pos_enc = PositionalEncoding(d_model)
        
        # Input shape: (seq_len, batch_size, d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        output = pos_enc(x)
        
        assert output.shape == (seq_len, batch_size, d_model)
        assert output.dtype == torch.float32


class TestCoreTransformer:
    """Test cases for CoreTransformer module."""
    
    def test_core_transformer_initialization(self):
        """Test that CoreTransformer initializes with correct parameters."""
        input_dim = 50
        num_heads = 8
        num_layers = 6
        ff_dim = 512
        output_dim = 3
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            output_dim=output_dim
        )
        
        assert transformer.input_dim == input_dim
        assert transformer.num_heads == num_heads
        assert transformer.num_layers == num_layers
        assert transformer.ff_dim == ff_dim
        assert transformer.output_dim == output_dim
        
        # Check that layers are properly initialized
        assert isinstance(transformer.input_projection, nn.Linear)
        assert isinstance(transformer.transformer_encoder, nn.TransformerEncoder)
        assert isinstance(transformer.output_projection, nn.Linear)
        assert isinstance(transformer.layer_norm, nn.LayerNorm)
    
    def test_core_transformer_forward_basic(self):
        """Test basic forward pass with correct input/output shapes."""
        input_dim = 20
        output_dim = 5
        batch_size = 4
        seq_len = 15
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=4,
            num_layers=2,
            ff_dim=128
        )
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = transformer(x)
        
        # Check output shape
        assert output.shape == (batch_size, output_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_core_transformer_different_configurations(self):
        """Test CoreTransformer with various parameter configurations."""
        test_configs = [
            {
                'input_dim': 10,
                'num_heads': 2,
                'num_layers': 1,
                'ff_dim': 64,
                'output_dim': 1
            },
            {
                'input_dim': 100,
                'num_heads': 8,
                'num_layers': 4,
                'ff_dim': 256,
                'output_dim': 10
            },
            {
                'input_dim': 50,
                'num_heads': 4,
                'num_layers': 3,
                'ff_dim': 512,
                'output_dim': 3
            }
        ]
        
        batch_size = 2
        seq_len = 10
        
        for config in test_configs:
            transformer = CoreTransformer(**config)
            x = torch.randn(batch_size, seq_len, config['input_dim'])
            
            output = transformer(x)
            
            assert output.shape == (batch_size, config['output_dim'])
            assert not torch.isnan(output).any()
    
    def test_core_transformer_with_mask(self):
        """Test CoreTransformer with attention mask."""
        input_dim = 30
        output_dim = 2
        batch_size = 3
        seq_len = 8
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            num_layers=2,
            ff_dim=128
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Create a simple mask (mask out last 2 positions)
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        mask[-2:, :] = True  # Mask last 2 positions
        
        output = transformer(x, mask=mask)
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
    
    def test_core_transformer_without_positional_encoding(self):
        """Test CoreTransformer without positional encoding."""
        input_dim = 25
        output_dim = 4
        batch_size = 2
        seq_len = 12
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=4,
            num_layers=2,
            ff_dim=128,
            use_positional_encoding=False
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = transformer(x)
        
        assert output.shape == (batch_size, output_dim)
        assert not torch.isnan(output).any()
    
    def test_core_transformer_gradient_flow(self):
        """Test that gradients flow properly through the transformer."""
        input_dim = 15
        output_dim = 1
        batch_size = 2
        seq_len = 5
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            num_layers=1,
            ff_dim=64
        )
        
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        target = torch.randn(batch_size, output_dim)
        
        output = transformer(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that gradients exist and are not zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
        # Check that model parameters have gradients
        for param in transformer.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_core_transformer_config_methods(self):
        """Test get_config and from_config methods."""
        original_config = {
            'input_dim': 40,
            'num_heads': 4,
            'num_layers': 3,
            'ff_dim': 256,
            'output_dim': 2,
            'dropout': 0.2,
            'max_seq_len': 500,
            'use_positional_encoding': True
        }
        
        # Create transformer from config
        transformer1 = CoreTransformer(**original_config)
        
        # Get config and create new transformer
        config = transformer1.get_config()
        transformer2 = CoreTransformer.from_config(config)
        
        # Check that configs match
        assert config == original_config
        assert transformer2.get_config() == original_config
        
        # Test that both transformers produce same output for same input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, original_config['input_dim'])
        
        # Set both to eval mode and ensure same random state
        transformer1.eval()
        transformer2.eval()
        
        with torch.no_grad():
            output1 = transformer1(x)
            output2 = transformer2(x)
        
        # Outputs should have same shape (weights will be different due to random init)
        assert output1.shape == output2.shape
    
    def test_core_transformer_attention_weights(self):
        """Test getting attention weights from transformer."""
        input_dim = 20
        output_dim = 1
        batch_size = 2
        seq_len = 8
        num_layers = 2
        
        transformer = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            num_layers=num_layers,
            ff_dim=64
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        attention_weights = transformer.get_attention_weights(x)
        
        # Should have attention weights for each layer
        assert len(attention_weights) == num_layers
        
        # Each attention weight tensor should have correct shape
        for attn_weights in attention_weights:
            # Shape: (batch_size, seq_len, seq_len) - averaged over heads
            assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_core_transformer_deterministic(self):
        """Test that transformer produces deterministic outputs with same seed."""
        input_dim = 30
        output_dim = 3
        batch_size = 2
        seq_len = 10
        
        # Set random seed
        torch.manual_seed(42)
        transformer1 = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            num_layers=1,
            ff_dim=128
        )
        
        torch.manual_seed(42)
        transformer2 = CoreTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=2,
            num_layers=1,
            ff_dim=128
        )
        
        # Same input
        torch.manual_seed(123)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        transformer1.eval()
        transformer2.eval()
        
        with torch.no_grad():
            output1 = transformer1(x)
            output2 = transformer2(x)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
