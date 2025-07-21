import pytest
import torch
from src.models.transformer_models import TransformerModel, ActorTransformerModel, CriticTransformerModel

def test_transformer_model_output_shape():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    batch_size = 3
    sequence_length = 7

    model = TransformerModel(input_dim, hidden_dim, output_dim, num_heads=4, num_layers=2)

    # Create a dummy input tensor
    # Expected input shape: (batch_size, sequence_length, input_dim)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)

    output = model(dummy_input)

    # Expected output shape: (batch_size, output_dim)
    assert output.shape == (batch_size, output_dim)

def test_transformer_model_multiple_layers():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 3  # Test with multiple layers
    batch_size = 3
    sequence_length = 7

    model = TransformerModel(input_dim, hidden_dim, output_dim, num_heads=4, num_layers=num_layers)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    output = model(dummy_input)
    assert output.shape == (batch_size, output_dim)

def test_actor_transformer_model():
    input_dim = 15
    hidden_dim = 32
    action_dim = 3
    batch_size = 2
    sequence_length = 5

    model = ActorTransformerModel(input_dim, hidden_dim, action_dim, num_heads=2, num_layers=2)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    output = model(dummy_input)

    # Should output action probabilities
    assert output.shape == (batch_size, action_dim)
    # Should sum to 1 (probabilities)
    assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)

def test_critic_transformer_model():
    input_dim = 15
    hidden_dim = 32
    batch_size = 2
    sequence_length = 5

    model = CriticTransformerModel(input_dim, hidden_dim, num_heads=2, num_layers=2)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    output = model(dummy_input)

    # Should output single value per batch
    assert output.shape == (batch_size, 1)