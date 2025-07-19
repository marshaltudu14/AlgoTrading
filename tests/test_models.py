import pytest
import torch
from src.models.lstm_model import LSTMModel

def test_lstm_model_output_shape():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 1
    batch_size = 3
    sequence_length = 7

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    
    # Create a dummy input tensor
    # Expected input shape: (batch_size, sequence_length, input_dim)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    
    output = model(dummy_input)
    
    # Expected output shape: (batch_size, output_dim)
    assert output.shape == (batch_size, output_dim)

def test_lstm_model_multiple_layers():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    num_layers = 2  # Test with multiple layers
    batch_size = 3
    sequence_length = 7

    model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    output = model(dummy_input)
    assert output.shape == (batch_size, output_dim)