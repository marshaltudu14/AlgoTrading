"""
Unit tests for TransformerWorldModel.

Tests the functionality, input/output shapes, and prediction capabilities
of the TransformerWorldModel class.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from src.models.world_model import TransformerWorldModel


class TestTransformerWorldModel:
    """Test cases for TransformerWorldModel."""
    
    def test_world_model_initialization(self):
        """Test that TransformerWorldModel initializes correctly."""
        input_dim = 50
        action_dim = 3
        prediction_horizon = 5
        market_features = 5
        hidden_dim = 128
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            market_features=market_features,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2
        )
        
        assert model.input_dim == input_dim
        assert model.action_dim == action_dim
        assert model.prediction_horizon == prediction_horizon
        assert model.market_features == market_features
        assert model.hidden_dim == hidden_dim
        
        # Check that components are properly initialized
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'prediction_head')
        assert hasattr(model, 'policy_head')
        assert hasattr(model, 'confidence_head')
        
        # Check prediction head components
        assert 'market_state' in model.prediction_head
        assert 'market_regime' in model.prediction_head
        assert 'volatility' in model.prediction_head
        
        # Check policy head components
        assert 'actions' in model.policy_head
        assert 'value' in model.policy_head
    
    def test_world_model_forward_basic(self):
        """Test basic forward pass with correct input/output shapes."""
        input_dim = 30
        action_dim = 4
        prediction_horizon = 3
        market_features = 5
        batch_size = 2
        seq_len = 10
        hidden_dim = 64
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            market_features=market_features,
            hidden_dim=hidden_dim,
            num_heads=2,
            num_layers=2
        )
        
        # Create dummy input
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Forward pass
        output = model(x)
        
        # Check output structure
        assert isinstance(output, dict)
        assert 'predictions' in output
        assert 'policy' in output
        assert 'confidence' in output
        
        # Check prediction outputs
        predictions = output['predictions']
        assert 'market_state' in predictions
        assert 'market_regime' in predictions
        assert 'volatility' in predictions
        
        # Check prediction shapes
        assert predictions['market_state'].shape == (batch_size, prediction_horizon, market_features)
        assert predictions['market_regime'].shape == (batch_size, 4)  # Default 4 regimes
        assert predictions['volatility'].shape == (batch_size, prediction_horizon)
        
        # Check policy outputs
        policy = output['policy']
        assert 'actions' in policy
        assert 'value' in policy
        
        # Check policy shapes
        assert policy['actions'].shape == (batch_size, action_dim)
        assert policy['value'].shape == (batch_size, 1)
        
        # Check confidence shape
        assert output['confidence'].shape == (batch_size, 1)
        
        # Check that outputs are valid
        assert not torch.isnan(output['predictions']['market_state']).any()
        assert not torch.isnan(output['policy']['actions']).any()
        assert not torch.isnan(output['confidence']).any()
        
        # Check that action probabilities sum to 1
        assert torch.allclose(policy['actions'].sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Check that confidence is between 0 and 1
        assert (output['confidence'] >= 0).all() and (output['confidence'] <= 1).all()
    
    def test_world_model_single_timestep_input(self):
        """Test world model with single timestep input."""
        input_dim = 20
        action_dim = 2
        batch_size = 3
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=64,
            num_heads=2,
            num_layers=1
        )
        
        # Single timestep input (batch_size, input_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = model(x)
        
        # Should still work and produce correct shapes
        assert output['predictions']['market_state'].shape == (batch_size, 5, 5)  # Default values
        assert output['policy']['actions'].shape == (batch_size, action_dim)
    
    def test_world_model_with_attention_weights(self):
        """Test world model returning attention weights."""
        input_dim = 25
        action_dim = 3
        batch_size = 2
        seq_len = 8
        num_layers = 2
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=64,
            num_heads=2,
            num_layers=num_layers
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x, return_attention=True)
        
        # Check that attention weights are returned
        assert 'attention_weights' in output
        assert len(output['attention_weights']) == num_layers
        
        # Check attention weight shapes
        for attn_weights in output['attention_weights']:
            assert attn_weights.shape == (batch_size, seq_len, seq_len)
    
    def test_world_model_predict_future_states(self):
        """Test future state prediction functionality."""
        input_dim = 15
        action_dim = 2
        batch_size = 2
        seq_len = 5
        prediction_horizon = 4
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Test default prediction
        future_states = model.predict_future_states(x)
        
        assert 'market_state' in future_states
        assert 'volatility' in future_states
        assert 'confidence' in future_states
        assert 'market_regime' in future_states
        
        # Check shapes
        assert future_states['market_state'].shape == (batch_size, prediction_horizon, 5)
        assert future_states['volatility'].shape == (batch_size, prediction_horizon)
        assert future_states['confidence'].shape == (batch_size, 1)
        
        # Test custom number of steps
        custom_steps = 2
        future_states_custom = model.predict_future_states(x, num_steps=custom_steps)
        assert future_states_custom['market_state'].shape == (batch_size, custom_steps, 5)
        assert future_states_custom['volatility'].shape == (batch_size, custom_steps)
    
    def test_world_model_simulate_action_outcomes(self):
        """Test action outcome simulation."""
        input_dim = 20
        action_dim = 3
        batch_size = 2
        seq_len = 6
        num_actions = 3
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        actions = torch.randint(0, action_dim, (batch_size, num_actions))
        
        outcomes = model.simulate_action_outcomes(x, actions)
        
        assert 'predicted_returns' in outcomes
        assert 'predicted_volatility' in outcomes
        assert 'action_values' in outcomes
        assert 'confidence' in outcomes
        
        # Check shapes
        assert outcomes['predicted_returns'].shape == (batch_size, 5)  # Default prediction horizon
        assert outcomes['predicted_volatility'].shape == (batch_size, 5)
        assert outcomes['action_values'].shape == (batch_size, num_actions)
        assert outcomes['confidence'].shape == (batch_size, num_actions)
    
    def test_world_model_without_market_regime(self):
        """Test world model without market regime prediction."""
        input_dim = 10
        action_dim = 2
        batch_size = 2
        seq_len = 4
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            predict_market_regime=False,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        
        # Market regime should not be in predictions
        assert 'market_regime' not in output['predictions']
        
        # Other predictions should still be there
        assert 'market_state' in output['predictions']
        assert 'volatility' in output['predictions']
    
    def test_world_model_config_methods(self):
        """Test get_config and from_config methods."""
        original_config = {
            'input_dim': 40,
            'action_dim': 4,
            'prediction_horizon': 6,
            'market_features': 5,
            'hidden_dim': 128,
            'predict_market_regime': True,
            'num_market_regimes': 3
        }
        
        # Create model from config
        model1 = TransformerWorldModel(**original_config)
        
        # Get config and create new model
        config = model1.get_config()
        model2 = TransformerWorldModel.from_config(config)
        
        # Check that configs match
        assert config['input_dim'] == original_config['input_dim']
        assert config['action_dim'] == original_config['action_dim']
        assert config['prediction_horizon'] == original_config['prediction_horizon']
        assert config['market_features'] == original_config['market_features']
        assert config['hidden_dim'] == original_config['hidden_dim']
        assert config['predict_market_regime'] == original_config['predict_market_regime']
        assert config['num_market_regimes'] == original_config['num_market_regimes']
        
        # Test that both models have same structure
        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, original_config['input_dim'])
        
        output1 = model1(x)
        output2 = model2(x)
        
        # Outputs should have same shapes
        assert output1['predictions']['market_state'].shape == output2['predictions']['market_state'].shape
        assert output1['policy']['actions'].shape == output2['policy']['actions'].shape
    
    def test_world_model_gradient_flow(self):
        """Test that gradients flow properly through the world model."""
        input_dim = 15
        action_dim = 2
        batch_size = 2
        seq_len = 4

        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_dim=32,
            num_heads=2,
            num_layers=1
        )

        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

        output = model(x)

        # Create a more complex loss that ensures gradients flow
        market_state_loss = output['predictions']['market_state'].pow(2).mean()
        policy_loss = -output['policy']['actions'].log().mean()  # Cross-entropy style loss
        value_loss = output['policy']['value'].pow(2).mean()
        confidence_loss = output['confidence'].pow(2).mean()

        total_loss = market_state_loss + policy_loss + value_loss + confidence_loss

        total_loss.backward()

        # Check that gradients exist and are not zero
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

        # Check that at least some model parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "At least some model parameters should have gradients"
    
    def test_world_model_interpretability(self):
        """Test that prediction outputs are interpretable for what-if scenarios."""
        input_dim = 20
        action_dim = 3
        batch_size = 1
        seq_len = 10
        
        model = TransformerWorldModel(
            input_dim=input_dim,
            action_dim=action_dim,
            prediction_horizon=3,
            market_features=5,
            hidden_dim=64,
            num_heads=2,
            num_layers=2
        )
        
        # Create realistic market data (normalized)
        x = torch.randn(batch_size, seq_len, input_dim) * 0.1  # Small variations
        
        output = model(x)
        
        # Check that market state predictions are reasonable
        market_state = output['predictions']['market_state']
        
        # Market state should have shape (batch, horizon, features)
        assert market_state.shape == (batch_size, 3, 5)
        
        # Volatility should be positive
        volatility = output['predictions']['volatility']
        assert (volatility >= 0).all(), "Volatility should be non-negative"
        
        # Market regime probabilities should sum to 1
        market_regime = output['predictions']['market_regime']
        assert torch.allclose(market_regime.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Action probabilities should sum to 1
        actions = output['policy']['actions']
        assert torch.allclose(actions.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Confidence should be between 0 and 1
        confidence = output['confidence']
        assert (confidence >= 0).all() and (confidence <= 1).all()
        
        print("✓ All interpretability checks passed")


if __name__ == "__main__":
    pytest.main([__file__])
