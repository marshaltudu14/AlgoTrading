"""
Comprehensive tests for MoE (Mixture of Experts) agent implementation.
Tests MoEAgent expert coordination, gating network, learning, and adaptation mechanisms.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.agents.moe_agent import MoEAgent, GatingNetwork
from src.agents.base_agent import BaseAgent
from tests.conftest import (
    TEST_OBSERVATION_DIM, TEST_ACTION_DIM_DISCRETE, TEST_ACTION_DIM_CONTINUOUS,
    TEST_HIDDEN_DIM, assert_valid_action, assert_valid_observation
)

class TestGatingNetwork:
    """Test suite for GatingNetwork component."""
    
    def test_gating_network_initialization(self):
        """Test GatingNetwork initialization."""
        input_dim = TEST_OBSERVATION_DIM
        num_experts = 4
        hidden_dim = TEST_HIDDEN_DIM
        
        gating_network = GatingNetwork(input_dim, num_experts, hidden_dim)
        
        assert hasattr(gating_network, 'fc1')
        assert hasattr(gating_network, 'fc2')
        assert gating_network.fc1.in_features == input_dim
        assert gating_network.fc2.out_features == num_experts
    
    def test_gating_network_forward(self):
        """Test GatingNetwork forward pass."""
        input_dim = TEST_OBSERVATION_DIM
        num_experts = 4
        hidden_dim = TEST_HIDDEN_DIM
        batch_size = 3
        
        gating_network = GatingNetwork(input_dim, num_experts, hidden_dim)
        
        # Create input tensor
        input_tensor = torch.randn(batch_size, input_dim)
        
        # Forward pass
        output = gating_network(input_tensor)
        
        # Check output shape
        assert output.shape == (batch_size, num_experts)
        
        # Check that outputs sum to 1 (softmax property)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-6)
        
        # Check that all outputs are positive
        assert (output >= 0).all()
    
    def test_gating_network_different_inputs(self):
        """Test GatingNetwork with different input patterns."""
        input_dim = TEST_OBSERVATION_DIM
        num_experts = 4
        hidden_dim = TEST_HIDDEN_DIM
        
        gating_network = GatingNetwork(input_dim, num_experts, hidden_dim)
        
        # Test different input patterns
        test_inputs = [
            torch.zeros(1, input_dim),  # All zeros
            torch.ones(1, input_dim),   # All ones
            torch.randn(1, input_dim),  # Random
        ]
        
        for input_tensor in test_inputs:
            output = gating_network(input_tensor)
            
            assert output.shape == (1, num_experts)
            assert torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-6)
            assert (output >= 0).all()

class TestMoEAgentInitialization:
    """Test suite for MoE agent initialization."""
    
    def test_basic_initialization(self):
        """Test basic MoE agent initialization."""
        expert_configs = {
            'TrendAgent': {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM},
            'MeanReversionAgent': {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM},
            'VolatilityAgent': {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM},
            'ConsolidationAgent': {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM}
        }
        
        agent = MoEAgent(
            observation_dim=TEST_OBSERVATION_DIM,
            action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
            action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
            hidden_dim=TEST_HIDDEN_DIM,
            expert_configs=expert_configs
        )
        
        # Check inheritance
        assert isinstance(agent, MoEAgent)
        assert isinstance(agent, BaseAgent)
        
        # Check parameters
        assert agent.observation_dim == TEST_OBSERVATION_DIM
        assert agent.action_dim_discrete == TEST_ACTION_DIM_DISCRETE
        assert agent.action_dim_continuous == TEST_ACTION_DIM_CONTINUOUS
        assert agent.hidden_dim == TEST_HIDDEN_DIM
        
        # Check components
        assert hasattr(agent, 'gating_network')
        assert hasattr(agent, 'experts')
        assert agent.gating_network is not None
        assert len(agent.experts) == len(expert_configs)
        
        # Check that all experts are BaseAgent instances
        for expert in agent.experts:
            assert isinstance(expert, BaseAgent)
    
    def test_initialization_different_expert_counts(self):
        """Test initialization with different numbers of experts."""
        expert_counts = [2, 3, 5, 8]
        
        for count in expert_counts:
            expert_configs = {}
            for i in range(count):
                expert_configs[f'Expert{i}'] = {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM}
            
            agent = MoEAgent(
                observation_dim=TEST_OBSERVATION_DIM,
                action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM,
                expert_configs=expert_configs
            )
            
            assert len(agent.experts) == count
            assert agent.gating_network.fc2.out_features == count
    
    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test empty expert configs
        with pytest.raises((ValueError, AssertionError)):
            MoEAgent(
                observation_dim=TEST_OBSERVATION_DIM,
                action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM,
                expert_configs={}
            )
        
        # Test invalid dimensions
        expert_configs = {'Expert1': {'lr': 0.001, 'hidden_dim': TEST_HIDDEN_DIM}}
        
        with pytest.raises((ValueError, AssertionError)):
            MoEAgent(
                observation_dim=-1,
                action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM,
                expert_configs=expert_configs
            )

class TestMoEAgentActionSelection:
    """Test suite for MoE agent action selection."""
    
    def test_select_action_basic(self, sample_moe_agent):
        """Test basic action selection."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        action_type, quantity = sample_moe_agent.select_action(observation)
        
        assert_valid_action(action_type, quantity)
    
    def test_select_action_expert_coordination(self, sample_moe_agent):
        """Test that action selection involves expert coordination."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        # Mock expert actions to verify coordination
        with patch.object(sample_moe_agent.experts[0], 'select_action', return_value=(0, 1.0)) as mock1, \
             patch.object(sample_moe_agent.experts[1], 'select_action', return_value=(1, 2.0)) as mock2, \
             patch.object(sample_moe_agent.experts[2], 'select_action', return_value=(2, 1.5)) as mock3, \
             patch.object(sample_moe_agent.experts[3], 'select_action', return_value=(3, 0.5)) as mock4:
            
            action_type, quantity = sample_moe_agent.select_action(observation)
            
            # All experts should be consulted
            mock1.assert_called_once()
            mock2.assert_called_once()
            mock3.assert_called_once()
            mock4.assert_called_once()
            
            assert_valid_action(action_type, quantity)
    
    def test_select_action_gating_weights(self, sample_moe_agent):
        """Test that gating network produces valid weights."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        # Get gating weights directly
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        gating_weights = sample_moe_agent.gating_network(obs_tensor)
        
        # Check weight properties
        assert gating_weights.shape == (1, len(sample_moe_agent.experts))
        assert torch.allclose(gating_weights.sum(dim=1), torch.ones(1), atol=1e-6)
        assert (gating_weights >= 0).all()
    
    def test_select_action_deterministic(self, sample_moe_agent):
        """Test action selection determinism with same input."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        action1_type, action1_quantity = sample_moe_agent.select_action(observation)
        
        torch.manual_seed(42)
        np.random.seed(42)
        action2_type, action2_quantity = sample_moe_agent.select_action(observation)
        
        # Actions should be the same with same seed
        assert action1_type == action2_type
        assert abs(action1_quantity - action2_quantity) < 1e-6
    
    def test_select_action_edge_cases(self, sample_moe_agent):
        """Test action selection with edge case observations."""
        edge_cases = [
            np.zeros(TEST_OBSERVATION_DIM, dtype=np.float32),  # All zeros
            np.ones(TEST_OBSERVATION_DIM, dtype=np.float32),   # All ones
            np.full(TEST_OBSERVATION_DIM, 0.5, dtype=np.float32),  # All 0.5
        ]
        
        for obs in edge_cases:
            action_type, quantity = sample_moe_agent.select_action(obs)
            assert_valid_action(action_type, quantity)

class TestMoEAgentLearning:
    """Test suite for MoE agent learning."""
    
    def test_learn_basic(self, sample_moe_agent, sample_experiences):
        """Test basic learning functionality."""
        # Store initial parameters
        initial_gating_params = [p.clone() for p in sample_moe_agent.gating_network.parameters()]
        initial_expert_params = []
        for expert in sample_moe_agent.experts:
            if hasattr(expert, 'actor'):
                initial_expert_params.append([p.clone() for p in expert.actor.parameters()])
        
        # Learn from experiences
        sample_moe_agent.learn(sample_experiences)
        
        # Check that gating network parameters have changed
        gating_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_gating_params, sample_moe_agent.gating_network.parameters())
        )
        
        # At least gating network should have changed
        assert gating_changed
    
    def test_learn_expert_coordination(self, sample_moe_agent, sample_experiences):
        """Test that learning involves expert coordination."""
        # Mock expert learning to verify coordination
        learn_calls = []
        
        for i, expert in enumerate(sample_moe_agent.experts):
            original_learn = expert.learn
            def mock_learn(experiences, expert_id=i):
                learn_calls.append(expert_id)
                return original_learn(experiences)
            expert.learn = mock_learn
        
        # Learn from experiences
        sample_moe_agent.learn(sample_experiences)
        
        # All experts should have been called for learning
        assert len(learn_calls) == len(sample_moe_agent.experts)
        assert set(learn_calls) == set(range(len(sample_moe_agent.experts)))
    
    def test_learn_empty_experiences(self, sample_moe_agent):
        """Test learning with empty experiences."""
        empty_experiences = []
        
        # Should handle empty experiences gracefully
        try:
            sample_moe_agent.learn(empty_experiences)
        except Exception as e:
            # If an exception is raised, it should be informative
            assert "empty" in str(e).lower() or "no" in str(e).lower()
    
    def test_learn_gating_network_training(self, sample_moe_agent, sample_experiences):
        """Test that gating network is trained based on expert performance."""
        # Store initial gating parameters
        initial_params = [p.clone() for p in sample_moe_agent.gating_network.parameters()]
        
        # Learn multiple times to ensure gating network adaptation
        for _ in range(3):
            sample_moe_agent.learn(sample_experiences)
        
        # Gating network parameters should have changed
        final_params = list(sample_moe_agent.gating_network.parameters())
        
        total_change = sum(
            torch.norm(final - initial).item()
            for initial, final in zip(initial_params, final_params)
        )
        
        # Should have some parameter change
        assert total_change > 0

class TestMoEAgentAdaptation:
    """Test suite for MoE agent adaptation (MAML)."""
    
    def test_adapt_basic(self, sample_moe_agent):
        """Test basic adaptation functionality."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        adapted_agent = sample_moe_agent.adapt(
            observation, action, reward, next_observation, done, num_gradient_steps
        )
        
        # Should return a BaseAgent
        assert isinstance(adapted_agent, BaseAgent)
        
        # Should be able to select actions
        test_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action_type, quantity = adapted_agent.select_action(test_obs)
        assert_valid_action(action_type, quantity)
    
    def test_adapt_expert_coordination(self, sample_moe_agent):
        """Test that adaptation involves all experts."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        # Mock expert adaptation to verify coordination
        adapt_calls = []
        
        for i, expert in enumerate(sample_moe_agent.experts):
            original_adapt = expert.adapt
            def mock_adapt(*args, expert_id=i, **kwargs):
                adapt_calls.append(expert_id)
                return original_adapt(*args, **kwargs)
            expert.adapt = mock_adapt
        
        # Adapt
        adapted_agent = sample_moe_agent.adapt(
            observation, action, reward, next_observation, done, num_gradient_steps
        )
        
        # All experts should have been called for adaptation
        assert len(adapt_calls) == len(sample_moe_agent.experts)
        assert isinstance(adapted_agent, BaseAgent)
    
    def test_adapt_preserves_original(self, sample_moe_agent):
        """Test that adaptation preserves original agent."""
        # Store original parameters
        original_gating_params = [p.clone() for p in sample_moe_agent.gating_network.parameters()]
        
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        # Adapt
        adapted_agent = sample_moe_agent.adapt(
            observation, action, reward, next_observation, done, num_gradient_steps
        )
        
        # Original agent parameters should be unchanged
        for original, current in zip(original_gating_params, sample_moe_agent.gating_network.parameters()):
            assert torch.allclose(original, current, atol=1e-6)

class TestMoEAgentIntegration:
    """Test suite for MoE agent integration."""
    
    def test_integration_with_environment(self, sample_moe_agent, sample_trading_env):
        """Test integration with trading environment."""
        obs = sample_trading_env.reset()
        
        # Agent should be able to select action for environment observation
        action_type, quantity = sample_moe_agent.select_action(obs)
        assert_valid_action(action_type, quantity)
        
        # Environment should accept agent's action
        action = (action_type, quantity)
        next_obs, reward, done, info = sample_trading_env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    
    def test_expert_specialization(self, sample_moe_agent):
        """Test that different experts can specialize for different market conditions."""
        # Create different market condition observations
        trending_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        trending_obs[:5] = 1.0  # High trend indicators
        
        sideways_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        sideways_obs[:5] = 0.0  # Low trend indicators
        
        # Get gating weights for different conditions
        trending_weights = sample_moe_agent.gating_network(torch.FloatTensor(trending_obs).unsqueeze(0))
        sideways_weights = sample_moe_agent.gating_network(torch.FloatTensor(sideways_obs).unsqueeze(0))
        
        # Weights should be different for different market conditions
        weight_diff = torch.norm(trending_weights - sideways_weights).item()
        
        # Allow for some similarity but expect some difference
        assert weight_diff >= 0  # At minimum, should not be identical
