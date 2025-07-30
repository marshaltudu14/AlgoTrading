"""
Comprehensive tests for PPO agent implementation.
Tests PPOAgent initialization, action selection, learning, adaptation, and model persistence.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.agents.ppo_agent import PPOAgent
from src.agents.base_agent import BaseAgent
from tests.conftest import (
    TEST_OBSERVATION_DIM, TEST_ACTION_DIM_DISCRETE, TEST_ACTION_DIM_CONTINUOUS,
    TEST_HIDDEN_DIM, assert_valid_action, assert_valid_observation
)

class TestPPOAgentInitialization:
    """Test suite for PPO agent initialization."""
    
    def test_basic_initialization(self):
        """Test basic PPO agent initialization."""
        agent = PPOAgent(
            observation_dim=TEST_OBSERVATION_DIM,
            action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
            action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
            hidden_dim=TEST_HIDDEN_DIM,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=0.99,
            epsilon_clip=0.2,
            k_epochs=3
        )
        
        # Check inheritance
        assert isinstance(agent, PPOAgent)
        assert isinstance(agent, BaseAgent)
        
        # Check parameters
        assert agent.observation_dim == TEST_OBSERVATION_DIM
        assert agent.action_dim_discrete == TEST_ACTION_DIM_DISCRETE
        assert agent.action_dim_continuous == TEST_ACTION_DIM_CONTINUOUS
        assert agent.hidden_dim == TEST_HIDDEN_DIM
        assert agent.gamma == 0.99
        assert agent.epsilon_clip == 0.2
        assert agent.k_epochs == 3
        
        # Check networks
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'critic')
        assert hasattr(agent, 'policy_old')
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.policy_old is not None
        
        # Check optimizers
        assert hasattr(agent, 'optimizer_actor')
        assert hasattr(agent, 'optimizer_critic')
        assert agent.optimizer_actor is not None
        assert agent.optimizer_critic is not None
    
    def test_initialization_with_different_parameters(self):
        """Test initialization with different parameter values."""
        params = [
            (10, 3, 1, 32, 0.0001, 0.0005, 0.95, 0.1, 5),
            (50, 10, 2, 128, 0.01, 0.01, 0.999, 0.3, 10),
        ]
        
        for obs_dim, act_disc, act_cont, hidden, lr_a, lr_c, gamma, eps, k in params:
            agent = PPOAgent(
                observation_dim=obs_dim,
                action_dim_discrete=act_disc,
                action_dim_continuous=act_cont,
                hidden_dim=hidden,
                lr_actor=lr_a,
                lr_critic=lr_c,
                gamma=gamma,
                epsilon_clip=eps,
                k_epochs=k
            )
            
            assert agent.observation_dim == obs_dim
            assert agent.action_dim_discrete == act_disc
            assert agent.action_dim_continuous == act_cont
            assert agent.hidden_dim == hidden
            assert agent.gamma == gamma
            assert agent.epsilon_clip == eps
            assert agent.k_epochs == k
    
    def test_initialization_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        # Test negative dimensions
        with pytest.raises((ValueError, AssertionError)):
            PPOAgent(
                observation_dim=-1,
                action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM
            )
        
        # Test zero dimensions
        with pytest.raises((ValueError, AssertionError)):
            PPOAgent(
                observation_dim=TEST_OBSERVATION_DIM,
                action_dim_discrete=0,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM
            )
        
        # Test invalid learning rates
        with pytest.raises((ValueError, AssertionError)):
            PPOAgent(
                observation_dim=TEST_OBSERVATION_DIM,
                action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
                action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
                hidden_dim=TEST_HIDDEN_DIM,
                lr_actor=-0.001
            )

class TestPPOAgentActionSelection:
    """Test suite for PPO agent action selection."""
    
    def test_select_action_basic(self, sample_ppo_agent):
        """Test basic action selection."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        action_type, quantity = sample_ppo_agent.select_action(observation)
        
        assert_valid_action(action_type, quantity)
    
    def test_select_action_deterministic(self, sample_ppo_agent):
        """Test action selection determinism with same input."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        action1_type, action1_quantity = sample_ppo_agent.select_action(observation)
        
        torch.manual_seed(42)
        action2_type, action2_quantity = sample_ppo_agent.select_action(observation)
        
        # Actions should be the same with same seed
        assert action1_type == action2_type
        assert abs(action1_quantity - action2_quantity) < 1e-6
    
    def test_select_action_different_observations(self, sample_ppo_agent):
        """Test action selection with different observations."""
        observations = [
            np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32),
            np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32),
            np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32),
        ]
        
        actions = []
        for obs in observations:
            action_type, quantity = sample_ppo_agent.select_action(obs)
            actions.append((action_type, quantity))
            assert_valid_action(action_type, quantity)
        
        # Actions might be different for different observations
        # (though not guaranteed due to randomness)
        assert len(actions) == 3
    
    def test_select_action_edge_cases(self, sample_ppo_agent):
        """Test action selection with edge case observations."""
        edge_cases = [
            np.zeros(TEST_OBSERVATION_DIM, dtype=np.float32),  # All zeros
            np.ones(TEST_OBSERVATION_DIM, dtype=np.float32),   # All ones
            np.full(TEST_OBSERVATION_DIM, 0.5, dtype=np.float32),  # All 0.5
        ]
        
        for obs in edge_cases:
            action_type, quantity = sample_ppo_agent.select_action(obs)
            assert_valid_action(action_type, quantity)
    
    def test_select_action_invalid_input(self, sample_ppo_agent):
        """Test action selection with invalid input."""
        invalid_inputs = [
            np.random.rand(TEST_OBSERVATION_DIM - 1).astype(np.float32),  # Wrong size
            np.random.rand(TEST_OBSERVATION_DIM + 1).astype(np.float32),  # Wrong size
            np.array([np.nan] * TEST_OBSERVATION_DIM, dtype=np.float32),  # NaN values
            np.array([np.inf] * TEST_OBSERVATION_DIM, dtype=np.float32),  # Inf values
        ]
        
        for invalid_obs in invalid_inputs:
            try:
                action_type, quantity = sample_ppo_agent.select_action(invalid_obs)
                # If no exception, should still return valid action
                assert_valid_action(action_type, quantity)
            except (ValueError, RuntimeError):
                # Expected for invalid inputs
                pass

class TestPPOAgentLearning:
    """Test suite for PPO agent learning."""
    
    def test_learn_basic(self, sample_ppo_agent, sample_experiences):
        """Test basic learning functionality."""
        # Store initial parameters
        initial_actor_params = [p.clone() for p in sample_ppo_agent.actor.parameters()]
        initial_critic_params = [p.clone() for p in sample_ppo_agent.critic.parameters()]
        
        # Learn from experiences
        sample_ppo_agent.learn(sample_experiences)
        
        # Check that parameters have changed (learning occurred)
        actor_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_actor_params, sample_ppo_agent.actor.parameters())
        )
        critic_changed = any(
            not torch.allclose(initial, current, atol=1e-6)
            for initial, current in zip(initial_critic_params, sample_ppo_agent.critic.parameters())
        )
        
        # At least one network should have changed
        assert actor_changed or critic_changed
    
    def test_learn_empty_experiences(self, sample_ppo_agent):
        """Test learning with empty experiences."""
        empty_experiences = []
        
        # Should handle empty experiences gracefully
        try:
            sample_ppo_agent.learn(empty_experiences)
        except Exception as e:
            # If an exception is raised, it should be informative
            assert "empty" in str(e).lower() or "no" in str(e).lower()
    
    def test_learn_single_experience(self, sample_ppo_agent):
        """Test learning with single experience."""
        single_experience = [
            (
                np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32),
                (0, 1.0),
                1.0,
                np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32),
                False
            )
        ]
        
        # Should handle single experience
        try:
            sample_ppo_agent.learn(single_experience)
        except Exception as e:
            # Some implementations might require batch size > 1
            assert "batch" in str(e).lower() or "size" in str(e).lower()
    
    def test_learn_multiple_epochs(self, sample_ppo_agent, sample_experiences):
        """Test learning with multiple epochs."""
        # Store initial parameters
        initial_params = [p.clone() for p in sample_ppo_agent.actor.parameters()]
        
        # Learn multiple times
        for _ in range(3):
            sample_ppo_agent.learn(sample_experiences)
        
        # Parameters should have changed more significantly
        final_params = list(sample_ppo_agent.actor.parameters())
        
        total_change = sum(
            torch.norm(final - initial).item()
            for initial, final in zip(initial_params, final_params)
        )
        
        # Should have some parameter change
        assert total_change > 0
    
    def test_learn_gradient_flow(self, sample_ppo_agent, sample_experiences):
        """Test that gradients flow properly during learning."""
        # Enable gradient tracking
        for param in sample_ppo_agent.actor.parameters():
            param.requires_grad_(True)
        for param in sample_ppo_agent.critic.parameters():
            param.requires_grad_(True)
        
        # Learn from experiences
        sample_ppo_agent.learn(sample_experiences)
        
        # Check that gradients were computed
        actor_has_grads = any(
            p.grad is not None and torch.norm(p.grad) > 1e-8
            for p in sample_ppo_agent.actor.parameters()
        )
        critic_has_grads = any(
            p.grad is not None and torch.norm(p.grad) > 1e-8
            for p in sample_ppo_agent.critic.parameters()
        )
        
        # At least one network should have gradients
        assert actor_has_grads or critic_has_grads

class TestPPOAgentAdaptation:
    """Test suite for PPO agent adaptation (MAML)."""
    
    def test_adapt_basic(self, sample_ppo_agent):
        """Test basic adaptation functionality."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        adapted_agent = sample_ppo_agent.adapt(
            observation, action, reward, next_observation, done, num_gradient_steps
        )
        
        # Should return a BaseAgent
        assert isinstance(adapted_agent, BaseAgent)
        
        # Should be able to select actions
        test_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action_type, quantity = adapted_agent.select_action(test_obs)
        assert_valid_action(action_type, quantity)
    
    def test_adapt_multiple_steps(self, sample_ppo_agent):
        """Test adaptation with multiple gradient steps."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        
        # Test different numbers of gradient steps
        for num_steps in [1, 3, 5]:
            adapted_agent = sample_ppo_agent.adapt(
                observation, action, reward, next_observation, done, num_steps
            )
            
            assert isinstance(adapted_agent, BaseAgent)
            
            # Test action selection
            test_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
            action_type, quantity = adapted_agent.select_action(test_obs)
            assert_valid_action(action_type, quantity)
    
    def test_adapt_different_rewards(self, sample_ppo_agent):
        """Test adaptation with different reward values."""
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        rewards = [-1.0, 0.0, 1.0, 10.0, -10.0]
        
        for reward in rewards:
            adapted_agent = sample_ppo_agent.adapt(
                observation, action, reward, next_observation, done, num_gradient_steps
            )
            
            assert isinstance(adapted_agent, BaseAgent)
    
    def test_adapt_preserves_original(self, sample_ppo_agent):
        """Test that adaptation preserves original agent."""
        # Store original parameters
        original_actor_params = [p.clone() for p in sample_ppo_agent.actor.parameters()]
        
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (0, 1.0)
        reward = 1.0
        next_observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = False
        num_gradient_steps = 1
        
        # Adapt
        adapted_agent = sample_ppo_agent.adapt(
            observation, action, reward, next_observation, done, num_gradient_steps
        )
        
        # Original agent parameters should be unchanged
        for original, current in zip(original_actor_params, sample_ppo_agent.actor.parameters()):
            assert torch.allclose(original, current, atol=1e-6)

class TestPPOAgentPersistence:
    """Test suite for PPO agent model persistence."""
    
    def test_save_model(self, sample_ppo_agent, temp_model_dir):
        """Test model saving functionality."""
        model_path = os.path.join(temp_model_dir, "test_ppo_model.pth")
        
        # Save model
        sample_ppo_agent.save_model(model_path)
        
        # Check that file was created
        assert os.path.exists(model_path)
        
        # Check that file is not empty
        assert os.path.getsize(model_path) > 0
    
    def test_load_model(self, sample_ppo_agent, temp_model_dir):
        """Test model loading functionality."""
        model_path = os.path.join(temp_model_dir, "test_ppo_model.pth")
        
        # Save model first
        sample_ppo_agent.save_model(model_path)
        
        # Create new agent
        new_agent = PPOAgent(
            observation_dim=TEST_OBSERVATION_DIM,
            action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
            action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
            hidden_dim=TEST_HIDDEN_DIM
        )
        
        # Load model
        new_agent.load_model(model_path)
        
        # Test that loaded agent works
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action_type, quantity = new_agent.select_action(observation)
        assert_valid_action(action_type, quantity)
    
    def test_save_load_consistency(self, sample_ppo_agent, temp_model_dir):
        """Test save/load consistency."""
        model_path = os.path.join(temp_model_dir, "test_ppo_model.pth")
        
        # Get action from original agent
        observation = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        torch.manual_seed(42)
        original_action = sample_ppo_agent.select_action(observation)
        
        # Save and load model
        sample_ppo_agent.save_model(model_path)
        
        new_agent = PPOAgent(
            observation_dim=TEST_OBSERVATION_DIM,
            action_dim_discrete=TEST_ACTION_DIM_DISCRETE,
            action_dim_continuous=TEST_ACTION_DIM_CONTINUOUS,
            hidden_dim=TEST_HIDDEN_DIM
        )
        new_agent.load_model(model_path)
        
        # Get action from loaded agent
        torch.manual_seed(42)
        loaded_action = new_agent.select_action(observation)
        
        # Actions should be the same
        assert original_action[0] == loaded_action[0]
        assert abs(original_action[1] - loaded_action[1]) < 1e-6
    
    def test_load_nonexistent_model(self, sample_ppo_agent):
        """Test loading non-existent model."""
        nonexistent_path = "nonexistent_model.pth"
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            sample_ppo_agent.load_model(nonexistent_path)

class TestPPOAgentIntegration:
    """Test suite for PPO agent integration."""
    
    def test_integration_with_environment(self, sample_ppo_agent, sample_trading_env):
        """Test integration with trading environment."""
        obs = sample_trading_env.reset()
        
        # Agent should be able to select action for environment observation
        action_type, quantity = sample_ppo_agent.select_action(obs)
        assert_valid_action(action_type, quantity)
        
        # Environment should accept agent's action
        action = (action_type, quantity)
        next_obs, reward, done, info = sample_trading_env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    
    def test_training_loop_simulation(self, sample_ppo_agent, sample_trading_env):
        """Test simulation of training loop."""
        experiences = []
        
        obs = sample_trading_env.reset()
        
        # Collect experiences
        for _ in range(10):
            action_type, quantity = sample_ppo_agent.select_action(obs)
            action = (action_type, quantity)
            
            next_obs, reward, done, info = sample_trading_env.step(action)
            
            experiences.append((obs, action, reward, next_obs, done))
            
            obs = next_obs
            if done:
                obs = sample_trading_env.reset()
        
        # Learn from experiences
        sample_ppo_agent.learn(experiences)
        
        # Should complete without errors
        assert len(experiences) == 10
