"""
Comprehensive tests for the trading environment.
Tests TradingEnv initialization, state transitions, action handling, and reward calculation.
"""

import pytest
import numpy as np
import pandas as pd
import gym
from unittest.mock import Mock, patch

from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from tests.conftest import (
    TEST_INITIAL_CAPITAL, TEST_LOOKBACK_WINDOW, TEST_EPISODE_LENGTH,
    assert_valid_observation, assert_valid_action
)

class TestTradingEnvInitialization:
    """Test suite for TradingEnv initialization."""
    
    def test_basic_initialization(self, mock_data_loader):
        """Test basic TradingEnv initialization."""
        env = TradingEnv(
            data_loader=mock_data_loader,
            symbol="Bank_Nifty",
            initial_capital=TEST_INITIAL_CAPITAL,
            lookback_window=TEST_LOOKBACK_WINDOW,
            episode_length=TEST_EPISODE_LENGTH
        )

        assert env.data_loader == mock_data_loader
        assert env.symbol == "Bank_Nifty"
        assert env.initial_capital == TEST_INITIAL_CAPITAL
        assert env.lookback_window == TEST_LOOKBACK_WINDOW
        assert env.episode_length == TEST_EPISODE_LENGTH
        assert env.current_step == 0
        assert env.done == False
    
    def test_initialization_with_streaming(self, mock_data_loader):
        """Test TradingEnv initialization with streaming enabled."""
        env = TradingEnv(
            data_loader=mock_data_loader,
            symbol="Bank_Nifty",
            initial_capital=TEST_INITIAL_CAPITAL,
            lookback_window=TEST_LOOKBACK_WINDOW,
            episode_length=TEST_EPISODE_LENGTH,
            use_streaming=True
        )
        
        assert env.use_streaming == True
    
    def test_initialization_invalid_parameters(self, mock_data_loader):
        """Test TradingEnv initialization with invalid parameters."""
        # Test negative initial capital
        with pytest.raises((ValueError, AssertionError)):
            TradingEnv(
                data_loader=mock_data_loader,
                symbol="Bank_Nifty",
                initial_capital=-1000,
                lookback_window=TEST_LOOKBACK_WINDOW,
                episode_length=TEST_EPISODE_LENGTH
            )
        
        # Test zero lookback window
        with pytest.raises((ValueError, AssertionError)):
            TradingEnv(
                data_loader=mock_data_loader,
                symbol="Bank_Nifty",
                initial_capital=TEST_INITIAL_CAPITAL,
                lookback_window=0,
                episode_length=TEST_EPISODE_LENGTH
            )
        
        # Test zero episode length
        with pytest.raises((ValueError, AssertionError)):
            TradingEnv(
                data_loader=mock_data_loader,
                symbol="Bank_Nifty",
                initial_capital=TEST_INITIAL_CAPITAL,
                lookback_window=TEST_LOOKBACK_WINDOW,
                episode_length=0
            )
    
    def test_action_space_definition(self, sample_trading_env):
        """Test that action space is properly defined."""
        assert hasattr(sample_trading_env, 'action_space')
        
        # Action space should be a Box or Discrete space
        assert isinstance(sample_trading_env.action_space, (gym.spaces.Box, gym.spaces.Discrete, gym.spaces.MultiDiscrete))
    
    def test_observation_space_definition(self, sample_trading_env):
        """Test that observation space is properly defined."""
        assert hasattr(sample_trading_env, 'observation_space')
        
        # Observation space should be a Box space
        assert isinstance(sample_trading_env.observation_space, gym.spaces.Box)
        
        # Should have reasonable dimensions
        assert len(sample_trading_env.observation_space.shape) == 1
        assert sample_trading_env.observation_space.shape[0] > 0

class TestTradingEnvReset:
    """Test suite for TradingEnv reset functionality."""
    
    def test_reset_basic(self, sample_trading_env):
        """Test basic reset functionality."""
        observation = sample_trading_env.reset()
        
        # Check observation validity
        assert_valid_observation(observation)
        
        # Check environment state
        assert sample_trading_env.current_step == 0
        assert sample_trading_env.done == False
        
        # Check that engine is initialized
        assert hasattr(sample_trading_env, 'engine')
        assert sample_trading_env.engine is not None
    
    def test_reset_multiple_times(self, sample_trading_env):
        """Test multiple resets."""
        observations = []
        
        for _ in range(3):
            obs = sample_trading_env.reset()
            observations.append(obs)
            assert_valid_observation(obs)
            assert sample_trading_env.current_step == 0
            assert sample_trading_env.done == False
        
        # Observations might be different due to random episode start points
        # but should all be valid
        for obs in observations:
            assert_valid_observation(obs)
    
    def test_reset_after_episode_completion(self, sample_trading_env):
        """Test reset after completing an episode."""
        # First reset
        obs1 = sample_trading_env.reset()
        
        # Run some steps
        for _ in range(10):
            action = (0, 1.0)  # HOLD action
            obs, reward, done, info = sample_trading_env.step(action)
            if done:
                break
        
        # Reset again
        obs2 = sample_trading_env.reset()
        
        assert_valid_observation(obs1)
        assert_valid_observation(obs2)
        assert sample_trading_env.current_step == 0
        assert sample_trading_env.done == False

class TestTradingEnvStep:
    """Test suite for TradingEnv step functionality."""
    
    def test_step_basic(self, sample_trading_env):
        """Test basic step functionality."""
        sample_trading_env.reset()
        
        action = (0, 1.0)  # HOLD action with quantity 1.0
        observation, reward, done, info = sample_trading_env.step(action)
        
        # Check return values
        assert_valid_observation(observation)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        
        # Check environment state progression
        assert sample_trading_env.current_step == 1
    
    def test_step_all_actions(self, sample_trading_env):
        """Test all possible actions."""
        sample_trading_env.reset()
        
        # Test different action types (0=BUY, 1=SELL, 2=CLOSE_LONG, 3=CLOSE_SHORT, 4=HOLD)
        actions = [
            (0, 1.0),  # BUY
            (1, 1.0),  # SELL
            (2, 1.0),  # CLOSE_LONG
            (3, 1.0),  # CLOSE_SHORT
            (4, 1.0),  # HOLD
        ]
        
        for action in actions:
            sample_trading_env.reset()
            obs, reward, done, info = sample_trading_env.step(action)
            
            assert_valid_observation(obs)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
    
    def test_step_invalid_actions(self, sample_trading_env):
        """Test handling of invalid actions."""
        sample_trading_env.reset()
        
        # Test invalid action type
        invalid_actions = [
            (-1, 1.0),  # Negative action
            (10, 1.0),  # Action out of range
            (0, -1.0),  # Negative quantity
            (0, 0.0),   # Zero quantity
        ]
        
        for action in invalid_actions:
            try:
                obs, reward, done, info = sample_trading_env.step(action)
                # If no exception, check that environment handled it gracefully
                assert_valid_observation(obs)
            except (ValueError, AssertionError):
                # Expected for invalid actions
                pass
    
    def test_step_sequence(self, sample_trading_env):
        """Test a sequence of steps."""
        sample_trading_env.reset()
        
        total_reward = 0
        step_count = 0
        
        for i in range(20):
            action = (4, 1.0)  # HOLD action
            obs, reward, done, info = sample_trading_env.step(action)
            
            assert_valid_observation(obs)
            assert isinstance(reward, (int, float))
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        assert step_count > 0
        assert isinstance(total_reward, (int, float))
    
    def test_episode_termination(self, sample_trading_env):
        """Test episode termination conditions."""
        sample_trading_env.reset()
        
        done = False
        step_count = 0
        
        while not done and step_count < sample_trading_env.episode_length + 10:
            action = (4, 1.0)  # HOLD action
            obs, reward, done, info = sample_trading_env.step(action)
            step_count += 1
        
        # Episode should terminate within expected length
        assert done or step_count >= sample_trading_env.episode_length

class TestTradingEnvRewards:
    """Test suite for TradingEnv reward calculation."""
    
    def test_reward_calculation_hold(self, sample_trading_env):
        """Test reward calculation for HOLD actions."""
        sample_trading_env.reset()
        
        # Take several HOLD actions
        rewards = []
        for _ in range(5):
            action = (4, 1.0)  # HOLD
            obs, reward, done, info = sample_trading_env.step(action)
            rewards.append(reward)
            if done:
                break
        
        # Rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards)
        
        # For HOLD actions, rewards might be small or zero
        # but should not be extreme
        assert all(abs(r) < 1000 for r in rewards)
    
    def test_reward_calculation_trading(self, sample_trading_env):
        """Test reward calculation for trading actions."""
        sample_trading_env.reset()
        
        # Execute a buy-sell sequence
        actions = [
            (0, 1.0),  # BUY
            (4, 1.0),  # HOLD
            (4, 1.0),  # HOLD
            (1, 1.0),  # SELL
        ]
        
        rewards = []
        for action in actions:
            obs, reward, done, info = sample_trading_env.step(action)
            rewards.append(reward)
            if done:
                break
        
        # All rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards)
        
        # Trading should generate some reward (positive or negative)
        total_reward = sum(rewards)
        assert isinstance(total_reward, (int, float))
    
    def test_reward_consistency(self, sample_trading_env):
        """Test reward calculation consistency."""
        # Run the same sequence multiple times
        action_sequence = [(0, 1.0), (4, 1.0), (1, 1.0)]
        
        reward_sequences = []
        for _ in range(3):
            sample_trading_env.reset()
            rewards = []
            
            for action in action_sequence:
                obs, reward, done, info = sample_trading_env.step(action)
                rewards.append(reward)
                if done:
                    break
            
            reward_sequences.append(rewards)
        
        # Rewards should be consistent for the same actions
        # (allowing for small numerical differences)
        if len(reward_sequences) > 1:
            for i in range(len(reward_sequences[0])):
                if i < len(reward_sequences[1]):
                    # Allow for small differences due to floating point precision
                    diff = abs(reward_sequences[0][i] - reward_sequences[1][i])
                    assert diff < 1e-6 or diff / max(abs(reward_sequences[0][i]), abs(reward_sequences[1][i])) < 1e-3

class TestTradingEnvState:
    """Test suite for TradingEnv state management."""
    
    def test_state_observation_consistency(self, sample_trading_env):
        """Test consistency of state observations."""
        obs1 = sample_trading_env.reset()
        
        # Take a step and check observation
        action = (4, 1.0)  # HOLD
        obs2, reward, done, info = sample_trading_env.step(action)
        
        # Observations should have same shape
        assert obs1.shape == obs2.shape
        
        # Both should be valid
        assert_valid_observation(obs1)
        assert_valid_observation(obs2)
    
    def test_state_information_content(self, sample_trading_env):
        """Test that state contains meaningful information."""
        obs = sample_trading_env.reset()
        
        # Observation should contain market information
        assert len(obs) > 0
        
        # Should not be all zeros or all the same value
        assert not np.allclose(obs, 0)
        assert not np.allclose(obs, obs[0])
        
        # Should contain reasonable values (not extreme)
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
        assert np.all(np.abs(obs) < 1e6)  # Reasonable magnitude
    
    def test_state_evolution(self, sample_trading_env):
        """Test that state evolves over time."""
        sample_trading_env.reset()
        
        observations = []
        for _ in range(5):
            action = (4, 1.0)  # HOLD
            obs, reward, done, info = sample_trading_env.step(action)
            observations.append(obs.copy())
            if done:
                break
        
        # States should evolve (not all identical)
        if len(observations) > 1:
            differences = []
            for i in range(1, len(observations)):
                diff = np.linalg.norm(observations[i] - observations[i-1])
                differences.append(diff)
            
            # At least some states should be different
            assert any(diff > 1e-6 for diff in differences)

class TestTradingEnvIntegration:
    """Test suite for TradingEnv integration with other components."""
    
    def test_integration_with_data_loader(self, mock_data_loader):
        """Test integration with DataLoader."""
        env = TradingEnv(
            data_loader=mock_data_loader,
            symbol="Bank_Nifty",
            initial_capital=TEST_INITIAL_CAPITAL,
            lookback_window=TEST_LOOKBACK_WINDOW,
            episode_length=TEST_EPISODE_LENGTH
        )
        
        # Should be able to reset and get valid data
        obs = env.reset()
        assert_valid_observation(obs)
        
        # Should be able to step
        action = (4, 1.0)
        obs, reward, done, info = env.step(action)
        assert_valid_observation(obs)
    
    def test_gym_interface_compliance(self, sample_trading_env):
        """Test compliance with OpenAI Gym interface."""
        # Should have required methods
        assert hasattr(sample_trading_env, 'reset')
        assert hasattr(sample_trading_env, 'step')
        assert hasattr(sample_trading_env, 'action_space')
        assert hasattr(sample_trading_env, 'observation_space')
        
        # Should follow Gym interface
        obs = sample_trading_env.reset()
        assert isinstance(obs, np.ndarray)
        
        action = (4, 1.0)
        step_result = sample_trading_env.step(action)
        assert len(step_result) == 4  # obs, reward, done, info
        
        obs, reward, done, info = step_result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_error_handling(self, sample_trading_env):
        """Test error handling in various scenarios."""
        # Test step before reset
        try:
            action = (4, 1.0)
            sample_trading_env.step(action)
        except Exception:
            # This might raise an exception, which is acceptable
            pass
        
        # Test with invalid data
        sample_trading_env.reset()
        
        # Should handle edge cases gracefully
        action = (4, 1.0)
        obs, reward, done, info = sample_trading_env.step(action)
        
        # Should return valid values even in edge cases
        assert_valid_observation(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
