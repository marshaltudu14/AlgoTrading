"""
Comprehensive test configuration and fixtures for the AlgoTrading system.
This file provides shared fixtures and utilities for testing all components.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Tuple
import logging

# Import all the components we need to test
from src.utils.data_loader import DataLoader
from src.backtesting.environment import TradingEnv
from src.agents.ppo_agent import PPOAgent
from src.agents.base_agent import BaseAgent
from src.training.trainer import Trainer
from src.config.config import INITIAL_CAPITAL

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test constants
TEST_OBSERVATION_DIM = 20
TEST_ACTION_DIM_DISCRETE = 5
TEST_ACTION_DIM_CONTINUOUS = 1
TEST_HIDDEN_DIM = 32
TEST_INITIAL_CAPITAL = 10000.0
TEST_LOOKBACK_WINDOW = 10
TEST_EPISODE_LENGTH = 100

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory with test data files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create test data structure
    final_dir = os.path.join(temp_dir, "final")
    raw_dir = os.path.join(temp_dir, "raw")
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    # Generate sample market data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1min')
    
    # Create multiple symbol datasets (use symbols from instruments.yaml)
    symbols = ['Bank_Nifty', 'Nifty']
    
    for symbol in symbols:
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible tests
        base_price = np.random.uniform(100, 500)
        
        # Generate price series with some trend and volatility
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 10000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Save raw data
        df.to_csv(os.path.join(raw_dir, f"{symbol}.csv"), index=False)
        
        # Create features for final data
        df_features = df.copy()
        df_features['returns'] = df_features['close'].pct_change()
        df_features['sma_10'] = df_features['close'].rolling(10).mean()
        df_features['sma_20'] = df_features['close'].rolling(20).mean()
        df_features['rsi'] = calculate_rsi(df_features['close'])
        df_features['volatility'] = df_features['returns'].rolling(20).std()
        
        # Add technical indicators
        for i in range(1, 6):  # Add some lag features
            df_features[f'close_lag_{i}'] = df_features['close'].shift(i)
            df_features[f'volume_lag_{i}'] = df_features['volume'].shift(i)
        
        # Drop NaN values
        df_features = df_features.dropna()
        
        # Save features data
        df_features.to_csv(os.path.join(final_dir, f"features_{symbol}.csv"), index=False)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

def calculate_rsi(prices, window=14):
    """Calculate RSI for test data."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='1min'),
        'open': np.random.uniform(100, 200, n_samples),
        'high': np.random.uniform(200, 250, n_samples),
        'low': np.random.uniform(50, 100, n_samples),
        'close': np.random.uniform(100, 200, n_samples),
        'volume': np.random.randint(1000, 10000, n_samples)
    }
    
    df = pd.DataFrame(data)
    # Ensure OHLC consistency
    for i in range(len(df)):
        high = max(df.loc[i, 'open'], df.loc[i, 'close']) * 1.02
        low = min(df.loc[i, 'open'], df.loc[i, 'close']) * 0.98
        df.loc[i, 'high'] = high
        df.loc[i, 'low'] = low
    
    return df

@pytest.fixture
def mock_data_loader(test_data_dir):
    """Create a DataLoader with test data."""
    final_dir = os.path.join(test_data_dir, "final")
    raw_dir = os.path.join(test_data_dir, "raw")
    return DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)

@pytest.fixture
def sample_trading_env(mock_data_loader):
    """Create a TradingEnv for testing."""
    return TradingEnv(
        data_loader=mock_data_loader,
        symbol="Bank_Nifty",
        initial_capital=TEST_INITIAL_CAPITAL,
        lookback_window=TEST_LOOKBACK_WINDOW,
        episode_length=TEST_EPISODE_LENGTH,
        use_streaming=False
    )

@pytest.fixture
def sample_ppo_agent():
    """Create a PPO agent for testing."""
    return PPOAgent(
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

# MoE agent fixture removed - only using PPO for now

@pytest.fixture
def sample_experiences():
    """Generate sample experiences for agent learning."""
    experiences = []
    for _ in range(10):
        obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        action = (np.random.randint(0, TEST_ACTION_DIM_DISCRETE), np.random.uniform(0.1, 2.0))
        reward = np.random.uniform(-1.0, 1.0)
        next_obs = np.random.rand(TEST_OBSERVATION_DIM).astype(np.float32)
        done = np.random.choice([True, False])
        experiences.append((obs, action, reward, next_obs, done))
    return experiences

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for saving models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_ray():
    """Mock Ray for testing parallel components."""
    with patch('ray.init'), \
         patch('ray.shutdown'), \
         patch('ray.tune.run'), \
         patch('ray.get'):
        yield

class MockAgent(BaseAgent):
    """Mock agent for testing purposes."""
    
    def __init__(self):
        self.action_calls = 0
        self.learn_calls = 0
        self.adapt_calls = 0
        
    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        self.action_calls += 1
        return 0, 1.0
        
    def learn(self, experiences: List[Tuple[np.ndarray, Tuple[int, float], float, np.ndarray, bool]]) -> None:
        self.learn_calls += 1
        
    def adapt(self, observation: np.ndarray, action: Tuple[int, float], reward: float, 
              next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        self.adapt_calls += 1
        return self
        
    def save_model(self, path: str) -> None:
        pass
        
    def load_model(self, path: str) -> None:
        pass

@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return MockAgent()

# Utility functions for tests
def assert_valid_action(action_type: int, quantity: float):
    """Assert that an action is valid."""
    assert isinstance(action_type, int)
    assert 0 <= action_type < TEST_ACTION_DIM_DISCRETE
    assert isinstance(quantity, float)
    assert quantity > 0

def assert_valid_observation(observation: np.ndarray):
    """Assert that an observation is valid."""
    assert isinstance(observation, np.ndarray)
    assert observation.shape[0] == TEST_OBSERVATION_DIM
    assert not np.isnan(observation).any()
    assert not np.isinf(observation).any()

def create_mock_training_config():
    """Create a mock training configuration."""
    return {
        'env_config': {
            'symbol': 'AAPL',
            'initial_capital': TEST_INITIAL_CAPITAL,
            'lookback_window': TEST_LOOKBACK_WINDOW,
            'episode_length': TEST_EPISODE_LENGTH
        },
        'training_config': {
            'algorithm': 'PPO',
            'num_workers': 2,
            'episodes': 10
        },
        'checkpoint_config': {
            'checkpoint_freq': 5,
            'evaluation_episodes': 3
        }
    }
