import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import random
import json
import warnings

# Ignore RuntimeWarning from Sharpe calculation with zero std dev
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")


# --- Configuration ---
PROCESSED_DATA_DIR = 'data/historical_processed'
STATS_FILE = 'normalization_stats.json'
LOOKBACK_WINDOW = 50
INITIAL_BALANCE = 100000  # Example initial balance
TRANSACTION_COST_PERCENT = 0.0005 # Example: 0.05% per trade (buy/sell) - adjust as needed
RISK_FREE_RATE = 0.0 # Assuming 0 risk-free rate for Sharpe calculation

# Features to use in the state representation (must match preprocess_norm_stats.py + derived)
MARKET_FEATURES = [
    'close', 'atr_14', 'rsi_14', 'MACDh_12_26_9', 'adx_14',
    'close_ema50_diff', 'close_ema200_diff' # Derived features
]
AGENT_STATE_FEATURES = ['position', 'trade_pnl_norm']
ALL_STATE_FEATURES = MARKET_FEATURES * LOOKBACK_WINDOW + AGENT_STATE_FEATURES

# Action space definition
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2
ACTION_CLOSE = 3
# --- End Configuration ---


class TradingEnv(gym.Env):
    """
    A custom Gymnasium environment for training a trading RL agent.

    Handles dynamic loading of instrument/timeframe data, state normalization,
    action execution, and Sharpe Ratio reward calculation.
    """
    metadata = {'render_modes': ['human', 'ansi', 'none'], 'render_fps': 1}

    def __init__(self, render_mode='none'):
        super().__init__()

        self.render_mode = render_mode
        self._validate_config()

        # Load normalization statistics
        try:
            with open(STATS_FILE, 'r') as f:
                self.norm_stats = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Normalization stats file not found at {STATS_FILE}. "
                                    "Run preprocess_norm_stats.py first.")
        except json.JSONDecodeError:
             raise ValueError(f"Error decoding JSON from {STATS_FILE}.")


        # List available data files
        self.available_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_processed.csv')]
        if not self.available_files:
            raise FileNotFoundError(f"No '*_processed.csv' files found in {PROCESSED_DATA_DIR}.")

        # Define action space: 0: Hold, 1: Buy, 2: Sell, 3: Close
        self.action_space = spaces.Discrete(4)

        # Define observation space: Box shape based on lookback window and features
        # Market features * lookback + agent state features
        num_market_features = len(MARKET_FEATURES)
        num_agent_features = len(AGENT_STATE_FEATURES)
        obs_shape = (num_market_features * LOOKBACK_WINDOW + num_agent_features,)
        # Using -inf to +inf because Z-score normalization doesn't guarantee bounds
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Environment state variables (will be reset)
        self.current_file = None
        self.current_data = None
        self.current_norm_stats = None
        self.current_step = 0
        self.balance = INITIAL_BALANCE
        self.position = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = 0.0
        self.trade_pnl = 0.0 # PnL of the current open trade in points/currency
        self.episode_returns = [] # List of portfolio returns per step
        self.total_steps = 0
        self.num_trades = 0

    def _validate_config(self):
        if not os.path.exists(PROCESSED_DATA_DIR):
             raise FileNotFoundError(f"Processed data directory not found: {PROCESSED_DATA_DIR}")
        if LOOKBACK_WINDOW <= 0:
            raise ValueError("LOOKBACK_WINDOW must be positive.")
        if INITIAL_BALANCE <= 0:
             raise ValueError("INITIAL_BALANCE must be positive.")
        if not (0 <= TRANSACTION_COST_PERCENT < 1):
            raise ValueError("TRANSACTION_COST_PERCENT must be between 0 and 1.")

    def _load_data(self, filename):
        """Loads data for a specific file and its normalization stats."""
        file_path = os.path.join(PROCESSED_DATA_DIR, filename)
        try:
            data = pd.read_csv(file_path)
            # Pre-calculate derived features needed for state
            data['close_ema50_diff'] = data['close'] - data['ema_50']
            data['close_ema200_diff'] = data['close'] - data['ema_200']
            # Ensure required columns exist
            required_cols = MARKET_FEATURES + ['open', 'high', 'low'] # Need OHLC for PnL calc
            missing_cols = [col for col in required_cols if col not in data.columns and col not in ['close_ema50_diff', 'close_ema200_diff']] # exclude derived
            if missing_cols:
                 raise ValueError(f"File {filename} missing required columns: {missing_cols}")

            stats = self.norm_stats.get(filename)
            if not stats:
                raise ValueError(f"Normalization stats not found for file: {filename}")

            # Verify stats exist for all needed features
            for feature in MARKET_FEATURES:
                 # Check base features used in derived features too
                 base_feature = feature.split('_')[0] # e.g., 'close' from 'close_ema50_diff'
                 if feature not in stats and base_feature not in stats:
                     # Allow derived features if base features exist in stats
