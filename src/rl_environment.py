import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import random
import json
import warnings
import time # Import time for potential delays if needed

# Ignore RuntimeWarning from Sharpe calculation with zero std dev
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom <= 0 for slice")

# Import config variables
from src.config import (
    RL_PROCESSED_DATA_DIR, RL_STATS_FILE, RL_LOOKBACK_WINDOW, RL_INITIAL_BALANCE,
    RL_TRANSACTION_COST_PERCENT, RL_RISK_FREE_RATE, RL_COST_PENALTY_MULTIPLIER,
    DYNAMIC_CONFIG_FILE, INDEX_MAPPING # Need dynamic config path
)

# --- Configuration ---
# Constants are now imported from src.config


# Features to use in the state representation (must match features used for normalization stats)
MARKET_FEATURES = [
    'close', 'atr_14', 'rsi_14', 'MACDh_12_26_9', 'adx_14',
    'close_ema50_diff', 'close_ema200_diff' # Derived features
]
AGENT_STATE_FEATURES = ['position', 'trade_pnl_norm']
# ALL_STATE_FEATURES calculation removed as it's done within __init__ using config

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

    def __init__(self, render_mode='none', data_df: pd.DataFrame | None = None, norm_stats_override: dict | None = None):
        """
        Initializes the Trading Environment.

        Args:
            render_mode (str): Rendering mode ('human', 'ansi', 'none').
            data_df (pd.DataFrame, optional): If provided, use this DataFrame directly
                                              instead of loading from files. Defaults to None.
            norm_stats_override (dict, optional): If provided alongside data_df, use these
                                                  normalization stats. Defaults to None.
        """
        super().__init__()

        # Store provided data and stats if any
        self.data_df = data_df
        self.norm_stats_override = norm_stats_override

        # Store config values
        self.processed_data_dir = RL_PROCESSED_DATA_DIR
        self.stats_file = RL_STATS_FILE
        self.lookback_window = RL_LOOKBACK_WINDOW
        self.initial_balance = RL_INITIAL_BALANCE
        self.transaction_cost_percent = RL_TRANSACTION_COST_PERCENT
        self.risk_free_rate = RL_RISK_FREE_RATE
        self.cost_penalty_multiplier = RL_COST_PENALTY_MULTIPLIER
        self.dynamic_config_path = DYNAMIC_CONFIG_FILE # Store path

        self.render_mode = render_mode
        self._validate_config() # Validate loaded config values

        # Load dynamic config directly in each instance
        try:
            with open(self.dynamic_config_path, 'r') as f:
                self.dynamic_config = json.load(f)
            if not self.dynamic_config or 'instruments' not in self.dynamic_config:
                 raise ValueError("Dynamic config is empty or missing 'instruments' key.")
            # print(f"Dynamic config loaded successfully in TradingEnv instance from {self.dynamic_config_path}") # Optional debug print
        except FileNotFoundError:
             raise FileNotFoundError(f"Dynamic config file not found at {self.dynamic_config_path}. Run run_data_setup.py.")
        except Exception as e:
             raise RuntimeError(f"Failed to load or parse dynamic config {self.dynamic_config_path} in TradingEnv: {e}")

        # Load normalization statistics from file (needed for file-based loading)
        # If norm_stats_override is provided, it will be used in reset for the provided data_df
        try:
            with open(self.stats_file, 'r') as f:
                self.norm_stats_from_file = json.load(f)
        except FileNotFoundError:
             # Only raise error if we are NOT providing data_df directly
             if self.data_df is None:
                 raise FileNotFoundError(f"Normalization stats file not found at {self.stats_file} and no override provided. "
                                         "Run preprocess_norm_stats.py first.")
             else:
                 print(f"Warning: Normalization stats file {self.stats_file} not found, but using provided override.")
                 self.norm_stats_from_file = None # Set to None if file not found but override exists
        except json.JSONDecodeError:
             raise ValueError(f"Error decoding JSON from {self.stats_file}.")

        # List available data files (only needed if data_df is not provided)
        if self.data_df is None:
            self.available_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith('_processed.csv')]
            if not self.available_files:
                raise FileNotFoundError(f"No '*_processed.csv' files found in {self.processed_data_dir} and no data_df provided.")
        else:
             self.available_files = [] # Not needed if data_df is provided

        # Define action space: 0: Hold, 1: Buy, 2: Sell, 3: Close
        self.action_space = spaces.Discrete(4)

        # Define observation space: Box shape based on lookback window and features
        num_market_features = len(MARKET_FEATURES)
        num_agent_features = len(AGENT_STATE_FEATURES)
        obs_shape = (num_market_features * self.lookback_window + num_agent_features,)
        # Using -inf to +inf because Z-score normalization doesn't guarantee bounds
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        # Environment state variables (will be reset)
        self.current_file = None
        self.current_data = None
        self.current_norm_stats = None
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: Short, 0: Flat, 1: Long
        self.entry_price = 0.0
        self.trade_pnl_points = 0.0 # PnL of the current open trade in points (before lot size)
        self.lot_size = 1 # Default, will be updated in reset
        self.episode_portfolio_values = [] # List of portfolio values per step
        self.total_steps = 0
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_cost = 0.0

    def _validate_config(self):
        """Validates the configuration parameters loaded during init."""
        if not os.path.exists(self.processed_data_dir):
             raise FileNotFoundError(f"Processed data directory not found: {self.processed_data_dir}")
        if self.lookback_window <= 0:
            raise ValueError("RL_LOOKBACK_WINDOW must be positive.")
        if self.initial_balance <= 0:
             raise ValueError("RL_INITIAL_BALANCE must be positive.")
        if not (0 <= self.transaction_cost_percent < 1):
            raise ValueError("RL_TRANSACTION_COST_PERCENT must be between 0 and 1.")
        # Add more checks if needed (e.g., for risk_free_rate, cost_penalty)

    # Removed _load_dynamic_config_data method as loading is now done in __init__

    def _get_instrument_details(self, filename):
         """Extracts instrument details (lot size) from the loaded dynamic config based on filename."""
         # Extract base instrument name and timeframe from filename (e.g., "Nifty_5_processed.csv")
         try:
             parts = filename.replace("_processed.csv", "").split('_')
             if len(parts) < 2:
                 raise ValueError("Filename format incorrect")
             timeframe = parts[-1]
             instrument_base_name = "_".join(parts[:-1]) # Handle names like "Bank_Nifty"
             dynamic_name_pattern = f"{instrument_base_name.upper()}_{timeframe}M"

             for instrument in self.dynamic_config.get("instruments", []):
                 if instrument.get("name") == dynamic_name_pattern:
                     lot_size = instrument.get("lot_size")
                     if lot_size is None:
                          raise ValueError(f"Lot size not found for {dynamic_name_pattern}")
                     return {"lot_size": int(lot_size)} # Ensure integer

             # Fallback: Try matching with INDEX_MAPPING if dynamic config fails (less ideal)
             print(f"Warning: Could not find {dynamic_name_pattern} in dynamic_config. Trying INDEX_MAPPING fallback.")
             if instrument_base_name in INDEX_MAPPING:
                  lot_size = INDEX_MAPPING[instrument_base_name].get("quantity")
                  if lot_size:
                       return {"lot_size": int(lot_size)}

             raise ValueError(f"Instrument details not found for pattern {dynamic_name_pattern} or base {instrument_base_name}")

         except Exception as e:
             raise ValueError(f"Error parsing filename '{filename}' or finding instrument details: {e}")


    def _load_data(self, filename):
        """Loads data for a specific file and its normalization stats."""
        file_path = os.path.join(self.processed_data_dir, filename)
        try:
            data = pd.read_csv(file_path)
            # --- Data Cleaning ---
            # Ensure essential columns are numeric, coercing errors
            for col in ['open', 'high', 'low', 'close', 'atr_14', 'rsi_14', 'MACDh_12_26_9', 'adx_14', 'ema_50', 'ema_200']:
                 if col in data.columns:
                     data[col] = pd.to_numeric(data[col], errors='coerce')

            # Forward fill NaNs that might result from coercion or were already present
            # Limit ffill to avoid excessive propagation at the start
            data.ffill(limit=self.lookback_window, inplace=True)
            # Drop any remaining rows with NaNs in essential columns, especially at the beginning
            essential_cols = ['open', 'high', 'low', 'close', 'atr_14'] # ATR needed for PnL norm
            data.dropna(subset=essential_cols, inplace=True)
            data.reset_index(drop=True, inplace=True) # Reset index after dropping rows

            if len(data) < self.lookback_window + 1: # Need at least one step after lookback
                 raise ValueError(f"File {filename} has insufficient data after cleaning ({len(data)} rows). Needs > {self.lookback_window}.")

            # --- Feature Engineering ---
            data['close_ema50_diff'] = data['close'] - data['ema_50']
            data['close_ema200_diff'] = data['close'] - data['ema_200']

            # Ensure required columns exist after engineering
            required_cols = MARKET_FEATURES + ['open', 'high', 'low'] # Need OHLC for PnL calc
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                 raise ValueError(f"File {filename} missing required columns after processing: {missing_cols}")

            # Use self.norm_stats_from_file which was loaded in __init__
            stats = self.norm_stats_from_file.get(filename)
            if not stats:
                raise ValueError(f"Normalization stats not found for file: {filename} in {self.stats_file}")

            # Verify stats exist for all needed features
            for feature in MARKET_FEATURES:
                 # Check if feature exists directly or if base feature exists (for derived like close_ema_diff)
                 base_feature = feature.split('_')[0] # e.g., 'close' from 'close_ema50_diff'
                 feature_present = feature in stats
                 base_present = base_feature in stats
                 derived_present = feature in ['close_ema50_diff', 'close_ema200_diff'] and 'close' in stats # Check base 'close' for derived

                 if not feature_present and not derived_present:
                     # If the feature itself isn't there AND it's not a derived feature with its base present
                     raise ValueError(f"Normalization stats missing for feature '{feature}' in file: {filename}")

            return data, stats
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading data or stats for {filename}: {e}")

    def _normalize_feature(self, feature_data, feature_name, stats):
        """Applies Z-score normalization using pre-calculated stats."""
        if feature_name not in stats:
             if feature_name == 'close_ema50_diff' and 'close' in stats:
                 stats_to_use = stats['close']
             elif feature_name == 'close_ema200_diff' and 'close' in stats:
                 stats_to_use = stats['close']
             else:
                 print(f"Warning: Stats missing for {feature_name}. Returning zeros.")
                 return np.zeros_like(feature_data)
        else:
            stats_to_use = stats[feature_name]

        mean = stats_to_use['mean']
        std = stats_to_use['std']
        std = max(std, 1e-8) # Prevent division by zero
        # Ensure feature_data is numeric before normalization, replace NaNs resulting from coercion with the mean
        feature_data_numeric = pd.to_numeric(feature_data, errors='coerce')
        feature_data_filled = np.nan_to_num(feature_data_numeric, nan=mean)
        return (feature_data_filled - mean) / std

    def _normalize_trade_pnl(self, pnl_points):
        """Normalizes the current trade PnL using ATR."""
        if self.current_step < len(self.current_data):
            current_atr = self.current_data.loc[self.current_step, 'atr_14']
        else: # Handle edge case if step somehow goes beyond data length
            current_atr = self.current_data['atr_14'].iloc[-1]

        if pd.isna(current_atr) or current_atr < 1e-8:
            current_atr = 1.0 # Avoid division by zero, use a default scaling

        norm_pnl = pnl_points / current_atr
        return np.clip(norm_pnl, -5.0, 5.0) # Clip to range [-5, 5] ATRs


    def _get_observation(self):
        """Constructs the normalized state vector for the current step."""
        # Ensure current_step is valid
        if self.current_step < self.lookback_window -1:
             # Should not happen if reset logic is correct, but handle defensively
             print(f"Warning: Current step {self.current_step} is less than lookback window {self.lookback_window}. This might indicate an issue.")
             # Return a zero observation or handle appropriately
             obs_shape = self.observation_space.shape
             return np.zeros(obs_shape, dtype=np.float32)


        start_idx = self.current_step - self.lookback_window + 1
        # Cap end_idx to prevent index out of bounds at the very end of an episode
        end_idx = min(self.current_step + 1, self.total_steps) # Include current step, but don't exceed total steps

        # Get window data
        window_data = self.current_data.iloc[start_idx:end_idx]

        # If at the very end, the window might be short, handle this
        actual_window_len = len(window_data)

        # Normalize market features
        norm_market_features = []
        for feature in MARKET_FEATURES:
            # Ensure the feature exists in the dataframe for the window
            if feature not in window_data.columns:
                 raise ValueError(f"Feature '{feature}' not found in window_data at step {self.current_step}.")

            feature_series = window_data[feature].values

            # Pad if window is shorter than lookback_window (can happen at the end)
            if actual_window_len < self.lookback_window:
                 padding_len = self.lookback_window - actual_window_len
                 # Pad with zeros (or another appropriate value like the first value) at the beginning
                 padding = np.zeros(padding_len)
                 feature_series = np.concatenate((padding, feature_series))
            elif len(feature_series) != self.lookback_window: # Should not happen if start/end logic is correct otherwise
                 raise ValueError(f"Window data length mismatch for {feature}. Expected {self.lookback_window}, got {len(feature_series)} at step {self.current_step}.")


            norm_feature = self._normalize_feature(feature_series, feature, self.current_norm_stats)
            # Ensure norm_feature is numpy array of correct length
            if not isinstance(norm_feature, np.ndarray) or len(norm_feature) != self.lookback_window:
                 # Attempt conversion or handle error
                 try:
                     norm_feature = np.array(norm_feature, dtype=np.float32)
                     if len(norm_feature) != self.lookback_window:
                         raise ValueError("Length mismatch after conversion.")
                 except Exception as e:
                     raise TypeError(f"Normalization for {feature} did not return a valid numpy array of length {self.lookback_window}. Error: {e}")

            norm_market_features.append(norm_feature)

        # Flatten market features
        try:
            flat_market_features = np.concatenate(norm_market_features)
        except ValueError as e:
             print(f"Error concatenating normalized features at step {self.current_step}: {e}")
             # Print shapes for debugging
             for i, arr in enumerate(norm_market_features):
                 print(f"Shape of norm_market_features[{i}] ({MARKET_FEATURES[i]}): {arr.shape if isinstance(arr, np.ndarray) else type(arr)}")
             raise

        # Normalize current trade PnL
        norm_trade_pnl = self._normalize_trade_pnl(self.trade_pnl_points)

        # Agent state features
        agent_features = np.array([self.position, norm_trade_pnl], dtype=np.float32)

        # Combine and ensure correct shape and type
        observation = np.concatenate((flat_market_features, agent_features)).astype(np.float32)

        # Final check for shape
        expected_len = len(MARKET_FEATURES) * self.lookback_window + len(AGENT_STATE_FEATURES)
        if len(observation) != expected_len:
             raise ValueError(f"Final observation length mismatch. Expected {expected_len}, got {len(observation)} at step {self.current_step}.")

        # Check for NaNs/Infs
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"Warning: NaN or Inf detected in observation at step {self.current_step}. Replacing with 0.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)


        return observation

    def _calculate_sharpe(self):
        """Calculates the annualized Sharpe ratio for the episode."""
        # Calculate step returns from portfolio values
        if len(self.episode_portfolio_values) < 2:
            return 0.0
        portfolio_values_arr = np.array(self.episode_portfolio_values)
        step_returns = np.diff(portfolio_values_arr) / portfolio_values_arr[:-1]
        # Handle potential division by zero if portfolio value was zero
        step_returns = np.nan_to_num(step_returns, nan=0.0, posinf=0.0, neginf=0.0)


        if len(step_returns) < 2:
            return 0.0  # Not enough returns data

        mean_return = np.mean(step_returns)
        std_return = np.std(step_returns)

        if std_return < 1e-9: # Use small threshold instead of exact zero
             # If std is effectively zero, return 0 Sharpe
             return 0.0
        else:
            # Annualization: Needs careful thought based on step frequency
            # If steps are 5-min, 252 * (trading hours * 60 / 5) steps per year?
            # Let's skip annualization for now, calculate raw Sharpe per step.
            # annualization_factor = 1.0 # No annualization
            # sharpe_ratio = (mean_return - self.risk_free_rate) / std_return * np.sqrt(annualization_factor)
            sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
            return sharpe_ratio if np.isfinite(sharpe_ratio) else 0.0


    def step(self, action):
        """Executes one time step within the environment."""
        terminated = False
        truncated = False # Use truncated for time limit, terminated for goal/fail state
        reward = 0.0
        info = {'trades': 0, 'cost': 0.0}

        # Ensure current step is valid
        if self.current_step >= self.total_steps:
             print(f"Warning: Step called at or beyond total_steps ({self.current_step}/{self.total_steps}). Truncating.")
             truncated = True
             observation = self._get_observation() # Get last valid observation
             return observation, 0.0, terminated, truncated, info


        current_price = self.current_data.loc[self.current_step, 'close']
        if pd.isna(current_price):
             # If price is NaN, we can't proceed. End the episode.
             print(f"Warning: NaN price encountered at step {self.current_step}. Terminating episode.")
             terminated = True
             reward = -1.0 # Penalize for bad data state
             observation = self._get_observation() # Get last valid observation
             # Ensure portfolio value doesn't change due to NaN price
             self.episode_portfolio_values.append(self.episode_portfolio_values[-1] if self.episode_portfolio_values else self.initial_balance)
             return observation, reward, terminated, truncated, info


        cost = 0.0
        trade_executed = False
        trade_profit_points = 0.0 # Profit of a closed trade in points
        trade_profit_value = 0.0 # Profit of a closed trade in currency (points * lot_size)

        # --- Action Execution ---
        if action == ACTION_BUY:
            if self.position == 0: # Open long from flat
                self.position = 1
                self.entry_price = current_price
                self.trade_pnl_points = 0.0
                cost = self.entry_price * self.lot_size * self.transaction_cost_percent
                self.num_trades += 1
                trade_executed = True
            elif self.position == -1: # Close short and open long (reverse)
                trade_profit_points = self.entry_price - current_price # Profit from short trade (points)
                trade_profit_value = trade_profit_points * self.lot_size
                self.balance += trade_profit_value
                cost = self.entry_price * self.lot_size * self.transaction_cost_percent # Cost for closing short
                self.position = 0
                self.trade_pnl_points = 0.0
                if trade_profit_value > 0: self.winning_trades += 1
                else: self.losing_trades += 1
                # Now enter long
                self.position = 1
                self.entry_price = current_price
                cost += current_price * self.lot_size * self.transaction_cost_percent # Cost for opening long
                self.num_trades += 2 # Closed one, opened one
                trade_executed = True

        elif action == ACTION_SELL:
            if self.position == 0: # Open short from flat
                self.position = -1
                self.entry_price = current_price
                self.trade_pnl_points = 0.0
                cost = self.entry_price * self.lot_size * self.transaction_cost_percent
                self.num_trades += 1
                trade_executed = True
            elif self.position == 1: # Close long and open short (reverse)
                trade_profit_points = current_price - self.entry_price # Profit from long trade (points)
                trade_profit_value = trade_profit_points * self.lot_size
                self.balance += trade_profit_value
                cost = self.entry_price * self.lot_size * self.transaction_cost_percent # Cost for closing long
                self.position = 0
                self.trade_pnl_points = 0.0
                if trade_profit_value > 0: self.winning_trades += 1
                else: self.losing_trades += 1
                # Now enter short
                self.position = -1
                self.entry_price = current_price
                cost += current_price * self.lot_size * self.transaction_cost_percent # Cost for opening short
                self.num_trades += 2
                trade_executed = True

        elif action == ACTION_CLOSE:
            if self.position == 1: # Close long
                trade_profit_points = current_price - self.entry_price
                trade_profit_value = trade_profit_points * self.lot_size
                self.balance += trade_profit_value
                cost = current_price * self.lot_size * self.transaction_cost_percent # Cost on closing value
                self.position = 0
                self.trade_pnl_points = 0.0
                self.num_trades += 1
                trade_executed = True
                if trade_profit_value > 0: self.winning_trades += 1
                else: self.losing_trades += 1
            elif self.position == -1: # Close short
                trade_profit_points = self.entry_price - current_price
                trade_profit_value = trade_profit_points * self.lot_size
                self.balance += trade_profit_value
                cost = current_price * self.lot_size * self.transaction_cost_percent # Cost on closing value
                self.position = 0
                self.trade_pnl_points = 0.0
                self.num_trades += 1
                trade_executed = True
                if trade_profit_value > 0: self.winning_trades += 1
                else: self.losing_trades += 1

        # --- Update State & PnL ---
        self.balance -= cost
        self.total_cost += cost
        info['cost'] = cost
        if trade_executed:
             # Count actual trades based on cost > 0 (opening/closing)
             info['trades'] = 1 if cost > 0 else 0 # Simplified: counts actions involving cost

        # Update PnL points for the current open trade
        if self.position == 1:
            self.trade_pnl_points = current_price - self.entry_price
        elif self.position == -1:
            self.trade_pnl_points = self.entry_price - current_price
        else: # Flat
            self.trade_pnl_points = 0.0

        # Calculate current portfolio value (Balance + Unrealized PnL * Lot Size)
        unrealized_pnl_value = self.trade_pnl_points * self.lot_size * self.position # Position handles sign
        current_portfolio_value = self.balance + unrealized_pnl_value
        self.episode_portfolio_values.append(current_portfolio_value)

        # --- Check Termination/Truncation ---
        self.current_step += 1
        if self.current_step >= self.total_steps:
            truncated = True # Reached end of data

        # Capital Drawdown Check
        if current_portfolio_value < self.initial_balance * 0.25:
            print(f"Terminating episode due to capital drawdown below 25% ({current_portfolio_value:.2f} < {self.initial_balance * 0.25:.2f})")
            terminated = True
            reward = -10.0 # Assign large penalty

        # --- Reward Calculation (Episode End Only) ---
        if terminated or truncated:
            sharpe_ratio = self._calculate_sharpe()
            # Penalize based on total costs relative to initial balance? Or per trade?
            # Let's use a penalty proportional to the number of trades
            cost_penalty = self.num_trades * self.cost_penalty_multiplier
            reward = sharpe_ratio - cost_penalty
            # Ensure reward is finite
            reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=-1.0) # Penalize non-finite Sharpe
            info['episode_sharpe'] = sharpe_ratio
            info['total_trades'] = self.num_trades # Renamed for clarity
            info['episode_cost_penalty'] = cost_penalty
            info['final_capital'] = current_portfolio_value # Renamed for clarity
            info['total_cost'] = self.total_cost
            info['winning_trades'] = self.winning_trades
            info['losing_trades'] = self.losing_trades
            total_closed_trades = self.winning_trades + self.losing_trades
            info['win_rate'] = self.winning_trades / total_closed_trades if total_closed_trades > 0 else 0.0

        # --- Get Next Observation ---
        # Need to get observation *after* potentially incrementing step
        observation = self._get_observation()

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        super().reset(seed=seed)

        if self.data_df is not None:
            # --- Use provided DataFrame ---
            print("Resetting environment with provided DataFrame.")
            if self.norm_stats_override is None:
                raise ValueError("If data_df is provided, norm_stats_override must also be provided.")

            # Perform cleaning and feature engineering similar to _load_data
            # Assume data_df is already processed by FullFeaturePipeline before passing
            self.current_data = self.data_df.copy()

            # --- Data Validation/Cleaning (on the provided df) ---
            essential_cols = ['open', 'high', 'low', 'close', 'atr_14'] # Add others if needed by logic/norm
            missing_essentials = [col for col in essential_cols if col not in self.current_data.columns]
            if missing_essentials:
                 raise ValueError(f"Provided data_df is missing essential columns: {missing_essentials}")

            # Ensure numeric types and handle potential NaNs introduced before passing
            for col in self.current_data.columns:
                 # Only attempt conversion if not already numeric-like
                 if not pd.api.types.is_numeric_dtype(self.current_data[col]):
                     self.current_data[col] = pd.to_numeric(self.current_data[col], errors='coerce')

            # Drop rows with NaNs in essential columns (should ideally be clean already)
            initial_len = len(self.current_data)
            self.current_data.dropna(subset=essential_cols, inplace=True)
            if len(self.current_data) < initial_len:
                 print(f"Warning: Dropped {initial_len - len(self.current_data)} rows with NaNs from provided data_df.")

            self.current_data.reset_index(drop=True, inplace=True)

            if len(self.current_data) < self.lookback_window + 1:
                 raise ValueError(f"Provided data_df has insufficient data after cleaning ({len(self.current_data)} rows). Needs > {self.lookback_window}.")

            # --- Feature Engineering (ensure derived features exist) ---
            # Assuming pipeline already added these, but double-check/add if missing
            if 'close_ema50_diff' not in self.current_data.columns and 'close' in self.current_data.columns and 'ema_50' in self.current_data.columns:
                 self.current_data['close_ema50_diff'] = self.current_data['close'] - self.current_data['ema_50']
            if 'close_ema200_diff' not in self.current_data.columns and 'close' in self.current_data.columns and 'ema_200' in self.current_data.columns:
                 self.current_data['close_ema200_diff'] = self.current_data['close'] - self.current_data['ema_200']

            # Verify required market features exist
            missing_market_features = [f for f in MARKET_FEATURES if f not in self.current_data.columns]
            if missing_market_features:
                 raise ValueError(f"Provided data_df is missing required market features after processing: {missing_market_features}")

            self.current_norm_stats = self.norm_stats_override
            self.current_file = "provided_data" # Indicate data was provided
            # Try to infer lot size from provided stats dict if possible, or default
            # This assumes norm_stats_override might contain metadata, which isn't ideal
            # A better approach would be to pass lot_size explicitly with data_df
            self.lot_size = self.norm_stats_override.get("_metadata", {}).get("lot_size", 1)
            if self.lot_size == 1:
                 print("Warning: Using default lot size 1 for provided data. Pass lot size via norm_stats_override['_metadata'] if needed.")


        else:
            # --- Load data from random file ---
            print("Resetting environment by loading random file.")
            if not self.available_files:
                 raise RuntimeError("No data files available and no data_df provided.")
            if self.norm_stats_from_file is None:
                 raise RuntimeError(f"Stats file {self.stats_file} failed to load, cannot proceed with file-based reset.")

            # Select a random file
            self.current_file = random.choice(self.available_files)
            try:
                # Load data and stats from file
                self.current_data, self.current_norm_stats = self._load_data(self.current_file)
                # Get instrument details (lot size)
                instrument_details = self._get_instrument_details(self.current_file)
                self.lot_size = instrument_details["lot_size"]
                print(f"Loaded file: {self.current_file}, Lot Size: {self.lot_size}")
            except Exception as e:
                # If loading fails, try another file? Or raise error?
                # Let's raise for now, indicates a data/stats issue.
                raise RuntimeError(f"Failed to load data/stats/details for {self.current_file} during reset: {e}")

        self.total_steps = len(self.current_data)

        # Start at index allowing for full lookback window
        self.current_step = self.lookback_window - 1

        # Reset financial state
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.trade_pnl_points = 0.0
        # Initialize portfolio value list with initial balance for the lookback period
        self.episode_portfolio_values = [self.initial_balance] * self.lookback_window
        self.num_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_cost = 0.0

        observation = self._get_observation()
        info = {'current_file': self.current_file}

        return observation, info

    def render(self):
        """Renders the environment state (optional)."""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            # Calculate unrealized PnL value for rendering
            unrealized_pnl_value = self.trade_pnl_points * self.lot_size * self.position
            current_portfolio_value = self.balance + unrealized_pnl_value
            profit = current_portfolio_value - self.initial_balance

            print(f"Step: {self.current_step}/{self.total_steps} | "
                  f"File: {self.current_file} | "
                  f"Lot Size: {self.lot_size} | "
                  f"Position: {self.position} | "
                  f"Balance: {self.balance:.2f} | "
                  f"Unrealized PnL: {unrealized_pnl_value:.2f} ({self.trade_pnl_points:.2f} pts) | "
                  f"Portfolio Value: {current_portfolio_value:.2f} | "
                  f"Profit: {profit:.2f} | "
                  f"Trades: {self.num_trades}")
        elif self.render_mode == 'none':
            pass # No rendering

    def close(self):
        """Cleans up environment resources (optional)."""
        pass

# Example usage (for testing the environment)
# Need to import check_env if running this directly
# from stable_baselines3.common.env_checker import check_env
if __name__ == '__main__':
    # Ensure normalization stats exist
    if not os.path.exists(RL_STATS_FILE): # Use config variable
        print(f"Error: {RL_STATS_FILE} not found. Run preprocess_norm_stats.py first.")
    else:
        try:
            # Import check_env here if needed for testing
            from stable_baselines3.common.env_checker import check_env
            env = TradingEnv(render_mode='ansi')
            check_env(env) # Check if environment conforms to Gymnasium API
            print("Environment check passed!")

            # --- Test Reset ---
            print("\n--- Testing Reset ---")
            obs, info = env.reset()
            print(f"Reset successful. Initial observation shape: {obs.shape}")
            print(f"Initial info: {info}")
            env.render()

            # --- Test Step ---
            print("\n--- Testing Step ---")
            action = env.action_space.sample() # Sample random action
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step successful. Action taken: {action}")
            print(f"Next observation shape: {obs.shape}")
            print(f"Reward: {reward}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print(f"Info: {info}")
            env.render()

            # --- Test Full Episode ---
            print("\n--- Testing Full Episode ---")
            obs, info = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            while not done:
                action = env.action_space.sample() # Use random actions for testing
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward # Accumulate reward (though only final reward matters here)
                done = terminated or truncated
                step_count += 1
                # Optional: Render every N steps
                # if step_count % 100 == 0:
                #     env.render()

            print("\n--- Episode Finished ---")
            env.render() # Render final state
            print(f"Total Steps: {step_count}")
            print(f"Final Reward (Sharpe - Cost Penalty): {reward}") # Print the final reward
            print(f"Final Info: {info}")

        except Exception as e:
            print(f"An error occurred during environment testing: {e}")
            import traceback
            traceback.print_exc()

# Note: This code includes basic error handling, data cleaning (ffill, dropna), and configuration. It also adds a `check_env` import and example usage block under `if __name__ == '__main__':` to help verify the environment conforms to the Gymnasium API. You'll need `stable-baselines3` installed (`pip install stable-baselines3[extra]`) for `check_env`.
# I've made assumptions about how to handle derived features during normalization and how to normalize PnL (using ATR). These might need refinement during testing. The Sharpe ratio calculation currently doesn't annualize; this is common for RL where rewards are often compared relatively within training, but could be added if needed.
