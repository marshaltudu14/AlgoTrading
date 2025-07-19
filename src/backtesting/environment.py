import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple

from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader
from src.config.config import INITIAL_CAPITAL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, symbol: str, data_loader: DataLoader, initial_capital: float = INITIAL_CAPITAL, lookback_window: int = 50):
        super().__init__()
        self.symbol = symbol
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window

        self.df = self.data_loader.load_raw_data_for_symbol(self.symbol)
        if self.df.empty:
            raise ValueError(f"Failed to load data for symbol {self.symbol}. Environment cannot be initialized.")

        self.engine = BacktestingEngine(self.initial_capital)
        self.current_step = self.lookback_window  # Start after lookback window

        # Action Space: BUY_LONG, SELL_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        self.action_space = spaces.Discrete(5)

        # Observation Space: OHLCV data for lookback window + current account state
        # Assuming OHLCV are normalized later. For now, define shape.
        # OHLCV (5 features: open, high, low, close, volume - assuming volume is present in raw data)
        # Account state (4 features: capital, current_position_quantity, current_position_entry_price, unrealized_pnl)
        # Total features per step: 5 (OHLCV) + 4 (account state) = 9
        # Shape: (lookback_window, 5) for OHLCV + 4 for account state
        # Let's simplify for now and assume a flat observation space after normalization
        # This will need careful design based on actual feature engineering.
        # For now, let's assume observation is a flat array of (lookback_window * 5) + 4
        # Max value for OHLCV can be large, min 0. Capital can be large.
        # We will normalize these values within _get_observation

        # Define a placeholder observation space. Actual bounds will depend on normalization.
        # A more robust way would be to analyze data first to get min/max for normalization.
        # For now, using float32 with large bounds.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=((self.lookback_window * 5) + 4,), dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        # Get OHLCV data for the lookback window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        obs_data = self.df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values.flatten() # Assuming 'volume' column exists

        # Get current account state
        current_price = self.df.iloc[self.current_step]['close']
        account_state = self.engine.get_account_state(current_price=current_price)

        # Normalize account state values (simple min-max for now, can be improved)
        # This is a very basic normalization. A proper one would use historical min/max or Z-score.
        norm_capital = account_state['capital'] / self.initial_capital
        norm_position = account_state['current_position_quantity'] / 100 # Assuming max position of 100 for normalization
        norm_entry_price = account_state['current_position_entry_price'] / current_price if current_price != 0 else 0
        norm_unrealized_pnl = account_state['unrealized_pnl'] / self.initial_capital

        account_obs = np.array([norm_capital, norm_position, norm_entry_price, norm_unrealized_pnl], dtype=np.float32)

        # Concatenate OHLCV and account state
        observation = np.concatenate((obs_data, account_obs))
        return observation

    def _get_observation_at_step(self, step: int) -> np.ndarray:
        # Get OHLCV data for the lookback window at a specific step
        start_idx = step - self.lookback_window
        end_idx = step
        obs_data = self.df.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values.flatten() # Assuming 'volume' column exists

        # Get current account state
        current_price = self.df.iloc[step]['close']
        account_state = self.engine.get_account_state(current_price=current_price)

        # Normalize account state values (simple min-max for now, can be improved)
        norm_capital = account_state['capital'] / self.initial_capital
        norm_position = account_state['current_position_quantity'] / 100 # Assuming max position of 100 for normalization
        norm_entry_price = account_state['current_position_entry_price'] / current_price if current_price != 0 else 0
        norm_unrealized_pnl = account_state['unrealized_pnl'] / self.initial_capital

        account_obs = np.array([norm_capital, norm_position, norm_entry_price, norm_unrealized_pnl], dtype=np.float32)

        # Concatenate OHLCV and account state
        observation = np.concatenate((obs_data, account_obs))
        return observation

    def _calculate_reward(self, prev_total_pnl: float, current_total_pnl: float, is_invalid_action: bool) -> float:
        if is_invalid_action:
            return -1.0  # Penalty for invalid actions
        
        # Reward is the change in total P&L (realized + unrealized)
        reward = current_total_pnl - prev_total_pnl
        return reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.engine.reset()
        self.current_step = self.lookback_window
        
        # Ensure 'volume' column exists in the DataFrame
        if 'volume' not in self.df.columns:
            logging.warning("'volume' column not found in raw data. Adding a placeholder column of zeros.")
            self.df['volume'] = 0

        observation = self._get_observation()
        info = self.engine.get_account_state(current_price=self.df.iloc[self.current_step]['close'])
        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1
        done = False
        reward = 0.0
        is_invalid_action = False

        if self.current_step >= len(self.df):
            done = True
            # Calculate final P&L for reward
            final_price = self.df.iloc[len(self.df) - 1]['close']
            final_account_state = self.engine.get_account_state(current_price=final_price)
            reward = final_account_state['total_pnl'] # Reward based on final total P&L
            # Return the observation from the last valid step, which is len(self.df) - 1
            observation = self._get_observation_at_step(len(self.df) - 1) 
            info = final_account_state
            return observation, reward, done, info

        current_price = self.df.iloc[self.current_step]['close']
        prev_account_state = self.engine.get_account_state(current_price=self.df.iloc[self.current_step - 1]['close'])
        prev_total_pnl = prev_account_state['total_pnl']

        # Map action integer to string for BacktestingEngine
        action_map = {
            0: "BUY_LONG",
            1: "SELL_SHORT",
            2: "CLOSE_LONG",
            3: "CLOSE_SHORT",
            4: "HOLD"
        }
        action_str = action_map.get(action, "HOLD")

        # Execute trade based on action
        if action_str == "HOLD":
            pass # No trade, no cost
        else:
            # For simplicity, assume quantity is 1 for now. This needs to be dynamic.
            # Or, the RL agent could output quantity as well.
            # For MVP, let's assume fixed quantity or full position close.
            quantity = 1 # Placeholder quantity
            if action_str == "CLOSE_LONG" and self.engine.get_account_state()['current_position_quantity'] > 0:
                quantity = self.engine.get_account_state()['current_position_quantity']
            elif action_str == "CLOSE_SHORT" and self.engine.get_account_state()['current_position_quantity'] < 0:
                quantity = abs(self.engine.get_account_state()['current_position_quantity'])
            elif action_str in ["BUY_LONG", "SELL_SHORT"]:
                # Check if enough capital for BUY/SELL
                trade_cost = (current_price * quantity) + self.engine.BROKERAGE_ENTRY
                if self.engine.get_account_state()['capital'] < trade_cost:
                    is_invalid_action = True
                    logging.warning(f"Invalid action: Insufficient capital for {action_str} at step {self.current_step}")

            if not is_invalid_action:
                _, _ = self.engine.execute_trade(action_str, current_price, quantity)
            else:
                # If invalid action, ensure no trade is executed and apply penalty
                pass # Penalty handled in _calculate_reward

        current_account_state = self.engine.get_account_state(current_price=current_price)
        current_total_pnl = current_account_state['total_pnl']

        reward = self._calculate_reward(prev_total_pnl, current_total_pnl, is_invalid_action)
        observation = self._get_observation()
        info = current_account_state

        return observation, reward, done, info

    def render(self, mode='human'):
        # This is a CLI-only environment, so rendering is minimal.
        # Could print current state or trade info if needed for debugging.
        pass

    def close(self):
        pass