import gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict

from src.utils.data_loader import DataLoader
from src.backtesting.engine import BacktestingEngine

class TradingEnv(gym.Env):
    def __init__(self, data_loader: DataLoader, symbol: str, initial_capital: float, lookback_window: int = 50):
        super(TradingEnv, self).__init__()
        self.data_loader = data_loader
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window

        self.engine = BacktestingEngine(initial_capital)
        self.data = None  # This will store the loaded data for the current episode
        self.current_step = 0

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(5)  # BUY_LONG, SELL_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        
        # Let's assume observation_dim is 10 for now, matching the test.
        # The actual observation space will include OHLCV, features (like ATR), and normalized account state.
        # For simplicity, let's define a fixed observation_dim for now that includes:
        # lookback_window * (OHLCV + ATR) + current_capital + current_position_quantity + current_position_entry_price + unrealized_pnl + is_position_open
        # Assuming OHLCV + ATR = 5 features per step
        features_per_step = 5 # open, high, low, close, atr
        account_state_features = 4 # capital, position_quantity, position_entry_price, unrealized_pnl
        self.observation_dim = (self.lookback_window * features_per_step) + account_state_features + 1 # +1 for is_position_open
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.data = self.data_loader.load_raw_data_for_symbol(self.symbol)
        self.engine.reset()
        self.current_step = self.lookback_window - 1 # Start from where lookback window is full
        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1

        if self.current_step >= len(self.data):
            done = True
            reward = 0.0 # No more data to process
            info = {"message": "End of data"}
            return self._get_observation(), reward, done, info

        current_price = self.data['close'].iloc[self.current_step]
        current_atr = self.data['atr'].iloc[self.current_step]
        prev_capital = self.engine.get_account_state()['capital']

        # Simplified action mapping for now
        if action == 0: # BUY_LONG
            self.engine.execute_trade("BUY_LONG", current_price, 1, current_atr) # Assuming quantity 1 for simplicity
        elif action == 1: # SELL_SHORT
            self.engine.execute_trade("SELL_SHORT", current_price, 1, current_atr)
        elif action == 2: # CLOSE_LONG
            self.engine.execute_trade("CLOSE_LONG", current_price, 1, current_atr)
        elif action == 3: # CLOSE_SHORT
            self.engine.execute_trade("CLOSE_SHORT", current_price, 1, current_atr)
        elif action == 4: # HOLD
            pass

        # Calculate reward based on P&L change
        current_capital = self.engine.get_account_state(current_price=current_price)['capital']
        reward = current_capital - prev_capital

        done = False # For now, only done at end of data
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        # Get OHLCV and ATR data for the lookback window
        start_index = self.current_step - self.lookback_window + 1
        end_index = self.current_step + 1
        
        # Ensure we don't go out of bounds
        if start_index < 0:
            # Pad with zeros if not enough history
            padding_needed = abs(start_index)
            ohlcv_atr_data = np.zeros((self.lookback_window, 5)) # 5 for OHLCV + ATR
            actual_data = self.data[['open', 'high', 'low', 'close', 'atr']].iloc[0:end_index].values
            ohlcv_atr_data[padding_needed:] = actual_data
        else:
            ohlcv_atr_data = self.data[['open', 'high', 'low', 'close', 'atr']].iloc[start_index:end_index].values

        # Get account state
        account_state = self.engine.get_account_state(current_price=self.data['close'].iloc[self.current_step])

        # Normalize account state features (simple normalization for now)
        normalized_capital = account_state['capital'] / self.initial_capital
        normalized_position_quantity = account_state['current_position_quantity'] # Assuming quantity is already somewhat normalized or small
        normalized_position_entry_price = account_state['current_position_entry_price'] / self.data['close'].iloc[self.current_step] if self.data['close'].iloc[self.current_step] != 0 else 0
        normalized_unrealized_pnl = account_state['unrealized_pnl'] / self.initial_capital
        is_position_open = 1.0 if account_state['is_position_open'] else 0.0

        # Combine all features into a single observation vector
        observation = np.concatenate([
            ohlcv_atr_data.flatten(),
            np.array([
                normalized_capital,
                normalized_position_quantity,
                normalized_position_entry_price,
                normalized_unrealized_pnl,
                is_position_open
            ])
        ])
        return observation.astype(np.float32)
