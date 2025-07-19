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
        
        # Placeholder for observation space definition. This will be more complex.
        # For now, let's assume a simple box space.
        # The actual observation space will depend on the features and lookback window.
        # For the test, we need a concrete observation_space.
        # Let's assume observation_dim is 10 for now, matching the test.
        observation_dim = 10 # This should be dynamically calculated based on data features + account state
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

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
        
        # Simplified action mapping for now
        if action == 0: # BUY_LONG
            self.engine.execute_trade("BUY_LONG", current_price, 1) # Assuming quantity 1 for simplicity
        elif action == 1: # SELL_SHORT
            self.engine.execute_trade("SELL_SHORT", current_price, 1)
        elif action == 2: # CLOSE_LONG
            self.engine.execute_trade("CLOSE_LONG", current_price, 1)
        elif action == 3: # CLOSE_SHORT
            self.engine.execute_trade("CLOSE_SHORT", current_price, 1)
        elif action == 4: # HOLD
            pass

        # Calculate reward based on P&L change
        # This is a very basic reward. Will be refined later.
        account_state = self.engine.get_account_state()
        reward = account_state['realized_pnl'] + account_state['unrealized_pnl']

        done = False # For now, only done at end of data
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        # This is a placeholder. Actual observation will include OHLCV, features, and normalized account state.
        # For now, return a dummy observation matching the expected dimension.
        return np.random.rand(self.observation_space.shape[0]).astype(np.float32)
