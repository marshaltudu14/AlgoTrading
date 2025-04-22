import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import PROCESSED_DIR, INITIAL_CAPITAL, BROKERAGE_ENTRY, BROKERAGE_EXIT, QUANTITIES, RLHF_WEIGHT


class TradingEnv(gym.Env):
    """Gymnasium environment for options-buying RL (CE/PE)."""
    metadata = {'render.modes': ['human']}

    def __init__(self, instrument, timeframe, window_size=50):
        super().__init__()
        self.instrument = instrument
        self.timeframe = timeframe
        self.window_size = window_size
        path = os.path.join(PROCESSED_DIR, f"{instrument.replace(' ', '_')}_{timeframe}.csv")
        # Load processed data; datetime as string, keep 'signal' for shaping
        self.df = pd.read_csv(path)
        # Features: open, high, low, close, ATR
        self.feature_cols = ['open', 'high', 'low', 'close', 'ATR']
        self.data = self.df[self.feature_cols].values.astype(np.float32)
        self.quantity = QUANTITIES.get(instrument, 0)

        # Trading parameters
        self.initial_capital = INITIAL_CAPITAL
        self.brokerage_entry = BROKERAGE_ENTRY
        self.brokerage_exit = BROKERAGE_EXIT
        # Encourage action: small penalty for holding and bonus for exits
        self.hold_penalty = float(os.getenv('HOLD_PENALTY', 1e-4))
        self.exit_bonus   = float(os.getenv('EXIT_BONUS',   1e-2))

        # Spaces: actions: 0=hold,1=buy,2=sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, len(self.feature_cols)),
            dtype=np.float32
        )
        self.entry_index = None
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.prev_capital = self.initial_capital
        self.peak = self.initial_capital
        self.max_drawdown = 0.0
        self.trade_count = 0
        self.win_count = 0
        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        return self.data[self.current_step-self.window_size:self.current_step]

    def step(self, action):
        done = False
        had_exit = False
        had_trade = False
        price = float(self.data[self.current_step][3])  # close
        # Entry
        if action == 1 and self.position == 0:
            # record entry index for shaping
            had_trade = True
            self.entry_index = self.current_step
            self.entry_price = price
            self.position = 1
            self.capital -= self.brokerage_entry
        # Exit
        elif action == 2 and self.position == 1:
            had_trade = True
            had_exit = True
            pnl = (price - self.entry_price) * self.quantity
            self.capital += pnl - self.brokerage_exit
            self.position = 0
            self.trade_count += 1
            if pnl > 0:
                self.win_count += 1
        # Update drawdown
        self.peak = max(self.peak, self.capital)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.capital)
        # Reward as normalized PnL
        reward = (self.capital - self.prev_capital) / self.initial_capital
        # Small penalty if no trade occurred this step
        if not had_trade:
            reward -= self.hold_penalty
        # RLHF shaping based on signal labels
        if had_exit and self.entry_index is not None:
            signal = int(self.df.loc[self.entry_index, 'signal'])
            if signal in (1, 3):
                reward += RLHF_WEIGHT
            elif signal in (2, 4):
                reward -= RLHF_WEIGHT
        self.prev_capital = self.capital
        # Bonus for any trade exit to encourage trading activity
        if had_exit:
            reward += self.exit_bonus
        # Advance
        self.current_step += 1
        # Check for episode end
        if self.current_step >= len(self.data):
            # Force exit of any open position at last price
            if self.position == 1:
                last_price = float(self.data[-1][3])
                pnl = (last_price - self.entry_price) * self.quantity
                self.capital += pnl - self.brokerage_exit
                self.position = 0
                self.trade_count += 1
                if pnl > 0:
                    self.win_count += 1
                # reward for forced exit
                exit_reward = (self.capital - self.prev_capital) / self.initial_capital + self.exit_bonus
                reward += exit_reward
                self.prev_capital = self.capital
            done = True
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        info = {
            'capital': self.capital,
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / self.trade_count if self.trade_count else 0.0
        }
        # Return per Gymnasium API (5-tuple) for Monitor compatibility
        return obs, reward, done, False, info

    def render(self, mode='human'):
        print(f"Step {self.current_step}: Capital={self.capital:.2f}, Position={self.position}")
