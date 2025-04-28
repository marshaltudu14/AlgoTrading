import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import (
    PROCESSED_DIR,
    INITIAL_CAPITAL,
    BROKERAGE_ENTRY,
    BROKERAGE_EXIT,
    QUANTITIES,
    RLHF_WEIGHT,
    RLHF_CONFIG
)


class TradingEnv(gym.Env):
    """
    Gymnasium environment for options-buying RL (CE/PE) with enhanced features
    for Indian market instruments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        instrument,
        timeframe,
        window_size=50,
        include_position_features=True,
        use_enhanced_features=True,
        normalize_rewards=True,
        asymmetric_rewards=True
    ):
        super().__init__()
        self.instrument = instrument
        self.timeframe = timeframe
        self.window_size = window_size
        self.include_position_features = include_position_features
        self.use_enhanced_features = use_enhanced_features
        self.normalize_rewards = normalize_rewards
        self.asymmetric_rewards = asymmetric_rewards

        # Load processed data
        if use_enhanced_features:
            # Use enhanced processor data if available
            enhanced_path = os.path.join(PROCESSED_DIR, f"enhanced_{instrument.replace(' ', '_')}_{timeframe}.csv")
            default_path = os.path.join(PROCESSED_DIR, f"{instrument.replace(' ', '_')}_{timeframe}.csv")

            if os.path.exists(enhanced_path):
                path = enhanced_path
                print(f"Using enhanced features for {instrument} @ {timeframe}min")
            else:
                path = default_path
                print(f"Enhanced features not found for {instrument} @ {timeframe}min, using default")
        else:
            path = os.path.join(PROCESSED_DIR, f"{instrument.replace(' ', '_')}_{timeframe}.csv")

        # Load processed data; datetime as string, keep 'signal' for shaping
        self.df = pd.read_csv(path)

        # Determine feature columns based on available data
        if use_enhanced_features and 'BB_POSITION' in self.df.columns:
            # Use a more comprehensive set of features for enhanced data
            self.feature_cols = [
                'open', 'high', 'low', 'close', 'ATR',
                'RSI_14', 'BB_POSITION', 'NATR', 'EMA_CROSS_20',
                'VOL_REGIME', 'TREND_REGIME'
            ]
            # Filter to only include columns that exist in the dataframe
            self.feature_cols = [col for col in self.feature_cols if col in self.df.columns]
        else:
            # Basic features for standard data
            self.feature_cols = ['open', 'high', 'low', 'close', 'ATR']

        # Extract feature data
        self.data = self.df[self.feature_cols].values.astype(np.float32)

        # Get instrument-specific quantity
        self.quantity = QUANTITIES.get(instrument, 0)
        if self.quantity == 0:
            print(f"Warning: No quantity defined for {instrument}, using default of 50")
            self.quantity = 50

        # Indian market-specific parameters
        # Margin requirements vary by instrument
        self.margin_requirement = self._get_margin_requirement(instrument)

        # Trading parameters
        self.initial_capital = INITIAL_CAPITAL
        self.brokerage_entry = BROKERAGE_ENTRY
        self.brokerage_exit = BROKERAGE_EXIT

        # Reward shaping parameters
        self.hold_penalty = float(os.getenv('HOLD_PENALTY', 1e-4))
        self.exit_bonus = float(os.getenv('EXIT_BONUS', 1e-2))
        self.drawdown_penalty = float(os.getenv('DRAWDOWN_PENALTY', 1e-3))
        self.sl_penalty = float(os.getenv('SL_PENALTY', 0.01))
        self.curiosity_bonus = float(os.getenv('CURIOSITY_BONUS', 1e-3))

        # RLHF parameters
        if asymmetric_rewards and RLHF_CONFIG.get('asymmetric_loss', False):
            self.target_weight = RLHF_CONFIG.get('target_weight', 1.0)
            self.sl_weight = RLHF_CONFIG.get('sl_weight', 2.0)
        else:
            self.target_weight = 1.0
            self.sl_weight = 1.0

        # Position tracking
        self.position_duration = 0
        self.unrealized_pnl = 0.0
        self.entry_time = None

        # Determine observation space shape based on whether position features are included
        feature_dim = len(self.feature_cols)
        if self.include_position_features:
            # Add position, position_duration, entry_price, unrealized_pnl
            feature_dim += 4

        # Spaces: actions: 0=hold,1=buy,2=sell
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.window_size, feature_dim),
            dtype=np.float32
        )

        # Trading statistics
        self.entry_index = None
        self.trade_history = []
        self.reset()

    def _get_margin_requirement(self, instrument):
        """
        Get margin requirement for the instrument.
        Different Indian indices have different margin requirements.
        """
        # Default margin requirements (percentage of contract value)
        margin_requirements = {
            'Nifty': 0.12,       # 12% of contract value
            'Bank Nifty': 0.15,  # 15% of contract value
            'Finnifty': 0.12,    # 12% of contract value
            'Sensex': 0.15,      # 15% of contract value
            'Bankex': 0.15       # 15% of contract value
        }

        return margin_requirements.get(instrument, 0.15)

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
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
        self.position_duration = 0
        self.unrealized_pnl = 0.0
        self.entry_index = None
        self.entry_time = None
        self.trade_history = []

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self):
        """
        Get observation with optional position features.

        Returns:
            Observation array
        """
        # Get market data
        market_data = self.data[self.current_step-self.window_size:self.current_step]

        if not self.include_position_features:
            return market_data

        # Create position features
        batch_size = market_data.shape[0]
        position_features = np.zeros((batch_size, 4), dtype=np.float32)

        # Fill the last row with current position features
        position_features[-1, 0] = self.position
        position_features[-1, 1] = self.position_duration
        position_features[-1, 2] = self.entry_price if self.position > 0 else 0.0
        position_features[-1, 3] = self.unrealized_pnl

        # Concatenate market data and position features
        return np.concatenate([market_data, position_features], axis=1)

    def _get_signal_at_current_step(self):
        """
        Get the signal at the current step (for training only).

        Returns:
            Signal value (0=hold, 1=buy target hit, 2=buy SL hit, 3=sell target hit, 4=sell SL hit)
        """
        if 'signal' not in self.df.columns:
            return 0
        return int(self.df.iloc[self.current_step]['signal'])

    def _calculate_unrealized_pnl(self, current_price):
        """
        Calculate unrealized PnL for the current position.
        Takes into account the instrument-specific quantity.

        Args:
            current_price: Current price

        Returns:
            Unrealized PnL
        """
        if self.position == 0:
            return 0.0

        # Calculate points gained/lost
        points = current_price - self.entry_price

        # Multiply by quantity to get money value
        return points * self.quantity

    def _calculate_margin_used(self):
        """
        Calculate margin used for the current position.
        Different Indian indices have different margin requirements.

        Returns:
            Margin used
        """
        if self.position == 0:
            return 0.0

        # Calculate contract value
        contract_value = self.entry_price * self.quantity

        # Apply margin requirement
        return contract_value * self.margin_requirement

    def _record_trade(self, entry_price, exit_price, duration, pnl, win):
        """
        Record trade details for analysis.

        Args:
            entry_price: Entry price
            exit_price: Exit price
            duration: Trade duration
            pnl: Profit/loss
            win: Whether the trade was a win
        """
        self.trade_history.append({
            'instrument': self.instrument,
            'timeframe': self.timeframe,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'duration': duration,
            'pnl': pnl,
            'win': win,
            'entry_step': self.entry_index,
            'exit_step': self.current_step
        })

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take (0=hold, 1=buy, 2=sell)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        done = False
        had_exit = False
        had_trade = False

        # Get close price from the feature data
        # Find the index of 'close' in feature_cols
        close_idx = self.feature_cols.index('close') if 'close' in self.feature_cols else 3
        price = float(self.data[self.current_step][close_idx])

        # Update unrealized PnL
        self.unrealized_pnl = self._calculate_unrealized_pnl(price)

        # Entry
        if action == 1 and self.position == 0:
            # Record entry details
            had_trade = True
            self.entry_index = self.current_step
            self.entry_price = price
            self.position = 1
            self.position_duration = 0

            # Apply brokerage fee
            self.capital -= self.brokerage_entry

            # Calculate margin used
            margin_used = self._calculate_margin_used()

            # Check if we have enough capital for margin
            if margin_used > self.capital:
                # Not enough capital, revert the trade
                self.position = 0
                self.entry_price = 0.0
                self.capital += self.brokerage_entry  # Refund brokerage
                had_trade = False

                # Apply penalty for failed trade
                failed_trade_penalty = self.brokerage_entry / self.initial_capital
                reward = -failed_trade_penalty

                # Advance step and return
                self.current_step += 1
                obs = self._get_observation()
                info = {
                    'capital': self.capital,
                    'max_drawdown': self.max_drawdown,
                    'trade_count': self.trade_count,
                    'win_rate': self.win_count / self.trade_count if self.trade_count else 0.0,
                    'position': self.position,
                    'position_duration': self.position_duration,
                    'unrealized_pnl': self.unrealized_pnl,
                    'failed_trade': True
                }
                return obs, reward, done, False, info

            # Add curiosity bonus for taking action
            curiosity_reward = self.curiosity_bonus

        # Exit
        elif action == 2 and self.position == 1:
            had_trade = True
            had_exit = True

            # Calculate PnL
            pnl = self._calculate_unrealized_pnl(price)

            # Apply brokerage fee
            self.capital += pnl - self.brokerage_exit

            # Record trade
            win = pnl > 0
            self._record_trade(
                entry_price=self.entry_price,
                exit_price=price,
                duration=self.position_duration,
                pnl=pnl,
                win=win
            )

            # Update statistics
            self.position = 0
            self.position_duration = 0
            self.trade_count += 1
            if win:
                self.win_count += 1

            # Add curiosity bonus for taking action
            curiosity_reward = self.curiosity_bonus

        else:
            curiosity_reward = 0.0

            # Update position duration if in a position
            if self.position > 0:
                self.position_duration += 1

        # Update drawdown
        self.peak = max(self.peak, self.capital)
        self.max_drawdown = max(self.max_drawdown, self.peak - self.capital)

        # Calculate drawdown penalty
        drawdown_penalty = self.drawdown_penalty * (self.max_drawdown / self.initial_capital)

        # Reward as normalized PnL
        reward = (self.capital - self.prev_capital) / self.initial_capital

        # Small penalty if no trade occurred this step
        if not had_trade:
            reward -= self.hold_penalty

        # RLHF shaping based on signal labels
        current_signal = self._get_signal_at_current_step()

        # Enhanced RLHF shaping with asymmetric rewards
        if had_exit and self.entry_index is not None:
            # Get signal at entry
            entry_signal = int(self.df.loc[self.entry_index, 'signal'])

            # Reward for exiting on profitable signals
            if entry_signal in (1, 3):  # Buy target hit, Sell target hit
                reward += RLHF_WEIGHT * self.target_weight
            elif entry_signal in (2, 4):  # Buy SL hit, Sell SL hit
                reward -= RLHF_WEIGHT * self.sl_weight
                # Extra penalty for stop-loss hits
                reward -= self.sl_penalty

        # Entry signal guidance (only during training)
        if action == 1:
            if current_signal in (1, 3):  # Buy on potential target hit
                reward += RLHF_WEIGHT * 0.5 * self.target_weight  # Half weight for entry guidance
            elif current_signal in (2, 4):  # Buy on potential SL hit
                reward -= RLHF_WEIGHT * 0.5 * self.sl_weight  # Penalize bad entries

        # Apply drawdown penalty
        reward -= drawdown_penalty

        # Add curiosity bonus
        reward += curiosity_reward

        # Update previous capital
        self.prev_capital = self.capital

        # Bonus for any trade exit to encourage trading activity
        if had_exit:
            reward += self.exit_bonus

        # Normalize rewards if enabled
        if self.normalize_rewards and self.initial_capital > 0:
            # Scale reward to be in a reasonable range
            reward = reward * 100  # Scale to percentage

            # Clip reward to prevent extreme values
            reward = np.clip(reward, -10.0, 10.0)

        # Advance
        self.current_step += 1

        # Check for episode end
        if self.current_step >= len(self.data):
            # Force exit of any open position at last price
            if self.position == 1:
                # Get last price
                last_price = float(self.data[-1][close_idx])

                # Calculate PnL
                pnl = self._calculate_unrealized_pnl(last_price)

                # Apply brokerage fee
                self.capital += pnl - self.brokerage_exit

                # Record trade
                win = pnl > 0
                self._record_trade(
                    entry_price=self.entry_price,
                    exit_price=last_price,
                    duration=self.position_duration,
                    pnl=pnl,
                    win=win
                )

                # Update statistics
                self.position = 0
                self.trade_count += 1
                if win:
                    self.win_count += 1

                # Reward for forced exit
                exit_reward = (self.capital - self.prev_capital) / self.initial_capital + self.exit_bonus

                # Apply asymmetric rewards for forced exit
                if pnl > 0:
                    exit_reward *= self.target_weight
                else:
                    exit_reward *= self.sl_weight

                reward += exit_reward
                self.prev_capital = self.capital

            done = True

        # Get next observation
        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())

        # Prepare info dictionary
        info = {
            'capital': self.capital,
            'max_drawdown': self.max_drawdown,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / self.trade_count if self.trade_count else 0.0,
            'position': self.position,
            'position_duration': self.position_duration,
            'unrealized_pnl': self.unrealized_pnl,
            'current_signal': current_signal,
            'instrument': self.instrument,
            'timeframe': self.timeframe
        }

        # Return per Gymnasium API (5-tuple) for Monitor compatibility
        return obs, reward, done, False, info

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode: Rendering mode
        """
        # Get current price
        close_idx = self.feature_cols.index('close') if 'close' in self.feature_cols else 3
        price = float(self.data[self.current_step][close_idx])

        # Get current signal
        signal = self._get_signal_at_current_step()
        signal_map = {
            0: "HOLD",
            1: "BUY_TARGET_HIT",
            2: "BUY_SL_HIT",
            3: "SELL_TARGET_HIT",
            4: "SELL_SL_HIT"
        }
        signal_str = signal_map.get(signal, "UNKNOWN")

        # Print detailed information
        print(f"Step {self.current_step}/{len(self.data)-1} | {self.instrument} @ {self.timeframe}min")
        print(f"Price: {price:.2f} | Capital: {self.capital:.2f} | Drawdown: {self.max_drawdown:.2f}")
        print(f"Position: {self.position} | Duration: {self.position_duration} | PnL: {self.unrealized_pnl:.2f}")
        win_rate = (self.win_count/self.trade_count*100) if self.trade_count else 0.0
        print(f"Trades: {self.trade_count} | Win Rate: {win_rate:.1f}%")
        print(f"Signal: {signal_str} (for training only)")
        print("-" * 50)
