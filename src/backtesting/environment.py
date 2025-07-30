import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
import random

from src.utils.data_loader import DataLoader
from src.backtesting.engine import BacktestingEngine
from src.config.instrument import Instrument
from src.utils.instrument_loader import load_instruments
from src.utils.data_feeding_strategy import DataFeedingStrategyManager, FeedingStrategy
from src.config.config import RISK_REWARD_CONFIG, INITIAL_CAPITAL
from src.utils.capital_aware_quantity import adjust_quantity_for_capital

# Configure logger
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, data_loader: DataLoader, symbol: str, initial_capital: float,
                 lookback_window: int = 50, trailing_stop_percentage: float = None,
                 reward_function: str = "pnl", episode_length: int = 1000,
                 use_streaming: bool = True):
        super(TradingEnv, self).__init__()
        self.data_loader = data_loader
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        self.reward_function = reward_function
        self.episode_length = episode_length
        self.use_streaming = use_streaming

        # For streaming mode
        self.total_data_length = 0
        self.current_episode_start = 0
        self.current_episode_end = 0

        # Track returns for sophisticated reward calculations
        self.returns_history = []
        self.equity_history = [initial_capital]

        # Track for reward shaping
        self.idle_steps = 0
        self.trade_count = 0
        self.last_action_type = 4  # Start with HOLD
        self.previous_trailing_stop = 0.0  # Track trailing stop improvement

        # Track statistics for z-score normalization
        self.feature_stats = {}  # Will store mean and std for each feature
        self.observation_history = []  # Store raw observations for statistics

        # Track for episode termination conditions
        self.peak_equity = initial_capital
        self.max_drawdown_pct = 0.20  # 20% maximum drawdown

        # Load instrument details
        self.instruments = load_instruments('config/instruments.yaml')

        # Get base symbol (remove timeframe suffix)
        self.base_symbol = self.data_loader.get_base_symbol(self.symbol)

        if self.base_symbol not in self.instruments:
            raise ValueError(f"Instrument {self.base_symbol} not found in instruments.yaml (original symbol: {self.symbol})")
        self.instrument = self.instruments[self.base_symbol]

        # Use centralized trailing stop configuration if not provided
        if trailing_stop_percentage is None:
            trailing_stop_percentage = RISK_REWARD_CONFIG['trailing_stop_percentage']

        self.engine = BacktestingEngine(initial_capital, self.instrument, trailing_stop_percentage)
        self.data = None  # This will store the loaded data for the current episode
        self.current_step = 0

        # Initialize data feeding strategy manager
        self.feeding_strategy_manager = None  # Will be initialized when data is first loaded

        # Initialize data length for streaming mode using FINAL data
        if self.use_streaming:
            self.total_data_length = self.data_loader.get_data_length(self.symbol, data_type="final")
            if self.total_data_length == 0:
                logging.warning(f"No final data found for symbol {self.symbol}, falling back to full loading")
                self.use_streaming = False

        # Define action and observation space
        # Action space: [action_type, quantity]
        # action_type: 0=BUY_LONG, 1=SELL_SHORT, 2=CLOSE_LONG, 3=CLOSE_SHORT, 4=HOLD
        # quantity: continuous value between 0 and max_quantity (e.g., 10 lots)
        self.max_quantity = 10  # Maximum lots that can be traded
        self.action_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([4, self.max_quantity]),
            dtype=np.float32
        )
        
        # Define observation space dimensions dynamically based on available data
        # We'll update this after loading data to get the actual number of features
        self.features_per_step = None  # Will be set after data loading
        self.account_state_features = 5 # capital, position_quantity, position_entry_price, unrealized_pnl, is_position_open
        self.trailing_features = 1 # distance_to_trail
        self.observation_dim = None  # Will be calculated after data loading
        self.observation_space = None  # Will be set after data loading
        self.fixed_feature_columns = None  # Fixed feature columns to ensure consistency

    def reset(self, performance_metrics: Optional[Dict] = None) -> np.ndarray:
        """Reset environment and load data segment for new episode with intelligent data feeding."""
        if self.use_streaming:
            self._load_episode_data_segment()
        else:
            self.data = self.data_loader.load_final_data_for_symbol(self.symbol)



            # Initialize feeding strategy manager if not already done
            if self.feeding_strategy_manager is None:
                self.feeding_strategy_manager = DataFeedingStrategyManager(
                    data=self.data,
                    lookback_window=self.lookback_window,
                    episode_length=self.episode_length
                )
                logger.info(f"🎯 Data feeding strategy initialized: {self.feeding_strategy_manager.current_strategy.value}")

        # Set observation space dimensions based on actual data
        if self.observation_space is None:
            feature_columns = [col for col in self.data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp']]
            self.features_per_step = len(feature_columns)
            self.observation_dim = (self.lookback_window * self.features_per_step) + self.account_state_features + self.trailing_features
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
            logger.info(f"Observation space set: {self.features_per_step} features per step, total dim: {self.observation_dim}")

        self.engine.reset()

        # Use intelligent data feeding strategy for episode start position
        if self.feeding_strategy_manager is not None:
            start_idx, end_idx = self.feeding_strategy_manager.get_next_episode_data(performance_metrics)
            # Ensure we have enough data for lookback
            self.current_step = max(start_idx + self.lookback_window - 1, self.lookback_window - 1)
            self.episode_end_step = min(end_idx, len(self.data) - 1)
            logger.debug(f"Strategic episode: steps {self.current_step} to {self.episode_end_step} "
                        f"(strategy: {self.feeding_strategy_manager.current_strategy.value})")
        else:
            # Fallback to random positioning
            max_start = len(self.data) - self.episode_length - self.lookback_window
            if max_start > self.lookback_window:
                random_start = random.randint(self.lookback_window - 1, max_start)
                self.current_step = random_start
                self.episode_end_step = min(self.current_step + self.episode_length, len(self.data) - 1)
                logger.debug(f"Episode starting at step {self.current_step} (randomized)")
            else:
                self.current_step = self.lookback_window - 1
                self.episode_end_step = len(self.data) - 1
                logger.debug(f"Episode starting at step {self.current_step} (fixed - insufficient data for randomization)")

        # Reset reward tracking
        self.returns_history = []
        self.equity_history = [self.initial_capital]
        self.idle_steps = 0
        self.trade_count = 0
        self.last_action_type = 4
        self.feature_stats = {}
        self.observation_history = []
        self.peak_equity = self.initial_capital
        self.previous_trailing_stop = 0.0

        return self._get_observation()

    def _load_episode_data_segment(self) -> None:
        """Load only the necessary data segment for the current episode."""
        # Calculate episode boundaries
        max_start = max(0, self.total_data_length - self.episode_length - self.lookback_window)

        if max_start <= 0:
            # Not enough data for streaming, load all FINAL data (not raw!)
            logging.warning(f"Insufficient data for streaming mode, loading all FINAL data for {self.symbol}")
            self.data = self.data_loader.load_final_data_for_symbol(self.symbol)
            self.use_streaming = False
            return

        # Randomly select episode start point
        import random
        self.current_episode_start = random.randint(0, max_start)
        self.current_episode_end = self.current_episode_start + self.episode_length + self.lookback_window

        # Load only the required segment from FINAL processed data
        self.data = self.data_loader.load_data_segment(
            symbol=self.symbol,
            start_idx=self.current_episode_start,
            end_idx=self.current_episode_end,
            data_type="final"
        )

        if self.data.empty:
            from src.utils.error_logger import log_error
            log_error(f"Failed to load data segment for {self.symbol}, falling back to full FINAL data loading",
                     f"Episode: {self.current_episode_start}-{self.current_episode_end}")
            self.data = self.data_loader.load_final_data_for_symbol(self.symbol)
            self.use_streaming = False
        else:
            # Reset data index for consistent access
            self.data = self.data.reset_index(drop=True)
            logging.info(f"Loaded data segment for {self.symbol}: rows {self.current_episode_start}-{self.current_episode_end}")

    def get_episode_info(self) -> Dict:
        """Get information about the current episode data segment."""
        info = {
            "use_streaming": self.use_streaming,
            "total_data_length": self.total_data_length,
            "episode_length": self.episode_length,
            "current_data_length": len(self.data) if self.data is not None else 0
        }

        if self.use_streaming:
            info.update({
                "episode_start": self.current_episode_start,
                "episode_end": self.current_episode_end,
                "data_utilization": f"{self.episode_length}/{self.total_data_length} ({100*self.episode_length/self.total_data_length:.1f}%)"
            })

        return info

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        self.current_step += 1

        # Check if detailed logging is enabled
        import os
        detailed_logging = os.environ.get('DETAILED_BACKTEST_LOGGING', 'false').lower() == 'true'

        if self.current_step >= len(self.data):
            # Force close any open positions before episode ends
            self._force_close_open_positions()
            done = True
            reward = 0.0 # No more data to process
            info = {"message": "End of data"}
            return self._get_observation(), reward, done, info

        # Use bounds checking to prevent indexing errors
        safe_step = min(self.current_step, len(self.data) - 1)
        current_price = self.data['close'].iloc[safe_step]

        # Get ATR if available, otherwise use a default volatility estimate
        if 'atr' in self.data.columns:
            current_atr = self.data['atr'].iloc[safe_step]
        else:
            # Fallback: estimate volatility from recent price range
            lookback = min(14, safe_step + 1)
            recent_highs = self.data['high'].iloc[max(0, safe_step - lookback + 1):safe_step + 1]
            recent_lows = self.data['low'].iloc[max(0, safe_step - lookback + 1):safe_step + 1]
            current_atr = (recent_highs.max() - recent_lows.min()) / lookback

        prev_capital = self.engine.get_account_state()['capital']

        # Calculate proxy premium for options simulation (based on ATR)
        proxy_premium = self._calculate_proxy_premium(current_price, current_atr)

        # Parse action: [action_type, quantity] or (action_type, quantity)
        if isinstance(action, (list, np.ndarray, tuple)) and len(action) >= 2:
            action_type = int(np.clip(action[0], 0, 4))
            quantity = max(0, float(action[1]))  # Ensure non-negative quantity
            # Force quantity to be integer for production use
            quantity = float(int(round(quantity)))
        else:
            # Backward compatibility: treat as discrete action with quantity 1
            try:
                action_type = int(action) if isinstance(action, (int, float)) else 4
            except (ValueError, TypeError):
                action_type = 4  # Default to HOLD if conversion fails
            quantity = 1.0 if action_type != 4 else 0.0

        # Apply capital-aware quantity adjustment for trading actions (not HOLD)
        if action_type != 4 and quantity > 0:
            available_capital = prev_capital
            adjusted_quantity = adjust_quantity_for_capital(
                predicted_quantity=quantity,
                available_capital=available_capital,
                current_price=current_price,
                instrument=self.instrument,
                proxy_premium=proxy_premium if self.instrument.type == "OPTION" else None
            )

            # Log quantity adjustment if it was changed
            if adjusted_quantity != quantity:
                logger.info(f"💰 Quantity adjusted for capital: {quantity} → {adjusted_quantity} lots")
                logger.info(f"   Available capital: ₹{available_capital:.2f}")

            quantity = float(adjusted_quantity)

        # DEBUG: Log every action to see what the agent is doing (only if detailed logging enabled)
        if detailed_logging:
            action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
            action_name = action_names[action_type]

            if self.current_step % 10 == 0 or action_type != 4:  # Log every 10 steps or non-HOLD actions
                logger.info(f"🎯 Step {self.current_step}: Agent action: {action} -> {action_name} (qty: {quantity})")
                logger.info(f"   Current price: ₹{current_price:.2f}, Capital: ₹{prev_capital:.2f}")

        # Get current position state to validate actions
        account_state = self.engine.get_account_state()
        current_position = account_state['current_position_quantity']

        if action_type == 0: # BUY_LONG
            self.engine.execute_trade("BUY_LONG", current_price, quantity, current_atr, proxy_premium)
        elif action_type == 1: # SELL_SHORT
            self.engine.execute_trade("SELL_SHORT", current_price, quantity, current_atr, proxy_premium)
        elif action_type == 2: # CLOSE_LONG
            # Only execute if we have a long position to close
            if current_position > 0:
                self.engine.execute_trade("CLOSE_LONG", current_price, quantity, current_atr, proxy_premium)
            else:
                # Convert invalid action to HOLD to reduce warnings
                self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)
        elif action_type == 3: # CLOSE_SHORT
            # Only execute if we have a short position to close
            if current_position < 0:
                self.engine.execute_trade("CLOSE_SHORT", current_price, quantity, current_atr, proxy_premium)
            else:
                # Convert invalid action to HOLD to reduce warnings
                self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)
        elif action_type == 4: # HOLD
            self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)

        # Log significant trading actions (not HOLD) every 50 steps or for trades (only if detailed logging enabled)
        if detailed_logging and (action_type != 4 or self.current_step % 50 == 0):
            action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
            action_name = action_names[action_type]
            account_state = self.engine.get_account_state(current_price=current_price)
            if action_type != 4:  # Actual trade
                logger.info(f"🎯 Step {self.current_step}: {action_name} @ ₹{current_price:.2f} (Qty: {quantity})")
                logger.info(f"   Position: {account_state['current_position_quantity']}, Capital: ₹{account_state['capital']:.2f}")
            else:  # Periodic status update
                logger.info(f"📊 Step {self.current_step}: Capital: ₹{account_state['capital']:.2f}, Position: {account_state['current_position_quantity']}")

        # Calculate reward using selected reward function
        current_capital = self.engine.get_account_state(current_price=current_price)['capital']
        self.equity_history.append(current_capital)

        # Calculate return for this step
        if len(self.equity_history) > 1:
            step_return = (current_capital - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(step_return)

        # Track action for reward shaping
        if action_type == 4:  # HOLD
            self.idle_steps += 1
        else:
            self.idle_steps = 0
            self.trade_count += 1

        base_reward = self._calculate_reward(current_capital, prev_capital)
        shaped_reward = self._apply_reward_shaping(base_reward, action_type, current_capital, prev_capital)
        reward = shaped_reward

        # DEBUG: Log reward calculation (only if detailed logging enabled)
        if detailed_logging and (self.current_step % 20 == 0 or abs(reward) > 0.1):
            logger.info(f"💰 Step {self.current_step}: Reward = {reward:.4f} (base: {base_reward:.4f}, shaped: {shaped_reward:.4f})")
            logger.info(f"   Capital: {prev_capital:.2f} -> {current_capital:.2f} (change: {current_capital - prev_capital:.2f})")

        self.last_action_type = action_type

        # Check termination conditions
        done, termination_reason = self._check_termination_conditions(current_capital)

        # Force close any open positions if episode is ending
        if done:
            self._force_close_open_positions()

        info = {"termination_reason": termination_reason} if termination_reason else {}

        # Return 4 values as expected by standard gym environments
        return self._get_observation(), reward, done, info

    def _force_close_open_positions(self):
        """Force close any open positions at the end of an episode to ensure accurate final capital."""
        # Check if detailed logging is enabled
        import os
        detailed_logging = os.environ.get('DETAILED_BACKTEST_LOGGING', 'false').lower() == 'true'

        account_state = self.engine.get_account_state()

        if account_state['is_position_open']:
            current_price = self.data['close'].iloc[min(self.current_step, len(self.data) - 1)]
            position_quantity = account_state['current_position_quantity']

            if position_quantity > 0:
                # Close long position
                if detailed_logging:
                    logging.info(f"Force closing long position at episode end. Price: {current_price}")
                self.engine.execute_trade("CLOSE_LONG", current_price, abs(position_quantity))
            elif position_quantity < 0:
                # Close short position
                if detailed_logging:
                    logging.info(f"Force closing short position at episode end. Price: {current_price}")
                self.engine.execute_trade("CLOSE_SHORT", current_price, abs(position_quantity))

    def _calculate_reward(self, current_capital: float, prev_capital: float) -> float:
        """Calculate reward based on selected reward function."""
        if self.reward_function == "pnl":
            return current_capital - prev_capital
        elif self.reward_function == "sharpe":
            return self._calculate_sharpe_ratio()
        elif self.reward_function == "sortino":
            return self._calculate_sortino_ratio()
        elif self.reward_function == "profit_factor":
            return self._calculate_profit_factor()
        elif self.reward_function == "trading_focused":
            return self._calculate_trading_focused_reward(current_capital, prev_capital)
        else:
            return current_capital - prev_capital  # Default to P&L

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on recent returns."""
        if len(self.returns_history) < 10:  # Need minimum history
            return 0.0

        returns = np.array(self.returns_history[-30:])  # Use last 30 returns
        if np.std(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return
        return sharpe * 10  # Scale for RL training

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(self.returns_history) < 10:
            return 0.0

        returns = np.array(self.returns_history[-30:])
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return mean_return * 10  # No downside, return scaled mean

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        sortino = mean_return / downside_std
        return sortino * 10  # Scale for RL training

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.returns_history) < 5:
            return 0.0

        returns = np.array(self.returns_history[-30:])
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return gross_profit * 10 if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return (profit_factor - 1.0) * 5  # Center around 0 and scale

    def _calculate_trading_focused_reward(self, current_capital: float, prev_capital: float) -> float:
        """Calculate reward focused on key trading metrics: profit factor, drawdown, win rate."""
        base_reward = current_capital - prev_capital

        # Get current trade history for real-time metrics
        trade_history = self.engine.get_trade_history()

        if len(trade_history) < 2:
            return base_reward  # Not enough trades for metrics

        # Calculate real-time profit factor
        recent_trades = trade_history[-10:]  # Last 10 trades
        gross_profit = sum(trade['pnl'] for trade in recent_trades if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in recent_trades if trade['pnl'] < 0))

        profit_factor_bonus = 0.0
        if gross_loss > 0:
            pf = gross_profit / gross_loss
            if pf > 1.5:  # Good profit factor
                profit_factor_bonus = (pf - 1.0) * 10
            elif pf < 0.8:  # Poor profit factor
                profit_factor_bonus = (pf - 1.0) * 20  # Penalty

        # Calculate real-time win rate
        win_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades)
        win_rate_bonus = 0.0
        if win_rate > 0.6:  # Good win rate
            win_rate_bonus = (win_rate - 0.5) * 20
        elif win_rate < 0.4:  # Poor win rate
            win_rate_bonus = (win_rate - 0.5) * 30  # Penalty

        # Calculate drawdown penalty using existing equity_history
        if len(self.equity_history) > 5:
            recent_capitals = self.equity_history[-20:]  # Last 20 steps
            peak = max(recent_capitals)
            current_dd = (peak - current_capital) / peak if peak > 0 else 0
            drawdown_penalty = -current_dd * 100 if current_dd > 0.05 else 0  # Penalty if DD > 5%
        else:
            drawdown_penalty = 0.0

        total_reward = base_reward + profit_factor_bonus + win_rate_bonus + drawdown_penalty
        return total_reward

    def _apply_reward_shaping(self, base_reward: float, action_type: int, current_capital: float, prev_capital: float) -> float:
        """Apply reward shaping to guide agent behavior."""
        shaped_reward = base_reward

        # Penalty for idleness (holding no position for too long)
        if action_type == 4 and self.idle_steps > 10:  # HOLD for more than 10 steps
            if not self.engine.get_account_state()['is_position_open']:
                shaped_reward -= 0.1 * (self.idle_steps - 10)  # Increasing penalty

        # Bonus for realizing profits
        if action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
            pnl_change = current_capital - prev_capital
            if pnl_change > 0:
                shaped_reward += 0.5  # Bonus for profitable trade

        # Penalty for over-trading (too many trades in short period)
        if self.trade_count > 0:
            recent_trade_rate = self.trade_count / max(1, self.current_step - self.lookback_window + 1)
            if recent_trade_rate > 0.5:  # More than 50% of steps are trades
                shaped_reward -= 0.2  # Over-trading penalty

        # Trailing stop reward shaping (AC 1.3.9)
        trailing_reward = self._calculate_trailing_stop_reward_shaping(action_type)
        shaped_reward += trailing_reward

        return shaped_reward

    def _calculate_trailing_stop_reward_shaping(self, action_type: int) -> float:
        """Calculate reward shaping for trailing stops."""
        if not self.engine._is_position_open:
            return 0.0

        current_trailing_stop = self.engine._trailing_stop_price
        reward_adjustment = 0.0

        # Bonus for holding profitable position as trailing stop improves
        if action_type == 4:  # HOLD action
            if self.previous_trailing_stop > 0 and current_trailing_stop > 0:
                # For long positions, trailing stop moving up is good
                if self.engine._current_position_quantity > 0:
                    if current_trailing_stop > self.previous_trailing_stop:
                        reward_adjustment += 0.1  # Bonus for improving trailing stop
                # For short positions, trailing stop moving down is good
                else:
                    if current_trailing_stop < self.previous_trailing_stop:
                        reward_adjustment += 0.1  # Bonus for improving trailing stop

        # Penalty for closing profitable position prematurely when trend is strong
        elif action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
            if hasattr(self, 'data') and self.current_step < len(self.data):
                safe_step = min(self.current_step, len(self.data) - 1)
                current_price = self.data['close'].iloc[safe_step]
                distance_to_trail = self._calculate_distance_to_trail(current_price)

                # If distance to trail is large (trend is strong) and position is profitable
                if distance_to_trail > 0.02:  # More than 2% away from trailing stop
                    unrealized_pnl = self.engine.get_account_state()['unrealized_pnl']
                    if unrealized_pnl > 0:  # Position is profitable
                        reward_adjustment -= 0.3  # Penalty for premature closing

        # Update previous trailing stop for next iteration
        self.previous_trailing_stop = current_trailing_stop

        return reward_adjustment

    def _check_termination_conditions(self, current_capital: float) -> tuple:
        """Check if episode should terminate due to risk management conditions or strategic episode end."""
        # Check strategic episode end (if using data feeding strategy)
        if hasattr(self, 'episode_end_step') and self.current_step >= self.episode_end_step:
            return True, f"strategic_episode_end_step_{self.current_step}"

        # Check if we've reached the end of available data
        if self.current_step >= len(self.data) - 1:
            return True, f"end_of_data_step_{self.current_step}"

        # Update peak equity
        if current_capital > self.peak_equity:
            self.peak_equity = current_capital

        # Check maximum drawdown
        current_drawdown = (self.peak_equity - current_capital) / self.peak_equity
        if current_drawdown > self.max_drawdown_pct:
            return True, f"max_drawdown_exceeded_{current_drawdown:.2%}"

        # Check insufficient capital (AC 1.3.7)
        # Estimate minimum capital needed for one trade
        if hasattr(self, 'data') and self.current_step < len(self.data):
            safe_step = min(self.current_step, len(self.data) - 1)
            current_price = self.data['close'].iloc[safe_step]

            # Get ATR if available, otherwise use a default volatility estimate
            if 'atr' in self.data.columns:
                current_atr = self.data['atr'].iloc[safe_step]
            else:
                # Fallback: use 2% of current price as volatility estimate
                current_atr = current_price * 0.02

            proxy_premium = self._calculate_proxy_premium(current_price, current_atr)

            if self.instrument.type == "OPTION":
                min_trade_cost = (proxy_premium * 1 * self.instrument.lot_size) + self.engine.BROKERAGE_ENTRY
            else:
                min_trade_cost = (current_price * 1 * self.instrument.lot_size) + self.engine.BROKERAGE_ENTRY

            if current_capital < min_trade_cost:
                return True, f"insufficient_capital_{current_capital:.2f}_needed_{min_trade_cost:.2f}"

        return False, None

    def _calculate_proxy_premium(self, current_price: float, atr: float) -> float:
        """
        Calculate realistic proxy premium for options simulation.

        Even for futures trading, we simulate the cost as if trading options
        to provide realistic trading costs and risk management.
        """
        # Calculate volatility-based premium regardless of instrument type
        # ATR represents daily volatility, scale it to option premium
        volatility_factor = atr / current_price

        # Base premium: 1.5% of underlying (typical for ATM options)
        base_premium_pct = 0.015

        # Adjust based on volatility (higher volatility = higher premium)
        # Scale volatility factor to reasonable range (0.5x to 3x base premium)
        volatility_multiplier = max(0.5, min(3.0, 1 + (volatility_factor * 10)))

        # Calculate final premium percentage
        premium_percentage = base_premium_pct * volatility_multiplier

        # Ensure premium is within realistic bounds (0.5% to 5%)
        premium_percentage = max(0.005, min(0.05, premium_percentage))

        proxy_premium = current_price * premium_percentage

        # Minimum premium based on instrument (Bank Nifty vs Nifty)
        if "Bank_Nifty" in self.symbol:
            min_premium = 50.0  # Bank Nifty options typically cost at least ₹50
        else:
            min_premium = 25.0  # Nifty options typically cost at least ₹25

        return max(proxy_premium, min_premium)

    def _get_observation(self) -> np.ndarray:
        # Get market data for the lookback window
        start_index = self.current_step - self.lookback_window + 1
        end_index = self.current_step + 1

        # Use ALL available columns for RL training (except datetime if present)
        feature_columns = [col for col in self.data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp']]

        # Ensure we don't go out of bounds
        if start_index < 0:
            # Pad with zeros if not enough history
            padding_needed = abs(start_index)
            market_data = np.zeros((self.lookback_window, len(feature_columns)))
            actual_data = self.data[feature_columns].iloc[0:end_index].values
            market_data[padding_needed:] = actual_data
        else:
            market_data = self.data[feature_columns].iloc[start_index:end_index].values

        # Get account state with bounds checking
        safe_step = min(self.current_step, len(self.data) - 1)
        current_price = self.data['close'].iloc[safe_step]
        account_state = self.engine.get_account_state(current_price=current_price)

        # Calculate distance to trailing stop
        distance_to_trail = self._calculate_distance_to_trail(current_price)

        # Create raw observation with proper validation
        try:
            # Ensure market_data is properly shaped
            market_features = market_data.flatten().astype(np.float32)

            # Create account state features
            account_features = np.array([
                float(account_state['capital']),
                float(account_state['current_position_quantity']),
                float(account_state['current_position_entry_price']),
                float(account_state['unrealized_pnl']),
                1.0 if account_state['is_position_open'] else 0.0,
                float(distance_to_trail)
            ], dtype=np.float32)

            # Concatenate all features
            raw_observation = np.concatenate([market_features, account_features])

            # Ensure no invalid values
            if not np.isfinite(raw_observation).all():
                # Replace invalid values with zeros
                raw_observation = np.where(np.isfinite(raw_observation), raw_observation, 0.0)

            # Ensure consistent shape
            if self.observation_dim is not None and len(raw_observation) != self.observation_dim:
                if len(raw_observation) < self.observation_dim:
                    # Pad with zeros
                    raw_observation = np.pad(raw_observation, (0, self.observation_dim - len(raw_observation)), 'constant')
                else:
                    # Truncate
                    raw_observation = raw_observation[:self.observation_dim]

            # Store raw observation for statistics
            self.observation_history.append(raw_observation.copy())

            # Apply z-score normalization
            normalized_observation = self._apply_zscore_normalization(raw_observation)

            return normalized_observation.astype(np.float32)

        except Exception as e:
            print(f"Error creating observation: {e}")
            # Return a zero observation with correct shape
            if self.observation_dim is not None:
                return np.zeros(self.observation_dim, dtype=np.float32)
            else:
                return np.zeros(1246, dtype=np.float32)  # Default fallback

    def _apply_zscore_normalization(self, observation: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to observation."""
        if len(self.observation_history) < 10:  # Need minimum history
            return observation  # Return raw observation initially

        try:
            # Use recent history for statistics (last 100 observations)
            # Ensure all observations have the same shape
            recent_observations = []
            target_shape = observation.shape

            for obs in self.observation_history[-100:]:
                if obs.shape == target_shape:
                    recent_observations.append(obs)
                else:
                    # Skip observations with different shapes
                    continue

            if len(recent_observations) < 5:  # Need minimum valid history
                return observation

            recent_history = np.stack(recent_observations, axis=0)

            # Calculate mean and std for each feature
            means = np.mean(recent_history, axis=0)
            stds = np.std(recent_history, axis=0)

            # Avoid division by zero
            stds = np.where(stds == 0, 1.0, stds)

            # Apply z-score normalization
            normalized = (observation - means) / stds

            # Clip extreme values to prevent instability
            normalized = np.clip(normalized, -5.0, 5.0)

            return normalized

        except Exception as e:
            print(f"Warning: Z-score normalization failed: {e}, returning raw observation")
            return observation

    def _calculate_distance_to_trail(self, current_price: float) -> float:
        """Calculate normalized distance to trailing stop."""
        if not self.engine._is_position_open:
            return 0.0  # No position, no trailing stop

        trailing_stop_price = self.engine._trailing_stop_price
        if trailing_stop_price == 0:
            return 0.0  # No trailing stop set

        # Calculate distance as percentage of current price
        if self.engine._current_position_quantity > 0:  # Long position
            distance = (current_price - trailing_stop_price) / current_price
        else:  # Short position
            distance = (trailing_stop_price - current_price) / current_price

        # Normalize to reasonable range [-1, 1]
        return np.clip(distance, -1.0, 1.0)
