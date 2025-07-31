import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
import random
from enum import Enum

from src.utils.data_loader import DataLoader
from src.backtesting.engine import BacktestingEngine
from src.config.instrument import Instrument
from src.utils.instrument_loader import load_instruments
from src.utils.data_feeding_strategy import DataFeedingStrategyManager, FeedingStrategy
from src.config.config import RISK_REWARD_CONFIG, INITIAL_CAPITAL

# Configure logger
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading environment modes."""
    TRAINING = "training"
    BACKTESTING = "backtesting"
    LIVE = "live"

class TradingEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = None, symbol: str = None, initial_capital: float = None,
                 lookback_window: int = 50, trailing_stop_percentage: float = None,
                 reward_function: str = "pnl", episode_length: int = 1000,
                 use_streaming: bool = True, mode: TradingMode = TradingMode.TRAINING,
                 external_data: pd.DataFrame = None, smart_action_filtering: bool = False):
        super(TradingEnv, self).__init__()

        # Mode and data handling
        self.mode = mode
        self.external_data = external_data

        # For TRAINING mode, require data_loader and symbol
        if mode == TradingMode.TRAINING:
            if data_loader is None or symbol is None:
                raise ValueError("TRAINING mode requires data_loader and symbol")
            self.data_loader = data_loader
            self.symbol = symbol
        else:
            # For BACKTESTING/LIVE modes, external_data can be provided
            self.data_loader = data_loader  # Optional for backtesting/live
            self.symbol = symbol or "INDEX"  # Default symbol for backtesting/live

        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.lookback_window = lookback_window
        self.reward_function = reward_function
        self.episode_length = episode_length
        self.use_streaming = use_streaming and (mode == TradingMode.TRAINING)  # Only use streaming for training
        self.smart_action_filtering = smart_action_filtering  # Prevent redundant position attempts

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

        # Handle different modes for instrument loading
        if self.mode == TradingMode.TRAINING:
            # Get base symbol (remove timeframe suffix)
            self.base_symbol = self.data_loader.get_base_symbol(self.symbol)
            if self.base_symbol not in self.instruments:
                raise ValueError(f"Instrument {self.base_symbol} not found in instruments.yaml (original symbol: {self.symbol})")
            self.instrument = self.instruments[self.base_symbol]
        else:
            # For BACKTESTING/LIVE modes, use generic data instrument
            # Create a simple generic instrument for point-based trading
            self.base_symbol = "GENERIC_DATA"
            self.instrument = Instrument(
                symbol="GENERIC_DATA",
                lot_size=1,  # 1 point = 1 unit
                tick_size=0.01
            )

        # Use centralized trailing stop configuration if not provided
        if trailing_stop_percentage is None:
            trailing_stop_percentage = RISK_REWARD_CONFIG['trailing_stop_percentage']

        self.engine = BacktestingEngine(self.initial_capital, self.instrument, trailing_stop_percentage)
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

        # Reward normalization parameters for different instrument types
        self.reward_normalization_factor = self._calculate_reward_normalization_factor(symbol)

    def reset(self, performance_metrics: Optional[Dict] = None) -> np.ndarray:
        """Reset environment and load data segment for new episode with intelligent data feeding."""

        # Handle data loading based on mode
        if self.mode == TradingMode.TRAINING:
            # TRAINING mode: Use data_loader (current behavior)
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

        elif self.mode == TradingMode.BACKTESTING:
            # BACKTESTING mode: Use external data
            if self.external_data is None:
                raise ValueError("BACKTESTING mode requires external_data")
            self.data = self.external_data.copy()
            self.feeding_strategy_manager = None  # No episodes in backtesting
            logger.info(f"🎯 BACKTESTING mode: Using external data with {len(self.data)} rows")

        elif self.mode == TradingMode.LIVE:
            # LIVE mode: Use external data (placeholder)
            if self.external_data is None:
                raise ValueError("LIVE mode requires external_data")
            self.data = self.external_data.copy()
            self.feeding_strategy_manager = None  # No episodes in live trading
            logger.info(f"🎯 LIVE mode: Using external data with {len(self.data)} rows")

        # Set observation space dimensions based on actual data
        if self.observation_space is None:
            # CRITICAL: Exclude raw OHLC prices for universal model - only use derived features
            # NOTE: datetime_epoch is INCLUDED as a feature for temporal learning
            excluded_columns = ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in self.data.columns if col.lower() not in [x.lower() for x in excluded_columns]]
            self.features_per_step = len(feature_columns)
            self.observation_dim = (self.lookback_window * self.features_per_step) + self.account_state_features + self.trailing_features
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
            logger.info(f"🔧 Universal Model: Excluded raw OHLC prices, using {len(feature_columns)} derived features")
            logger.info(f"📊 Observation space set: {self.features_per_step} features per step, total dim: {self.observation_dim}")
            logger.info(f"🎯 Feature columns: {feature_columns[:10]}...")  # Show first 10 features

        self.engine.reset()

        # Handle episode positioning based on mode
        if self.mode == TradingMode.TRAINING:
            # TRAINING mode: Use intelligent data feeding strategy for episode start position
            if self.feeding_strategy_manager is not None:
                start_idx, end_idx = self.feeding_strategy_manager.get_next_episode_data(performance_metrics)
                # Ensure we have enough data for lookback
                self.current_step = max(start_idx + self.lookback_window - 1, self.lookback_window - 1)
                self.episode_end_step = min(end_idx, len(self.data) - 1)
                logger.debug(f"Strategic episode: steps {self.current_step} to {self.episode_end_step} "
                            f"(strategy: {self.feeding_strategy_manager.current_strategy.value})")
            else:
                # Fallback to random positioning for training
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
        else:
            # BACKTESTING/LIVE modes: Sequential processing from start
            self.current_step = self.lookback_window - 1  # Start after lookback window
            self.episode_end_step = len(self.data) - 1  # Process all data
            logger.debug(f"Sequential processing: steps {self.current_step} to {self.episode_end_step}")

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
            # PRESERVE datetime index - do NOT reset it!
            # The datetime index contains readable timestamps for logging
            logging.info(f"Loaded data segment for {self.symbol}: rows {self.current_episode_start}-{self.current_episode_end}")
            logging.info(f"✅ Preserved datetime index with {len(self.data)} rows")

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

        # Use point-based calculation for all modes (no proxy premium)
        # Direct price for index trading - no option premium complexity
        proxy_premium = current_price

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

            # Import capital-aware quantity adjustment
            from src.utils.capital_aware_quantity import adjust_quantity_for_capital

            # Calculate dynamically adjusted quantity based on available capital
            adjusted_quantity = adjust_quantity_for_capital(
                predicted_quantity=quantity,
                available_capital=available_capital,
                current_price=current_price,
                instrument=self.instrument
            )

            # Log quantity adjustment if it was changed
            if adjusted_quantity != int(round(quantity)):
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

        # Smart action filtering to prevent redundant position attempts
        if self.smart_action_filtering:
            if action_type == 0 and current_position != 0:  # BUY_LONG when already have position
                action_type = 4  # Convert to HOLD
            elif action_type == 1 and current_position != 0:  # SELL_SHORT when already have position
                action_type = 4  # Convert to HOLD

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

        # Add quantity prediction feedback reward
        quantity_reward = self._calculate_quantity_feedback_reward(action, prev_capital)
        shaped_reward += quantity_reward

        # Apply normalization for universal model training across different instruments
        reward = self._normalize_reward(shaped_reward)

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

        # Create info dictionary with termination reason and exit reason
        info = {}
        if termination_reason:
            info["termination_reason"] = termination_reason

        # Get exit reason from the latest decision log entry
        if self.engine._decision_log:
            latest_decision = self.engine._decision_log[-1]
            exit_reason = latest_decision.get('exit_reason')
            if exit_reason:
                info["exit_reason"] = exit_reason

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
            # Include unrealized P&L in reward calculation
            account_state = self.engine.get_account_state()
            total_pnl_change = (current_capital + account_state['unrealized_pnl']) - prev_capital
            return total_pnl_change
        elif self.reward_function == "sharpe":
            return self._calculate_sharpe_ratio()
        elif self.reward_function == "sortino":
            return self._calculate_sortino_ratio()
        elif self.reward_function == "profit_factor":
            return self._calculate_profit_factor()
        elif self.reward_function == "trading_focused":
            return self._calculate_trading_focused_reward(current_capital, prev_capital)
        elif self.reward_function == "enhanced_trading_focused":
            return self._calculate_enhanced_trading_focused_reward(current_capital, prev_capital)
        else:
            # Default to P&L including unrealized gains/losses
            account_state = self.engine.get_account_state()
            total_pnl_change = (current_capital + account_state['unrealized_pnl']) - prev_capital
            return total_pnl_change

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
        # Include unrealized P&L in base reward calculation
        account_state = self.engine.get_account_state()
        base_reward = (current_capital + account_state['unrealized_pnl']) - prev_capital

        # Get current trade history for real-time metrics
        trade_history = self.engine.get_trade_history()

        if len(trade_history) < 2:
            return base_reward  # Not enough trades for metrics

        # Calculate real-time profit factor (only for closing trades)
        recent_trades = trade_history[-20:]  # Last 20 trades for better sample size
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']

        profit_factor_bonus = 0.0
        if len(closing_trades) >= 5:  # Need minimum trades for meaningful PF
            gross_profit = sum(trade['pnl'] for trade in closing_trades if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in closing_trades if trade['pnl'] < 0))

            if gross_loss > 0:
                pf = gross_profit / gross_loss
                # ENHANCED: Much stronger focus on profit factor since profits matter most
                if pf > 3.0:  # Exceptional profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # Massive bonus for exceptional performance
                elif pf > 2.0:  # Excellent profit factor
                    profit_factor_bonus = (pf - 1.0) * 75   # Very high bonus
                elif pf > 1.5:  # Good profit factor
                    profit_factor_bonus = (pf - 1.0) * 50   # High bonus
                elif pf > 1.2:  # Decent profit factor
                    profit_factor_bonus = (pf - 1.0) * 30   # Moderate bonus
                elif pf > 1.0:  # Barely profitable
                    profit_factor_bonus = (pf - 1.0) * 15   # Small bonus
                elif pf < 0.5:  # Terrible profit factor
                    profit_factor_bonus = (pf - 1.0) * 80   # Severe penalty
                elif pf < 0.7:  # Very poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 60   # High penalty
                else:  # Poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 40   # Moderate penalty

        # Calculate real-time win rate (only for closing trades)
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']
        if closing_trades:
            win_rate = sum(1 for trade in closing_trades if trade['pnl'] > 0) / len(closing_trades)
            win_rate_bonus = 0.0
            # REDUCED: Lower emphasis on win rate since profit factor matters more
            if win_rate > 0.7:  # Excellent win rate
                win_rate_bonus = (win_rate - 0.5) * 30  # Moderate bonus
            elif win_rate > 0.6:  # Good win rate
                win_rate_bonus = (win_rate - 0.5) * 20  # Small bonus
            elif win_rate > 0.5:  # Decent win rate
                win_rate_bonus = (win_rate - 0.5) * 10  # Very small bonus
            elif win_rate < 0.3:  # Very poor win rate
                win_rate_bonus = (win_rate - 0.5) * 25  # Moderate penalty
            elif win_rate < 0.4:  # Poor win rate
                win_rate_bonus = (win_rate - 0.5) * 15  # Small penalty
        else:
            win_rate_bonus = 0.0

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

    def _calculate_enhanced_trading_focused_reward(self, current_capital: float, prev_capital: float) -> float:
        """Enhanced reward function targeting 70%+ win rate with better profit factor."""
        # Include unrealized P&L in base reward calculation
        account_state = self.engine.get_account_state()
        base_reward = (current_capital + account_state['unrealized_pnl']) - prev_capital

        # Get current trade history for real-time metrics
        trade_history = self.engine.get_trade_history()

        if len(trade_history) < 2:
            return base_reward  # Not enough trades for metrics

        # Calculate real-time profit factor with enhanced bonuses
        recent_trades = trade_history[-30:]  # Increased sample size for better metrics
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']

        profit_factor_bonus = 0.0
        if len(closing_trades) >= 3:  # Reduced minimum for faster feedback
            gross_profit = sum(trade['pnl'] for trade in closing_trades if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in closing_trades if trade['pnl'] < 0))

            if gross_loss > 0:
                pf = gross_profit / gross_loss
                # ULTRA-ENHANCED: Massive focus on profit factor for 70%+ performance
                if pf > 4.0:  # Exceptional profit factor
                    profit_factor_bonus = (pf - 1.0) * 200  # Massive bonus
                elif pf > 3.0:  # Excellent profit factor
                    profit_factor_bonus = (pf - 1.0) * 150  # Very high bonus
                elif pf > 2.5:  # Very good profit factor
                    profit_factor_bonus = (pf - 1.0) * 120  # High bonus
                elif pf > 2.0:  # Good profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # Good bonus
                elif pf > 1.5:  # Decent profit factor
                    profit_factor_bonus = (pf - 1.0) * 80   # Moderate bonus
                elif pf > 1.2:  # Barely good
                    profit_factor_bonus = (pf - 1.0) * 50   # Small bonus
                elif pf > 1.0:  # Barely profitable
                    profit_factor_bonus = (pf - 1.0) * 25   # Tiny bonus
                elif pf < 0.4:  # Terrible profit factor
                    profit_factor_bonus = (pf - 1.0) * 150  # Severe penalty
                elif pf < 0.6:  # Very poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # High penalty
                else:  # Poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 75   # Moderate penalty

        # Enhanced win rate calculation targeting 70%+
        win_rate_bonus = 0.0
        if closing_trades:
            win_rate = sum(1 for trade in closing_trades if trade['pnl'] > 0) / len(closing_trades)
            # ENHANCED: Strong bonuses for high win rates (targeting 70%+)
            if win_rate >= 0.8:  # Exceptional win rate (80%+)
                win_rate_bonus = (win_rate - 0.5) * 100  # Massive bonus
            elif win_rate >= 0.7:  # Target win rate (70%+)
                win_rate_bonus = (win_rate - 0.5) * 80   # High bonus
            elif win_rate >= 0.6:  # Good win rate
                win_rate_bonus = (win_rate - 0.5) * 60   # Moderate bonus
            elif win_rate >= 0.5:  # Decent win rate
                win_rate_bonus = (win_rate - 0.5) * 40   # Small bonus
            elif win_rate < 0.3:  # Very poor win rate
                win_rate_bonus = (win_rate - 0.5) * 60   # High penalty
            elif win_rate < 0.4:  # Poor win rate
                win_rate_bonus = (win_rate - 0.5) * 40   # Moderate penalty

        # Enhanced drawdown penalty for better risk management
        drawdown_penalty = 0.0
        if len(self.equity_history) > 5:
            recent_capitals = self.equity_history[-30:]  # Longer history for better DD calculation
            peak = max(recent_capitals)
            current_dd = (peak - current_capital) / peak if peak > 0 else 0
            # Stricter drawdown penalties
            if current_dd > 0.1:  # 10%+ drawdown
                drawdown_penalty = -current_dd * 200  # Severe penalty
            elif current_dd > 0.05:  # 5%+ drawdown
                drawdown_penalty = -current_dd * 100  # High penalty
            elif current_dd > 0.02:  # 2%+ drawdown
                drawdown_penalty = -current_dd * 50   # Moderate penalty

        # Risk-reward ratio bonus
        risk_reward_bonus = 0.0
        if len(closing_trades) >= 3:
            avg_win = np.mean([trade['pnl'] for trade in closing_trades if trade['pnl'] > 0]) if any(trade['pnl'] > 0 for trade in closing_trades) else 0
            avg_loss = abs(np.mean([trade['pnl'] for trade in closing_trades if trade['pnl'] < 0])) if any(trade['pnl'] < 0 for trade in closing_trades) else 1

            if avg_loss > 0:
                risk_reward_ratio = avg_win / avg_loss
                if risk_reward_ratio > 3.0:  # Excellent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 20
                elif risk_reward_ratio > 2.0:  # Good risk-reward
                    risk_reward_bonus = risk_reward_ratio * 15
                elif risk_reward_ratio > 1.5:  # Decent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 10

        total_reward = base_reward + profit_factor_bonus + win_rate_bonus + drawdown_penalty + risk_reward_bonus
        return total_reward

    def _apply_reward_shaping(self, base_reward: float, action_type: int, current_capital: float, prev_capital: float) -> float:
        """Apply reward shaping to guide agent behavior."""
        shaped_reward = base_reward

        # Penalty for idleness (holding no position for too long)
        if action_type == 4 and self.idle_steps > 10:  # HOLD for more than 10 steps
            if not self.engine.get_account_state()['is_position_open']:
                shaped_reward -= 0.1 * (self.idle_steps - 10)  # Increasing penalty

        # Enhanced bonus/penalty for trade outcomes
        if action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
            pnl_change = current_capital - prev_capital
            if pnl_change > 0:
                # Larger bonus for profitable trades to encourage winning
                shaped_reward += min(pnl_change * 0.01, 5.0)  # Scale with profit, cap at 5
            else:
                # Penalty for losing trades to discourage bad exits
                shaped_reward += max(pnl_change * 0.005, -2.0)  # Scale with loss, cap at -2

        # Penalty for over-trading (too many trades in short period)
        if self.trade_count > 0:
            recent_trade_rate = self.trade_count / max(1, self.current_step - self.lookback_window + 1)
            if recent_trade_rate > 0.3:  # More than 30% of steps are trades
                shaped_reward -= 0.5 * (recent_trade_rate - 0.3)  # Scaled over-trading penalty

        # Bonus for maintaining profitable positions
        account_state = self.engine.get_account_state()
        if account_state['is_position_open'] and action_type == 4:  # HOLD with open position
            unrealized_pnl = account_state['unrealized_pnl']
            if unrealized_pnl > 0:
                # Small bonus for holding profitable positions
                shaped_reward += min(unrealized_pnl * 0.001, 0.5)  # Scale with unrealized profit

        # Trailing stop reward shaping (AC 1.3.9)
        trailing_reward = self._calculate_trailing_stop_reward_shaping(action_type)
        shaped_reward += trailing_reward

        return shaped_reward

    def _calculate_quantity_feedback_reward(self, action, available_capital: float) -> float:
        """
        Calculate reward feedback for quantity predictions to guide better capital utilization.

        Args:
            action: The action taken (action_type, quantity)
            available_capital: Available capital at the time of action

        Returns:
            Reward component based on quantity prediction quality
        """
        if isinstance(action, (list, np.ndarray, tuple)) and len(action) >= 2:
            action_type = int(action[0])
            predicted_quantity = float(action[1])
        else:
            return 0.0  # No quantity feedback for non-trading actions

        # Only provide feedback for trading actions (not HOLD)
        if action_type == 4:  # HOLD action
            return 0.0

        if predicted_quantity <= 0:
            return 0.0

        # Get current price for cost calculation
        current_price = self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        if current_price <= 0:
            return 0.0

        # Import capital-aware quantity calculation
        from src.utils.capital_aware_quantity import CapitalAwareQuantitySelector
        selector = CapitalAwareQuantitySelector()

        # Calculate what the optimal quantity would be
        max_affordable = selector.adjust_quantity_for_capital(
            predicted_quantity=5.0,  # Max possible
            available_capital=available_capital,
            current_price=current_price,
            instrument=self.instrument
        )

        # Calculate capital utilization efficiency
        if max_affordable > 0:
            # Reward efficient capital utilization
            actual_quantity = min(int(predicted_quantity), max_affordable)
            utilization_ratio = actual_quantity / max_affordable

            # Reward good utilization (0.6-0.9 is optimal, not too conservative, not too aggressive)
            if 0.6 <= utilization_ratio <= 0.9:
                quantity_reward = 0.1 * utilization_ratio  # Small positive reward
            elif utilization_ratio > 0.9:
                quantity_reward = 0.05  # Slightly less reward for being too aggressive
            else:
                quantity_reward = 0.02 * utilization_ratio  # Small reward for conservative approach

            # Penalize predictions that exceed affordable quantity
            if predicted_quantity > max_affordable:
                over_prediction_penalty = -0.05 * (predicted_quantity - max_affordable) / max_affordable
                quantity_reward += over_prediction_penalty

            return quantity_reward

        return 0.0

    def _calculate_reward_normalization_factor(self, symbol: str) -> float:
        """Calculate normalization factor based on instrument type and typical price ranges."""
        symbol_lower = symbol.lower()

        # Index instruments (typically higher values, more volatile)
        if any(keyword in symbol_lower for keyword in ['nifty', 'sensex', 'bank_nifty', 'fin_nifty']):
            # Indices typically range from 15,000-25,000+ points
            # Normalize to make rewards comparable across different price levels
            return 0.0001  # Scale down large index movements

        # Stock instruments (typically lower values per share but higher lot sizes)
        elif any(keyword in symbol_lower for keyword in ['reliance', 'sbi', 'hdfc', 'icici', 'tcs', 'infy']):
            # Individual stocks typically range from 100-3000 per share
            # But lot sizes can vary significantly
            return 0.001  # Moderate scaling for stocks

        # Default for unknown instruments
        else:
            return 0.001  # Conservative default scaling

    def _normalize_reward(self, reward: float) -> float:
        """Apply normalization to make rewards consistent across different instruments."""
        return reward * self.reward_normalization_factor

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

        # Check if we've reached the end of available data
        if self.current_step >= len(self.data) - 1:
            return True, f"end_of_data_step_{self.current_step}"

        if self.mode == TradingMode.TRAINING:
            # TRAINING mode: Use episode-based termination
            # Check strategic episode end (if using data feeding strategy)
            if hasattr(self, 'episode_end_step') and self.current_step >= self.episode_end_step:
                return True, f"strategic_episode_end_step_{self.current_step}"

            # Update peak equity
            if current_capital > self.peak_equity:
                self.peak_equity = current_capital

            # Check maximum drawdown
            current_drawdown = (self.peak_equity - current_capital) / self.peak_equity
            if current_drawdown > self.max_drawdown_pct:
                return True, f"max_drawdown_exceeded_{current_drawdown:.2%}"

            # No capital constraints for index trading - removed insufficient capital check

        else:
            # BACKTESTING/LIVE modes: No early termination, process all data
            # Update peak equity for tracking
            if current_capital > self.peak_equity:
                self.peak_equity = current_capital

        return False, None


    def get_backtest_results(self) -> Dict:
        """Get comprehensive backtesting results for BACKTESTING/LIVE modes."""
        if self.mode == TradingMode.TRAINING:
            return {}  # No backtest results for training

        account_state = self.engine.get_account_state()

        # Calculate performance metrics
        total_return = (account_state['capital'] - self.initial_capital) / self.initial_capital
        max_drawdown = (self.peak_equity - account_state['capital']) / self.peak_equity if self.peak_equity > 0 else 0

        # Get trade history from engine
        trades = self.engine.get_trade_history()

        results = {
            'mode': self.mode.value,
            'symbol': self.symbol,
            'initial_capital': self.initial_capital,
            'final_capital': account_state['capital'],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'peak_equity': self.peak_equity,
            'total_trades': self.engine._trade_count,
            'equity_curve': self.equity_history.copy(),
            'trades': trades,
            'current_position': account_state['current_position_quantity'],
            'total_steps': self.current_step + 1,
            'data_length': len(self.data) if self.data is not None else 0
        }

        return results

    def _get_observation(self) -> np.ndarray:
        # Get market data for the lookback window
        start_index = self.current_step - self.lookback_window + 1
        end_index = self.current_step + 1

        # Use ALL available columns for RL training (except datetime if present)
        # NOTE: datetime_epoch is INCLUDED as a feature for temporal learning
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

            # Apply selective z-score normalization (exclude datetime_epoch)
            normalized_observation = self._apply_selective_zscore_normalization(raw_observation)

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

    def _apply_selective_zscore_normalization(self, observation: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to observation, excluding datetime_epoch features."""
        if len(self.observation_history) < 10:  # Need minimum history
            return observation  # Return raw observation initially

        try:
            # Identify datetime_epoch feature indices
            datetime_epoch_indices = self._get_datetime_epoch_indices()

            # Use recent history for statistics (last 100 observations)
            recent_observations = []
            target_shape = observation.shape

            for obs in self.observation_history[-100:]:
                if obs.shape == target_shape:
                    recent_observations.append(obs)

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

            # CRITICAL: Keep datetime_epoch features unnormalized
            for idx in datetime_epoch_indices:
                if idx < len(normalized):
                    normalized[idx] = observation[idx]  # Keep original datetime_epoch value

            # Clip extreme values to prevent instability (except datetime_epoch)
            for i in range(len(normalized)):
                if i not in datetime_epoch_indices:
                    normalized[i] = np.clip(normalized[i], -5.0, 5.0)

            return normalized

        except Exception as e:
            print(f"Warning: Selective z-score normalization failed: {e}, returning raw observation")
            return observation

    def _get_datetime_epoch_indices(self) -> list:
        """Get indices of datetime_epoch features in the flattened observation."""
        try:
            # Find datetime_epoch column index in feature_columns
            excluded_columns = ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in self.data.columns if col.lower() not in [x.lower() for x in excluded_columns]]

            datetime_epoch_indices = []
            for i, col in enumerate(feature_columns):
                if 'datetime_epoch' in col.lower():
                    # Calculate indices in flattened observation
                    # Each timestep contributes len(feature_columns) features
                    for t in range(self.lookback_window):
                        idx = t * len(feature_columns) + i
                        datetime_epoch_indices.append(idx)

            return datetime_epoch_indices
        except Exception as e:
            print(f"Warning: Could not identify datetime_epoch indices: {e}")
            return []

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
