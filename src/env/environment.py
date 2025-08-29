import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging
import random

from src.utils.data_loader import DataLoader
from src.env.engine import BacktestingEngine
from src.config.instrument import Instrument
from src.utils.instrument_loader import load_instruments
from src.utils.data_feeding_strategy import DataFeedingStrategyManager, FeedingStrategy
from src.utils.config_loader import ConfigLoader

# Import refactored modules
from .trading_mode import TradingMode
from .reward_calculator import RewardCalculator
from .observation_handler import ObservationHandler
from .termination_manager import TerminationManager

# Configure logger
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    def __init__(self, data_loader: DataLoader = None, symbol: str = None, initial_capital: float = None,
                 lookback_window: int = 50, trailing_stop_percentage: float = None,
                 reward_function: str = "pnl", episode_length: int = 1000,
                 use_streaming: bool = True, mode: TradingMode = TradingMode.TRAINING,
                 external_data: pd.DataFrame = None, smart_action_filtering: bool = False):
        super(TradingEnv, self).__init__()
        
        # Load centralized configuration
        config_loader = ConfigLoader()
        config = config_loader.get_config()

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

        self.initial_capital = initial_capital or config.get('environment', {}).get('initial_capital', 100000.0)
        self.lookback_window = lookback_window
        self.reward_function = reward_function
        self.episode_length = episode_length
        self.use_streaming = use_streaming and (mode == TradingMode.TRAINING)  # Only use streaming for training
        self.smart_action_filtering = smart_action_filtering  # Prevent redundant position attempts

        # For streaming mode
        self.total_data_length = 0
        self.current_episode_start = 0
        self.current_episode_end = 0

        # Initialize component managers
        self.reward_calculator = RewardCalculator(reward_function, symbol)
        self.observation_handler = ObservationHandler(lookback_window)
        self.termination_manager = TerminationManager(mode, max_drawdown_pct=0.20)

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
            trailing_stop_percentage = config.get('environment', {}).get('trailing_stop_percentage', 0.02)

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
        # quantity: continuous value representing desired number of lots to trade
        # Environment will clamp to actual max_affordable based on capital and price
        self.max_theoretical_quantity = 100000  # Consistent upper bound across codebase
        self.action_space = gym.spaces.Box(
            low=np.array([0, 1]),
            high=np.array([4, self.max_theoretical_quantity]),
            dtype=np.float32
        )
        
        # Observation space will be initialized by ObservationHandler
        self.observation_space = None

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
            # Get hierarchical processing configuration
            from src.utils.config_loader import ConfigLoader
            config_loader = ConfigLoader()
            config = config_loader.get_config()
            
            # Initialize observation handler with data and config
            feature_columns = self.observation_handler.initialize_observation_space(self.data, config)
            self.observation_space = self.observation_handler.observation_space
            self.lookback_window = self.observation_handler.lookback_window
            
            logger.info(f"🔧 Universal Model: Excluded raw OHLC prices, using {len(feature_columns)} derived features")
            logger.info(f"📊 Observation space set: {self.observation_handler.features_per_step} features per step, total dim: {self.observation_handler.observation_dim}")
            logger.info(f"📈 Hierarchical lookback - High: {self.observation_handler.high_level_lookback}, Low: {self.observation_handler.low_level_lookback}")
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

        # Reset component managers
        self.reward_calculator.reset(self.initial_capital)
        self.observation_handler.reset()
        self.termination_manager.reset(self.initial_capital, getattr(self, 'episode_end_step', None))

        return self.observation_handler.get_observation(self.data, self.current_step, self.engine)

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

    def step(self, action: Tuple[int, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        try:
            self.current_step += 1

            if self.current_step >= len(self.data):
                # Force close any open positions before episode ends
                self.termination_manager.force_close_positions(self.engine, self.data, self.current_step)
                done = True
                reward = 0.0 # No more data to process
                info = {"message": "End of data"}
                return self.observation_handler.get_observation(self.data, self.current_step, self.engine), reward, done, info

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

            # Action is now directly from agent.select_action: (action_type, quantity)
            action_type, predicted_quantity = action

            # Adjust predicted quantity to available capital for trading actions (not HOLD)
            if action_type != 4 and predicted_quantity > 0:
                available_capital = prev_capital

                # Import capital-aware quantity calculation
                from src.utils.capital_aware_quantity import CapitalAwareQuantitySelector
                selector = CapitalAwareQuantitySelector()

                # Calculate maximum affordable quantity (no artificial limits)
                max_affordable_quantity = selector.get_max_affordable_quantity(
                    available_capital=available_capital,
                    current_price=current_price,
                    instrument=self.instrument
                )

                # Clamp predicted quantity to what's actually affordable
                if max_affordable_quantity > 0:
                    # Use the smaller of predicted quantity or max affordable
                    actual_quantity = min(int(predicted_quantity), max_affordable_quantity)
                    actual_quantity = max(1, actual_quantity)  # Ensure at least 1 lot

                    # Store values for reward calculation
                    self._last_predicted_quantity = predicted_quantity
                    self._last_max_affordable = max_affordable_quantity

                    quantity = float(actual_quantity)
                else:
                    # No capital available for trading
                    quantity = 0.0
                    self._last_predicted_quantity = predicted_quantity
                    self._last_max_affordable = 0
            else:
                quantity = 0.0
                self._last_predicted_quantity = 0.0
                self._last_max_affordable = 0

            # Get current position state to validate actions
            account_state = self.engine.get_account_state()
            current_position = account_state['current_position_quantity']

            # Smart action filtering to prevent redundant position attempts and invalid closes
            if self.smart_action_filtering:
                if action_type == 0 and current_position != 0:  # BUY_LONG when already have position
                    action_type = 4  # Convert to HOLD
                elif action_type == 1 and current_position != 0:  # SELL_SHORT when already have position
                    action_type = 4  # Convert to HOLD
                elif action_type == 2 and current_position <= 0: # CLOSE_LONG when not long
                    action_type = 4 # Convert to HOLD
                elif action_type == 3 and current_position >= 0: # CLOSE_SHORT when not short
                    action_type = 4 # Convert to HOLD

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

            # Calculate reward using RewardCalculator
            current_capital = self.engine.get_account_state(current_price=current_price)['capital']
            
            # Update tracking data in reward calculator
            self.reward_calculator.update_tracking_data(
                action_type, current_capital, predicted_quantity, 
                getattr(self, '_last_max_affordable', 0)
            )

            # Calculate base reward
            base_reward = self.reward_calculator.calculate_reward(current_capital, prev_capital, self.engine)
            
            # Apply reward shaping
            shaped_reward = self.reward_calculator.apply_reward_shaping(
                base_reward, action_type, current_capital, prev_capital, self.engine, current_price
            )

            # Add quantity prediction feedback reward
            quantity_reward = self.reward_calculator.calculate_quantity_feedback_reward(action, prev_capital)
            shaped_reward += quantity_reward

            # Apply normalization for universal model training
            reward = self.reward_calculator.normalize_reward(shaped_reward)

            self.reward_calculator.last_action_type = action_type

            # Check termination conditions using TerminationManager
            done, termination_reason = self.termination_manager.check_termination_conditions(
                self.current_step, len(self.data), current_capital
            )

            # Force close any open positions if episode is ending
            if done:
                self.termination_manager.force_close_positions(self.engine, self.data, self.current_step)

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
            return self.observation_handler.get_observation(self.data, self.current_step, self.engine, current_price), reward, done, info
        except Exception as e:
            logger.error(f"Error in step: {e}")
            return self.observation_handler.get_observation(self.data, self.current_step, self.engine), 0.0, True, {"error": str(e)}

    def get_backtest_results(self) -> Dict:
        """Get comprehensive backtesting results for BACKTESTING/LIVE modes."""
        if self.mode == TradingMode.TRAINING:
            return {}  # No backtest results for training

        account_state = self.engine.get_account_state()

        # Calculate performance metrics
        total_return = (account_state['capital'] - self.initial_capital) / self.initial_capital
        peak_equity = self.termination_manager.peak_equity
        max_drawdown = (peak_equity - account_state['capital']) / peak_equity if peak_equity > 0 else 0

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
            'peak_equity': peak_equity,
            'total_trades': self.engine._trade_count,
            'equity_curve': self.reward_calculator.equity_history.copy(),
            'trades': trades,
            'current_position': account_state['current_position_quantity'],
            'total_steps': self.current_step + 1,
            'data_length': len(self.data) if self.data is not None else 0
        }

        return results