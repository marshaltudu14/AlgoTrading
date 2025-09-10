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
        # Action space: action_type only (no quantity prediction)
        # Get number of actions from configuration
        num_actions = config.get('actions', {}).get('num_actions', 5)
        self.action_space = gym.spaces.Discrete(num_actions)
        
        # Load action configuration
        self.action_config = config.get('actions', {})
        self.action_names = self.action_config.get('action_names', 
            ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"])
        self.action_types = self.action_config.get('action_types', {
            'BUY_LONG': 0, 'SELL_SHORT': 1, 'CLOSE_LONG': 2, 'CLOSE_SHORT': 3, 'HOLD': 4
        })
        
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
            print(f"Warning: Failed to load data segment for {self.symbol}, falling back to full FINAL data loading. Episode: {self.current_episode_start}-{self.current_episode_end}")
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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        try:
            # Get action indices from configuration for consistent use throughout step
            buy_long_idx = self.action_types.get('BUY_LONG', 0)
            sell_short_idx = self.action_types.get('SELL_SHORT', 1)
            close_long_idx = self.action_types.get('CLOSE_LONG', 2)
            close_short_idx = self.action_types.get('CLOSE_SHORT', 3)
            hold_action_idx = self.action_types.get('HOLD', 4)
            
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

            # Action is now directly an integer representing the action type
            action_type = action

            # Use fixed quantity of 1 for all trading actions (except HOLD)
            if action_type != hold_action_idx:
                available_capital = prev_capital

                # Import capital-aware quantity calculation to check if trade is affordable
                from src.utils.capital_aware_quantity import CapitalAwareQuantitySelector
                selector = CapitalAwareQuantitySelector()

                # Calculate maximum affordable quantity to check if trade is possible
                max_affordable_quantity = selector.get_max_affordable_quantity(
                    available_capital=available_capital,
                    current_price=current_price,
                    instrument=self.instrument
                )

                # Use quantity of 1 if affordable, otherwise 0
                if max_affordable_quantity >= 1:
                    quantity = 1.0
                    self._last_max_affordable = max_affordable_quantity
                else:
                    # No capital available for trading
                    quantity = 0.0
                    self._last_max_affordable = 0
            else:
                quantity = 0.0
                self._last_max_affordable = 0

            # Get current position state to validate actions
            account_state = self.engine.get_account_state()
            current_position = account_state['current_position_quantity']

            # Smart action filtering to prevent redundant position attempts and invalid closes
            if self.smart_action_filtering:
                if action_type == buy_long_idx and current_position != 0:  # BUY_LONG when already have position
                    action_type = hold_action_idx  # Convert to HOLD
                elif action_type == sell_short_idx and current_position != 0:  # SELL_SHORT when already have position
                    action_type = hold_action_idx  # Convert to HOLD
                elif action_type == close_long_idx and current_position <= 0: # CLOSE_LONG when not long
                    action_type = hold_action_idx # Convert to HOLD
                elif action_type == close_short_idx and current_position >= 0: # CLOSE_SHORT when not short
                    action_type = hold_action_idx # Convert to HOLD

            # Get datetime early for position blocking logic
            try:
                safe_step = min(self.current_step, len(self.data) - 1)
                if hasattr(self.data.index, 'strftime') or pd.api.types.is_datetime64_any_dtype(self.data.index):
                    # DatetimeIndex
                    current_datetime = self.data.index[safe_step]
                elif self.data.index.name == 'datetime_readable':
                    # Regular index with datetime_readable name - convert to datetime
                    datetime_str = self.data.index[safe_step]
                    current_datetime = pd.to_datetime(datetime_str)
                elif 'datetime_readable' in self.data.columns:
                    # Use datetime_readable column
                    datetime_val = self.data['datetime_readable'].iloc[safe_step]
                    current_datetime = pd.to_datetime(datetime_val)
                elif 'datetime' in self.data.columns:
                    # DateTime column (fallback)
                    datetime_val = self.data['datetime'].iloc[safe_step]
                    current_datetime = pd.to_datetime(datetime_val)
                else:
                    # Fallback datetime calculation
                    total_minutes = safe_step * 5  # 5-minute intervals
                    hours = 9 + (total_minutes // 60)  # Start at 9 AM
                    minutes = total_minutes % 60
                    current_datetime = pd.to_datetime(f"2024-01-01 {hours:02d}:{minutes:02d}:00")
            except Exception as e:
                current_datetime = pd.to_datetime("2024-01-01 09:00:00")
            
            # Market timing filter: Prevent new positions between 3:15-3:30 PM
            # Allow closing existing positions but block opening new ones
            if self.termination_manager.is_new_position_blocked(current_datetime):
                if action_type in [buy_long_idx, sell_short_idx]:  # New position actions
                    action_type = hold_action_idx  # Convert to HOLD
                    # Note: CLOSE_LONG and CLOSE_SHORT are still allowed to close existing positions

            # Increment candle counter for position lock mechanism
            self.engine.increment_candle()
            
            # Execute trade based on action type using configuration
            if action_type == buy_long_idx:
                self.engine.execute_trade("BUY_LONG", current_price, quantity, current_atr, proxy_premium)
            elif action_type == sell_short_idx:
                self.engine.execute_trade("SELL_SHORT", current_price, quantity, current_atr, proxy_premium)
            elif action_type == close_long_idx:
                # Only execute if we have a long position to close
                if current_position > 0:
                    self.engine.execute_trade("CLOSE_LONG", current_price, quantity, current_atr, proxy_premium)
                else:
                    # Convert invalid action to HOLD to reduce warnings
                    self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)
            elif action_type == close_short_idx:
                # Only execute if we have a short position to close
                if current_position < 0:
                    self.engine.execute_trade("CLOSE_SHORT", current_price, quantity, current_atr, proxy_premium)
                else:
                    # Convert invalid action to HOLD to reduce warnings
                    self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)
            elif action_type == hold_action_idx:
                self.engine.execute_trade("HOLD", current_price, 0, current_atr, proxy_premium)

            # Calculate reward using RewardCalculator
            current_capital = self.engine.get_account_state(current_price=current_price)['capital']
            
            # Update tracking data in reward calculator
            self.reward_calculator.update_tracking_data(action_type, current_capital, self.engine)
            
            # Calculate base reward
            try:
                base_reward = self.reward_calculator.calculate_reward(current_capital, prev_capital, self.engine)
            except NameError as ne:
                logger.error(f"NameError in calculate_reward: {ne}")
                logger.error(f"Parameters: current_capital={current_capital}, prev_capital={prev_capital}")
                logger.error(f"Engine type: {type(self.engine)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                base_reward = 0.0  # Fallback to no reward
            
            # Apply reward shaping
            try:
                shaped_reward = self.reward_calculator.apply_reward_shaping(
                    base_reward, action_type, current_capital, prev_capital, self.engine, current_price
                )
            except NameError as ne:
                logger.error(f"NameError in apply_reward_shaping: {ne}")
                logger.error(f"Parameters: base_reward={base_reward}, action_type={action_type}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                shaped_reward = base_reward  # Fallback to base reward

            # Calculate percentage-based P&L for universal reward scaling
            capital_pct_change = self.reward_calculator.calculate_percentage_pnl(current_capital, prev_capital)
            
            # Apply normalization using percentage-based approach
            reward = self.reward_calculator.normalize_reward(shaped_reward, capital_pct_change)

            self.reward_calculator.last_action_type = action_type

            # Create comprehensive info dictionary with all debug information
            info = {}
            info['datetime'] = current_datetime  # Use datetime calculated earlier

            # Check termination conditions using TerminationManager
            done, termination_reason = self.termination_manager.check_termination_conditions(
                self.current_step, len(self.data), current_capital, current_datetime
            )

            # Force close any open positions if episode is ending
            if done:
                self.termination_manager.force_close_positions(self.engine, self.data, self.current_step)
            if termination_reason:
                info["termination_reason"] = termination_reason

            # Add comprehensive trading information for debug logging
            account_state = self.engine.get_account_state(current_price=current_price)
            
            # Add action mapping using configured action names
            if action_type < len(self.action_names):
                info['action'] = self.action_names[action_type]
            else:
                info['action'] = 'UNKNOWN'
            info['action_idx'] = action_type
            info['quantity'] = quantity
            
            # Add account state information
            info['account_state'] = account_state
            info['initial_capital'] = self.initial_capital
            info['current_step'] = self.current_step
            
            # Add current price information (datetime already set above)
            info['current_price'] = current_price
                
            # Add engine decision log information
            if self.engine._decision_log:
                latest_decision = self.engine._decision_log[-1]
                info['engine_decision'] = latest_decision
                exit_reason = latest_decision.get('exit_reason')
                if exit_reason:
                    info["exit_reason"] = exit_reason
            
            # Add trade statistics and position change tracking
            trade_history = self.engine.get_trade_history()
            if trade_history:
                # Calculate win rate from closed trades
                closed_trades = [trade for trade in trade_history if trade.get('trade_type') == 'CLOSE']
                if closed_trades:
                    winning_trades = sum(1 for trade in closed_trades if trade.get('pnl', 0) > 0)
                    info['win_rate'] = (winning_trades / len(closed_trades)) if closed_trades else 0.0
                    info['total_trades'] = len(closed_trades)
                else:
                    info['win_rate'] = 0.0
                    info['total_trades'] = 0
                    
                info['trade_history'] = trade_history[-5:]  # Last 5 trades for context
            else:
                info['win_rate'] = 0.0 
                info['total_trades'] = 0
                info['trade_history'] = []
                
            # Track actual position changes for accurate reason display
            prev_capital = getattr(self, '_prev_capital', self.initial_capital)
            prev_position = getattr(self, '_prev_position', 0.0)
            current_position = account_state.get('current_position_quantity', 0.0)
            
            # Store for next step
            self._prev_capital = account_state.get('capital', self.initial_capital)
            self._prev_position = current_position
            
            # Determine if this was an actual position change
            info['position_changed'] = abs(current_position - prev_position) > 0.001
            info['position_opened'] = prev_position == 0 and current_position != 0
            info['position_closed'] = prev_position != 0 and current_position == 0
            info['previous_position'] = prev_position
                
            # Add risk management information - only show when position is actually open
            if account_state['is_position_open'] and account_state['current_position_entry_price'] > 0:
                info['entry_price'] = account_state['current_position_entry_price']
                info['target_price'] = getattr(self.engine, '_target_profit_price', 0.0)
                info['sl_price'] = getattr(self.engine, '_stop_loss_price', 0.0)
                info['unrealized_pnl'] = account_state['unrealized_pnl']
            else:
                # No position or invalid position state - clear all position-related data
                info['entry_price'] = 0.0
                info['target_price'] = 0.0
                info['sl_price'] = 0.0
                info['unrealized_pnl'] = 0.0

            # Return 4 values as expected by standard gym environments
            return self.observation_handler.get_observation(self.data, self.current_step, self.engine, current_price), reward, done, info
        except NameError as ne:
            if "'engine' is not defined" in str(ne):
                logger.error(f"SPECIFIC ENGINE ERROR: {ne}")
                logger.error(f"self.engine exists: {hasattr(self, 'engine')}")
                logger.error(f"self.engine type: {type(getattr(self, 'engine', 'NOT_FOUND'))}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Return safe fallback
                obs = self.observation_handler.get_fallback_observation()
                return obs, 0.0, True, {"error": f"Engine NameError: {str(ne)}"}
            else:
                raise  # Re-raise if it's a different NameError
        except Exception as e:
            logger.error(f"Error in step: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return a safe observation in case of error
            try:
                obs = self.observation_handler.get_observation(self.data, self.current_step, self.engine)
            except:
                # Fallback observation if engine access fails
                obs = self.observation_handler.get_fallback_observation()
            return obs, 0.0, True, {"error": str(e)}

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