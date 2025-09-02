import logging
import yaml
import os
from typing import Dict, Tuple, Optional
from datetime import datetime
import time

from src.config.instrument import Instrument
from src.utils.config_loader import ConfigLoader

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Warning deduplication to prevent spam
_warning_cache = set()

class BacktestingEngine:
    def __init__(self, initial_capital: float, instrument: Instrument, trailing_stop_percentage: float = 0.02):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self.instrument = instrument
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = [] # To store details of each trade for reporting/metrics
        self._stop_loss_price = 0.0
        self._target_profit_price = 0.0
        self._trailing_stop_price = 0.0
        self._peak_price = 0.0
        self._is_position_open = False
        self.trailing_stop_percentage = trailing_stop_percentage

        # Point-based trailing stop variables
        self._initial_sl_points = 0.0  # Fixed point gap from initial SL
        self._use_point_based_trailing = True  # Use point-based instead of percentage-based
        self._premium_paid_per_lot = 0.0 # For options P&L calculation

        # Load configuration for brokerage and risk management
        self.config = self._load_config()
        trading_config = self.config.get('trading', {})
        risk_config = self.config.get('risk_management', {})

        self.BROKERAGE_ENTRY = trading_config.get('brokerage_entry', 25.0)  # INR
        self.BROKERAGE_EXIT = trading_config.get('brokerage_exit', 35.0)   # INR

        # Risk management configuration
        self.risk_multiplier = risk_config.get('risk_multiplier', 1.0)
        self.reward_multiplier = risk_config.get('reward_multiplier', 2.0)
        self.use_atr_based_stops = risk_config.get('use_atr_based_stops', True)

        # Enhanced P&L tracking
        self._total_realized_pnl = 0.0  # Total P&L from all closed trades
        self._trade_count = 0  # Number of completed trades

        # Real-time trade decision logging
        self._current_step = 0
        self._decision_log = []  # Detailed log of all decisions

    def _load_config(self) -> dict:
        """Load configuration from training_sequence.yaml"""
        config_path = "config/training_sequence.yaml"
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {
                'trading': {'brokerage_entry': 25.0, 'brokerage_exit': 35.0},
                'risk_management': {'risk_multiplier': 1.0, 'reward_multiplier': 2.0, 'use_atr_based_stops': True}
            }
        from src.config.settings import get_settings

class BacktestingEngine:
    def __init__(self, initial_capital: float, instrument: Instrument, trailing_stop_percentage: float = 0.02):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self.instrument = instrument
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = [] # To store details of each trade for reporting/metrics
        self._stop_loss_price = 0.0
        self._target_profit_price = 0.0
        self._trailing_stop_price = 0.0
        self._peak_price = 0.0
        self._is_position_open = False
        self.trailing_stop_percentage = trailing_stop_percentage

        # Point-based trailing stop variables
        self._initial_sl_points = 0.0  # Fixed point gap from initial SL
        self._use_point_based_trailing = True  # Use point-based instead of percentage-based
        self._premium_paid_per_lot = 0.0 # For options P&L calculation

        # Load configuration for brokerage and risk management
        self.config = self._load_config()
        trading_config = self.config.get('trading', {})
        risk_config = self.config.get('risk_management', {})

        self.BROKERAGE_ENTRY = trading_config.get('brokerage_entry', 25.0)  # INR
        self.BROKERAGE_EXIT = trading_config.get('brokerage_exit', 35.0)   # INR

        # Risk management configuration
        self.risk_multiplier = risk_config.get('risk_multiplier', 1.0)
        self.reward_multiplier = risk_config.get('reward_multiplier', 2.0)
        self.use_atr_based_stops = risk_config.get('use_atr_based_stops', True)

        # Enhanced P&L tracking
        self._total_realized_pnl = 0.0  # Total P&L from all closed trades
        self._trade_count = 0  # Number of completed trades

        # Real-time trade decision logging
        self._current_step = 0
        self._decision_log = []  # Detailed log of all decisions
        
        # Position lock mechanism - prevent closing within 5 candles of entry
        self._position_entry_candle = 0  # Candle number when position was opened
        self._current_candle = 0  # Current candle counter
        self.POSITION_LOCK_CANDLES = 5  # Minimum candles to hold position

    def _load_config(self) -> dict:
        """Load configuration from settings.yaml"""
        config_loader = ConfigLoader()
        return config_loader.get_config()
    
    def increment_candle(self):
        """Increment candle counter - should be called at each time step"""
        self._current_candle += 1
    
    def can_close_position(self, is_sl_hit: bool = False) -> bool:
        """Check if position can be closed based on candle lock"""
        if not self._is_position_open:
            return True  # No position to close
        
        # SL hits are always allowed to override position lock
        if is_sl_hit:
            return True
        
        candles_held = self._current_candle - self._position_entry_candle
        return candles_held >= self.POSITION_LOCK_CANDLES

    def reset(self, verbose: bool = False):
        if verbose:
            logger.info(f"🔄 Engine reset: Position before reset: {self._current_position_quantity}")
        self._capital = self._initial_capital
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = []
        self._stop_loss_price = 0.0
        self._target_profit_price = 0.0
        self._trailing_stop_price = 0.0
        self._peak_price = 0.0
        self._is_position_open = False

        # Enhanced P&L tracking
        self._total_realized_pnl = 0.0  # Total P&L from all closed trades
        self._trade_count = 0  # Number of completed trades

        if verbose:
            logger.info(f"✅ Engine reset complete: Position after reset: {self._current_position_quantity}, Capital: ₹{self._capital:.2f}")

    def _update_trailing_stop(self, current_price: float):
        """
        Update trailing stop using point-based logic that maintains fixed point gaps.
        Trail moves up/down maintaining the same point gap but doesn't move when price comes closer.
        """
        if not self._is_position_open or self._initial_sl_points == 0:
            return

        if self._current_position_quantity > 0:  # Long position
            if current_price > self._peak_price:
                # Price moved up, update peak and trail
                self._peak_price = current_price
                new_trailing_stop = self._peak_price - self._initial_sl_points

                # Only move trailing stop up, never down (maintain or improve)
                if new_trailing_stop > self._trailing_stop_price:
                    self._trailing_stop_price = new_trailing_stop

                # Ensure trailing stop doesn't go below initial SL
                if self._trailing_stop_price < self._stop_loss_price:
                    self._trailing_stop_price = self._stop_loss_price

        elif self._current_position_quantity < 0:  # Short position
            if current_price < self._peak_price:
                # Price moved down, update peak and trail
                self._peak_price = current_price
                new_trailing_stop = self._peak_price + self._initial_sl_points

                # Only move trailing stop down, never up (maintain or improve)
                if new_trailing_stop < self._trailing_stop_price:
                    self._trailing_stop_price = new_trailing_stop

                # Ensure trailing stop doesn't go above initial SL
                if self._trailing_stop_price > self._stop_loss_price:
                    self._trailing_stop_price = self._stop_loss_price

    def execute_trade(self, action: str, price: float, quantity: float, atr_value: float = 0.0, proxy_premium: float = 0.0) -> Tuple[float, float]:
        """Execute trade with detailed real-time logging."""
        self._current_step += 1
        self._last_decision_timestamp = datetime.now()

        # Check if detailed logging is enabled
        import os
        detailed_logging = os.environ.get('DETAILED_BACKTEST_LOGGING', 'false').lower() == 'true'

        # Log the decision being made
        decision_log = {
            'step': self._current_step,
            'timestamp': self._last_decision_timestamp,
            'action': action,
            'price': price,
            'quantity': quantity,
            'atr': atr_value,
            'proxy_premium': proxy_premium,
            'capital_before': self._capital,
            'position_before': self._current_position_quantity,
            'is_position_open_before': self._is_position_open
        }

        # Always update trailing stop if a position is open, regardless of action or quantity
        if self._is_position_open:
            self._update_trailing_stop(price)

        if price <= 0 or (quantity <= 0 and action != "HOLD"):
            logger.warning(f"WARNING Step {self._current_step}: Invalid price ({price}) or quantity ({quantity} lots) for {action}")
            decision_log['result'] = 'INVALID_PARAMETERS'
            decision_log['reason'] = f"Invalid price ({price}) or quantity ({quantity})"
            self._decision_log.append(decision_log)
            return 0.0, self._unrealized_pnl

        realized_pnl_this_trade = 0.0
        cost = 0.0

        # Store original action for logging
        original_action = action

        # Check for SL/Trailing Stop hit if a position is open (removed TP check when using trailing stops)
        exit_reason = None
        is_automatic_close = False  # Flag to track if this is an automatic SL/trail hit
        if self._is_position_open:
            if self._current_position_quantity > 0: # Long position
                if price <= self._stop_loss_price:
                    exit_reason = f"STOP_LOSS_HIT"
                    is_automatic_close = True
                    if detailed_logging:
                        logger.info(f"🛑 Step {self._current_step}: SL hit for LONG position at ₹{price:.2f} (SL: ₹{self._stop_loss_price:.2f})")
                    action = "CLOSE_LONG"
                    quantity = self._current_position_quantity
                    price = self._stop_loss_price  # Use SL price for execution, not market price
                elif price <= self._trailing_stop_price:
                    exit_reason = f"TRAILING_STOP_HIT"
                    is_automatic_close = True
                    if detailed_logging:
                        logger.info(f"📉 Step {self._current_step}: Trailing SL hit for LONG position at ₹{price:.2f} (Trail: ₹{self._trailing_stop_price:.2f})")
                    action = "CLOSE_LONG"
                    quantity = self._current_position_quantity
                    price = self._trailing_stop_price  # Use trailing stop price for execution
            elif self._current_position_quantity < 0: # Short position
                if price >= self._stop_loss_price:
                    exit_reason = f"STOP_LOSS_HIT"
                    is_automatic_close = True
                    if detailed_logging:
                        logger.info(f"🛑 Step {self._current_step}: SL hit for SHORT position at ₹{price:.2f} (SL: ₹{self._stop_loss_price:.2f})")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)
                    price = self._stop_loss_price  # Use SL price for execution, not market price
                elif price >= self._trailing_stop_price:
                    exit_reason = f"TRAILING_STOP_HIT"
                    is_automatic_close = True
                    if detailed_logging:
                        logger.info(f"📈 Step {self._current_step}: Trailing SL hit for SHORT position at ₹{price:.2f} (Trail: ₹{self._trailing_stop_price:.2f})")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)
                    price = self._trailing_stop_price  # Use trailing stop price for execution

        if action == "BUY_LONG":
            if self._current_position_quantity != 0:
                # Suppress detailed logging during training - info will be in step logs
                pass
                decision_log['result'] = 'REJECTED_EXISTING_POSITION'
                decision_log['reason'] = f"Already have position: {self._current_position_quantity}"
                self._decision_log.append(decision_log)
                return 0.0, self._unrealized_pnl

            # Deduct entry brokerage for realistic trading costs
            cost = self.BROKERAGE_ENTRY
            self._capital -= cost  # Deduct entry brokerage from capital

            # Execute the trade
            self._current_position_quantity = quantity
            self._current_position_entry_price = price
            self._is_position_open = True
            self._position_entry_time = self._last_decision_timestamp
            self._position_entry_reason = f"BUY_LONG signal at step {self._current_step}"
            # Record entry candle for position lock
            self._position_entry_candle = self._current_candle

            # Use centralized risk-reward configuration
            risk_multiplier = self.risk_multiplier
            reward_multiplier = self.reward_multiplier

            self._stop_loss_price = price - (atr_value * risk_multiplier)  # SL = risk_multiplier * ATR below entry for long
            self._target_profit_price = price + (atr_value * reward_multiplier)  # TP = reward_multiplier * ATR above entry for long
            self._peak_price = price # Initialize peak price for trailing stop

            # Calculate initial SL points for point-based trailing
            self._initial_sl_points = price - self._stop_loss_price  # Fixed point gap
            self._trailing_stop_price = self._stop_loss_price  # Start with initial SL

            # Detailed position entry logging (only if detailed logging enabled)
            # Log key information for research logger to capture
            if detailed_logging:
                logger.debug(f"🟢 Step {self._current_step}: BUY_LONG EXECUTED")
                logger.debug(f"   📊 Position: {quantity} lots @ ₹{price:.2f} (Entry Brokerage: ₹{cost:.2f})")
                logger.debug(f"   🛑 Stop Loss: ₹{self._stop_loss_price:.2f} ({risk_multiplier}x ATR)")
                logger.debug(f"   🎯 Target: ₹{self._target_profit_price:.2f} ({reward_multiplier}x ATR)")
                logger.debug(f"   📉 Trailing SL: ₹{self._trailing_stop_price:.2f} ({self.trailing_stop_percentage:.1%})")
                logger.debug(f"   💰 Capital remaining: ₹{self._capital:.2f}")
                logger.debug(f"   📈 Risk-Reward Ratio: 1:{reward_multiplier/risk_multiplier:.1f}")

            # Update decision log
            decision_log['result'] = 'POSITION_OPENED'
            decision_log['position_type'] = 'LONG'
            decision_log['entry_price'] = price
            decision_log['stop_loss'] = self._stop_loss_price
            decision_log['target_profit'] = self._target_profit_price
            decision_log['trailing_stop'] = self._trailing_stop_price
            decision_log['cost'] = cost
            decision_log['capital_after'] = self._capital

        elif action == "SELL_SHORT":
            if self._current_position_quantity != 0:
                # Suppress detailed logging during training - info will be in step logs
                pass
                return 0.0, self._unrealized_pnl

            # Deduct entry brokerage for realistic trading costs
            cost = self.BROKERAGE_ENTRY
            self._capital -= cost  # Deduct entry brokerage from capital

            self._current_position_quantity = -quantity # Negative for short position
            self._current_position_entry_price = price
            self._is_position_open = True
            # Record entry candle for position lock
            self._position_entry_candle = self._current_candle


            # Use centralized risk-reward configuration
            risk_multiplier = self.risk_multiplier
            reward_multiplier = self.reward_multiplier

            self._stop_loss_price = price + (atr_value * risk_multiplier)  # SL = risk_multiplier * ATR above entry for short
            self._target_profit_price = price - (atr_value * reward_multiplier)  # TP = reward_multiplier * ATR below entry for short
            self._peak_price = price # Initialize peak price for trailing stop

            # Calculate initial SL points for point-based trailing
            self._initial_sl_points = self._stop_loss_price - price  # Fixed point gap
            self._trailing_stop_price = self._stop_loss_price  # Start with initial SL
            # Log key information for research logger to capture
            if detailed_logging:
                logging.debug(f"🔴 Step {self._current_step}: SELL_SHORT EXECUTED")
                logging.debug(f"   📊 Position: {quantity} lots @ ₹{price:.2f} (Entry Brokerage: ₹{cost:.2f})")
                logging.debug(f"   🛑 Stop Loss: ₹{self._stop_loss_price:.2f} ({risk_multiplier}x ATR)")
                logging.debug(f"   🎯 Target: ₹{self._target_profit_price:.2f} ({reward_multiplier}x ATR)")
                logging.debug(f"   📈 Trailing SL: ₹{self._trailing_stop_price:.2f} ({self.trailing_stop_percentage:.1%})")
                logging.debug(f"   💰 Capital remaining: ₹{self._capital:.2f}")
                logging.debug(f"   📉 Risk-Reward Ratio: 1:{reward_multiplier/risk_multiplier:.1f}")

        elif action == "CLOSE_LONG":
            if self._current_position_quantity <= 0: # No long position to close
                warning_key = f"no_long_position_to_close_{self._current_position_quantity}"
                if warning_key not in _warning_cache:
                    logging.warning(f"No long position to CLOSE_LONG. Current position: {self._current_position_quantity}. Trade not executed.")
                    _warning_cache.add(warning_key)
                return 0.0, self._unrealized_pnl
            
            # Position lock check - prevent manual closing within 5 candles (SL hits are allowed)
            if not self.can_close_position(is_sl_hit=is_automatic_close):
                candles_held = self._current_candle - self._position_entry_candle
                if detailed_logging:
                    logging.info(f"Position lock active: held for {candles_held} candles, need {self.POSITION_LOCK_CANDLES}. Manual close blocked.")
                return 0.0, self._unrealized_pnl
            
            if quantity > self._current_position_quantity:
                logging.warning(f"Attempted to close {quantity} long, but only {self._current_position_quantity} held. Closing full position.")
                quantity = self._current_position_quantity

            # Point-based P&L calculation for index trading
            realized_pnl_this_trade = (price - self._current_position_entry_price) * quantity * self.instrument.lot_size

            # Calculate net P&L after brokerage
            net_pnl_this_trade = realized_pnl_this_trade - self.BROKERAGE_EXIT

            # Apply the P&L directly to capital
            self._realized_pnl += net_pnl_this_trade
            self._capital += net_pnl_this_trade  # Add profit or subtract loss
            self._total_realized_pnl += net_pnl_this_trade
            self._trade_count += 1

            self._current_position_quantity -= quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
                self._is_position_open = False # Close position

            # Calculate total P&L from initial capital
            total_pnl_from_initial = self._capital - self._initial_capital

            # Log key information for research logger to capture
            if detailed_logging:
                logging.debug(f"Executed CLOSE_LONG. Quantity: {quantity}, Price: {price}.")
                logging.debug(f"  📊 Current Trade P&L: ₹{net_pnl_this_trade:.2f}")
                logging.debug(f"  💰 Total P&L from Initial: ₹{total_pnl_from_initial:.2f}")
                logging.debug(f"  🏦 Capital: ₹{self._capital:.2f} (Trade #{self._trade_count})")

        elif action == "CLOSE_SHORT":
            if self._current_position_quantity >= 0: # No short position to close
                warning_key = f"no_short_position_to_close_{self._current_position_quantity}"
                if warning_key not in _warning_cache:
                    logging.warning(f"No short position to CLOSE_SHORT. Current position: {self._current_position_quantity}. Trade not executed.")
                    _warning_cache.add(warning_key)
                return 0.0, self._unrealized_pnl
            
            # Position lock check - prevent manual closing within 5 candles (SL hits are allowed)
            if not self.can_close_position(is_sl_hit=is_automatic_close):
                candles_held = self._current_candle - self._position_entry_candle
                if detailed_logging:
                    logging.info(f"Position lock active: held for {candles_held} candles, need {self.POSITION_LOCK_CANDLES}. Manual close blocked.")
                return 0.0, self._unrealized_pnl
            
            if quantity > abs(self._current_position_quantity):
                logging.warning(f"Attempted to close {quantity} short, but only {abs(self._current_position_quantity)} held. Closing full position.")
                quantity = abs(self._current_position_quantity)

            # Point-based P&L calculation for index trading (short position)
            realized_pnl_this_trade = (self._current_position_entry_price - price) * quantity * self.instrument.lot_size

            # Calculate net P&L after brokerage
            net_pnl_this_trade = realized_pnl_this_trade - self.BROKERAGE_EXIT

            # Apply the P&L directly to capital
            self._realized_pnl += net_pnl_this_trade
            self._capital += net_pnl_this_trade  # Add profit or subtract loss
            self._total_realized_pnl += net_pnl_this_trade
            self._trade_count += 1

            self._current_position_quantity += quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
                self._is_position_open = False # Close position

            # Calculate total P&L from initial capital
            total_pnl_from_initial = self._capital - self._initial_capital

            # Log key information for research logger to capture
            if detailed_logging:
                logging.debug(f"Executed CLOSE_SHORT. Quantity: {quantity}, Price: {price}.")
                logging.debug(f"  📊 Current Trade P&L: ₹{net_pnl_this_trade:.2f}")
                logging.debug(f"  💰 Total P&L from Initial: ₹{total_pnl_from_initial:.2f}")
                logging.debug(f"  🏦 Capital: ₹{self._capital:.2f} (Trade #{self._trade_count})")

        elif action == "HOLD":
            # HOLD action just updates trailing stop and checks for stop hits (already done above)
            # No new trade is executed, just price update
            pass

        else:
            logging.warning(f"Unknown action: {action}. No trade executed.")
            return 0.0, self._unrealized_pnl

        # Update unrealized P&L based on current market price
        self._update_unrealized_pnl(price)

        # Record trade for history (both openings and closings for better tracking)
        if action in ["BUY_LONG", "SELL_SHORT"]:
            # Record position opening
            self._trade_history.append({
                "action": action,
                "price": price,
                "quantity": quantity,
                "pnl": -cost,  # Entry cost as negative P&L (brokerage paid)
                "realized_pnl_this_trade": 0.0,  # No realized P&L on opening
                "cost": cost,
                "capital_after_trade": self._capital,
                "position_after_trade": self._current_position_quantity,
                "entry_price_after_trade": self._current_position_entry_price,
                "unrealized_pnl_after_trade": self._unrealized_pnl,
                "trade_type": "OPEN"
            })
        elif action in ["CLOSE_LONG", "CLOSE_SHORT"]:
            # Record position closing
            net_pnl_for_history = realized_pnl_this_trade - self.BROKERAGE_EXIT
            self._trade_history.append({
                "action": action,
                "price": price,
                "quantity": quantity,
                "pnl": net_pnl_for_history,  # Net P&L after brokerage for metrics compatibility
                "realized_pnl_this_trade": realized_pnl_this_trade,  # Gross P&L for backward compatibility
                "cost": cost,
                "capital_after_trade": self._capital,
                "position_after_trade": self._current_position_quantity,
                "entry_price_after_trade": self._current_position_entry_price,
                "unrealized_pnl_after_trade": self._unrealized_pnl,
                "trade_type": "CLOSE"
            })

        # Complete decision log entry
        decision_log.update({
            'capital_after': self._capital,
            'position_after': self._current_position_quantity,
            'realized_pnl': realized_pnl_this_trade,
            'unrealized_pnl': self._unrealized_pnl,
            'exit_reason': exit_reason,
            'action_executed': action,
            'original_action': original_action
        })

        # Set result if not already set
        if 'result' not in decision_log:
            if action == "HOLD":
                decision_log['result'] = 'HOLD_ACTION'
            elif action in ["CLOSE_LONG", "CLOSE_SHORT"]:
                decision_log['result'] = 'POSITION_CLOSED'
            elif action in ["BUY_LONG", "SELL_SHORT"]:
                decision_log['result'] = 'POSITION_OPENED'
            else:
                decision_log['result'] = 'ACTION_EXECUTED'

        self._decision_log.append(decision_log)

        return realized_pnl_this_trade, self._unrealized_pnl

    def _update_unrealized_pnl(self, current_price: float):
        # Generic P&L calculation for all data types
        if self._current_position_quantity > 0:  # Long position
            self._unrealized_pnl = (current_price - self._current_position_entry_price) * self._current_position_quantity * self.instrument.lot_size
        elif self._current_position_quantity < 0:  # Short position
            self._unrealized_pnl = (self._current_position_entry_price - current_price) * abs(self._current_position_quantity) * self.instrument.lot_size
        else:
            self._unrealized_pnl = 0.0

    def get_account_state(self, current_price: float = None) -> Dict[str, float]:
        if current_price is not None:
            self._update_unrealized_pnl(current_price)

        return {
            "capital": self._capital,
            "current_position_quantity": self._current_position_quantity,
            "current_position_entry_price": self._current_position_entry_price,
            "realized_pnl": self._realized_pnl,
            "unrealized_pnl": self._unrealized_pnl,
            "total_pnl": self._realized_pnl + self._unrealized_pnl,
            "is_position_open": self._is_position_open,
            # Enhanced P&L tracking
            "total_realized_pnl": self._total_realized_pnl,
            "total_pnl_from_initial": self._capital - self._initial_capital,
            "trade_count": self._trade_count,
            "initial_capital": self._initial_capital
        }

    def get_trade_history(self) -> list:
        return self._trade_history

    def get_decision_log(self):
        """Get detailed log of all trading decisions."""
        return self._decision_log

    def get_recent_decisions(self, count: int = 10):
        """Get the most recent trading decisions."""
        return self._decision_log[-count:] if self._decision_log else []

    def get_trade_summary(self):
        """Get summary of trading activity with detailed decision analysis."""
        if not self._decision_log:
            return {"message": "No trading decisions recorded"}

        total_decisions = len(self._decision_log)
        position_opens = len([d for d in self._decision_log if d.get('result') == 'POSITION_OPENED'])
        position_closes = len([d for d in self._decision_log if d.get('result') == 'POSITION_CLOSED'])
        holds = len([d for d in self._decision_log if d.get('result') == 'HOLD_ACTION'])
        rejections = len([d for d in self._decision_log if 'REJECTED' in d.get('result', '')])

        # Analyze exit reasons
        exit_reasons = {}
        for decision in self._decision_log:
            if decision.get('exit_reason'):
                reason = decision['exit_reason']
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            'total_decisions': total_decisions,
            'position_opens': position_opens,
            'position_closes': position_closes,
            'hold_actions': holds,
            'rejected_actions': rejections,
            'exit_reasons': exit_reasons,
            'current_step': self._current_step,
            'total_trades_completed': len(self._trade_history),
            'current_position': self._current_position_quantity,
            'is_position_open': self._is_position_open
        }

