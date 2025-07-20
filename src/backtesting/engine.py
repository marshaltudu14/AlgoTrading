import logging
from typing import Dict, Tuple

from src.config.instrument import Instrument
from src.config.config import RISK_REWARD_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        self._premium_paid_per_lot = 0.0 # For options P&L calculation

        self.BROKERAGE_ENTRY = 25.0  # INR
        self.BROKERAGE_EXIT = 35.0   # INR

        # Enhanced P&L tracking
        self._total_realized_pnl = 0.0  # Total P&L from all closed trades
        self._trade_count = 0  # Number of completed trades

    def reset(self):
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

    def _update_trailing_stop(self, current_price: float):
        if self._current_position_quantity > 0:  # Long position
            if current_price > self._peak_price:
                self._peak_price = current_price
                self._trailing_stop_price = self._peak_price * (1 - self.trailing_stop_percentage)
        elif self._current_position_quantity < 0:  # Short position
            if current_price < self._peak_price:
                self._peak_price = current_price
                self._trailing_stop_price = self._peak_price * (1 + self.trailing_stop_percentage)

    def execute_trade(self, action: str, price: float, quantity: float, atr_value: float = 0.0, proxy_premium: float = 0.0) -> Tuple[float, float]:
        # Always update trailing stop if a position is open, regardless of action or quantity
        if self._is_position_open:
            self._update_trailing_stop(price)

        if price <= 0 or (quantity <= 0 and action != "HOLD"):
            logging.warning(f"Invalid price ({price}) or quantity ({quantity} lots) for trade action {action}. Trade not executed.")
            return 0.0, self._unrealized_pnl

        realized_pnl_this_trade = 0.0
        cost = 0.0

        # Check for SL/TP/Trailing Stop hit if a position is open
        if self._is_position_open:
            if self._current_position_quantity > 0: # Long position
                if price <= self._stop_loss_price:
                    logging.info(f"SL hit for long position at {price:.2f}. Closing position.")
                    action = "CLOSE_LONG"
                    quantity = self._current_position_quantity
                elif price >= self._target_profit_price:
                    logging.info(f"TP hit for long position at {price:.2f}. Closing position.")
                    action = "CLOSE_LONG"
                    quantity = self._current_position_quantity
                elif price <= self._trailing_stop_price:
                    logging.info(f"Trailing SL hit for long position at {price:.2f}. Closing position.")
                    action = "CLOSE_LONG"
                    quantity = self._current_position_quantity
            elif self._current_position_quantity < 0: # Short position
                if price >= self._stop_loss_price:
                    logging.info(f"SL hit for short position at {price:.2f}. Closing position.")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)
                elif price <= self._target_profit_price:
                    logging.info(f"TP hit for short position at {price:.2f}. Closing position.")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)
                elif price >= self._trailing_stop_price:
                    logging.info(f"Trailing SL hit for short position at {price:.2f}. Closing position.")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)

        if action == "BUY_LONG":
            if self._current_position_quantity != 0:
                logging.warning(f"Cannot BUY_LONG. Already have an open position ({self._current_position_quantity}). Trade not executed.")
                return 0.0, self._unrealized_pnl
            if self.instrument.type == "OPTION":
                cost = (proxy_premium * quantity * self.instrument.lot_size) + self.BROKERAGE_ENTRY
                self._premium_paid_per_lot = proxy_premium # Store premium for options P&L
            else:
                # For STOCK, cost = stock price * quantity * lot_size (lot_size = 1 for stocks)
                cost = (price * quantity * self.instrument.lot_size) + self.BROKERAGE_ENTRY
            if self._capital < cost:
                logging.warning(f"Insufficient capital to BUY_LONG {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            self._capital -= cost
            self._current_position_quantity = quantity
            self._current_position_entry_price = price
            self._is_position_open = True



            # Use centralized risk-reward configuration
            risk_multiplier = RISK_REWARD_CONFIG['risk_multiplier']
            reward_multiplier = RISK_REWARD_CONFIG['reward_multiplier']

            self._stop_loss_price = price - (atr_value * risk_multiplier)  # SL = risk_multiplier * ATR below entry for long
            self._target_profit_price = price + (atr_value * reward_multiplier)  # TP = reward_multiplier * ATR above entry for long
            self._peak_price = price # Initialize peak price for trailing stop
            self._trailing_stop_price = self._peak_price * (1 - self.trailing_stop_percentage)
            logging.info(f"Executed BUY_LONG. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}. SL: {self._stop_loss_price:.2f}, TP: {self._target_profit_price:.2f}, Trailing SL: {self._trailing_stop_price:.2f}")
            logging.info(f"RR Config - Risk: {risk_multiplier}x ATR ({atr_value:.2f}), Reward: {reward_multiplier}x ATR, RR Ratio: 1:{reward_multiplier/risk_multiplier:.1f}")

        elif action == "SELL_SHORT":
            if self._current_position_quantity != 0:
                logging.warning(f"Cannot SELL_SHORT. Already have an open position ({self._current_position_quantity}). Trade not executed.")
                return 0.0, self._unrealized_pnl
            if self.instrument.type == "OPTION":
                cost = (proxy_premium * quantity * self.instrument.lot_size) + self.BROKERAGE_ENTRY
                self._premium_paid_per_lot = proxy_premium # Store premium for options P&L
            else:
                # For STOCK short selling, cost = stock price * quantity * lot_size (lot_size = 1 for stocks)
                cost = (price * quantity * self.instrument.lot_size) + self.BROKERAGE_ENTRY
            if self._capital < cost:
                logging.warning(f"Insufficient capital to SELL_SHORT {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            self._capital -= cost
            self._current_position_quantity = -quantity # Negative for short position
            self._current_position_entry_price = price
            self._is_position_open = True



            # Use centralized risk-reward configuration
            risk_multiplier = RISK_REWARD_CONFIG['risk_multiplier']
            reward_multiplier = RISK_REWARD_CONFIG['reward_multiplier']

            self._stop_loss_price = price + (atr_value * risk_multiplier)  # SL = risk_multiplier * ATR above entry for short
            self._target_profit_price = price - (atr_value * reward_multiplier)  # TP = reward_multiplier * ATR below entry for short
            self._peak_price = price # Initialize peak price for trailing stop
            self._trailing_stop_price = self._peak_price * (1 + self.trailing_stop_percentage)
            logging.info(f"Executed SELL_SHORT. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}. SL: {self._stop_loss_price:.2f}, TP: {self._target_profit_price:.2f}, Trailing SL: {self._trailing_stop_price:.2f}")
            logging.info(f"RR Config - Risk: {risk_multiplier}x ATR ({atr_value:.2f}), Reward: {reward_multiplier}x ATR, RR Ratio: 1:{reward_multiplier/risk_multiplier:.1f}")

        elif action == "CLOSE_LONG":
            if self._current_position_quantity <= 0: # No long position to close
                logging.warning(f"No long position to CLOSE_LONG. Current position: {self._current_position_quantity}. Trade not executed.")
                return 0.0, self._unrealized_pnl
            if quantity > self._current_position_quantity:
                logging.warning(f"Attempted to close {quantity} long, but only {self._current_position_quantity} held. Closing full position.")
                quantity = self._current_position_quantity

            if self.instrument.type == "OPTION":
                # For option trading simulation: calculate P&L as underlying movement
                # The premium is already paid as cost, so P&L is just the underlying movement
                realized_pnl_this_trade = (price - self._current_position_entry_price) * quantity * self.instrument.lot_size
                # Note: Premium cost is already deducted from capital when opening position
            else:
                realized_pnl_this_trade = (price - self._current_position_entry_price) * quantity * self.instrument.lot_size

            # Calculate net P&L after brokerage
            net_pnl_this_trade = realized_pnl_this_trade - self.BROKERAGE_EXIT

            # For OPTIONS, release the premium cost when closing position
            # For STOCKS, no premium to release
            if self.instrument.type == "OPTION":
                # Release the premium that was paid when opening the position
                premium_to_release = self._premium_paid_per_lot * quantity * self.instrument.lot_size
                self._capital += premium_to_release

            # Then apply the P&L
            self._realized_pnl += net_pnl_this_trade
            self._capital += net_pnl_this_trade  # Add profit or subtract loss
            self._total_realized_pnl += net_pnl_this_trade
            self._trade_count += 1

            self._current_position_quantity -= quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
                self._is_position_open = False # Close position

            # Calculate total P&L from initial capital
            # For OPTIONS, use total realized P&L (excludes premium costs)
            # For STOCKS, use capital difference
            if self.instrument.type == "OPTION":
                total_pnl_from_initial = self._total_realized_pnl
            else:
                total_pnl_from_initial = self._capital - self._initial_capital

            logging.info(f"Executed CLOSE_LONG. Quantity: {quantity}, Price: {price}.")
            logging.info(f"  📊 Current Trade P&L: ₹{net_pnl_this_trade:.2f}")
            logging.info(f"  💰 Total P&L from Initial: ₹{total_pnl_from_initial:.2f}")
            logging.info(f"  🏦 Capital: ₹{self._capital:.2f} (Trade #{self._trade_count})")

        elif action == "CLOSE_SHORT":
            if self._current_position_quantity >= 0: # No short position to close
                logging.warning(f"No short position to CLOSE_SHORT. Current position: {self._current_position_quantity}. Trade not executed.")
                return 0.0, self._unrealized_pnl
            if quantity > abs(self._current_position_quantity):
                logging.warning(f"Attempted to close {quantity} short, but only {abs(self._current_position_quantity)} held. Closing full position.")
                quantity = abs(self._current_position_quantity)

            if self.instrument.type == "OPTION":
                # For option trading simulation: calculate P&L as underlying movement
                # The premium is already paid as cost, so P&L is just the underlying movement
                realized_pnl_this_trade = (self._current_position_entry_price - price) * quantity * self.instrument.lot_size
                # Note: Premium cost is already deducted from capital when opening position
            else:
                realized_pnl_this_trade = (self._current_position_entry_price - price) * quantity * self.instrument.lot_size

            # Calculate net P&L after brokerage
            net_pnl_this_trade = realized_pnl_this_trade - self.BROKERAGE_EXIT

            # For OPTIONS, release the premium cost when closing position
            # For STOCKS, no premium to release
            if self.instrument.type == "OPTION":
                # Release the premium that was paid when opening the position
                premium_to_release = self._premium_paid_per_lot * quantity * self.instrument.lot_size
                self._capital += premium_to_release

            # Then apply the P&L
            self._realized_pnl += net_pnl_this_trade
            self._capital += net_pnl_this_trade  # Add profit or subtract loss
            self._total_realized_pnl += net_pnl_this_trade
            self._trade_count += 1

            self._current_position_quantity += quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
                self._is_position_open = False # Close position

            # Calculate total P&L from initial capital
            # For OPTIONS, use total realized P&L (excludes premium costs)
            # For STOCKS, use capital difference
            if self.instrument.type == "OPTION":
                total_pnl_from_initial = self._total_realized_pnl
            else:
                total_pnl_from_initial = self._capital - self._initial_capital

            logging.info(f"Executed CLOSE_SHORT. Quantity: {quantity}, Price: {price}.")
            logging.info(f"  📊 Current Trade P&L: ₹{net_pnl_this_trade:.2f}")
            logging.info(f"  💰 Total P&L from Initial: ₹{total_pnl_from_initial:.2f}")
            logging.info(f"  🏦 Capital: ₹{self._capital:.2f} (Trade #{self._trade_count})")

        elif action == "HOLD":
            # HOLD action just updates trailing stop and checks for stop hits (already done above)
            # No new trade is executed, just price update
            pass

        else:
            logging.warning(f"Unknown action: {action}. No trade executed.")
            return 0.0, self._unrealized_pnl

        # Update unrealized P&L based on current market price
        self._update_unrealized_pnl(price)

        # Record trade for history (only for completed trades)
        if action in ["CLOSE_LONG", "CLOSE_SHORT"]:
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
                "unrealized_pnl_after_trade": self._unrealized_pnl
            })

        return realized_pnl_this_trade, self._unrealized_pnl

    def _update_unrealized_pnl(self, current_price: float):
        if self.instrument.type == "OPTION":
            # For option trading simulation: calculate unrealized P&L as underlying movement
            # The premium cost is already accounted for in capital, so this is just mark-to-market
            if self._current_position_quantity > 0:  # Long position
                self._unrealized_pnl = (current_price - self._current_position_entry_price) * self._current_position_quantity * self.instrument.lot_size
            elif self._current_position_quantity < 0:  # Short position
                self._unrealized_pnl = (self._current_position_entry_price - current_price) * abs(self._current_position_quantity) * self.instrument.lot_size
            else:
                self._unrealized_pnl = 0.0
        else:
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
            "total_pnl_from_initial": self._total_realized_pnl if self.instrument.type == "OPTION" else self._capital - self._initial_capital,
            "trade_count": self._trade_count,
            "initial_capital": self._initial_capital
        }

    def get_trade_history(self) -> list:
        return self._trade_history

