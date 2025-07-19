import logging
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BacktestingEngine:
    def __init__(self, initial_capital: float):
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive.")
        self._initial_capital = initial_capital
        self._capital = initial_capital
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = [] # To store details of each trade for reporting/metrics

        self.BROKERAGE_ENTRY = 25.0  # INR
        self.BROKERAGE_EXIT = 35.0   # INR

    def reset(self):
        self._capital = self._initial_capital
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = []

    def execute_trade(self, action: str, price: float, quantity: float) -> Tuple[float, float]:
        if price <= 0 or quantity <= 0:
            logging.warning(f"Invalid price ({price}) or quantity ({quantity}) for trade action {action}. Trade not executed.")
            return 0.0, self._unrealized_pnl

        realized_pnl_this_trade = 0.0
        cost = 0.0

        if action == "BUY_LONG":
            cost = (price * quantity) + self.BROKERAGE_ENTRY
            if self._capital < cost:
                logging.warning(f"Insufficient capital to BUY_LONG {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            if self._current_position_quantity < 0: # Currently short, so this is a close short + new long
                # Close short position first
                realized_pnl_this_trade += (self._current_position_entry_price - price) * abs(self._current_position_quantity)
                self._realized_pnl += realized_pnl_this_trade
                self._capital += realized_pnl_this_trade - self.BROKERAGE_EXIT
                logging.info(f"Closed short position. PnL: {realized_pnl_this_trade:.2f}")
                self._current_position_quantity = 0.0 # Position is now flat
                self._current_position_entry_price = 0.0

            # Open new long position or add to existing long
            self._capital -= cost
            total_value = (self._current_position_quantity * self._current_position_entry_price) + (quantity * price)
            self._current_position_quantity += quantity
            self._current_position_entry_price = total_value / self._current_position_quantity
            logging.info(f"Executed BUY_LONG. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}")

        elif action == "SELL_SHORT":
            cost = (price * quantity) + self.BROKERAGE_ENTRY
            if self._capital < cost: # Shorting also requires margin/capital to cover potential losses
                logging.warning(f"Insufficient capital to SELL_SHORT {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            if self._current_position_quantity > 0: # Currently long, so this is a close long + new short
                # Close long position first
                realized_pnl_this_trade += (price - self._current_position_entry_price) * self._current_position_quantity
                self._realized_pnl += realized_pnl_this_trade
                self._capital += realized_pnl_this_trade - self.BROKERAGE_EXIT
                logging.info(f"Closed long position. PnL: {realized_pnl_this_trade:.2f}")
                self._current_position_quantity = 0.0 # Position is now flat
                self._current_position_entry_price = 0.0

            # Open new short position or add to existing short
            self._capital -= cost # Assuming capital is reduced for shorting as well (margin requirement)
            total_value = (abs(self._current_position_quantity) * self._current_position_entry_price) + (quantity * price)
            self._current_position_quantity -= quantity # Negative for short position
            self._current_position_entry_price = total_value / abs(self._current_position_quantity)
            logging.info(f"Executed SELL_SHORT. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}")

        elif action == "CLOSE_LONG":
            if self._current_position_quantity <= 0: # No long position to close
                logging.warning(f"No long position to CLOSE_LONG. Current position: {self._current_position_quantity}. Trade not executed.")
                return 0.0, self._unrealized_pnl
            if quantity > self._current_position_quantity:
                logging.warning(f"Attempted to close {quantity} long, but only {self._current_position_quantity} held. Closing full position.")
                quantity = self._current_position_quantity

            realized_pnl_this_trade = (price - self._current_position_entry_price) * quantity
            self._realized_pnl += realized_pnl_this_trade
            self._capital += realized_pnl_this_trade - self.BROKERAGE_EXIT
            self._current_position_quantity -= quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
            logging.info(f"Executed CLOSE_LONG. Quantity: {quantity}, Price: {price}. PnL: {realized_pnl_this_trade:.2f}")

        elif action == "CLOSE_SHORT":
            if self._current_position_quantity >= 0: # No short position to close
                logging.warning(f"No short position to CLOSE_SHORT. Current position: {self._current_position_quantity}. Trade not executed.")
                return 0.0, self._unrealized_pnl
            if quantity > abs(self._current_position_quantity):
                logging.warning(f"Attempted to close {quantity} short, but only {abs(self._current_position_quantity)} held. Closing full position.")
                quantity = abs(self._current_position_quantity)

            realized_pnl_this_trade = (self._current_position_entry_price - price) * quantity
            self._realized_pnl += realized_pnl_this_trade
            self._capital += realized_pnl_this_trade - self.BROKERAGE_EXIT # Corrected capital update for CLOSE_SHORT
            self._current_position_quantity += quantity
            if self._current_position_quantity == 0:
                self._current_position_entry_price = 0.0
            logging.info(f"Executed CLOSE_SHORT. Quantity: {quantity}, Price: {price}. PnL: {realized_pnl_this_trade:.2f}")

        else:
            logging.warning(f"Unknown action: {action}. No trade executed.")
            return 0.0, self._unrealized_pnl

        # Update unrealized P&L based on current market price
        self._update_unrealized_pnl(price)

        # Record trade for history
        self._trade_history.append({
            "action": action,
            "price": price,
            "quantity": quantity,
            "realized_pnl_this_trade": realized_pnl_this_trade,
            "cost": cost,
            "capital_after_trade": self._capital,
            "position_after_trade": self._current_position_quantity,
            "entry_price_after_trade": self._current_position_entry_price,
            "unrealized_pnl_after_trade": self._unrealized_pnl
        })

        return realized_pnl_this_trade, self._unrealized_pnl

    def _update_unrealized_pnl(self, current_price: float):
        if self._current_position_quantity > 0:  # Long position
            self._unrealized_pnl = (current_price - self._current_position_entry_price) * self._current_position_quantity
        elif self._current_position_quantity < 0:  # Short position
            self._unrealized_pnl = (self._current_position_entry_price - current_price) * abs(self._current_position_quantity)
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
            "total_pnl": self._realized_pnl + self._unrealized_pnl
        }

    def get_trade_history(self) -> list:
        return self._trade_history
