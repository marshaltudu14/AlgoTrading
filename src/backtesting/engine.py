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
        self._stop_loss_price = 0.0
        self._target_profit_price = 0.0
        self._is_position_open = False

        self.BROKERAGE_ENTRY = 25.0  # INR
        self.BROKERAGE_EXIT = 35.0   # INR

    def reset(self):
        self._capital = self._initial_capital
        self._current_position_quantity = 0.0
        self._current_position_entry_price = 0.0
        self._realized_pnl = 0.0
        self._unrealized_pnl = 0.0
        self._trade_history = []
        self._stop_loss_price = 0.0
        self._target_profit_price = 0.0
        self._is_position_open = False

    def execute_trade(self, action: str, price: float, quantity: float, atr_value: float = 0.0) -> Tuple[float, float]:
        if price <= 0 or quantity <= 0:
            logging.warning(f"Invalid price ({price}) or quantity ({quantity}) for trade action {action}. Trade not executed.")
            return 0.0, self._unrealized_pnl

        realized_pnl_this_trade = 0.0
        cost = 0.0

        # Check for SL/TP hit if a position is open
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
            elif self._current_position_quantity < 0: # Short position
                if price >= self._stop_loss_price:
                    logging.info(f"SL hit for short position at {price:.2f}. Closing position.")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)
                elif price <= self._target_profit_price:
                    logging.info(f"TP hit for short position at {price:.2f}. Closing position.")
                    action = "CLOSE_SHORT"
                    quantity = abs(self._current_position_quantity)

        if action == "BUY_LONG":
            if self._current_position_quantity != 0:
                logging.warning(f"Cannot BUY_LONG. Already have an open position ({self._current_position_quantity}). Trade not executed.")
                return 0.0, self._unrealized_pnl
            cost = (price * quantity) + self.BROKERAGE_ENTRY
            if self._capital < cost:
                logging.warning(f"Insufficient capital to BUY_LONG {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            self._capital -= cost
            self._current_position_quantity = quantity
            self._current_position_entry_price = price
            self._is_position_open = True
            self._stop_loss_price = price - atr_value # SL is ATR below entry for long
            self._target_profit_price = price + (atr_value * 2) # TP is 2*ATR above entry for long
            logging.info(f"Executed BUY_LONG. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}. SL: {self._stop_loss_price:.2f}, TP: {self._target_profit_price:.2f}")

        elif action == "SELL_SHORT":
            if self._current_position_quantity != 0:
                logging.warning(f"Cannot SELL_SHORT. Already have an open position ({self._current_position_quantity}). Trade not executed.")
                return 0.0, self._unrealized_pnl
            cost = (price * quantity) + self.BROKERAGE_ENTRY
            if self._capital < cost: # Shorting also requires margin/capital to cover potential losses
                logging.warning(f"Insufficient capital to SELL_SHORT {quantity} at {price}. Capital: {self._capital:.2f}, Cost: {cost:.2f}. Trade not executed.")
                return 0.0, self._unrealized_pnl

            self._capital -= cost # Assuming capital is reduced for shorting as well (margin requirement)
            self._current_position_quantity = -quantity # Negative for short position
            self._current_position_entry_price = price
            self._is_position_open = True
            self._stop_loss_price = price + atr_value # SL is ATR above entry for short
            self._target_profit_price = price - (atr_value * 2) # TP is 2*ATR below entry for short
            logging.info(f"Executed SELL_SHORT. Quantity: {quantity}, Price: {price}. New position: {self._current_position_quantity:.2f} at {self._current_position_entry_price:.2f}. SL: {self._stop_loss_price:.2f}, TP: {self._target_profit_price:.2f}")

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
            "total_pnl": self._realized_pnl + self._unrealized_pnl,
            "is_position_open": self._is_position_open
        }

    def get_trade_history(self) -> list:
        return self._trade_history
