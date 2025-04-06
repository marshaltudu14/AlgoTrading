# src/position_manager.py
"""
Tracks the bot's live positions, entry prices, quantities, and calculates P&L.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class PositionManager:
    def __init__(self, order_manager):
        """
        Initializes the PositionManager.

        Args:
            order_manager: An instance of OrderManager to fetch position data.
        """
        self.order_manager = order_manager
        self.positions = {} # Dictionary to store current positions {symbol: {'qty': int, 'entry_price': float, ...}}
        # TODO: Potentially load initial positions if restarting

    def update_positions(self):
        """
        Fetches the latest positions from the broker via OrderManager
        and updates the internal state.
        """
        # TODO: Implement fetching positions from OrderManager and updating self.positions
        logger.debug("Updating positions...")
        try:
            broker_positions = self.order_manager.get_positions()
            if broker_positions and broker_positions.get('s') == 'ok':
                # Process the position data from Fyers API format
                # Reset internal state and rebuild based on broker data
                self.positions = {}
                for pos in broker_positions.get('netPositions', []):
                    symbol = pos.get('symbol')
                    qty = pos.get('netQty', 0)
                    if qty != 0: # Only track open positions
                         self.positions[symbol] = {
                             'qty': qty,
                             'entry_price': pos.get('avgPrice', 0), # Or calculate based on trades if needed
                             'side': 1 if qty > 0 else -1,
                             # Add other relevant fields like realized/unrealized PnL if available
                             'unrealized_pnl': pos.get('unrealized_profit', 0),
                             'realized_pnl': pos.get('realized_profit', 0),
                         }
                logger.info(f"Positions updated: {self.positions}")
            elif broker_positions:
                 logger.warning(f"Could not update positions from broker. Response: {broker_positions}")
            else:
                 logger.error("Failed to fetch positions from broker (None response).")

        except Exception as e:
            logger.error(f"Error updating positions: {e}", exc_info=True)


    def get_position(self, symbol: str) -> dict | None:
        """
        Returns the current position details for a specific symbol.

        Args:
            symbol: The trading symbol.

        Returns:
            A dictionary containing position details (qty, entry_price, etc.)
            or None if no position exists for the symbol.
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> dict:
        """Returns the dictionary of all current positions."""
        return self.positions

    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """
        Calculates the unrealized P&L for a specific position.

        Args:
            symbol: The trading symbol.
            current_price: The current market price of the symbol.

        Returns:
            The unrealized P&L, or 0 if no position exists.
        """
        position = self.get_position(symbol)
        if position:
            entry_price = position.get('entry_price', 0)
            qty = position.get('qty', 0)
            side = position.get('side', 0)
            if side == 1: # Long
                return (current_price - entry_price) * qty
            elif side == -1: # Short
                return (entry_price - current_price) * abs(qty) # Qty is negative for short
        return 0.0

    # TODO: Add methods to manually update positions based on order fills if needed,
    # especially if broker position updates are delayed.

if __name__ == '__main__':
    # Example usage or testing
    logging.basicConfig(level=logging.INFO)
    logger.info("PositionManager module")
    # Add test code here if needed (requires mocked OrderManager or live connection)
