"""
Capital-aware quantity selection utility.
Adjusts predicted quantities based on available capital and instrument specifications.
"""

import logging
from typing import Tuple, Optional
from src.config.instrument import Instrument

logger = logging.getLogger(__name__)

class CapitalAwareQuantitySelector:
    """
    Utility class to adjust predicted quantities based on available capital.
    
    This ensures that agents don't predict quantities that exceed available capital,
    preventing trade rejections due to insufficient funds.
    """
    
    def __init__(self, brokerage_entry: float = 25.0):
        """
        Initialize the capital-aware quantity selector.
        
        Args:
            brokerage_entry: Entry brokerage cost per trade
        """
        self.brokerage_entry = brokerage_entry
    
    def adjust_quantity_for_capital(
        self,
        predicted_quantity: float,
        available_capital: float,
        current_price: float,
        instrument: Instrument,
        proxy_premium: Optional[float] = None
    ) -> int:
        """
        Adjust predicted quantity based on available capital.

        Args:
            predicted_quantity: Raw quantity prediction from model (can be any positive value)
            available_capital: Available capital for trading
            current_price: Current market price
            instrument: Instrument specification
            proxy_premium: Premium for options (if applicable)

        Returns:
            Adjusted integer quantity that fits within available capital
        """
        # Calculate cost per lot
        # Calculate cost per lot
        # For all instruments, use direct price calculation
        cost_per_lot = current_price * instrument.lot_size

        # Calculate maximum affordable quantity
        max_affordable_quantity = int((available_capital - self.brokerage_entry) // cost_per_lot)
        max_affordable_quantity = max(0, max_affordable_quantity)

        # If predicted_quantity is very high (like 100), return max affordable
        # This is used to find the maximum possible quantity
        if predicted_quantity >= 50:
            return max_affordable_quantity

        # For normal predictions, ensure predicted quantity is an integer in valid range
        predicted_quantity = max(1, int(round(predicted_quantity)))

        # Adjust predicted quantity to what's affordable
        adjusted_quantity = min(predicted_quantity, max_affordable_quantity)

        # Ensure at least 1 lot if any capital is available
        if adjusted_quantity == 0 and available_capital > (cost_per_lot + self.brokerage_entry):
            adjusted_quantity = 1

        # Log adjustment if quantity was reduced (only for normal predictions, not max queries)
        if predicted_quantity < 50 and adjusted_quantity < predicted_quantity:
            logger.info(f"💰 Quantity adjusted: {predicted_quantity} → {adjusted_quantity} lots")
            logger.info(f"   Available capital: ₹{available_capital:.2f}")
            logger.info(f"   Cost per lot: ₹{cost_per_lot:.2f}")
            logger.info(f"   Max affordable: {max_affordable_quantity} lots")

        return adjusted_quantity

    def get_max_affordable_quantity(
        self,
        available_capital: float,
        current_price: float,
        instrument: Instrument,
        proxy_premium: Optional[float] = None
    ) -> int:
        """
        Calculate the maximum quantity that can be afforded with available capital.
        No artificial limits - purely based on capital constraints.

        Args:
            available_capital: Available capital for trading
            current_price: Current market price
            instrument: Instrument specification
            proxy_premium: Premium for options (if applicable)

        Returns:
            Maximum affordable quantity (integer)
        """
        # Calculate cost per lot
        # Calculate cost per lot
        # For all instruments, use direct price calculation
        cost_per_lot = current_price * instrument.lot_size

        # Calculate maximum affordable quantity (no artificial limits)
        if cost_per_lot <= 0:
            return 0

        max_affordable_quantity = int((available_capital - self.brokerage_entry) // cost_per_lot)
        return max(0, max_affordable_quantity)

    def calculate_trade_cost(
        self,
        quantity: int,
        current_price: float,
        instrument: Instrument,
        proxy_premium: Optional[float] = None
    ) -> float:
        """
        Calculate the total cost of a trade.

        Args:
            quantity: Number of lots to trade
            current_price: Current market price
            instrument: Instrument specification
            proxy_premium: Premium for options (if applicable)

        Returns:
            Total cost including brokerage
        """
        # Calculate cost per lot based on instrument type
        # Calculate cost per lot
        # For all instruments, use direct price calculation
        cost_per_lot = current_price * instrument.lot_size

        # Total cost including brokerage
        total_cost = (cost_per_lot * quantity) + self.brokerage_entry

        return total_cost
    
    def get_affordable_quantities(
        self,
        available_capital: float,
        current_price: float,
        instrument: Instrument,
        proxy_premium: Optional[float] = None
    ) -> list:
        """
        Get list of affordable quantities (1-5) given available capital.
        
        Args:
            available_capital: Available capital for trading
            current_price: Current market price
            instrument: Instrument specification
            proxy_premium: Premium for options (if applicable)
            
        Returns:
            List of affordable quantities
        """
        affordable_quantities = []
        
        for quantity in range(1, 6):  # 1 to 5 lots
            cost = self.calculate_trade_cost(quantity, current_price, instrument, proxy_premium)
            if cost <= available_capital:
                affordable_quantities.append(quantity)
        
        return affordable_quantities
    
    def get_capital_utilization_ratio(
        self,
        quantity: int,
        available_capital: float,
        current_price: float,
        instrument: Instrument,
        proxy_premium: Optional[float] = None
    ) -> float:
        """
        Calculate what percentage of available capital would be used.
        
        Args:
            quantity: Number of lots to trade
            available_capital: Available capital for trading
            current_price: Current market price
            instrument: Instrument specification
            proxy_premium: Premium for options (if applicable)
            
        Returns:
            Capital utilization ratio (0.0 to 1.0)
        """
        if available_capital <= 0:
            return 1.0
        
        cost = self.calculate_trade_cost(quantity, current_price, instrument, proxy_premium)
        return min(1.0, cost / available_capital)

# Global instance for easy access
capital_aware_selector = CapitalAwareQuantitySelector()

def adjust_quantity_for_capital(
    predicted_quantity: float,
    available_capital: float,
    current_price: float,
    instrument: Instrument,
    proxy_premium: Optional[float] = None
) -> int:
    """
    Convenience function to adjust quantity for capital constraints.
    
    Args:
        predicted_quantity: Raw quantity prediction from model
        available_capital: Available capital for trading
        current_price: Current market price
        instrument: Instrument specification
        proxy_premium: Premium for options (if applicable)
        
    Returns:
        Adjusted integer quantity
    """
    return capital_aware_selector.adjust_quantity_for_capital(
        predicted_quantity, available_capital, current_price, instrument, proxy_premium
    )

def get_affordable_quantities(
    available_capital: float,
    current_price: float,
    instrument: Instrument,
    proxy_premium: Optional[float] = None
) -> list:
    """
    Convenience function to get affordable quantities.
    
    Args:
        available_capital: Available capital for trading
        current_price: Current market price
        instrument: Instrument specification
        proxy_premium: Premium for options (if applicable)
        
    Returns:
        List of affordable quantities
    """
    return capital_aware_selector.get_affordable_quantities(
        available_capital, current_price, instrument, proxy_premium
    )
