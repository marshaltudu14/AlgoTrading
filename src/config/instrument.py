"""
Instrument configuration and data structure for trading system.
"""

from dataclasses import dataclass


@dataclass
class Instrument:
    """
    Represents a trading instrument with its configuration parameters.
    
    Attributes:
        symbol: Trading symbol identifier
        lot_size: Number of units per lot
        tick_size: Minimum price movement
        instrument_type: Type of instrument ("stock" or "index")
        option_premium_range: For index instruments, [min_premium_%, max_premium_%]
    """
    symbol: str
    lot_size: int
    tick_size: float
    instrument_type: str = "stock"
    option_premium_range: list = None
    
    def __str__(self) -> str:
        """String representation of the instrument."""
        return (f"Instrument(symbol={self.symbol}, lot_size={self.lot_size}, "
                f"tick_size={self.tick_size}, type={self.instrument_type})")
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
    
    def is_stock(self) -> bool:
        """Check if this is a stock instrument."""
        return self.instrument_type.lower() == "stock"
    
    def is_index(self) -> bool:
        """Check if this is an index instrument."""
        return self.instrument_type.lower() == "index"
    
    def calculate_lot_value(self, price: float) -> float:
        """Calculate the total value of one lot at given price."""
        return price * self.lot_size
    
    def round_to_tick(self, price: float) -> float:
        """Round price to the nearest tick size."""
        return round(price / self.tick_size) * self.tick_size