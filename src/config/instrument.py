from typing import List, Optional

class Instrument:
    def __init__(self, symbol: str, lot_size: int, tick_size: float,
                 instrument_type: str = "stock", option_premium_range: Optional[List[float]] = None):
        self.symbol = symbol
        self.lot_size = lot_size
        self.tick_size = tick_size
        self.type = instrument_type  # "stock" or "index"
        self.option_premium_range = option_premium_range or [0.025, 0.05]  # Default 2.5% to 5%

    def __repr__(self):
        return f"Instrument(symbol='{self.symbol}', type='{self.type}', lot_size={self.lot_size}, tick_size={self.tick_size}, option_premium_range={self.option_premium_range})"