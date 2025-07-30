class Instrument:
    def __init__(self, symbol: str, lot_size: int, tick_size: float):
        self.symbol = symbol
        self.lot_size = lot_size
        self.tick_size = tick_size

    def __repr__(self):
        return f"Instrument(symbol='{self.symbol}', lot_size={self.lot_size}, tick_size={self.tick_size})"