class Instrument:
    def __init__(self, symbol: str, type: str, lot_size: int, tick_size: float):
        self.symbol = symbol
        self.type = type
        self.lot_size = lot_size
        self.tick_size = tick_size

    def __repr__(self):
        return f"Instrument(symbol='{self.symbol}', type='{self.type}', lot_size={self.lot_size}, tick_size={self.tick_size})"