"""
Instrument configuration for trading
"""

class InstrumentConfig:
    def __init__(self, id, name, symbol, exchangeSymbol, type, lotSize, tickSize):
        self.id = id
        self.name = name
        self.symbol = symbol
        self.exchangeSymbol = exchangeSymbol
        self.type = type
        self.lotSize = lotSize
        self.tickSize = tickSize


class Timeframe:
    def __init__(self, id, name, description, days):
        self.id = id
        self.name = name
        self.description = description
        self.days = days


INSTRUMENTS = [
    InstrumentConfig(
        id=0,
        name="Bank Nifty",
        symbol="Bank_Nifty",
        exchangeSymbol="NSE:NIFTYBANK-INDEX",
        type="index",
        lotSize=35,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=1,
        name="Nifty 50",
        symbol="Nifty",
        exchangeSymbol="NSE:NIFTY50-INDEX",
        type="index",
        lotSize=75,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=2,
        name="Bankex",
        symbol="Bankex",
        exchangeSymbol="NSE:BANKEX-INDEX",
        type="index",
        lotSize=30,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=3,
        name="Finnifty",
        symbol="Finnifty",
        exchangeSymbol="NSE:FINNIFTY-INDEX",
        type="index",
        lotSize=65,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=4,
        name="Sensex",
        symbol="Sensex",
        exchangeSymbol="BSE:SENSEX-INDEX",
        type="index",
        lotSize=20,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=5,
        name="Reliance Industries",
        symbol="RELIANCE",
        exchangeSymbol="NSE:RELIANCE-EQ",
        type="stock",
        lotSize=1,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=6,
        name="Tata Consultancy Services",
        symbol="TCS",
        exchangeSymbol="NSE:TCS-EQ",
        type="stock",
        lotSize=1,
        tickSize=0.05,
    ),
    InstrumentConfig(
        id=7,
        name="HDFC Bank",
        symbol="HDFC",
        exchangeSymbol="NSE:HDFCBANK-EQ",
        type="stock",
        lotSize=1,
        tickSize=0.05,
    ),
]

TIMEFRAMES = [
    Timeframe(id=0, name="1", description="1 minute", days=7),
    Timeframe(id=1, name="2", description="2 minutes", days=7),
    Timeframe(id=2, name="3", description="3 minutes", days=10),
    Timeframe(id=3, name="5", description="5 minutes", days=15),
    Timeframe(id=4, name="10", description="10 minutes", days=20),
    Timeframe(id=5, name="15", description="15 minutes", days=30),
    Timeframe(id=6, name="20", description="20 minutes", days=30),
    Timeframe(id=7, name="30", description="30 minutes", days=45),
    Timeframe(id=8, name="45", description="45 minutes", days=60),
    Timeframe(id=9, name="60", description="1 hour", days=90),
    Timeframe(id=10, name="120", description="2 hours", days=100),
    Timeframe(id=11, name="180", description="3 hours", days=100),
    Timeframe(id=12, name="240", description="4 hours", days=100),
]

DEFAULT_INSTRUMENT = INSTRUMENTS[1]  # Nifty 50
DEFAULT_TIMEFRAME = TIMEFRAMES[3]  # 5 minutes
DEFAULT_LOT_SIZE = DEFAULT_INSTRUMENT.lotSize