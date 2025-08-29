from enum import Enum


class TradingMode(Enum):
    """Trading environment modes."""
    TRAINING = "training"
    BACKTESTING = "backtesting"
    LIVE = "live"