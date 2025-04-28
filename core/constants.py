"""
Constants for the AlgoTrading system.
Defines enums and constants used throughout the application.
"""
from enum import Enum, auto


class OrderType(Enum):
    """Order types for trading."""
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4


class OrderSide(Enum):
    """Order sides for trading."""
    BUY = 1
    SELL = 2


class OptionType(Enum):
    """Option types for trading."""
    CALL = "CE"
    PUT = "PE"


class OrderStatus(Enum):
    """Order statuses for tracking."""
    PENDING = auto()
    OPEN = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    PARTIALLY_FILLED = auto()


class PositionStatus(Enum):
    """Position statuses for tracking."""
    OPEN = auto()
    CLOSED = auto()


class SignalType(Enum):
    """Signal types for model predictions."""
    HOLD = 0
    BUY_TARGET_HIT = 1
    BUY_SL_HIT = 2
    SELL_TARGET_HIT = 3
    SELL_SL_HIT = 4


class ActionType(Enum):
    """Action types for model predictions."""
    HOLD = 0
    BUY_CE = 1  # Buy Call Option
    BUY_PE = 2  # Buy Put Option


class MarketRegime(Enum):
    """Market regime types for feature engineering."""
    RANGING = 0
    TRENDING = 1
    VOLATILE = 2


# Fyers API constants
FYERS_EXCHANGE_NSE = 10
FYERS_EXCHANGE_BSE = 11
FYERS_SEGMENT_FO = 11  # Future & Options segment

# Fyers order types
FYERS_ORDER_TYPE_MARKET = 2
FYERS_ORDER_TYPE_LIMIT = 1
FYERS_ORDER_TYPE_STOP = 3
FYERS_ORDER_TYPE_STOP_LIMIT = 4

# Fyers order sides
FYERS_ORDER_SIDE_BUY = 1
FYERS_ORDER_SIDE_SELL = -1

# Fyers product types
FYERS_PRODUCT_TYPE_INTRADAY = "INTRADAY"
FYERS_PRODUCT_TYPE_MARGIN = "MARGIN"
FYERS_PRODUCT_TYPE_CNC = "CNC"

# Fyers order validity
FYERS_VALIDITY_DAY = "DAY"
FYERS_VALIDITY_IOC = "IOC"

# Fyers order status
FYERS_ORDER_STATUS_PENDING = 1
FYERS_ORDER_STATUS_FILLED = 2
FYERS_ORDER_STATUS_CANCELLED = 3
FYERS_ORDER_STATUS_REJECTED = 4
FYERS_ORDER_STATUS_PARTIALLY_FILLED = 5

# Fyers position types
FYERS_POSITION_TYPE_NET = "NET"
FYERS_POSITION_TYPE_DAY = "DAY"

# Fyers websocket data types
FYERS_WS_DATA_TYPE_LTP = 1
FYERS_WS_DATA_TYPE_QUOTE = 2
FYERS_WS_DATA_TYPE_DEPTH = 3

# Fyers websocket modes
FYERS_WS_MODE_SUBSCRIBE = 1
FYERS_WS_MODE_UNSUBSCRIBE = 2
