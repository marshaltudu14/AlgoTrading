"""
Position Data Model
"""
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    """Data model for an active trading position."""
    instrument: str
    direction: str
    entry_price: float
    quantity: int
    stop_loss: float
    target_price: float
    entry_time: datetime
    trade_type: str
    current_pnl: float = 0.0
