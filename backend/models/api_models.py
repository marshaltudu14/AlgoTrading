"""
API Models for AlgoTrading System
Pydantic models for request/response validation
"""
from pydantic import BaseModel
from typing import Optional
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode enumeration for paper vs real trading"""
    PAPER = "paper"
    REAL = "real"


class LoginRequest(BaseModel):
    app_id: str
    secret_key: str
    redirect_uri: str
    fy_id: str
    pin: str
    totp_secret: str


class BacktestRequest(BaseModel):
    instrument: str
    timeframe: str
    duration: int
    initial_capital: float = 100000


class LiveTradingRequest(BaseModel):
    instrument: str
    timeframe: str
    option_strategy: Optional[str] = "ITM"
    trading_mode: TradingMode = TradingMode.REAL


class ManualTradeRequest(BaseModel):
    instrument: str
    direction: str
    quantity: int
    stopLoss: Optional[float] = None
    target: Optional[float] = None
    trading_mode: TradingMode = TradingMode.REAL