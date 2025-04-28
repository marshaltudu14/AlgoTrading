"""
Risk management module for AlgoTrading.
Handles risk assessment, position sizing, and capital allocation.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, time
import pytz

from fyers_apiv3 import fyersModel

from core.logging_setup import get_logger
from core.config import (
    INSTRUMENTS, 
    QUANTITIES, 
    MARGIN_REQUIREMENTS,
    INITIAL_CAPITAL,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME
)
from trading.broker_api import get_funds, get_market_status
from trading.position_manager import PositionManager

logger = get_logger(__name__)
ist_timezone = pytz.timezone("Asia/Kolkata")


class RiskManager:
    """Class to manage trading risk."""
    
    def __init__(
        self,
        position_manager: PositionManager,
        max_positions: int = 3,
        max_drawdown_pct: float = 0.05,
        max_capital_per_trade_pct: float = 0.1,
        max_capital_per_instrument_pct: float = 0.2,
        max_daily_loss_pct: float = 0.02
    ):
        """
        Initialize risk manager.
        
        Args:
            position_manager: Position manager
            max_positions: Maximum number of open positions
            max_drawdown_pct: Maximum drawdown percentage
            max_capital_per_trade_pct: Maximum capital percentage per trade
            max_capital_per_instrument_pct: Maximum capital percentage per instrument
            max_daily_loss_pct: Maximum daily loss percentage
        """
        self.position_manager = position_manager
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.max_capital_per_instrument_pct = max_capital_per_instrument_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        
        # Initialize tracking variables
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_start_capital = position_manager.current_capital
        self.last_reset_date = datetime.now(ist_timezone).date()
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics."""
        today = datetime.now(ist_timezone).date()
        if today != self.last_reset_date:
            logger.info(f"Resetting daily metrics for {today}")
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_start_capital = self.position_manager.current_capital
            self.last_reset_date = today
    
    def update_daily_pnl(self) -> None:
        """Update daily P&L."""
        self.reset_daily_metrics()
        self.daily_pnl = self.position_manager.get_total_pnl()
    
    def can_trade(self, fyers: Optional[fyersModel.FyersModel] = None) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on risk parameters.
        
        Args:
            fyers: Fyers API client for market status check
            
        Returns:
            Tuple of (can_trade, reason)
        """
        self.reset_daily_metrics()
        
        # Check market hours
        if not self.is_market_open(fyers):
            return False, "Market is closed"
        
        # Check max positions
        if self.position_manager.get_position_count() >= self.max_positions:
            return False, f"Maximum positions reached ({self.max_positions})"
        
        # Check max drawdown
        if self.position_manager.get_max_drawdown() >= self.max_drawdown_pct:
            return False, f"Maximum drawdown reached ({self.max_drawdown_pct:.1%})"
        
        # Check daily loss
        daily_loss_pct = -self.daily_pnl / self.daily_start_capital
        if daily_loss_pct >= self.max_daily_loss_pct:
            return False, f"Maximum daily loss reached ({self.max_daily_loss_pct:.1%})"
        
        return True, "Trading allowed"
    
    def can_trade_instrument(self, instrument: str) -> Tuple[bool, str]:
        """
        Check if trading is allowed for a specific instrument.
        
        Args:
            instrument: Instrument to check
            
        Returns:
            Tuple of (can_trade, reason)
        """
        # Check if already has position for this instrument
        if self.position_manager.has_open_position(instrument):
            return False, f"Already has position for {instrument}"
        
        # Check max capital per instrument
        exposure = self.position_manager.get_exposure_by_instrument()
        instrument_exposure = exposure.get(instrument, 0)
        max_instrument_capital = self.position_manager.initial_capital * self.max_capital_per_instrument_pct
        
        if instrument_exposure >= max_instrument_capital:
            return False, f"Maximum capital for {instrument} reached ({self.max_capital_per_instrument_pct:.1%})"
        
        return True, "Trading allowed for instrument"
    
    def get_position_size(self, instrument: str, price: float) -> int:
        """
        Calculate position size based on risk parameters.
        
        Args:
            instrument: Instrument to trade
            price: Current price
            
        Returns:
            Position size (quantity)
        """
        # Get standard lot size
        standard_lot = QUANTITIES.get(instrument, 1)
        
        # Calculate maximum quantity based on capital per trade
        max_capital = self.position_manager.current_capital * self.max_capital_per_trade_pct
        max_qty = int(max_capital / (price * standard_lot)) * standard_lot
        
        # Ensure at least one lot
        return max(standard_lot, max_qty)
    
    def is_market_open(self, fyers: Optional[fyersModel.FyersModel] = None) -> bool:
        """
        Check if market is open.
        
        Args:
            fyers: Fyers API client for market status check
            
        Returns:
            True if market is open
        """
        # Check via API if available
        if fyers:
            try:
                response = get_market_status(fyers)
                if response.get('s') == 'ok':
                    market_status = response.get('marketStatus', [])
                    for status in market_status:
                        if status.get('exchange') in ['NSE', 'BSE']:
                            return status.get('status') == 'open'
            except Exception as e:
                logger.error(f"Error checking market status: {e}")
        
        # Fallback to time check
        now = datetime.now(ist_timezone).time()
        open_time = datetime.strptime(MARKET_OPEN_TIME, "%H:%M").time()
        close_time = datetime.strptime(MARKET_CLOSE_TIME, "%H:%M").time()
        
        return open_time <= now <= close_time
    
    def get_risk_assessment(self, instrument: str, action: str) -> Dict[str, Any]:
        """
        Assess risk for a potential trade.
        
        Args:
            instrument: Instrument to trade
            action: Action to take
            
        Returns:
            Risk assessment dictionary
        """
        # Get current capital and exposure
        current_capital = self.position_manager.current_capital
        total_exposure = self.position_manager.get_total_exposure()
        exposure_pct = total_exposure / self.position_manager.initial_capital
        
        # Get instrument exposure
        exposure_by_instrument = self.position_manager.get_exposure_by_instrument()
        instrument_exposure = exposure_by_instrument.get(instrument, 0)
        instrument_exposure_pct = instrument_exposure / self.position_manager.initial_capital
        
        # Get daily metrics
        daily_pnl_pct = self.daily_pnl / self.daily_start_capital
        
        # Calculate risk score (0-1, higher is riskier)
        position_risk = self.position_manager.get_position_count() / self.max_positions
        drawdown_risk = self.position_manager.get_max_drawdown() / self.max_drawdown_pct
        exposure_risk = exposure_pct / self.max_capital_per_instrument_pct
        instrument_risk = instrument_exposure_pct / self.max_capital_per_instrument_pct
        daily_loss_risk = max(0, -daily_pnl_pct / self.max_daily_loss_pct)
        
        # Combine risk factors
        risk_score = max(
            position_risk,
            drawdown_risk,
            exposure_risk,
            instrument_risk,
            daily_loss_risk
        )
        
        return {
            "risk_score": risk_score,
            "position_risk": position_risk,
            "drawdown_risk": drawdown_risk,
            "exposure_risk": exposure_risk,
            "instrument_risk": instrument_risk,
            "daily_loss_risk": daily_loss_risk,
            "current_capital": current_capital,
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "instrument_exposure": instrument_exposure,
            "instrument_exposure_pct": instrument_exposure_pct,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "max_drawdown": self.position_manager.get_max_drawdown(),
            "win_rate": self.position_manager.get_win_rate()
        }
    
    def should_exit_all(self) -> Tuple[bool, str]:
        """
        Check if all positions should be exited.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        # Check market hours
        if not self.is_market_open():
            return True, "Market is closed"
        
        # Check max drawdown
        if self.position_manager.get_max_drawdown() >= self.max_drawdown_pct * 1.5:
            return True, f"Critical drawdown reached ({self.max_drawdown_pct * 1.5:.1%})"
        
        # Check daily loss
        daily_loss_pct = -self.daily_pnl / self.daily_start_capital
        if daily_loss_pct >= self.max_daily_loss_pct * 1.5:
            return True, f"Critical daily loss reached ({self.max_daily_loss_pct * 1.5:.1%})"
        
        return False, "No need to exit all positions"
