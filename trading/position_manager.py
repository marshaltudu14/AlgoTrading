"""
Position management module for AlgoTrading.
Handles position tracking, risk management, and P&L calculation.
"""
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import pandas as pd

from fyers_apiv3 import fyersModel

from core.logging_setup import get_logger
from core.config import (
    INSTRUMENTS, 
    QUANTITIES, 
    MARGIN_REQUIREMENTS,
    INITIAL_CAPITAL
)
from core.constants import (
    PositionStatus,
    OptionType
)
from trading.broker_api import get_positions, get_funds

logger = get_logger(__name__)


class Position:
    """Class to track a single position."""
    
    def __init__(
        self,
        symbol: str,
        instrument: str,
        entry_price: float,
        quantity: int,
        entry_time: datetime,
        option_type: str,
        strike_price: float,
        expiry_date: str,
        stop_loss: float,
        take_profit: float
    ):
        """
        Initialize a position.
        
        Args:
            symbol: Symbol of the position
            instrument: Underlying instrument
            entry_price: Entry price
            quantity: Quantity
            entry_time: Entry time
            option_type: Option type (CE or PE)
            strike_price: Strike price
            expiry_date: Expiry date
            stop_loss: Stop loss price
            take_profit: Take profit price
        """
        self.symbol = symbol
        self.instrument = instrument
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = entry_time
        self.option_type = option_type
        self.strike_price = strike_price
        self.expiry_date = expiry_date
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Initialize tracking variables
        self.exit_price = None
        self.exit_time = None
        self.status = PositionStatus.OPEN
        self.duration = 0
        self.max_profit = 0.0
        self.max_loss = 0.0
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.exit_reason = None
    
    def update(self, current_price: float, current_time: datetime) -> None:
        """
        Update position with current price and time.
        
        Args:
            current_price: Current price
            current_time: Current time
        """
        self.current_price = current_price
        
        # Update duration
        if self.status == PositionStatus.OPEN:
            self.duration = (current_time - self.entry_time).total_seconds() / 60  # in minutes
        
        # Update unrealized P&L
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        
        # Update max profit/loss
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        elif self.unrealized_pnl < self.max_loss:
            self.max_loss = self.unrealized_pnl
        
        # Check for stop loss or take profit
        if self.status == PositionStatus.OPEN:
            if (self.option_type == OptionType.CALL.value and current_price <= self.stop_loss) or \
               (self.option_type == OptionType.PUT.value and current_price >= self.stop_loss):
                self.exit(current_price, current_time, "SL")
            elif (self.option_type == OptionType.CALL.value and current_price >= self.take_profit) or \
                 (self.option_type == OptionType.PUT.value and current_price <= self.take_profit):
                self.exit(current_price, current_time, "TP")
    
    def exit(self, exit_price: float, exit_time: datetime, reason: str) -> None:
        """
        Exit the position.
        
        Args:
            exit_price: Exit price
            exit_time: Exit time
            reason: Exit reason
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = PositionStatus.CLOSED
        self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        self.unrealized_pnl = 0.0
        self.exit_reason = reason
        
        logger.info(f"Exited position {self.symbol}: {reason} at {exit_price}, P&L: {self.realized_pnl:.2f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary.
        
        Returns:
            Dictionary representation of position
        """
        return {
            "symbol": self.symbol,
            "instrument": self.instrument,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "option_type": self.option_type,
            "strike_price": self.strike_price,
            "expiry_date": self.expiry_date,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_price": self.exit_price,
            "exit_time": self.exit_time,
            "status": self.status.name,
            "duration": self.duration,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "exit_reason": self.exit_reason
        }


class PositionManager:
    """Class to manage multiple positions."""
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        """
        Initialize position manager.
        
        Args:
            initial_capital: Initial capital
        """
        self.positions = {}  # symbol -> Position
        self.closed_positions = []  # list of closed Position objects
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_drawdown = 0.0
        self.peak_capital = initial_capital
    
    def add_position(
        self,
        symbol: str,
        instrument: str,
        entry_price: float,
        quantity: int,
        entry_time: datetime,
        option_type: str,
        strike_price: float,
        expiry_date: str,
        stop_loss: float,
        take_profit: float
    ) -> Position:
        """
        Add a new position.
        
        Args:
            symbol: Symbol of the position
            instrument: Underlying instrument
            entry_price: Entry price
            quantity: Quantity
            entry_time: Entry time
            option_type: Option type (CE or PE)
            strike_price: Strike price
            expiry_date: Expiry date
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            New position
        """
        # Check if we already have a position for this symbol
        if symbol in self.positions:
            logger.warning(f"Position already exists for {symbol}")
            return self.positions[symbol]
        
        # Check if we have enough capital
        position_cost = entry_price * quantity
        if position_cost > self.current_capital:
            logger.warning(f"Not enough capital for position: {position_cost} > {self.current_capital}")
            return None
        
        # Create new position
        position = Position(
            symbol=symbol,
            instrument=instrument,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=entry_time,
            option_type=option_type,
            strike_price=strike_price,
            expiry_date=expiry_date,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Add to positions
        self.positions[symbol] = position
        
        # Update capital
        self.current_capital -= position_cost
        
        logger.info(f"Added position {symbol}: {quantity} @ {entry_price}, capital: {self.current_capital:.2f}")
        
        return position
    
    def update_positions(
        self,
        prices: Dict[str, float],
        current_time: datetime
    ) -> None:
        """
        Update all positions with current prices.
        
        Args:
            prices: Dictionary of symbol -> price
            current_time: Current time
        """
        # Update each position
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                position.update(prices[symbol], current_time)
                
                # If position is closed, move to closed_positions
                if position.status == PositionStatus.CLOSED:
                    self.closed_positions.append(position)
                    del self.positions[symbol]
                    
                    # Update capital
                    self.current_capital += position.exit_price * position.quantity
        
        # Update peak capital and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
    
    def exit_position(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        reason: str
    ) -> Optional[Position]:
        """
        Exit a position.
        
        Args:
            symbol: Symbol of the position
            exit_price: Exit price
            exit_time: Exit time
            reason: Exit reason
            
        Returns:
            Closed position or None if not found
        """
        if symbol not in self.positions:
            logger.warning(f"Position not found for {symbol}")
            return None
        
        # Get position
        position = self.positions[symbol]
        
        # Exit position
        position.exit(exit_price, exit_time, reason)
        
        # Move to closed_positions
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Update capital
        self.current_capital += exit_price * position.quantity
        
        logger.info(f"Exited position {symbol}: {reason} at {exit_price}, capital: {self.current_capital:.2f}")
        
        return position
    
    def exit_all_positions(
        self,
        prices: Dict[str, float],
        exit_time: datetime,
        reason: str
    ) -> None:
        """
        Exit all positions.
        
        Args:
            prices: Dictionary of symbol -> price
            exit_time: Exit time
            reason: Exit reason
        """
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                self.exit_position(symbol, prices[symbol], exit_time, reason)
            else:
                logger.warning(f"No price for {symbol}, using current price")
                self.exit_position(symbol, position.current_price, exit_time, reason)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get a position by symbol.
        
        Args:
            symbol: Symbol of the position
            
        Returns:
            Position or None if not found
        """
        return self.positions.get(symbol)
    
    def get_positions_for_instrument(self, instrument: str) -> List[Position]:
        """
        Get all positions for an instrument.
        
        Args:
            instrument: Instrument name
            
        Returns:
            List of positions
        """
        return [p for p in self.positions.values() if p.instrument == instrument]
    
    def has_open_position(self, instrument: str) -> bool:
        """
        Check if there is an open position for an instrument.
        
        Args:
            instrument: Instrument name
            
        Returns:
            True if there is an open position
        """
        return any(p.instrument == instrument for p in self.positions.values())
    
    def get_position_count(self) -> int:
        """
        Get number of open positions.
        
        Returns:
            Number of open positions
        """
        return len(self.positions)
    
    def get_total_exposure(self) -> float:
        """
        Get total exposure (sum of position costs).
        
        Returns:
            Total exposure
        """
        return sum(p.entry_price * p.quantity for p in self.positions.values())
    
    def get_exposure_by_instrument(self) -> Dict[str, float]:
        """
        Get exposure by instrument.
        
        Returns:
            Dictionary of instrument -> exposure
        """
        exposure = {}
        for position in self.positions.values():
            if position.instrument not in exposure:
                exposure[position.instrument] = 0
            exposure[position.instrument] += position.entry_price * position.quantity
        return exposure
    
    def get_unrealized_pnl(self) -> float:
        """
        Get total unrealized P&L.
        
        Returns:
            Total unrealized P&L
        """
        return sum(p.unrealized_pnl for p in self.positions.values())
    
    def get_realized_pnl(self) -> float:
        """
        Get total realized P&L.
        
        Returns:
            Total realized P&L
        """
        return sum(p.realized_pnl for p in self.closed_positions)
    
    def get_total_pnl(self) -> float:
        """
        Get total P&L (realized + unrealized).
        
        Returns:
            Total P&L
        """
        return self.get_realized_pnl() + self.get_unrealized_pnl()
    
    def get_win_rate(self) -> float:
        """
        Get win rate (percentage of profitable closed positions).
        
        Returns:
            Win rate (0-1)
        """
        if not self.closed_positions:
            return 0.0
        
        wins = sum(1 for p in self.closed_positions if p.realized_pnl > 0)
        return wins / len(self.closed_positions)
    
    def get_average_trade(self) -> float:
        """
        Get average trade P&L.
        
        Returns:
            Average trade P&L
        """
        if not self.closed_positions:
            return 0.0
        
        return self.get_realized_pnl() / len(self.closed_positions)
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown.
        
        Returns:
            Maximum drawdown (0-1)
        """
        return self.max_drawdown
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Get Sharpe ratio.
        
        Args:
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if not self.closed_positions:
            return 0.0
        
        # Get daily returns
        returns = [p.realized_pnl / (p.entry_price * p.quantity) for p in self.closed_positions]
        
        # Calculate Sharpe ratio
        mean_return = sum(returns) / len(returns)
        std_return = (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert positions to DataFrame.
        
        Returns:
            DataFrame with positions
        """
        # Combine open and closed positions
        all_positions = list(self.positions.values()) + self.closed_positions
        
        # Convert to list of dicts
        position_dicts = [p.to_dict() for p in all_positions]
        
        # Create DataFrame
        return pd.DataFrame(position_dicts)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "total_trades": len(self.positions) + len(self.closed_positions),
            "unrealized_pnl": self.get_unrealized_pnl(),
            "realized_pnl": self.get_realized_pnl(),
            "total_pnl": self.get_total_pnl(),
            "win_rate": self.get_win_rate(),
            "average_trade": self.get_average_trade(),
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.get_sharpe_ratio(),
            "return_pct": (self.current_capital / self.initial_capital - 1) * 100
        }


def sync_positions_with_broker(
    position_manager: PositionManager,
    fyers: fyersModel.FyersModel
) -> None:
    """
    Synchronize position manager with broker positions.
    
    Args:
        position_manager: Position manager
        fyers: Fyers API client
    """
    logger.info("Synchronizing positions with broker")
    
    # Get positions from broker
    response = get_positions(fyers)
    
    if response.get('s') != 'ok':
        logger.error(f"Failed to get positions: {response}")
        return
    
    broker_positions = response.get('netPositions', [])
    
    # Get current time
    current_time = datetime.now()
    
    # Update prices for existing positions
    prices = {}
    for bp in broker_positions:
        symbol = bp.get('symbol')
        ltp = bp.get('ltp')
        if symbol and ltp:
            prices[symbol] = ltp
    
    # Update positions with current prices
    position_manager.update_positions(prices, current_time)
    
    # Check for positions in broker that are not in position manager
    broker_symbols = {bp.get('symbol') for bp in broker_positions}
    manager_symbols = set(position_manager.positions.keys())
    
    # Symbols in broker but not in manager
    new_symbols = broker_symbols - manager_symbols
    
    # Add new positions
    for bp in broker_positions:
        symbol = bp.get('symbol')
        if symbol in new_symbols:
            # Extract information from broker position
            instrument = bp.get('segment')
            entry_price = bp.get('buyAvg') or bp.get('sellAvg')
            quantity = bp.get('netQty')
            
            # Try to extract option details
            option_details = parse_option_symbol(symbol)
            
            if option_details:
                # Add position
                position_manager.add_position(
                    symbol=symbol,
                    instrument=option_details['instrument'],
                    entry_price=entry_price,
                    quantity=quantity,
                    entry_time=current_time,
                    option_type=option_details['option_type'],
                    strike_price=option_details['strike_price'],
                    expiry_date=option_details['expiry_date'],
                    stop_loss=0.0,  # Will be updated later
                    take_profit=0.0  # Will be updated later
                )
    
    # Symbols in manager but not in broker
    closed_symbols = manager_symbols - broker_symbols
    
    # Exit positions that are closed in broker
    for symbol in closed_symbols:
        position_manager.exit_position(
            symbol=symbol,
            exit_price=position_manager.positions[symbol].current_price,
            exit_time=current_time,
            reason="BROKER_CLOSED"
        )
    
    logger.info(f"Synchronized {len(broker_positions)} positions")


def parse_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Parse option symbol to extract details.
    
    Args:
        symbol: Option symbol
        
    Returns:
        Dictionary with option details or None if not an option
    """
    try:
        # Example: NSE:NIFTY2543017650CE
        parts = symbol.split(':')
        if len(parts) != 2:
            return None
        
        exchange, symbol_part = parts
        
        # Extract instrument
        for instrument, index_symbol in INSTRUMENTS.items():
            if index_symbol.startswith(f"{exchange}:") and symbol_part.startswith(instrument.upper()):
                # Extract option type (last 2 characters)
                option_type = symbol_part[-2:]
                if option_type not in [OptionType.CALL.value, OptionType.PUT.value]:
                    return None
                
                # Extract strike price (last digits before option type)
                strike_part = symbol_part[:-2]
                strike_price = float(''.join(filter(str.isdigit, strike_part[-5:])))
                
                # Extract expiry date (YYMMDD format)
                expiry_part = strike_part[len(instrument.upper()):]
                expiry_date = expiry_part[:6]
                
                return {
                    'instrument': instrument,
                    'option_type': option_type,
                    'strike_price': strike_price,
                    'expiry_date': expiry_date
                }
        
        return None
    except Exception as e:
        logger.error(f"Error parsing option symbol {symbol}: {e}")
        return None
