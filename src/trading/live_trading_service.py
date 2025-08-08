"""
Live Trading Service for AlgoTrading System
Handles live trading operations and real-time data streaming
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
import json
import os
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

# Import existing trading components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.fyers_auth_service import create_fyers_model, create_fyers_websocket
from backtesting.environment import TradingEnv
from backtesting.environment import TradingMode as EnvTradingMode
from utils.data_loader import DataLoader

from utils.capital_aware_quantity import CapitalAwareQuantitySelector
from src.trading.position import Position
from src.utils.option_utils import get_nearest_itm_strike, get_nearest_expiry, map_underlying_to_option_price
from agents.ppo_agent import PPOAgent


class TradingMode(str, Enum):
    """Trading mode enumeration for paper vs real trading"""
    PAPER = "paper"
    REAL = "real"


@dataclass
class PaperTrade:
    """Represents a paper trade for simulation"""
    trade_id: str
    instrument: str
    direction: str
    entry_price: float
    quantity: int
    entry_time: datetime
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    trade_type: str = "Automated"
    status: str = "Open"
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    reason: Optional[str] = None


@dataclass
class PaperPortfolio:
    """Tracks paper trading portfolio state"""
    initial_capital: float = 100000.0
    available_capital: float = field(default_factory=lambda: 100000.0)
    total_pnl: float = 0.0
    open_trades: List[PaperTrade] = field(default_factory=list)
    closed_trades: List[PaperTrade] = field(default_factory=list)
    trade_count: int = 0
    win_count: int = 0


from utils.metrics_calculator import MetricsCalculator
from utils.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

class LiveTradingService:
    """Service for live trading operations"""
    
    def __init__(
        self,
        user_id: str,
        access_token: str,
        app_id: str,
        instrument: str,
        timeframe: str,
        option_strategy: str = "ITM",
        trading_mode: TradingMode = TradingMode.REAL
    ):
        self.user_id = user_id
        self.access_token = access_token
        self.app_id = app_id
        self.instrument = instrument
        self.timeframe = timeframe
        self.option_strategy = option_strategy
        self.trading_mode = trading_mode
        
        self.status = "initialized"
        self.is_running = False
        self.websocket_clients = []
        self.trading_thread = None
        self.active_position: Optional[Position] = None
        
        # Trading state
        self.current_position = 0
        self.entry_price = 0
        self.current_pnl = 0
        self.today_trades = 0
        self.win_count = 0
        self.total_trades = 0
        
        # Fyers API instances (only for real trading)
        self.fyers_model = None
        self.fyers_socket = None
        
        # Trading environment
        self.trading_env = None
        
        # Real-time data
        self.current_price = 0
        self.market_data = {}
        
        # Paper trading portfolio (only for paper mode)
        self.paper_portfolio: Optional[PaperPortfolio] = None
        if self.trading_mode == TradingMode.PAPER:
            self.paper_portfolio = PaperPortfolio()
            logger.info(f"Initialized PAPER trading service for {user_id}: {instrument} {timeframe}")
        else:
            logger.info(f"Initialized REAL trading service for {user_id}: {instrument} {timeframe}")
    
    def add_websocket_client(self, websocket):
        """Add WebSocket client for real-time updates"""
        # Limit to maximum 3 concurrent connections per user to prevent resource exhaustion
        if len(self.websocket_clients) >= 3:
            logger.warning(f"Maximum WebSocket connections (3) reached for user {self.user_id}, removing oldest")
            oldest_client = self.websocket_clients.pop(0)
            try:
                # Close the oldest connection
                asyncio.create_task(oldest_client.close())
            except Exception as e:
                logger.warning(f"Failed to close oldest WebSocket connection: {e}")
        
        self.websocket_clients.append(websocket)
        logger.info(f"Added WebSocket client for live trading {self.user_id} (total: {len(self.websocket_clients)})")
    
    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            logger.info(f"Removed WebSocket client for live trading {self.user_id} (remaining: {len(self.websocket_clients)})")
    
    async def _broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected_clients = []
        
        for client in self.websocket_clients:
            try:
                await client.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket client: {e}")
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.remove_websocket_client(client)
    
    def _initialize_fyers_connection(self):
        """Initialize Fyers API connection (only for real trading)"""
        if self.trading_mode == TradingMode.PAPER:
            logger.info("Skipping Fyers API connection for paper trading mode")
            return
            
        try:
            logger.info("Initializing Fyers API connection")
            
            # Create Fyers model for API calls
            self.fyers_model = create_fyers_model(self.access_token)
            
            # Create WebSocket for real-time data
            self.fyers_socket = create_fyers_websocket(self.access_token)
            
            # Set up WebSocket callbacks
            self.fyers_socket.on_connect = self._on_websocket_connect
            self.fyers_socket.on_message = self._on_websocket_message
            self.fyers_socket.on_error = self._on_websocket_error
            self.fyers_socket.on_close = self._on_websocket_close
            
            logger.info("Fyers API connection initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Fyers connection: {e}")
            raise
    
    def _on_websocket_connect(self):
        """WebSocket connection callback"""
        logger.info("Fyers WebSocket connected")
        
        # Subscribe to instrument data
        symbol = self._get_fyers_symbol()
        if symbol:
            self.fyers_socket.subscribe([symbol])
            logger.info(f"Subscribed to {symbol}")
    
    def _on_websocket_message(self, message):
        """WebSocket message callback - handles tick data from FyersClient"""
        try:
            # Parse market data and immediately relay as tick data
            if isinstance(message, dict):
                symbol = message.get('symbol', '')
                ltp = message.get('ltp', 0)
                volume = message.get('volume', 0)
                
                if ltp > 0:
                    # Store previous price for position PnL updates
                    previous_price = self.current_price
                    self.current_price = ltp
                    self.market_data = message
                    
                    # Create tick data structure for frontend chart
                    tick_data = {
                        "symbol": symbol,
                        "price": float(ltp),
                        "volume": float(volume) if volume else 0.0,
                        "timestamp": datetime.now().isoformat(),
                        # Include additional tick data that might be useful for charts
                        "bid": float(message.get('bid', 0)),
                        "ask": float(message.get('ask', 0)),
                        "high": float(message.get('high', 0)),
                        "low": float(message.get('low', 0)),
                        "open": float(message.get('open', 0))
                    }
                    
                    # Immediately relay tick data to WebSocket manager
                    asyncio.create_task(self._broadcast_tick_data(tick_data))
                    
                    # Check position updates based on trading mode
                    if (hasattr(self, 'trading_env') and self.trading_env and abs(ltp - previous_price) > 0):
                        if self.trading_mode == TradingMode.REAL:
                            # Real trading position update
                            if (hasattr(self.trading_env, 'engine') and 
                                getattr(self.trading_env.engine, '_is_position_open', False)):
                                asyncio.create_task(self._broadcast_position_update())
                        else:
                            # Paper trading position update
                            if self.paper_portfolio and self.paper_portfolio.open_trades:
                                asyncio.create_task(self._broadcast_paper_position_update())
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _simulate_price_movement(self) -> float:
        """Simulate price movement for paper trading mode when no real data is available"""
        if self.current_price > 0:
            # Simple random walk with slight upward bias
            change_percent = np.random.normal(0.0001, 0.002)  # Small random changes
            new_price = self.current_price * (1 + change_percent)
            return max(new_price, 1.0)  # Ensure price stays positive
        else:
            # Start with a base price if none exists
            return 25000.0  # Default starting price for NIFTY-like instruments
    
    async def _broadcast_tick_data(self, tick_data: Dict[str, Any]):
        """Broadcast raw tick data to all connected WebSocket clients"""
        try:
            # Structure the tick message according to requirements
            tick_message = {
                "type": "tick",
                "data": tick_data
            }
            
            # Use existing broadcast mechanism
            await self._broadcast_update(tick_message)
            
        except Exception as e:
            logger.error(f"Error broadcasting tick data: {e}")
    
    def _calculate_paper_pnl(self, paper_trade: PaperTrade, current_price: float) -> float:
        """Calculate current PnL for a paper trade"""
        if paper_trade.direction == "Long":
            return (current_price - paper_trade.entry_price) * paper_trade.quantity
        else:  # Short
            return (paper_trade.entry_price - current_price) * paper_trade.quantity
    
    def _execute_paper_trade(self, action_name: str, adjusted_quantity: int, current_price: float, trade_type: str = "Automated") -> Optional[PaperTrade]:
        """Execute a paper trade (simulation)"""
        if not self.paper_portfolio or adjusted_quantity == 0:
            return None
        
        # Check if we have enough capital for the trade
        trade_value = adjusted_quantity * current_price
        if trade_value > self.paper_portfolio.available_capital:
            logger.warning(f"Insufficient paper capital for trade. Required: {trade_value}, Available: {self.paper_portfolio.available_capital}")
            return None
        
        # Generate trade ID
        trade_id = f"paper_{self.user_id}_{datetime.now().timestamp()}"
        
        # Get SL/TP from trading environment
        stop_loss = getattr(self.trading_env.engine, '_stop_loss_price', 0) if self.trading_env else 0
        target_price = getattr(self.trading_env.engine, '_target_profit_price', 0) if self.trading_env else 0
        
        # Create paper trade
        paper_trade = PaperTrade(
            trade_id=trade_id,
            instrument=self.instrument,
            direction="Long" if action_name == "BUY" else "Short",
            entry_price=current_price,
            quantity=adjusted_quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            target_price=target_price,
            trade_type=trade_type
        )
        
        # Update portfolio
        self.paper_portfolio.available_capital -= trade_value
        self.paper_portfolio.open_trades.append(paper_trade)
        self.paper_portfolio.trade_count += 1
        
        logger.info(f"Paper trade executed: {action_name} {adjusted_quantity} {self.instrument} at {current_price}")
        return paper_trade
    
    def _close_paper_trade(self, paper_trade: PaperTrade, exit_price: float, reason: str):
        """Close a paper trade"""
        if not self.paper_portfolio:
            return
        
        # Remove from open trades
        if paper_trade in self.paper_portfolio.open_trades:
            self.paper_portfolio.open_trades.remove(paper_trade)
        
        # Calculate final PnL
        paper_trade.exit_price = exit_price
        paper_trade.exit_time = datetime.now()
        paper_trade.pnl = self._calculate_paper_pnl(paper_trade, exit_price)
        paper_trade.status = "Closed"
        paper_trade.reason = reason
        
        # Update portfolio
        trade_value = paper_trade.quantity * paper_trade.entry_price
        self.paper_portfolio.available_capital += trade_value + paper_trade.pnl
        self.paper_portfolio.total_pnl += paper_trade.pnl
        self.paper_portfolio.closed_trades.append(paper_trade)
        
        if paper_trade.pnl > 0:
            self.paper_portfolio.win_count += 1
        
        logger.info(f"Paper trade closed: {paper_trade.direction} {paper_trade.quantity} {paper_trade.instrument} - PnL: {paper_trade.pnl:.2f} ({reason})")
    
    def _check_paper_sl_tp_triggers(self):
        """Check for stop-loss or target-profit triggers in paper trades"""
        if not self.paper_portfolio:
            return
        
        trades_to_close = []
        
        for trade in self.paper_portfolio.open_trades:
            if trade.direction == "Long":
                if trade.stop_loss and self.current_price <= trade.stop_loss:
                    trades_to_close.append((trade, "SL Hit"))
                elif trade.target_price and self.current_price >= trade.target_price:
                    trades_to_close.append((trade, "TP Hit"))
            elif trade.direction == "Short":
                if trade.stop_loss and self.current_price >= trade.stop_loss:
                    trades_to_close.append((trade, "SL Hit"))
                elif trade.target_price and self.current_price <= trade.target_price:
                    trades_to_close.append((trade, "TP Hit"))
        
        # Close triggered trades
        for trade, reason in trades_to_close:
            self._close_paper_trade(trade, self.current_price, reason)
            # Create position update for closed paper trade
            asyncio.create_task(self._broadcast_paper_position_update(trade))
    
    async def _broadcast_paper_position_update(self, paper_trade: Optional[PaperTrade] = None):
        """Broadcast paper trading position update"""
        if not self.paper_portfolio:
            return
        
        try:
            # If specific trade provided, broadcast that trade's status
            if paper_trade:
                current_pnl = paper_trade.pnl if paper_trade.status == "Closed" else self._calculate_paper_pnl(paper_trade, self.current_price)
                
                position_data = {
                    "instrument": paper_trade.instrument,
                    "direction": paper_trade.direction,
                    "entryPrice": float(paper_trade.entry_price),
                    "quantity": int(paper_trade.quantity),
                    "stopLoss": float(paper_trade.stop_loss) if paper_trade.stop_loss else 0.0,
                    "targetPrice": float(paper_trade.target_price) if paper_trade.target_price else 0.0,
                    "currentPnl": float(current_pnl),
                    "tradeType": paper_trade.trade_type,
                    "isOpen": paper_trade.status == "Open",
                    "entryTime": paper_trade.entry_time.isoformat(),
                    "tradingMode": "PAPER"
                }
                
                if paper_trade.status == "Closed":
                    position_data["exitPrice"] = float(paper_trade.exit_price)
                    position_data["exitTime"] = paper_trade.exit_time.isoformat()
                    position_data["pnl"] = float(paper_trade.pnl)
                    position_data["reason"] = paper_trade.reason
            else:
                # Broadcast aggregate position for open trades
                if self.paper_portfolio.open_trades:
                    # For simplicity, broadcast the first open trade
                    # In a real implementation, you might want to aggregate or handle multiple positions
                    first_trade = self.paper_portfolio.open_trades[0]
                    current_pnl = self._calculate_paper_pnl(first_trade, self.current_price)
                    
                    position_data = {
                        "instrument": first_trade.instrument,
                        "direction": first_trade.direction,
                        "entryPrice": float(first_trade.entry_price),
                        "quantity": int(first_trade.quantity),
                        "stopLoss": float(first_trade.stop_loss) if first_trade.stop_loss else 0.0,
                        "targetPrice": float(first_trade.target_price) if first_trade.target_price else 0.0,
                        "currentPnl": float(current_pnl),
                        "tradeType": first_trade.trade_type,
                        "isOpen": True,
                        "entryTime": first_trade.entry_time.isoformat(),
                        "tradingMode": "PAPER"
                    }
                else:
                    # No open positions
                    position_data = {
                        "instrument": self.instrument,
                        "direction": "",
                        "entryPrice": 0.0,
                        "quantity": 0,
                        "stopLoss": 0.0,
                        "targetPrice": 0.0,
                        "currentPnl": 0.0,
                        "tradeType": "",
                        "isOpen": False,
                        "entryTime": None,
                        "tradingMode": "PAPER"
                    }
            
            position_message = {
                "type": "position_update",
                "data": position_data
            }
            
            await self._broadcast_update(position_message)
            
        except Exception as e:
            logger.error(f"Error broadcasting paper position update: {e}")
    
    async def _broadcast_position_update(self):
        """Broadcast position update to all connected WebSocket clients"""
        try:
            if not hasattr(self, 'trading_env') or self.trading_env is None:
                return
                
            engine = self.trading_env.engine
            
            # Get position data according to the Position data model
            position_quantity = getattr(engine, '_current_position_quantity', 0)
            is_position_open = getattr(engine, '_is_position_open', False)
            entry_price = getattr(engine, '_current_position_entry_price', 0)
            stop_loss = getattr(engine, '_stop_loss_price', 0)
            target_price = getattr(engine, '_target_profit_price', 0)
            
            # Calculate current unrealized PnL
            current_pnl = 0.0
            if is_position_open and position_quantity != 0 and entry_price > 0 and self.current_price > 0:
                if position_quantity > 0:  # Long position
                    current_pnl = (self.current_price - entry_price) * position_quantity
                else:  # Short position
                    current_pnl = (entry_price - self.current_price) * abs(position_quantity)
            
            # Determine direction
            direction = ""
            if position_quantity > 0:
                direction = "Long"
            elif position_quantity < 0:
                direction = "Short"
            
            # Create position update message according to Position data model
            position_data = {
                "instrument": self.instrument,
                "direction": direction,
                "entryPrice": float(entry_price) if entry_price else 0.0,
                "quantity": int(abs(position_quantity)) if position_quantity else 0,
                "stopLoss": float(stop_loss) if stop_loss else 0.0,
                "targetPrice": float(target_price) if target_price else 0.0,
                "currentPnl": float(current_pnl),
                "tradeType": self.active_position.trade_type if self.active_position else "",
                "isOpen": bool(is_position_open),
                "entryTime": self.active_position.entry_time.isoformat() if self.active_position else None,
                "tradingMode": "REAL"
            }
            
            # Add exit price and final PnL for closed positions
            if not is_position_open and position_quantity == 0:
                # Position was closed, include exit information
                position_data["exitPrice"] = float(self.current_price)
                position_data["exitTime"] = datetime.now().isoformat()
                # For closed positions, currentPnL becomes the final PnL
                position_data["pnl"] = position_data["currentPnl"]
            
            position_message = {
                "type": "position_update",
                "data": position_data
            }
            
            logger.info(f"Broadcasting position update: {direction} position, quantity: {position_quantity}, PnL: {current_pnl:.2f}")
            
            # Use existing broadcast mechanism
            await self._broadcast_update(position_message)
            
        except Exception as e:
            logger.error(f"Error broadcasting position update: {e}")
    
    def _on_websocket_error(self, error):
        """WebSocket error callback"""
        logger.error(f"Fyers WebSocket error: {error}")
    
    def _on_websocket_close(self):
        """WebSocket close callback"""
        logger.info("Fyers WebSocket disconnected")
    
    def _get_fyers_symbol(self) -> str:
        """Get Fyers symbol format for the instrument"""
        # Map instrument names to Fyers symbols
        symbol_map = {
            "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
            "NIFTY50": "NSE:NIFTY50-INDEX",
            "RELIANCE": "NSE:RELIANCE-EQ"
        }
        
        return symbol_map.get(self.instrument.upper(), "")
    
    def _initialize_trading_environment(self):
        """Initialize the trading environment"""
        try:
            logger.info("Initializing trading environment")

            # Initialize data loader for live trading
            data_loader = DataLoader(final_data_dir="data/final", use_parquet=True)

            # Initialize trading environment in LIVE mode
            self.trading_env = TradingEnv(
                data_loader=data_loader,
                symbol=self.instrument,
                initial_capital=100000,
                lookback_window=50,
                mode=EnvTradingMode.LIVE,
                reward_function="trading_focused",
                trailing_stop_percentage=0.02,
                use_streaming=True
            )

            # Reset environment to get initial observation
            self.current_obs = self.trading_env.reset()

            logger.info("Trading environment initialized")

        except Exception as e:
            logger.error(f"Failed to initialize trading environment: {e}")
            raise
    
    def _load_model(self):
        """Load the universal trading model"""
        try:
            logger.info("Loading universal trading model")

            model_path = Path(__file__).parent.parent.parent / "models" / "universal_final_model.pth"

            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, using random actions")
                self.agent = None
                return

            # Get observation dimensions from environment
            if not hasattr(self, 'trading_env') or self.trading_env is None:
                logger.error("Trading environment must be initialized before loading model")
                self.agent = None
                return

            obs = self.trading_env.reset()
            observation_dim = obs.shape[0]
            action_dim_discrete = 5  # TradingEnv action space: [0,1,2,3,4]
            action_dim_continuous = 1
            # Load training config to get hidden_dim
            import yaml
            training_config_path = Path(__file__).parent.parent.parent / "config" / "training_sequence.yaml"
            with open(training_config_path, 'r') as f:
                training_config = yaml.safe_load(f)
            hidden_dim = training_config.get('model', {}).get('hidden_dim', 64)

            logger.info(f"Model dimensions: obs={observation_dim}, discrete={action_dim_discrete}, continuous={action_dim_continuous}")

            # Create agent with same parameters as training
            self.agent = PPOAgent(
                observation_dim=observation_dim,
                action_dim_discrete=action_dim_discrete,
                action_dim_continuous=action_dim_continuous,
                hidden_dim=hidden_dim
            )

            # Load the trained model
            self.agent.load_model(str(model_path))
            logger.info("✅ Universal model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using random actions as fallback")
            self.agent = None
    
    def _generate_trading_signal(self) -> int:
        """Generate trading signal using the model"""
        try:
            if not hasattr(self, 'current_obs') or self.current_obs is None:
                return 2  # HOLD action if no observation

            # Use trained model if available
            if hasattr(self, 'agent') and self.agent is not None:
                # Use trained model to predict action
                action, _ = self.agent.act(self.current_obs)
                return action
            else:
                # Fallback to random actions with bias towards HOLD
                # TradingEnv action space: [0: STRONG_SELL, 1: SELL, 2: HOLD, 3: BUY, 4: STRONG_BUY]
                action = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                return action

        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return 2  # Default to HOLD on error
    
    async def _execute_trade(self, action: int):
        """Execute trade based on action using TradingEnv"""
        try:
            if not hasattr(self, 'trading_env') or self.trading_env is None:
                logger.error("Trading environment not initialized")
                return

            # Get predicted quantity from model
            _, predicted_quantity = self.agent.act(self.current_obs)

            # Get available capital based on trading mode
            if self.trading_mode == TradingMode.REAL:
                funds = self.fyers_model.funds()
                available_capital = funds['fund_limit'][10]['equityAmount']
            else:  # Paper trading
                available_capital = self.paper_portfolio.available_capital

            # Adjust quantity for capital
            capital_aware_quantity = CapitalAwareQuantitySelector()
            adjusted_quantity = capital_aware_quantity.adjust_quantity_for_capital(
                predicted_quantity=predicted_quantity,
                available_capital=available_capital,
                current_price=self.current_price,
                instrument=self.trading_env.instrument
            )

            logger.info(f"Predicted quantity: {predicted_quantity}, Adjusted quantity: {adjusted_quantity}")

            if adjusted_quantity == 0:
                logger.info("Skipping trade due to insufficient capital or zero adjusted quantity.")
                return

            # Store previous position state for change detection
            previous_position_quantity = getattr(self.trading_env.engine, '_current_position_quantity', 0)
            previous_position_open = getattr(self.trading_env.engine, '_is_position_open', False)
            previous_entry_price = getattr(self.trading_env.engine, '_current_position_entry_price', 0)

            # Take step in trading environment
            obs, reward, done, info = self.trading_env.step(action)
            self.current_obs = obs

            # Update current price from environment if available
            if hasattr(self.trading_env, 'current_price'):
                self.current_price = self.trading_env.current_price

            # Check for position state changes and broadcast position updates
            current_position_quantity = getattr(self.trading_env.engine, '_current_position_quantity', 0)
            current_position_open = getattr(self.trading_env.engine, '_is_position_open', False)
            current_entry_price = getattr(self.trading_env.engine, '_current_position_entry_price', 0)
            
            # Detect position changes
            position_changed = (
                previous_position_quantity != current_position_quantity or
                previous_position_open != current_position_open or
                previous_entry_price != current_entry_price
            )
            
            if position_changed:
                # Broadcast position update
                asyncio.create_task(self._broadcast_position_update())

            # Check if trade was executed
            if info.get('trade_executed', False):
                trade_info = info.get('trade_info', {})

                self.today_trades += 1
                self.total_trades += 1

                # Update PnL
                trade_pnl = trade_info.get('pnl', 0)
                self.current_pnl += trade_pnl

                if trade_pnl > 0:
                    self.win_count += 1

                # Get action name for logging
                action_names = {0: "STRONG_SELL", 1: "SELL", 2: "HOLD", 3: "BUY", 4: "STRONG_BUY"}
                action_name = action_names.get(action, "UNKNOWN")

                # Check for trade entry signals
                if action_name in ["BUY", "SELL"] and adjusted_quantity > 0:
                    # Check if we can enter a new trade (no active position for simplicity)
                    can_enter_trade = (
                        (self.trading_mode == TradingMode.REAL and not self.active_position) or
                        (self.trading_mode == TradingMode.PAPER and not self.paper_portfolio.open_trades)
                    )
                    
                    if can_enter_trade:
                        if self.trading_mode == TradingMode.PAPER:
                            # Execute paper trade
                            paper_trade = self._execute_paper_trade(action_name, adjusted_quantity, self.current_price, "Automated")
                            if paper_trade:
                                # Broadcast paper position update
                                await self._broadcast_paper_position_update(paper_trade)
                        else:
                            # Execute real trade
                            # Options trading logic
                            if self.option_strategy != "None":
                                # 1. Get available strikes and expiries (mocked for now)
                                # In a real scenario, this would come from the Fyers API
                                available_strikes = [self.current_price - 200, self.current_price - 100, self.current_price, self.current_price + 100, self.current_price + 200]
                                available_expiries = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, 15)]

                                # 2. Determine option type
                                option_type = 'CE' if action_name == "BUY" else 'PE'

                                # 3. Select strike and expiry
                                strike_price = get_nearest_itm_strike(self.current_price, available_strikes, option_type)
                                expiry_date = get_nearest_expiry(available_expiries)

                                if not strike_price or not expiry_date:
                                    logger.error("Could not determine valid strike or expiry for options trade.")
                                    return

                                # 4. Construct the Fyers symbol for the option
                                # This is a simplified example. The actual symbol format might be more complex.
                                option_symbol = f"NSE:{self.instrument.split('-')[0].replace('NIFTY','NFO')}{expiry_date.replace('-','')}{strike_price}{option_type}"
                                
                                # 5. Map underlying SL/TP to option prices
                                # This requires fetching the current option price, which we will mock for now
                                mock_option_price = 150 
                                option_sl = map_underlying_to_option_price(self.trading_env.engine._stop_loss_price, self.current_price, mock_option_price, option_type)
                                option_tp = map_underlying_to_option_price(self.trading_env.engine._target_profit_price, self.current_price, mock_option_price, option_type)

                                trade_symbol = option_symbol
                                stop_loss_price = option_sl
                                target_price = option_tp
                                
                            else: # Futures/Equity trading logic
                                trade_symbol = self._get_fyers_symbol()
                                stop_loss_price = self.trading_env.engine._stop_loss_price
                                target_price = self.trading_env.engine._target_profit_price

                            product_type = "CNC" if "-EQ" in self.instrument.upper() else "INTRADAY"
                            side = 1 if action_name == "BUY" else -1
                            
                            try:
                                order_response = self.fyers_model.place_order(
                                    symbol=trade_symbol,
                                    qty=adjusted_quantity,
                                    side=side,
                                    productType=product_type
                                )

                                if order_response.get("s") == "ok":
                                    order_id = order_response.get("id")
                                    logger.info(f"Order placed successfully for {trade_symbol}. Order ID: {order_id}")
                                    
                                    self.active_position = Position(
                                        instrument=trade_symbol, # Store the actual traded symbol
                                        direction="Long" if action_name == "BUY" else "Short",
                                        entry_price=self.current_price,
                                        quantity=adjusted_quantity,
                                        stop_loss=stop_loss_price,
                                        target_price=target_price,
                                        entry_time=datetime.now(),
                                        trade_type="Automated"
                                    )
                                    logger.info(f"New position created: {self.active_position}")
                                    await self._broadcast_position_update()
                                else:
                                    logger.error(f"Failed to place order for {trade_symbol}: {order_response.get('message')}")

                            except Exception as e:
                                logger.error(f"Exception placing order for {trade_symbol}: {e}")

                logger.info(f"Trade executed: {action_name} at {self.current_price}, PnL: {trade_pnl}")

                # Broadcast trade execution
                asyncio.create_task(self._broadcast_update({
                    "type": "trade_executed",
                    "action": action_name,
                    "price": self.current_price,
                    "pnl": trade_pnl,
                    "reward": reward,
                    "timestamp": datetime.now().isoformat(),
                    "trade_info": trade_info,
                    "trading_mode": self.trading_mode.value.upper()
                }))

            # Update position from environment
            if hasattr(self.trading_env, 'current_position'):
                self.current_position = self.trading_env.current_position

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return

    def _check_sl_tp_triggers(self):
        """Check for stop-loss or target-profit triggers."""
        if not self.active_position:
            return

        # For options, the SL/TP is based on the underlying's price movement.
        # The self.active_position.stop_loss and target_price will hold the mapped option prices,
        # but the trigger is the underlying's price hitting the original SL/TP.
        
        underlying_sl = self.trading_env.engine._stop_loss_price
        underlying_tp = self.trading_env.engine._target_profit_price

        if self.active_position.direction == "Long":
            if self.current_price <= underlying_sl:
                logger.info(f"Stop-loss triggered for long position on underlying at {self.current_price}")
                asyncio.create_task(self._close_position("SL Hit"))
            elif self.current_price >= underlying_tp:
                logger.info(f"Target-profit triggered for long position on underlying at {self.current_price}")
                asyncio.create_task(self._close_position("TP Hit"))
        elif self.active_position.direction == "Short":
            if self.current_price >= underlying_sl:
                logger.info(f"Stop-loss triggered for short position on underlying at {self.current_price}")
                asyncio.create_task(self._close_position("SL Hit"))
            elif self.current_price <= underlying_tp:
                logger.info(f"Target-profit triggered for short position on underlying at {self.current_price}")
                asyncio.create_task(self._close_position("TP Hit"))

    async def _close_position(self, reason: str):
        """Close the active position."""
        if not self.active_position:
            return

        logger.info(f"Closing position for reason: {reason}")

        trade_data = {
            "trade_id": f"{self.user_id}_{datetime.now().timestamp()}",
            "instrument": self.active_position.instrument,
            "timeframe": self.timeframe,
            "direction": self.active_position.direction,
            "entry_price": self.active_position.entry_price,
            "exit_price": self.current_price,
            "quantity": self.active_position.quantity,
            "pnl": (self.current_price - self.active_position.entry_price) * self.active_position.quantity if self.active_position.direction == "Long" else (self.active_position.entry_price - self.current_price) * self.active_position.quantity,
            "entry_time": self.active_position.entry_time.isoformat(),
            "exit_time": datetime.now().isoformat(),
            "trade_type": self.active_position.trade_type,
            "status": "Closed",
            "reason": reason
        }

        trade_logger = TradeLogger()
        trade_logger.log_trade(trade_data)

        metrics_calculator = MetricsCalculator()
        metrics_calculator.update_metrics(trade_data)

        for i in range(3):
            try:
                exit_response = self.fyers_model.exit_positions(id=self.active_position.instrument)
                if exit_response.get("s") == "ok":
                    logger.info("Position exited successfully.")
                    self.active_position = None
                    await self._broadcast_position_update()
                    return
                else:
                    logger.error(f"Failed to exit position: {exit_response.get('message')}")
            except Exception as e:
                logger.error(f"Exception exiting position: {e}")
            
            logger.info(f"Retrying to exit position... ({i+1}/3)")
            await asyncio.sleep(1)

        logger.error("Failed to exit position after 3 attempts.")
        await self._broadcast_update({
            "type": "error",
            "message": f"Failed to close position for {self.active_position.instrument}. Please check manually."
        })
    
    async def _trading_loop(self):
        """Main trading loop running in separate thread"""
        logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                # For paper trading, simulate price movement if no real data
                if self.trading_mode == TradingMode.PAPER and self.current_price == 0:
                    self.current_price = self._simulate_price_movement()
                    
                # Get the latest observation which includes features
                self.current_obs = self.trading_env._get_observation()
                
                # Get feature columns to find the index of ATR
                feature_columns = [col for col in self.trading_env.data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                atr_index = feature_columns.index('atr')
                self.atr = self.current_obs[atr_index]

                # For paper trading, simulate price updates occasionally
                if self.trading_mode == TradingMode.PAPER:
                    previous_price = self.current_price
                    self.current_price = self._simulate_price_movement()
                    
                    # Broadcast simulated tick data
                    if abs(self.current_price - previous_price) > 0:
                        tick_data = {
                            "symbol": self.instrument,
                            "price": float(self.current_price),
                            "volume": float(np.random.randint(1000, 10000)),  # Simulated volume
                            "timestamp": datetime.now().isoformat(),
                            "bid": float(self.current_price - 0.5),
                            "ask": float(self.current_price + 0.5),
                            "high": float(self.current_price * 1.001),
                            "low": float(self.current_price * 0.999),
                            "open": float(previous_price)
                        }
                        await self._broadcast_tick_data(tick_data)

                # Check for SL/TP triggers based on trading mode
                if self.trading_mode == TradingMode.REAL and self.active_position:
                    self._check_sl_tp_triggers()
                elif self.trading_mode == TradingMode.PAPER and self.paper_portfolio.open_trades:
                    self._check_paper_sl_tp_triggers()

                # Generate trading action
                action = self._generate_trading_signal()

                # Execute trade action
                await self._execute_trade(action)
                
                # Broadcast current stats based on trading mode
                if self.trading_mode == TradingMode.PAPER and self.paper_portfolio:
                    win_rate = (self.paper_portfolio.win_count / self.paper_portfolio.trade_count * 100) if self.paper_portfolio.trade_count > 0 else 0
                    total_pnl = self.paper_portfolio.total_pnl
                    total_trades = self.paper_portfolio.trade_count
                    available_capital = self.paper_portfolio.available_capital
                    
                    asyncio.create_task(self._broadcast_update({
                        "type": "stats_update",
                        "current_pnl": total_pnl,
                        "today_trades": total_trades,
                        "win_rate": win_rate,
                        "current_price": self.current_price,
                        "position": len(self.paper_portfolio.open_trades),
                        "available_capital": available_capital,
                        "trading_mode": "PAPER",
                        "timestamp": datetime.now().isoformat()
                    }))
                else:
                    win_rate = (self.win_count / self.total_trades * 100) if self.total_trades > 0 else 0
                    
                    asyncio.create_task(self._broadcast_update({
                        "type": "stats_update",
                        "current_pnl": self.current_pnl,
                        "today_trades": self.today_trades,
                        "win_rate": win_rate,
                        "current_price": self.current_price,
                        "position": self.current_position,
                        "trading_mode": "REAL",
                        "timestamp": datetime.now().isoformat()
                    }))
                
                # Sleep for the specified timeframe
                time.sleep(self._get_sleep_duration())
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            logger.info("Trading loop stopped")
    
    def _get_sleep_duration(self) -> int:
        """Get sleep duration based on timeframe"""
        timeframe_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        return timeframe_map.get(self.timeframe, 300)  # Default 5 minutes

    async def initiate_manual_trade(self, instrument: str, direction: str, quantity: int, sl: Optional[float], tp: Optional[float], user_id: str, trading_mode: TradingMode = None):
        """Initiate a manual trade"""
        # Use the service's trading mode if not explicitly specified
        if trading_mode is None:
            trading_mode = self.trading_mode
            
        # Check for active positions based on trading mode
        if self.trading_mode == TradingMode.REAL:
            if self.active_position and self.active_position.trade_type == "Automated":
                raise Exception("Cannot initiate manual trade while an automated trade is active.")
        else:  # Paper mode
            # Allow manual trades even with automated trades in paper mode for testing
            pass

        # Pause the automated trading loop temporarily
        was_running = self.is_running
        self.is_running = False

        try:
            if trading_mode == TradingMode.PAPER:
                # Execute paper manual trade
                if not self.paper_portfolio:
                    raise Exception("Paper trading portfolio not initialized")
                
                # Check available capital
                trade_value = quantity * self.current_price
                if trade_value > self.paper_portfolio.available_capital:
                    raise Exception(f"Insufficient paper capital for manual trade. Required: {trade_value}, Available: {self.paper_portfolio.available_capital}")
                
                # Execute paper trade
                paper_trade = self._execute_paper_trade(
                    action_name="BUY" if direction.lower() == "buy" else "SELL",
                    adjusted_quantity=quantity,
                    current_price=self.current_price,
                    trade_type="Manual"
                )
                
                if paper_trade:
                    # Update with manual SL/TP if provided
                    if sl:
                        paper_trade.stop_loss = sl
                    if tp:
                        paper_trade.target_price = tp
                    
                    logger.info(f"Manual paper trade created: {paper_trade}")
                    await self._broadcast_paper_position_update(paper_trade)
                else:
                    raise Exception("Failed to create paper trade")
            else:
                # Execute real manual trade
                # Perform margin and risk validation
                funds = self.fyers_model.funds()
                available_capital = funds['fund_limit'][10]['equityAmount']

                capital_aware_quantity = CapitalAwareQuantitySelector()
                adjusted_quantity = capital_aware_quantity.adjust_quantity_for_capital(
                    predicted_quantity=quantity,
                    available_capital=available_capital,
                    current_price=self.current_price,
                    instrument=instrument
                )

                if adjusted_quantity < quantity:
                    raise Exception(f"Insufficient capital for the requested quantity. Maximum allowable quantity is {adjusted_quantity}.")

                # Place the manual trade
                product_type = "CNC" if "-EQ" in instrument.upper() else "INTRADAY"
                side = 1 if direction.lower() == "buy" else -1

                order_response = self.fyers_model.place_order(
                    symbol=instrument,
                    qty=quantity,
                    side=side,
                    productType=product_type
                )

                if order_response.get("s") == "ok":
                    order_id = order_response.get("id")
                    logger.info(f"Manual order placed successfully for {instrument}. Order ID: {order_id}")

                    self.active_position = Position(
                        instrument=instrument,
                        direction="Long" if direction.lower() == "buy" else "Short",
                        entry_price=self.current_price,
                        quantity=quantity,
                        stop_loss=sl,
                        target_price=tp,
                        entry_time=datetime.now(),
                        trade_type="Manual"
                    )
                    logger.info(f"New manual position created: {self.active_position}")
                    await self._broadcast_position_update()
                else:
                    raise Exception(f"Failed to place manual order: {order_response.get('message')}")

        except Exception as e:
            # Resume the automated trading loop if the manual trade fails
            self.is_running = was_running
            raise e
        finally:
            # Resume the automated trading loop
            self.is_running = was_running
    
    async def run(self):
        """Start live trading"""
        try:
            logger.info(f"Starting live trading for user {self.user_id}")
            
            self.status = "starting"
            self.is_running = True
            
            # Initialize connections
            self._initialize_fyers_connection()
            self._initialize_trading_environment()
            self._load_model()  # Load model after environment is initialized
            
            # Start WebSocket connection
            if self.fyers_socket:
                self.fyers_socket.connect()
            
            # Start trading loop in separate thread
            self.trading_thread = threading.Thread(target=lambda: asyncio.run(self._trading_loop()))
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.status = "running"
            
            # Broadcast start message
            await self._broadcast_update({
                "type": "started",
                "message": f"{self.trading_mode.value.title()} trading started",
                "instrument": self.instrument,
                "timeframe": self.timeframe,
                "option_strategy": self.option_strategy,
                "trading_mode": self.trading_mode.value.upper()
            })
            
            logger.info(f"Live trading started for user {self.user_id}")
            
        except Exception as e:
            self.status = "failed"
            error_message = f"Failed to start live trading: {str(e)}"
            logger.error(error_message)
            
            await self._broadcast_update({
                "type": "error",
                "message": error_message
            })
            
            raise
    
    async def stop(self):
        """Stop live trading"""
        try:
            logger.info(f"Stopping live trading for user {self.user_id}")
            
            self.status = "stopping"
            self.is_running = False
            
            # Stop WebSocket connection
            if self.fyers_socket:
                self.fyers_socket.disconnect()
            
            # Wait for trading thread to finish
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            self.status = "stopped"
            
            # Prepare final statistics based on trading mode
            if self.trading_mode == TradingMode.PAPER and self.paper_portfolio:
                final_pnl = self.paper_portfolio.total_pnl
                total_trades = self.paper_portfolio.trade_count
                final_message = f"{self.trading_mode.value.title()} trading stopped"
            else:
                final_pnl = self.current_pnl
                total_trades = self.today_trades
                final_message = f"{self.trading_mode.value.title()} trading stopped"
            
            # Broadcast stop message
            await self._broadcast_update({
                "type": "stopped",
                "message": final_message,
                "final_pnl": final_pnl,
                "total_trades": total_trades,
                "trading_mode": self.trading_mode.value.upper()
            })
            
            logger.info(f"Live trading stopped for user {self.user_id}")
            
        except Exception as e:
            error_message = f"Error stopping live trading: {str(e)}"
            logger.error(error_message)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        base_status = {
            "status": self.status,
            "is_running": self.is_running,
            "current_price": self.current_price,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "option_strategy": self.option_strategy,
            "trading_mode": self.trading_mode.value.upper(),
            "fetchIntervalSeconds": self._get_sleep_duration(),
            "nextFetchTimestamp": (datetime.now() + timedelta(seconds=self._get_sleep_duration())).isoformat()
        }
        
        if self.trading_mode == TradingMode.PAPER and self.paper_portfolio:
            # Paper trading specific status
            win_rate = (self.paper_portfolio.win_count / self.paper_portfolio.trade_count * 100) if self.paper_portfolio.trade_count > 0 else 0
            base_status.update({
                "current_pnl": self.paper_portfolio.total_pnl,
                "today_trades": self.paper_portfolio.trade_count,
                "win_rate": win_rate,
                "position": len(self.paper_portfolio.open_trades),
                "available_capital": self.paper_portfolio.available_capital,
                "open_positions": len(self.paper_portfolio.open_trades),
                "closed_positions": len(self.paper_portfolio.closed_trades)
            })
        else:
            # Real trading status
            win_rate = (self.win_count / self.total_trades * 100) if self.total_trades > 0 else 0
            base_status.update({
                "current_pnl": self.current_pnl,
                "today_trades": self.today_trades,
                "win_rate": win_rate,
                "position": self.current_position
            })
        
        return base_status
