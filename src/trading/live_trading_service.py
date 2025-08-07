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

# Import existing trading components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.fyers_auth_service import create_fyers_model, create_fyers_websocket
from backtesting.environment import TradingEnv, TradingMode
from utils.data_loader import DataLoader

from utils.capital_aware_quantity import CapitalAwareQuantitySelector
from src.trading.position import Position
from src.utils.option_utils import get_nearest_itm_strike, get_nearest_expiry, map_underlying_to_option_price
from agents.ppo_agent import PPOAgent



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
        option_strategy: str = "ITM"
    ):
        self.user_id = user_id
        self.access_token = access_token
        self.app_id = app_id
        self.instrument = instrument
        self.timeframe = timeframe
        self.option_strategy = option_strategy
        
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
        
        # Fyers API instances
        self.fyers_model = None
        self.fyers_socket = None
        
        # Trading environment
        self.trading_env = None
        
        # Real-time data
        self.current_price = 0
        self.market_data = {}
        
        logger.info(f"Initialized live trading service for {user_id}: {instrument} {timeframe}")
    
    def add_websocket_client(self, websocket):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.append(websocket)
        logger.info(f"Added WebSocket client for live trading {self.user_id}")
    
    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            logger.info(f"Removed WebSocket client for live trading {self.user_id}")
    
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
        """Initialize Fyers API connection"""
        try:
            logger.info("Initializing Fyers API connection")
            
            # Create Fyers model for API calls
            self.fyers_model = create_fyers_model(self.access_token, self.app_id)
            
            # Create WebSocket for real-time data
            self.fyers_socket = create_fyers_websocket(self.access_token, self.app_id)
            
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
                    
                    # Check if we have an open position and price changed significantly
                    # This will trigger position updates for real-time PnL tracking
                    if (hasattr(self, 'trading_env') and 
                        self.trading_env and 
                        hasattr(self.trading_env, 'engine') and
                        getattr(self.trading_env.engine, '_is_position_open', False) and
                        abs(ltp - previous_price) > 0):
                        # Broadcast position update for real-time PnL
                        asyncio.create_task(self._broadcast_position_update())
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
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
                "tradeType": "Automated",  # LiveTradingService generates automated trades
                "isOpen": bool(is_position_open)
            }
            
            # Add exit price and final PnL for closed positions
            if not is_position_open and position_quantity == 0:
                # Position was closed, include exit information
                position_data["exitPrice"] = float(self.current_price)
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
                mode=TradingMode.LIVE,
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

            # Get available capital
            funds = self.fyers_model.funds()
            available_capital = funds['fund_limit'][10]['equityAmount']

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
                if action_name in ["BUY", "SELL"] and adjusted_quantity > 0 and not self.active_position:
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
                    "trade_info": trade_info
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
                # Get the latest observation which includes features
                self.current_obs = self.trading_env._get_observation()
                
                # Get feature columns to find the index of ATR
                feature_columns = [col for col in self.trading_env.data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
                atr_index = feature_columns.index('atr')
                self.atr = self.current_obs[atr_index]

                # Check for SL/TP triggers
                if self.active_position:
                    self._check_sl_tp_triggers()

                # Generate trading action
                action = self._generate_trading_signal()

                # Execute trade action
                await self._execute_trade(action)
                
                # Broadcast current stats
                win_rate = (self.win_count / self.total_trades * 100) if self.total_trades > 0 else 0
                
                asyncio.create_task(self._broadcast_update({
                    "type": "stats_update",
                    "current_pnl": self.current_pnl,
                    "today_trades": self.today_trades,
                    "win_rate": win_rate,
                    "current_price": self.current_price,
                    "position": self.current_position,
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
                "message": "Live trading started",
                "instrument": self.instrument,
                "timeframe": self.timeframe,
                "option_strategy": self.option_strategy
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
            
            # Broadcast stop message
            await self._broadcast_update({
                "type": "stopped",
                "message": "Live trading stopped",
                "final_pnl": self.current_pnl,
                "total_trades": self.today_trades
            })
            
            logger.info(f"Live trading stopped for user {self.user_id}")
            
        except Exception as e:
            error_message = f"Error stopping live trading: {str(e)}"
            logger.error(error_message)
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        win_rate = (self.win_count / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            "status": self.status,
            "is_running": self.is_running,
            "current_pnl": self.current_pnl,
            "today_trades": self.today_trades,
            "win_rate": win_rate,
            "current_price": self.current_price,
            "position": self.current_position,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "option_strategy": self.option_strategy
        }
