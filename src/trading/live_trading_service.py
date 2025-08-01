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
        """WebSocket message callback"""
        try:
            # Parse market data
            if isinstance(message, dict):
                symbol = message.get('symbol', '')
                ltp = message.get('ltp', 0)
                
                if ltp > 0:
                    self.current_price = ltp
                    self.market_data = message
                    
                    # Broadcast price update
                    asyncio.create_task(self._broadcast_update({
                        "type": "price_update",
                        "symbol": symbol,
                        "price": ltp,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
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

            # Import PPOAgent
            from agents.ppo_agent import PPOAgent

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
    
    def _execute_trade(self, action: int):
        """Execute trade based on action using TradingEnv"""
        try:
            if not hasattr(self, 'trading_env') or self.trading_env is None:
                logger.error("Trading environment not initialized")
                return

            # Take step in trading environment
            obs, reward, done, info = self.trading_env.step(action)
            self.current_obs = obs

            # Update current price from environment if available
            if hasattr(self.trading_env, 'current_price'):
                self.current_price = self.trading_env.current_price

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
    
    def _trading_loop(self):
        """Main trading loop running in separate thread"""
        logger.info("Starting trading loop")
        
        try:
            while self.is_running:
                # Generate trading action
                action = self._generate_trading_signal()

                # Execute trade action
                self._execute_trade(action)
                
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
            self.trading_thread = threading.Thread(target=self._trading_loop)
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
