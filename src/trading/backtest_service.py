"""
Backtest Service for AlgoTrading System
Handles backtesting operations and result streaming
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

# Import existing trading environment
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.environment import TradingEnv, TradingMode

logger = logging.getLogger(__name__)

class BacktestService:
    """Service for running backtests and streaming results"""
    
    def __init__(self, user_id: str, instrument: str, timeframe: str, duration: int, initial_capital: float = 100000, access_token: str = None, app_id: str = None):
        self.user_id = user_id
        self.instrument = instrument
        self.timeframe = timeframe
        self.duration = duration
        self.initial_capital = initial_capital
        self.access_token = access_token
        self.app_id = app_id
        self.status = "initialized"
        self.results = {}
        self.progress = 0
        self.websocket_clients = []

        # Load instrument configuration
        self.config = self._load_instrument_config()

        logger.info(f"Initialized backtest service for {user_id}: {instrument} {timeframe} {duration}d, capital: ₹{initial_capital:,.0f}")
    
    def _load_instrument_config(self) -> Dict[str, Any]:
        """Load instrument configuration from config file"""
        try:
            import yaml
            config_path = Path(__file__).parent.parent.parent / "config" / "instruments.yaml"

            if config_path.exists():
                with open(config_path, 'r') as file:
                    config = yaml.safe_load(file)
                logger.info(f"Loaded instrument config from {config_path}")
                return config
            else:
                logger.warning(f"Config file not found at {config_path}")
                return {"instruments": []}

        except Exception as e:
            logger.error(f"Failed to load instrument config: {e}")
            return {"instruments": []}
    
    def add_websocket_client(self, websocket):
        """Add WebSocket client for real-time updates"""
        self.websocket_clients.append(websocket)
        logger.info(f"Added WebSocket client for backtest {self.user_id}")
    
    def remove_websocket_client(self, websocket):
        """Remove WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            logger.info(f"Removed WebSocket client for backtest {self.user_id}")
    
    async def _broadcast_update(self, message: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
        
        message_str = json.dumps(message)
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
    
    def _get_trading_symbol(self) -> str:
        """Get trading symbol for the instrument from config"""
        try:
            instruments = self.config.get('instruments', [])
            for instrument in instruments:
                if instrument.get('name', '').replace(' ', '_').upper() == self.instrument.upper():
                    return instrument.get('symbol', '')

            # Fallback mapping if not found in config
            symbol_map = {
                "BANK_NIFTY": "NSE:NIFTYBANK-INDEX",
                "NIFTY": "NSE:NIFTY50-INDEX",
                "RELIANCE": "NSE:RELIANCE-EQ",
                "TCS": "NSE:TCS-EQ",
                "HDFC": "NSE:HDFCBANK-EQ"
            }

            return symbol_map.get(self.instrument.upper(), "")

        except Exception as e:
            logger.error(f"Failed to get trading symbol: {e}")
            return ""

    async def _load_historical_data(self) -> pd.DataFrame:
        """Load historical data from trading API and process through feature generator"""
        try:
            logger.info(f"Loading historical data for {self.instrument}")

            # Get trading symbol
            trading_symbol = self._get_trading_symbol()
            if not trading_symbol:
                raise ValueError(f"No trading symbol found for instrument: {self.instrument}")

            logger.info(f"Using trading symbol: {trading_symbol}")

            # Import required components (currently using Fyers but can be swapped)
            from trading.fyers_client import FyersClient
            from data_processing.feature_generator import DynamicFileProcessor

            # Initialize data client and feature processor
            data_client = FyersClient(access_token=self.access_token, app_id=self.app_id)
            feature_processor = DynamicFileProcessor()

            # Fetch raw data from trading API
            logger.info(f"Fetching data from trading API...")
            raw_data = data_client.get_historical_data(
                symbol=trading_symbol,
                timeframe=self.timeframe,
                days=self.duration
            )

            if raw_data is None or raw_data.empty:
                raise ValueError("Failed to fetch data from trading API")

            logger.info(f"Fetched {len(raw_data)} rows of raw data")

            # Process features
            logger.info("Processing data through feature generator...")
            processed_data = feature_processor.process_dataframe(raw_data)

            if processed_data is None or processed_data.empty:
                raise ValueError("Failed to process data through feature generator")

            logger.info(f"Processed data shape: {processed_data.shape}")
            logger.info(f"Data columns: {list(processed_data.columns)}")

            return processed_data

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise
    
    def _load_model(self, env):
        """Load the universal trading model"""
        try:
            logger.info("Loading universal trading model")

            model_path = Path(__file__).parent.parent.parent / "models" / "universal_final_model.pth"

            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, using random actions")
                return None

            # Import PPOAgent
            from agents.ppo_agent import PPOAgent

            # Get observation dimensions from environment
            obs = env.reset()
            observation_dim = obs.shape[0]
            action_dim_discrete = 5  # TradingEnv action space: [0,1,2,3,4]
            action_dim_continuous = 1
            hidden_dim = 64  # Default from config

            logger.info(f"Model dimensions: obs={observation_dim}, discrete={action_dim_discrete}, continuous={action_dim_continuous}")

            # Create agent with same parameters as training
            agent = PPOAgent(
                observation_dim=observation_dim,
                action_dim_discrete=action_dim_discrete,
                action_dim_continuous=action_dim_continuous,
                hidden_dim=hidden_dim
            )

            # Load the trained model
            agent.load_model(str(model_path))
            logger.info("✅ Universal model loaded successfully")
            return agent

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.warning("Using random actions as fallback")
            return None

    async def _run_backtest_simulation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run the actual backtest simulation"""
        try:
            logger.info("Starting backtest simulation")

            # Initialize trading environment in BACKTESTING mode (like run_unified_backtest.py)
            env = TradingEnv(
                mode=TradingMode.BACKTESTING,
                external_data=data,
                symbol=self.instrument.replace('_', '').upper(),  # Extract symbol name for reward normalization
                initial_capital=self.initial_capital,
                lookback_window=50,
                trailing_stop_percentage=0.02,
                smart_action_filtering=False
            )

            # Load the trained model
            agent = self._load_model(env)

            # Reset environment and force clean start (like run_unified_backtest.py)
            obs = env.reset()

            # Force engine to start with zero position
            env.engine._current_position_quantity = 0.0
            env.engine._is_position_open = False
            logger.info(f"Force reset position to: {env.engine._current_position_quantity}")

            trades = []
            portfolio_values = []
            chart_data = []
            done = False
            step_count = 0

            while not done:
                # Update progress
                progress = int((step_count / len(data)) * 100) if len(data) > 0 else 0
                self.progress = progress

                # Send progress update every 5%
                if step_count % max(1, len(data) // 20) == 0:
                    await self._broadcast_update({
                        "type": "progress",
                        "progress": progress,
                        "message": f"Processing {step_count}/{len(data)} data points"
                    })

                # Generate action using trained model (like run_unified_backtest.py)
                if agent is not None:
                    action_type, quantity = agent.select_action(obs)
                    action = [action_type, quantity]
                else:
                    # Fallback to random actions
                    action_type = np.random.choice([0, 1, 2, 3, 4])
                    quantity = 1.0
                    action = [action_type, quantity]

                # Take step in environment
                obs, reward, done, info = env.step(action)
                step_count += 1

                # Get current datetime from data
                current_datetime = "N/A"
                current_price = 0
                try:
                    if hasattr(env, 'data') and env.data is not None and env.current_step < len(env.data):
                        current_datetime = str(env.data.index[env.current_step])
                        current_price = env.data['close'].iloc[env.current_step] if 'close' in env.data.columns else 0
                except Exception:
                    current_datetime = f"Step_{env.current_step}"

                # Collect chart data for real-time visualization
                chart_data.append({
                    'timestamp': current_datetime,
                    'price': current_price,
                    'action': action_type,
                    'portfolio_value': env.engine.get_account_state().get('total_capital', self.initial_capital)
                })

                # Collect trade information if available
                if info.get('trade_executed'):
                    trade_info = info.get('trade_info', {})
                    trades.append({
                        'timestamp': current_datetime,
                        'action': action_type,
                        'price': current_price,
                        'pnl': trade_info.get('pnl', 0),
                        'position': trade_info.get('position', 0)
                    })

                # Record portfolio value
                portfolio_values.append({
                    'timestamp': current_datetime,
                    'value': env.engine.get_account_state().get('total_capital', self.initial_capital)
                })

                # Send real-time chart updates every 50 steps
                if step_count % 50 == 0:
                    await self._broadcast_update({
                        "type": "chart_update",
                        "data": chart_data[-50:],  # Send last 50 points
                        "current_price": current_price,
                        "portfolio_value": portfolio_values[-1]['value']
                    })

                # Small delay to simulate processing time
                if step_count % 100 == 0:
                    await asyncio.sleep(0.01)
            
            # Calculate metrics from environment and trades
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # Get final capital from environment
            final_capital = env.current_capital if hasattr(env, 'current_capital') else 100000
            initial_capital = env.initial_capital if hasattr(env, 'initial_capital') else 100000
            total_pnl = final_capital - initial_capital
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            results = {
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(portfolio_values),
                'total_reward': sum([t.get('pnl', 0) for t in trades]),
                'trades': trades,
                'portfolio_values': portfolio_values,
                'final_capital': final_capital,
                'initial_capital': initial_capital
            }
            
            logger.info(f"Backtest completed: {total_trades} trades, {win_rate:.1f}% win rate")
            return results
            
        except Exception as e:
            logger.error(f"Backtest simulation failed: {e}")
            raise
    
    def _calculate_max_drawdown(self, portfolio_values: List[Dict]) -> float:
        """Calculate maximum drawdown"""
        if not portfolio_values:
            return 0
        
        values = [pv['value'] for pv in portfolio_values]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, portfolio_values: List[Dict]) -> float:
        """Calculate Sharpe ratio"""
        if len(portfolio_values) < 2:
            return 0
        
        values = [pv['value'] for pv in portfolio_values]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        if not returns:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualized Sharpe ratio (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    async def run(self):
        """Run the complete backtest process"""
        try:
            self.status = "running"
            
            # Broadcast start message
            await self._broadcast_update({
                "type": "started",
                "message": "Backtest started",
                "instrument": self.instrument,
                "timeframe": self.timeframe,
                "duration": self.duration
            })
            
            # Load historical data
            data = await self._load_historical_data()
            
            # Run simulation
            results = await self._run_backtest_simulation(data)
            
            # Store results
            self.results = results
            self.status = "completed"
            self.progress = 100
            
            # Broadcast completion
            await self._broadcast_update({
                "type": "completed",
                "results": results,
                "message": "Backtest completed successfully"
            })
            
            logger.info(f"Backtest completed for user {self.user_id}")
            
        except Exception as e:
            self.status = "failed"
            error_message = f"Backtest failed: {str(e)}"
            logger.error(error_message)
            
            await self._broadcast_update({
                "type": "error",
                "message": error_message
            })
    
    def get_status(self) -> Dict[str, Any]:
        """Get current backtest status"""
        return {
            "status": self.status,
            "progress": self.progress,
            "results": self.results
        }
