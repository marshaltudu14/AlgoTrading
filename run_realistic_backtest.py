#!/usr/bin/env python3
"""
Realistic Backtesting System for PPO Trading Models
Processes data row-by-row with real position management, SL/TP handling
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.ppo_agent import PPOAgent
from src.trading.fyers_client import FyersClient
from src.data_processing.feature_generator import DynamicFileProcessor
from src.config.config import INITIAL_CAPITAL, RISK_REWARD_CONFIG
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realistic_backtest_log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealisticBacktester:
    """
    Realistic backtesting system that processes data row-by-row
    with proper position management and risk controls.
    """

    def __init__(self):
        self.fyers_client = FyersClient()
        self.feature_processor = DynamicFileProcessor()

        # Hardcoded configuration - Bank Nifty, 2min, 30 days
        self.SYMBOL = 'NSE:NIFTYBANK-INDEX'
        self.TIMEFRAME = '2'  # 2 minutes
        self.DAYS = 30

        # Use config values for consistency with training
        self.INITIAL_CAPITAL = INITIAL_CAPITAL
        self.LOOKBACK_WINDOW = 20  # Same as training environment

        # Risk management settings from config
        self.STOP_LOSS_ATR_MULTIPLIER = RISK_REWARD_CONFIG['risk_multiplier']  # 1.0 ATR for SL
        self.TARGET_PROFIT_ATR_MULTIPLIER = RISK_REWARD_CONFIG['reward_multiplier']  # 2.0 ATR for TP
        self.TRAILING_STOP_PERCENTAGE = RISK_REWARD_CONFIG['trailing_stop_percentage']  # 0.02 (2%)

        # Model parameters (same as training)
        self.HIDDEN_DIM = 64
        self.ACTION_DIM_DISCRETE = 5  # BUY_LONG, SELL_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        self.ACTION_DIM_CONTINUOUS = 1  # Quantity

        # Trading state
        self.position = 0  # 0 = no position, >0 = long, <0 = short
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        self.target_price = 0.0
        self.trailing_stop_price = 0.0
        self.peak_price = 0.0  # For trailing stop

        # Portfolio state
        self.capital = self.INITIAL_CAPITAL
        self.initial_capital = self.INITIAL_CAPITAL

        # Trade tracking
        self.trades = []
        self.equity_curve = []
        
    def get_observation_dim_from_data(self, processed_data: pd.DataFrame) -> int:
        """Calculate observation dimension based on actual data structure."""
        try:
            # Get feature columns (exclude datetime columns)
            feature_columns = [col for col in processed_data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp']]
            features_per_step = len(feature_columns)

            # Calculate total observation dimension (same as TradingEnv)
            # market_features = lookback_window * features_per_step
            # account_features = 5 (capital, position_quantity, position_entry_price, unrealized_pnl, is_position_open)
            # trailing_features = 1 (distance_to_trail)
            observation_dim = (self.LOOKBACK_WINDOW * features_per_step) + 5 + 1

            logger.info(f"📏 Calculated observation dimensions:")
            logger.info(f"   Features per step: {features_per_step}")
            logger.info(f"   Lookback window: {self.LOOKBACK_WINDOW}")
            logger.info(f"   Total observation dim: {observation_dim}")

            return observation_dim

        except Exception as e:
            logger.error(f"Error calculating observation dimension: {e}")
            return 1246  # Fallback to default

    def load_model(self, model_path: str, observation_dim: int) -> Optional[PPOAgent]:
        """Load the trained PPO model with proper dimensions."""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return None

            # Use exact same parameters as training
            agent = PPOAgent(
                observation_dim=observation_dim,
                action_dim_discrete=self.ACTION_DIM_DISCRETE,
                action_dim_continuous=self.ACTION_DIM_CONTINUOUS,
                hidden_dim=self.HIDDEN_DIM
            )

            agent.load_model(model_path)
            logger.info(f"✅ Model loaded from {model_path}")
            logger.info(f"   Observation dim: {observation_dim}")
            logger.info(f"   Action discrete: {self.ACTION_DIM_DISCRETE}")
            logger.info(f"   Action continuous: {self.ACTION_DIM_CONTINUOUS}")
            logger.info(f"   Hidden dim: {self.HIDDEN_DIM}")
            return agent

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return None
    
    def calculate_position_size(self, predicted_quantity: float) -> int:
        """Use predicted quantity directly as integer (same as training)."""
        try:
            # Simply convert predicted quantity to integer (same as training environment)
            final_quantity = int(round(predicted_quantity))

            # Ensure non-negative
            final_quantity = max(final_quantity, 0)

            return final_quantity

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def calculate_sl_tp_prices(self, entry_price: float, action_type: int, atr: float) -> Tuple[float, float]:
        """
        Calculate stop loss and target prices based on ATR.

        Args:
            entry_price: Entry price
            action_type: 0=BUY_LONG, 1=SELL_SHORT
            atr: Average True Range value

        Returns:
            Tuple of (stop_loss_price, target_price)
        """
        try:
            if action_type == 0:  # BUY_LONG
                # For long: SL below entry, TP above entry
                stop_loss_price = entry_price - (self.STOP_LOSS_ATR_MULTIPLIER * atr)
                target_price = entry_price + (self.TARGET_PROFIT_ATR_MULTIPLIER * atr)
            else:  # SELL_SHORT
                # For short: SL above entry, TP below entry
                stop_loss_price = entry_price + (self.STOP_LOSS_ATR_MULTIPLIER * atr)
                target_price = entry_price - (self.TARGET_PROFIT_ATR_MULTIPLIER * atr)

            return stop_loss_price, target_price

        except Exception as e:
            logger.error(f"Error calculating SL/TP prices: {e}")
            return entry_price, entry_price
    
    def update_trailing_stop(self, current_price: float):
        """Update trailing stop loss based on current price."""
        if self.position == 0:
            return

        try:
            if self.position > 0:  # Long position
                if current_price > self.peak_price:
                    self.peak_price = current_price
                    self.trailing_stop_price = self.peak_price * (1 - self.TRAILING_STOP_PERCENTAGE)
            else:  # Short position
                if current_price < self.peak_price:
                    self.peak_price = current_price
                    self.trailing_stop_price = self.peak_price * (1 + self.TRAILING_STOP_PERCENTAGE)

        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
    
    def check_exit_conditions(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if any exit conditions are met.
        
        Returns:
            Tuple of (should_exit, exit_reason)
        """
        if self.position == 0:
            return False, ""
            
        try:
            if self.position > 0:  # Long position
                if current_price <= self.stop_loss_price:
                    return True, "STOP_LOSS_HIT"
                elif current_price >= self.target_price:
                    return True, "TARGET_PROFIT_HIT"
                elif current_price <= self.trailing_stop_price:
                    return True, "TRAILING_STOP_HIT"
            else:  # Short position
                if current_price >= self.stop_loss_price:
                    return True, "STOP_LOSS_HIT"
                elif current_price <= self.target_price:
                    return True, "TARGET_PROFIT_HIT"
                elif current_price >= self.trailing_stop_price:
                    return True, "TRAILING_STOP_HIT"
                    
            return False, ""
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return False, "ERROR"
    
    def execute_trade(self, action_type: int, quantity: int, current_price: float, atr: float, timestamp: pd.Timestamp) -> bool:
        """
        Execute a trade based on the action type.
        
        Returns:
            bool: True if trade was executed successfully
        """
        try:
            if action_type == 0:  # BUY_LONG
                if self.position != 0:
                    logger.warning(f"Cannot open long position - already have position: {self.position}")
                    return False
                    
                self.position = quantity
                self.entry_price = current_price
                self.stop_loss_price, self.target_price = self.calculate_sl_tp_prices(current_price, 0, atr)
                self.peak_price = current_price
                self.trailing_stop_price = current_price * (1 - self.TRAILING_STOP_PERCENTAGE)
                
                # Update capital (subtract cost)
                cost = quantity * current_price
                self.capital -= cost
                
                logger.info(f"🟢 LONG ENTRY: {quantity} @ ₹{current_price:.2f}")
                logger.info(f"   SL: ₹{self.stop_loss_price:.2f}, TP: ₹{self.target_price:.2f}")
                
                return True
                
            elif action_type == 1:  # SELL_SHORT
                if self.position != 0:
                    logger.warning(f"Cannot open short position - already have position: {self.position}")
                    return False
                    
                self.position = -quantity
                self.entry_price = current_price
                self.stop_loss_price, self.target_price = self.calculate_sl_tp_prices(current_price, 1, atr)
                self.peak_price = current_price
                self.trailing_stop_price = current_price * (1 + self.TRAILING_STOP_PERCENTAGE)
                
                # Update capital (add proceeds from short sale)
                proceeds = quantity * current_price
                self.capital += proceeds
                
                logger.info(f"🔴 SHORT ENTRY: {quantity} @ ₹{current_price:.2f}")
                logger.info(f"   SL: ₹{self.stop_loss_price:.2f}, TP: ₹{self.target_price:.2f}")
                
                return True
                
            elif action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
                return self.close_position(current_price, timestamp, "MODEL_SIGNAL")
                
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

    def close_position(self, current_price: float, timestamp: pd.Timestamp, reason: str) -> bool:
        """Close the current position."""
        try:
            if self.position == 0:
                return False

            # Calculate P&L
            if self.position > 0:  # Closing long
                pnl = self.position * (current_price - self.entry_price)
                proceeds = self.position * current_price
                self.capital += proceeds
                logger.info(f"🟢 LONG EXIT: {self.position} @ ₹{current_price:.2f} | Reason: {reason}")
            else:  # Closing short
                pnl = abs(self.position) * (self.entry_price - current_price)
                cost = abs(self.position) * current_price
                self.capital -= cost
                logger.info(f"🔴 SHORT EXIT: {abs(self.position)} @ ₹{current_price:.2f} | Reason: {reason}")

            # Record trade
            trade = {
                'timestamp': timestamp,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'quantity': abs(self.position),
                'side': 'LONG' if self.position > 0 else 'SHORT',
                'pnl': pnl,
                'reason': reason,
                'capital_after': self.capital
            }
            self.trades.append(trade)

            logger.info(f"   P&L: ₹{pnl:.2f}, Capital: ₹{self.capital:.2f}")

            # Reset position
            self.position = 0
            self.entry_price = 0.0
            self.stop_loss_price = 0.0
            self.target_price = 0.0
            self.trailing_stop_price = 0.0
            self.peak_price = 0.0

            return True

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    def run_backtest(self) -> Dict:
        """
        Run realistic backtesting on Bank Nifty 2-minute data.

        Returns:
            dict: Backtest results
        """
        try:
            logger.info(f"🚀 Starting realistic backtesting...")
            logger.info(f"   Symbol: {self.SYMBOL}")
            logger.info(f"   Timeframe: {self.TIMEFRAME} minutes")
            logger.info(f"   Days: {self.DAYS}")

            # Fetch and process data first to get observation dimensions
            logger.info(f"📡 Fetching real-time data...")
            raw_data = self.fyers_client.get_historical_data(
                symbol=self.SYMBOL,
                timeframe=self.TIMEFRAME,
                days=self.DAYS
            )

            if raw_data is None or raw_data.empty:
                logger.error("Failed to load data")
                return {}

            logger.info(f"📊 Raw data: {len(raw_data)} candles")

            # Process features
            logger.info(f"⚙️ Processing features...")
            processed_data = self.feature_processor.process_dataframe(raw_data)

            if processed_data is None or processed_data.empty:
                logger.error("Failed to process features")
                return {}

            logger.info(f"✅ Processed data: {len(processed_data)} rows, {processed_data.shape[1]} features")

            # Calculate observation dimensions from actual data
            observation_dim = self.get_observation_dim_from_data(processed_data)

            # Load model with proper dimensions
            model_path = 'models/universal_final_model.pth'
            agent = self.load_model(model_path, observation_dim)
            if agent is None:
                return {}



            # Sequential row-by-row processing
            for i in range(self.LOOKBACK_WINDOW, len(processed_data)):
                current_row = processed_data.iloc[i]
                timestamp = processed_data.index[i]

                # Get current price and ATR
                current_price = current_row.get('close', 0.0)
                atr = current_row.get('atr', current_price * 0.02)  # Default 2% if no ATR

                if current_price <= 0:
                    continue

                # Update trailing stop
                self.update_trailing_stop(current_price)

                # Check exit conditions first
                should_exit, exit_reason = self.check_exit_conditions(current_price)
                if should_exit:
                    self.close_position(current_price, timestamp, exit_reason)

                # Get observation for model (last LOOKBACK_WINDOW rows)
                obs_data = processed_data.iloc[i-self.LOOKBACK_WINDOW:i]
                observation = obs_data.values.flatten()

                # Ensure observation has correct size
                if len(observation) != 1246:  # Expected size
                    # Pad or truncate to correct size
                    if len(observation) < 1246:
                        observation = np.pad(observation, (0, 1246 - len(observation)), 'constant')
                    else:
                        observation = observation[:1246]

                # Get model prediction for THIS SINGLE ROW
                action = agent.select_action(observation)

                # Parse action
                if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
                    action_type = int(np.clip(action[0], 0, 4))
                    predicted_quantity = max(0, float(action[1]))
                else:
                    action_type = 4  # HOLD
                    predicted_quantity = 0

                # Execute trades based on model prediction for current row
                if action_type in [0, 1] and predicted_quantity > 0:  # BUY_LONG or SELL_SHORT
                    actual_quantity = self.calculate_position_size(predicted_quantity, current_price)

                    if actual_quantity > 0:
                        self.execute_trade(action_type, actual_quantity, current_price, atr, timestamp)
                elif action_type in [2, 3] and self.position != 0:  # CLOSE_LONG or CLOSE_SHORT
                    self.close_position(current_price, timestamp, "MODEL_SIGNAL")

                # Record equity curve for this row
                total_value = self.capital
                if self.position != 0:
                    if self.position > 0:
                        total_value += self.position * current_price
                    else:
                        total_value -= abs(self.position) * current_price

                self.equity_curve.append({
                    'timestamp': timestamp,
                    'capital': self.capital,
                    'total_value': total_value,
                    'position': self.position,
                    'price': current_price
                })

                # Log progress every 500 rows (not batched processing)
                if i % 500 == 0:
                    logger.info(f"📊 Row {i}/{len(processed_data)} | Capital: ₹{self.capital:.2f} | Position: {self.position}")

            # Close any remaining position
            if self.position != 0:
                final_price = processed_data.iloc[-1].get('close', 0.0)
                self.close_position(final_price, processed_data.index[-1], "END_OF_DATA")

            # Calculate results
            results = self.calculate_results()
            return results

        except Exception as e:
            logger.error(f"Error in backtesting: {e}", exc_info=True)
            return {}

    def calculate_results(self) -> Dict:
        """Calculate comprehensive backtest results."""
        try:
            final_capital = self.capital
            total_pnl = final_capital - self.initial_capital
            total_return_pct = (total_pnl / self.initial_capital) * 100

            # Trade statistics
            total_trades = len(self.trades)
            winning_trades = len([t for t in self.trades if t['pnl'] > 0])
            losing_trades = len([t for t in self.trades if t['pnl'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            # P&L statistics
            total_profit = sum([t['pnl'] for t in self.trades if t['pnl'] > 0])
            total_loss = sum([t['pnl'] for t in self.trades if t['pnl'] < 0])
            profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')

            # Drawdown calculation
            equity_values = [eq['total_value'] for eq in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0
            for value in equity_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

            results = {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_pnl': total_pnl,
                'total_return_percentage': total_return_pct,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown_percentage': max_drawdown * 100,
                'trades': self.trades,
                'equity_curve': self.equity_curve
            }

            return results

        except Exception as e:
            logger.error(f"Error calculating results: {e}")
            return {}

def main():
    logger.info("🎯 Hardcoded settings: Bank Nifty, 2-minute timeframe, 30 days")

    backtester = RealisticBacktester()
    results = backtester.run_backtest()

    if results:
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 REALISTIC BACKTEST RESULTS")
        logger.info(f"{'='*50}")
        logger.info(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
        logger.info(f"Final Capital:   ₹{results['final_capital']:,.2f}")
        logger.info(f"Total P&L:       ₹{results['total_pnl']:,.2f}")
        logger.info(f"Total Return:    {results['total_return_percentage']:.2f}%")
        logger.info(f"Total Trades:    {results['total_trades']}")
        logger.info(f"Win Rate:        {results['win_rate']:.2f}%")
        logger.info(f"Profit Factor:   {results['profit_factor']:.2f}")
        logger.info(f"Max Drawdown:    {results['max_drawdown_percentage']:.2f}%")
        logger.info(f"{'='*50}")
    else:
        logger.error("Backtesting failed")

if __name__ == "__main__":
    main()
