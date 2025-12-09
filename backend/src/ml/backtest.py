"""
Backtesting Module for Algo Trading
===================================

Implements backtesting logic using ML predictions for trade execution.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging
from .predictor import TradingPredictor

logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for ML-based trading strategies"""

    def __init__(self, target_pnl: float = 1000, stop_loss_pnl: float = -600,
                 ml_model_path: str = None, initial_capital: float = 20000,
                 brokerage_entry: float = 25, brokerage_exit: float = 25):
        """
        Initialize backtester

        Args:
            target_pnl: Target profit in Rs (per position)
            stop_loss_pnl: Stop loss in Rs (per position)
            ml_model_path: Path to saved ML models
            initial_capital: Starting capital for backtest
            brokerage_entry: Brokerage fee per trade entry
            brokerage_exit: Brokerage fee per trade exit
        """
        self.target_pnl = target_pnl
        self.stop_loss_pnl = stop_loss_pnl
        self.ml_model_path = ml_model_path
        self.initial_capital = initial_capital
        self.brokerage_entry = brokerage_entry
        self.brokerage_exit = brokerage_exit
        self.predictor = None
        self.trades = []
        self.equity_curve = []
        self.trade_log_path = None

    def load_ml_model(self, model_path: str):
        """Load pre-trained ML models"""
        try:
            self.predictor = TradingPredictor()
            self.predictor.load_models(model_path)
            logger.info(f"ML models loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            raise

    def simulate_trade(self, entry_price: float, direction: str,
                      high_prices: pd.Series, low_prices: pd.Series,
                      entry_time: pd.Timestamp, lot_size: int) -> Tuple[bool, float, int, Dict]:
        bars_held = 0
        exit_details = {
            'exit_price': entry_price,
            'target_price': None,
            'stop_loss_price': None,
            'exit_reason': None
        }

        # Calculate points needed based on P&L targets
        target_points = abs(self.target_pnl) / lot_size
        stop_loss_points = abs(self.stop_loss_pnl) / lot_size

        if direction == 'BUY':
            target_price = entry_price + target_points
            stop_loss_price = entry_price - stop_loss_points
            exit_details.update({
                'target_price': target_price,
                'stop_loss_price': stop_loss_price
            })

            for high, low in zip(high_prices, low_prices):
                bars_held += 1

                if high >= target_price:
                    exit_details.update({
                        'exit_price': target_price,
                        'exit_reason': 'TARGET'
                    })
                    return True, target_points, bars_held, exit_details

                if low <= stop_loss_price:
                    exit_details.update({
                        'exit_price': stop_loss_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -stop_loss_points, bars_held, exit_details

        else:
            target_price = entry_price - target_points
            stop_loss_price = entry_price + stop_loss_points
            exit_details.update({
                'target_price': target_price,
                'stop_loss_price': stop_loss_price
            })

            for high, low in zip(high_prices, low_prices):
                bars_held += 1

                if low <= target_price:
                    exit_details.update({
                        'exit_price': target_price,
                        'exit_reason': 'TARGET'
                    })
                    return True, target_points, bars_held, exit_details

                if high >= stop_loss_price:
                    exit_details.update({
                        'exit_price': stop_loss_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -stop_loss_points, bars_held, exit_details

        if len(high_prices) > 0:
            last_price = high_prices.iloc[-1] if direction == 'BUY' else low_prices.iloc[-1]
            pnl = last_price - entry_price if direction == 'BUY' else entry_price - last_price
            exit_details.update({
                'exit_price': last_price,
                'exit_reason': 'EXPIRY'
            })
            return pnl > 0, pnl, bars_held, exit_details

        return False, 0, bars_held, exit_details

    def run_backtest(self, csv_path: str, instrument=None) -> Dict:
        """
        Run backtest on historical data

        Args:
            csv_path: Path to processed data CSV
            instrument: Instrument object containing lot size and other info

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest with target=Rs.{self.target_pnl}, SL=Rs.{self.stop_loss_pnl}")

        # Load data
        df = pd.read_csv(csv_path)
        if 'datetime_readable' in df.columns:
            df['datetime_readable'] = pd.to_datetime(df['datetime_readable'])
            df.set_index('datetime_readable', inplace=True)

        # Load ML models
        if self.ml_model_path:
            self.load_ml_model(self.ml_model_path)
        else:
            raise ValueError("ML model path not provided")

        # Initialize tracking variables
        capital = self.initial_capital
        position = None
        entry_price = None
        entry_time = None
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        # Generate predictions for all data points
        predictions = self.predictor.predict(df)
        df['direction_signal'] = predictions['direction_signal']
        df['volatility_prediction'] = predictions['volatility']
        df['direction_proba'] = [proba.max() for proba in predictions['direction_proba']]
        df['volatility_proba'] = [proba.max() for proba in predictions['volatility_proba']]

        # Iterate through each data point (skip last 50 bars to ensure future data for trade simulation)
        for i in range(len(df) - 50):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            current_signal = df['direction_signal'].iloc[i]
            current_volatility = df['volatility_prediction'].iloc[i]
            direction_confidence = df['direction_proba'].iloc[i]

            # If not in position, check for entry signal
            if position is None:
                # Get current ATR if available
                current_atr = df['atr'].iloc[i] if 'atr' in df.columns else None
                
                # Check Volatility Prediction (0=Low, 1=Normal, 2=High)
                # Avoid trading in Low Volatility (0) unless confidence is reasonable
                if current_volatility == 0 and direction_confidence < 0.45:
                    continue

                # Only enter if we have strong directional signal and good confidence
                # Lowered threshold to 0.38 (slightly better than random 0.33) to allow more trades
                if direction_confidence < 0.38:
                    continue

                # Determine if we should enter based on 3-state signals and ATR
                should_buy = False
                should_sell = False

                if current_signal == 1:
                    # For BUY
                    should_buy = True
                    # Optional: Add ATR filter if available (ensure enough range)
                    if current_atr is not None and current_atr < 5:
                         if direction_confidence < 0.45: # require higher confidence if ATR is low
                            should_buy = False

                elif current_signal == -1:
                    # For SELL
                    should_sell = True
                    # Optional: Add ATR filter
                    if current_atr is not None and current_atr < 5:
                         if direction_confidence < 0.45:
                            should_sell = False

                if should_buy:
                    position = 'BUY'
                    entry_price = current_price
                    entry_time = current_time
                    entry_volatility = current_volatility
                    atr_info = f", ATR: {current_atr:.2f}" if current_atr else ""
                    logger.debug(f"{current_time}: BUY at {entry_price} (vol2state: {entry_volatility}, confidence: {direction_confidence:.2f}{atr_info})")

                elif should_sell:
                    position = 'SELL'
                    entry_price = current_price
                    entry_time = current_time
                    entry_volatility = current_volatility
                    atr_info = f", ATR: {current_atr:.2f}" if current_atr else ""
                    logger.debug(f"{current_time}: SELL at {entry_price} (vol2state: {entry_volatility}, confidence: {direction_confidence:.2f}{atr_info})")

            # If in position, check for exit
            elif position:
                # Use lot size from instrument
                if instrument and hasattr(instrument, 'lot_size'):
                    lot_size = instrument.lot_size
                else:
                    lot_size = 50  # Default fallback

                # Get future prices for trade simulation
                future_highs = df['high'].iloc[i+1:i+51]  # Next 50 bars
                future_lows = df['low'].iloc[i+1:i+51]

                # Simulate trade with P&L-based targets and stop loss
                is_win, pnl_points, bars_held, exit_price_detail = self.simulate_trade(
                    entry_price, position, future_highs, future_lows, entry_time, lot_size
                )

                # Calculate P&L: points * lot_size
                pnl_currency = pnl_points * lot_size

                # Add brokerage fees to P&L calculation
                total_brokerage = self.brokerage_entry + self.brokerage_exit

                # For winners: subtract brokerage
                # For losers: add brokerage to the loss
                if is_win:
                    net_pnl = pnl_currency - total_brokerage
                else:
                    net_pnl = pnl_currency - total_brokerage  # pnl_currency is already negative

                capital += net_pnl

                exit_time = df.index[i + bars_held] if i + bars_held < len(df) else df.index[-1]
                trade = {
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'position': position,
                    'entry_price': entry_price,
                    'target_price': exit_price_detail['target_price'],
                    'stop_loss_price': exit_price_detail['stop_loss_price'],
                    'lot_size': lot_size,
                    'pnl_points': round(pnl_points, 2),  # Points with 2 decimal places
                    'pnl_currency': round(net_pnl, 2),  # Net P&L in Rs with 2 decimal places
                    'bars_held': bars_held,
                    'exit_reason': exit_price_detail['exit_reason'],
                    'direction_signal': current_signal,
                    'volatility_state': entry_volatility,
                    'confidence': round(direction_confidence * 100, 1),  # Convert to percentage, 2 digits
                    'capital': capital  # Current capital after this trade
                }
                self.trades.append(trade)

                total_trades += 1
                if is_win:
                    winning_trades += 1
                else:
                    losing_trades += 1

                self.equity_curve.append({
                    'time': exit_time,
                    'capital': capital
                })

                # Save trade to CSV in real-time
                self._save_trade_to_csv(trade)

                position = None
                entry_price = None
                entry_time = None

        # Calculate final metrics
        results = self._calculate_metrics(self.initial_capital, total_trades, winning_trades, losing_trades)

        logger.info(f"Backtest completed - Total trades: {total_trades}, Win rate: {results['win_rate']:.2%}")
        logger.info(f"Max Winning Streak: {results.get('max_winning_streak', 0)} trades")
        logger.info(f"Max Losing Streak: {results.get('max_losing_streak', 0)} trades")

        return results

    def _calculate_metrics(self, initial_capital: float, total_trades: int,
                          winning_trades: int, losing_trades: int) -> Dict:
        """Calculate performance metrics"""

        if not self.trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'total_pnl_percent': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'avg_trade_pnl': 0
            }

        # Convert trades list to DataFrame for easier analysis
        trades_df = pd.DataFrame(self.trades)

        # Total P&L (from pnl_currency, which is after brokerage)
        total_pnl = trades_df['pnl_currency'].sum()
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average trade P&L (after brokerage)
        avg_trade_pnl = trades_df['pnl_currency'].mean()

        # Maximum drawdown
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = equity_df['capital'] - equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()

            # Sharpe ratio (assuming daily returns)
            equity_df['returns'] = equity_df['capital'].pct_change()
            sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()
                          * np.sqrt(252)) if equity_df['returns'].std() != 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0

        # Calculate winning and losing streaks
        max_winning_streak = 0
        max_losing_streak = 0
        current_winning_streak = 0
        current_losing_streak = 0

        for _, trade in trades_df.iterrows():
            # Determine if trade was winning based on pnl_currency
            is_winning_trade = trade['pnl_currency'] > 0
            if is_winning_trade:
                current_winning_streak += 1
                max_winning_streak = max(max_winning_streak, current_winning_streak)
                current_losing_streak = 0
            else:
                current_losing_streak += 1
                max_losing_streak = max(max_losing_streak, current_losing_streak)
                current_winning_streak = 0

        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_pnl_percent': total_pnl_percent,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_pnl': avg_trade_pnl,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak,
            'profit_factor': (trades_df[trades_df['pnl_currency'] > 0]['pnl_currency'].sum() /
                             abs(trades_df[trades_df['pnl_currency'] < 0]['pnl_currency'].sum()))
                             if trades_df[trades_df['pnl_currency'] < 0]['pnl_currency'].sum() != 0 else float('inf')
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Get detailed trade log"""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve data"""
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame(self.equity_curve)

    def _save_trade_to_csv(self, trade: Dict):
        """Save a single trade to CSV file"""
        try:
            # Convert trade dict to DataFrame
            trade_df = pd.DataFrame([trade])

            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(self.trade_log_path) if self.trade_log_path else False

            # Append to CSV
            if self.trade_log_path:
                trade_df.to_csv(self.trade_log_path, mode='a', header=not file_exists, index=False)
        except Exception as e:
            logger.error(f"Failed to save trade to CSV: {e}")