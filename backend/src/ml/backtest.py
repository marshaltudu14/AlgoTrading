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

    def __init__(self, target_points: float = 20, stop_loss_points: float = 15,
                 ml_model_path: str = None, initial_capital: float = 20000,
                 brokerage_entry: float = 25, brokerage_exit: float = 25):
        """
        Initialize backtester

        Args:
            target_points: Target profit in points
            stop_loss_points: Stop loss in points
            ml_model_path: Path to saved ML models
            initial_capital: Starting capital for backtest
            brokerage_entry: Brokerage fee per trade entry
            brokerage_exit: Brokerage fee per trade exit
        """
        self.target_points = target_points
        self.stop_loss_points = stop_loss_points
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
                      entry_time: pd.Timestamp) -> Tuple[bool, float, int, Dict]:
        bars_held = 0
        exit_details = {
            'exit_price': entry_price,
            'target_price': None,
            'stop_loss_price': None,
            'exit_reason': None
        }

        if direction == 'BUY':
            target_price = entry_price + self.target_points
            stop_loss_price = entry_price - self.stop_loss_points
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
                    return True, self.target_points, bars_held, exit_details

                if low <= stop_loss_price:
                    exit_details.update({
                        'exit_price': stop_loss_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -self.stop_loss_points, bars_held, exit_details

        else:
            target_price = entry_price - self.target_points
            stop_loss_price = entry_price + self.stop_loss_points
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
                    return True, self.target_points, bars_held, exit_details

                if high >= stop_loss_price:
                    exit_details.update({
                        'exit_price': stop_loss_price,
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -self.stop_loss_points, bars_held, exit_details

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
        logger.info(f"Starting backtest with target={self.target_points}, SL={self.stop_loss_points}")

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
                # Only enter if we have strong directional signal and good confidence
                if direction_confidence < 0.35:  # Skip low confidence signals
                    continue

                # Determine if we should enter based on 3-state signals
                should_buy = current_signal == 1 and direction_confidence > 0.4
                should_sell = current_signal == -1 and direction_confidence > 0.4

                if should_buy:
                    position = 'BUY'
                    entry_price = current_price
                    entry_time = current_time
                    entry_volatility = current_volatility
                    logger.debug(f"{current_time}: BUY at {entry_price} (volatility: {entry_volatility}, confidence: {direction_confidence:.2f})")

                elif should_sell:
                    position = 'SELL'
                    entry_price = current_price
                    entry_time = current_time
                    entry_volatility = current_volatility
                    logger.debug(f"{current_time}: SELL at {entry_price} (volatility: {entry_volatility}, confidence: {direction_confidence:.2f})")

            # If in position, check for exit
            elif position:
                # Get future prices for trade simulation
                future_highs = df['high'].iloc[i+1:i+51]  # Next 50 bars
                future_lows = df['low'].iloc[i+1:i+51]

                # Simulate trade with fixed targets and stop loss
                is_win, pnl_points, bars_held, exit_price_detail = self.simulate_trade(
                    entry_price, position, future_highs, future_lows, entry_time
                )

                # Use lot size from instrument
                if instrument and hasattr(instrument, 'lot_size'):
                    lot_size = instrument.lot_size
                else:
                    lot_size = 50  # Default fallback

                # Calculate P&L: points * lot_size
                # For NIFTY: 1 point = Rs. 50 per lot, so 20 points = Rs. 1000 per lot
                pnl_currency = pnl_points * lot_size

                # Subtract brokerage fees
                total_brokerage = self.brokerage_entry + self.brokerage_exit
                net_pnl = pnl_currency - total_brokerage
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
                    'pnl_points': pnl_points,
                    'pnl_currency': pnl_currency,
                    'bars_held': bars_held,
                    'exit_reason': exit_price_detail['exit_reason'],
                    'direction_signal': current_signal,
                    'volatility_state': entry_volatility,
                    'confidence': direction_confidence * 100,  # Convert to percentage
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

        # Total P&L (after brokerage)
        total_pnl = trades_df['net_pnl'].sum()
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average trade P&L (after brokerage)
        avg_trade_pnl = trades_df['net_pnl'].mean()

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
            if trade['is_win']:
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