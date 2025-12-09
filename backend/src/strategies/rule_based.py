"""
Rule-Based Trading Strategies
==============================

Uses pre-calculated features from feature_generator.py to implement
rule-based trading strategies without ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class RuleBasedBacktester:
    """Backtester for rule-based strategies using pre-calculated features"""

    def __init__(self, target_pnl: float = 500, stop_loss_pnl: float = -300,
                 initial_capital: float = 20000, brokerage_entry: float = 25,
                 brokerage_exit: float = 25):
        """
        Initialize backtester for rule-based strategies

        Args:
            target_pnl: Target profit in Rs (per position)
            stop_loss_pnl: Stop loss in Rs (per position)
            initial_capital: Starting capital for backtest
            brokerage_entry: Brokerage fee per trade entry
            brokerage_exit: Brokerage fee per trade exit
        """
        self.target_pnl = target_pnl
        self.stop_loss_pnl = stop_loss_pnl
        self.initial_capital = initial_capital
        self.brokerage_entry = brokerage_entry
        self.brokerage_exit = brokerage_exit
        self.trades = []
        self.equity_curve = []
        self.trade_log_path = None

    def rsi_mean_reversion(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """RSI Mean Reversion Strategy using pre-calculated RSI"""
        rsi = df['rsi_14'].iloc[i]
        price = df['close'].iloc[i]

        signal = 'HOLD'
        confidence = 0.0

        if rsi < 30:  # Oversold
            signal = 'BUY'
            confidence = min(0.8, (30 - rsi) / 20)
        elif rsi > 70:  # Overbought
            signal = 'SELL'
            confidence = min(0.8, (rsi - 70) / 20)

        return signal, confidence

    def macd_strategy(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """MACD Strategy using pre-calculated MACD features"""
        if i < 1:
            return 'HOLD', 0.0

        macd_pct = df['macd_pct'].iloc[i]
        macd_signal_pct = df['macd_signal_pct'].iloc[i]
        macd_hist_pct = df['macd_hist_pct'].iloc[i]

        prev_macd = df['macd_pct'].iloc[i-1]
        prev_signal = df['macd_signal_pct'].iloc[i-1]

        signal = 'HOLD'
        confidence = 0.0

        # Bullish crossover
        if macd_pct > macd_signal_pct and prev_macd <= prev_signal and macd_hist_pct > 0:
            signal = 'BUY'
            confidence = min(0.9, abs(macd_hist_pct) * 10)
        # Bearish crossover
        elif macd_pct < macd_signal_pct and prev_macd >= prev_signal and macd_hist_pct < 0:
            signal = 'SELL'
            confidence = min(0.9, abs(macd_hist_pct) * 10)

        return signal, confidence

    def bollinger_bands_strategy(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Bollinger Bands Strategy using pre-calculated BB features"""
        bb_position = df['bb_position'].iloc[i]  # 0 to 1, where 0 is at lower band
        bb_width_pct = df['bb_width_pct'].iloc[i]

        signal = 'HOLD'
        confidence = 0.0

        # Squeeze play - low volatility followed by breakout
        if i > 20:
            prev_bb_width = df['bb_width_pct'].iloc[i-20:i].mean()
            squeeze = bb_width_pct < 0.8 * prev_bb_width

            if squeeze and bb_position > 0.9:  # Breakout above upper band
                signal = 'BUY'
                confidence = min(0.85, bb_position - 0.9)
            elif squeeze and bb_position < 0.1:  # Breakout below lower band
                signal = 'SELL'
                confidence = min(0.85, 0.1 - bb_position)

        return signal, confidence

    def ema_crossover_strategy(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """EMA Crossover Strategy using distance from EMAs"""
        if i < 1:
            return 'HOLD', 0.0

        dist_ema_20 = df['dist_ema_20'].iloc[i]
        dist_sma_200 = df['dist_sma_200'].iloc[i]

        prev_dist_ema_20 = df['dist_ema_20'].iloc[i-1]
        prev_dist_sma_200 = df['dist_sma_200'].iloc[i-1]

        signal = 'HOLD'
        confidence = 0.0

        # Golden cross - price crosses above EMA20 while above SMA200
        if dist_ema_20 > 0 and prev_dist_ema_20 <= 0 and dist_sma_200 > 0:
            signal = 'BUY'
            confidence = min(0.8, abs(dist_ema_20) / 2)
        # Death cross - price crosses below EMA20 while below SMA200
        elif dist_ema_20 < 0 and prev_dist_ema_20 >= 0 and dist_sma_200 < 0:
            signal = 'SELL'
            confidence = min(0.8, abs(dist_ema_20) / 2)

        return signal, confidence

    def trend_following(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Trend Following Strategy using multiple indicators"""
        if i < 10:
            return 'HOLD', 0.0

        # Get trend strength and direction
        trend_strength = df['trend_strength'].iloc[i]
        trend_direction = df['trend_direction'].iloc[i]

        # Get ADX for trend strength confirmation
        adx = df['adx'].iloc[i] if 'adx' in df.columns else 25

        # Get price momentum
        price_change = df['price_change_pct'].iloc[i]

        signal = 'HOLD'
        confidence = 0.0

        # Strong uptrend
        if (trend_direction > 0 and trend_strength > 0.5 and
            adx > 25 and price_change > 0):
            signal = 'BUY'
            confidence = min(0.9, (trend_strength + adx/100) / 2)
        # Strong downtrend
        elif (trend_direction < 0 and trend_strength > 0.5 and
              adx > 25 and price_change < 0):
            signal = 'SELL'
            confidence = min(0.9, (trend_strength + adx/100) / 2)

        return signal, confidence

    def mean_reversion(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Mean Reversion Strategy using distance from moving averages"""
        dist_sma_20 = df['dist_sma_20'].iloc[i]
        dist_sma_50 = df['dist_sma_50'].iloc[i]
        atr_pct = df['atr_pct'].iloc[i]

        signal = 'HOLD'
        confidence = 0.0

        # Far from mean with normal volatility
        if abs(dist_sma_20) > 2 and atr_pct < 2:
            if dist_sma_20 < -2:  # Price below 20 SMA by 2%
                signal = 'BUY'
                confidence = min(0.7, abs(dist_sma_20) / 5)
            elif dist_sma_20 > 2:  # Price above 20 SMA by 2%
                signal = 'SELL'
                confidence = min(0.7, abs(dist_sma_20) / 5)

        return signal, confidence

    def momentum_breakout(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Momentum Breakout Strategy"""
        if i < 20:
            return 'HOLD', 0.0

        # Recent price performance
        price_change_abs = df['price_change_abs'].iloc[i]
        hl_range_pct = df['hl_range_pct'].iloc[i]

        # Volatility
        volatility_10 = df['volatility_10'].iloc[i] if 'volatility_10' in df.columns else 1
        volatility_20 = df['volatility_20'].iloc[i] if 'volatility_20' in df.columns else 1

        # Volume confirmation (if available)
        volume_confirmation = 1  # Default if no volume data

        signal = 'HOLD'
        confidence = 0.0

        # High momentum with expanding volatility
        if (price_change_abs > 1.5 and hl_range_pct > 2.0 and
            volatility_10 > volatility_20):
            if df['price_change_pct'].iloc[i] > 0:  # Positive momentum
                signal = 'BUY'
                confidence = min(0.8, price_change_abs / 3)
            else:  # Negative momentum
                signal = 'SELL'
                confidence = min(0.8, price_change_abs / 3)

        return signal, confidence

    def multi_indicator_combo(self, df: pd.DataFrame, i: int) -> Tuple[str, float]:
        """Enhanced Multi-Indicator Combination Strategy"""

        # Get signals from multiple strategies
        rsi_signal, rsi_conf = self.rsi_mean_reversion(df, i)
        macd_signal, macd_conf = self.macd_strategy(df, i)
        trend_signal, trend_conf = self.trend_following(df, i)

        # Get ATR for volatility filtering
        atr_pct = df['atr_pct'].iloc[i] if 'atr_pct' in df.columns else 1.0

        # Get price momentum
        price_change_pct = df['price_change_pct'].iloc[i] if 'price_change_pct' in df.columns else 0

        # Dynamic weighting based on volatility and momentum
        weight_multiplier = 1.0

        # In high volatility, reduce RSI weight (don't fade strong momentum)
        if atr_pct > 2.0 or abs(price_change_pct) > 1.5:
            # Strong trend - prioritize trend following
            trend_conf *= 1.5
            rsi_conf *= 0.3
            weight_multiplier = 1.2  # Require stronger consensus
        elif atr_pct < 0.5:
            # Low volatility - RSI mean reversion works better
            rsi_conf *= 1.3
            trend_conf *= 0.7
            weight_multiplier = 0.8  # Easier to enter

        # Enhanced RSI logic for momentum markets
        if abs(price_change_pct) > 2.0:  # Very strong momentum
            # Ignore RSI overbought/oversold in strong momentum
            if (rsi_signal == 'SELL' and price_change_pct > 2.0) or \
               (rsi_signal == 'BUY' and price_change_pct < -2.0):
                rsi_signal = 'HOLD'
                rsi_conf = 0

        # Weight voting with dynamic adjustments
        buy_votes = 0
        sell_votes = 0
        total_confidence = 0

        for signal, conf in [(rsi_signal, rsi_conf), (macd_signal, macd_conf), (trend_signal, trend_conf)]:
            if signal == 'BUY':
                buy_votes += conf
                total_confidence += conf
            elif signal == 'SELL':
                sell_votes += conf
                total_confidence += conf

        # Adjust thresholds based on volatility
        buy_threshold = 0.6 / weight_multiplier
        sell_threshold = 0.6 / weight_multiplier
        min_confidence = 0.5 * weight_multiplier

        # Decision based on weighted votes
        if total_confidence > 0:
            buy_ratio = buy_votes / total_confidence
            sell_ratio = sell_votes / total_confidence

            if buy_ratio > buy_threshold and total_confidence > min_confidence:
                # In high volatility, require trend alignment
                if atr_pct > 2.0 and trend_signal != 'BUY' and trend_conf > 0:
                    return 'HOLD', 0.0
                return 'BUY', min(0.9, buy_ratio * weight_multiplier)
            elif sell_ratio > sell_threshold and total_confidence > min_confidence:
                # In high volatility, require trend alignment
                if atr_pct > 2.0 and trend_signal != 'SELL' and trend_conf > 0:
                    return 'HOLD', 0.0
                return 'SELL', min(0.9, sell_ratio * weight_multiplier)

        return 'HOLD', 0.0

    def get_signal(self, df: pd.DataFrame, strategy: str, i: int) -> Tuple[str, float]:
        """Get trading signal for a specific strategy"""

        strategy_map = {
            'rsi_mean_reversion': self.rsi_mean_reversion,
            'ema_crossover': self.ema_crossover_strategy,
            'trend_following': self.trend_following,
            'multi_indicator_combo': self.multi_indicator_combo
        }

        if strategy not in strategy_map:
            logger.error(f"Strategy '{strategy}' not found")
            return 'HOLD', 0.0

        return strategy_map[strategy](df, i)

    def simulate_trade(self, entry_price: float, direction: str,
                      high_prices: pd.Series, low_prices: pd.Series,
                      entry_time: pd.Timestamp, lot_size: int) -> Tuple[bool, float, int, Dict]:
        """Simulate trade execution"""
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
                'target_price': round(target_price, 2),
                'stop_loss_price': round(stop_loss_price, 2)
            })

            for high, low in zip(high_prices, low_prices):
                bars_held += 1

                if high >= target_price:
                    exit_details.update({
                        'exit_price': round(target_price, 2),
                        'exit_reason': 'TARGET'
                    })
                    return True, target_points, bars_held, exit_details

                if low <= stop_loss_price:
                    exit_details.update({
                        'exit_price': round(stop_loss_price, 2),
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -stop_loss_points, bars_held, exit_details

        else:  # SELL
            target_price = entry_price - target_points
            stop_loss_price = entry_price + stop_loss_points
            exit_details.update({
                'target_price': round(target_price, 2),
                'stop_loss_price': round(stop_loss_price, 2)
            })

            for high, low in zip(high_prices, low_prices):
                bars_held += 1

                if low <= target_price:
                    exit_details.update({
                        'exit_price': round(target_price, 2),
                        'exit_reason': 'TARGET'
                    })
                    return True, target_points, bars_held, exit_details

                if high >= stop_loss_price:
                    exit_details.update({
                        'exit_price': round(stop_loss_price, 2),
                        'exit_reason': 'STOP_LOSS'
                    })
                    return False, -stop_loss_points, bars_held, exit_details

        # If we reach here, trade didn't hit target or SL
        if len(high_prices) > 0:
            last_price = high_prices.iloc[-1] if direction == 'BUY' else low_prices.iloc[-1]
            pnl = last_price - entry_price if direction == 'BUY' else entry_price - last_price
            exit_details.update({
                'exit_price': round(last_price, 2),
                'exit_reason': 'EXPIRY'
            })
            return pnl > 0, pnl, bars_held, exit_details

        return False, 0, bars_held, exit_details

    def run_backtest(self, csv_path: str, strategy: str = 'multi_indicator_combo',
                    min_confidence: float = 0.5, instrument=None) -> Dict:
        """
        Run backtest using rule-based strategy

        Args:
            csv_path: Path to processed data CSV with features
            strategy: Strategy name to use
            min_confidence: Minimum confidence threshold for taking trades
            instrument: Instrument object containing lot size and other info

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting rule-based backtest with strategy: {strategy}")
        logger.info(f"Target: Rs.{self.target_pnl}, Stop Loss: Rs.{self.stop_loss_pnl}")

        # Load data
        df = pd.read_csv(csv_path)

        # Handle datetime column
        if 'datetime_readable' in df.columns:
            df['datetime_readable'] = pd.to_datetime(df['datetime_readable'])
            df.set_index('datetime_readable', inplace=True)

        # Initialize tracking variables
        capital = self.initial_capital
        position = None
        entry_price = None
        entry_time = None
        entry_confidence = 0.0
        total_trades = 0
        winning_trades = 0
        losing_trades = 0

        # Iterate through each data point (skip last 50 bars for trade simulation)
        for i in range(len(df) - 50):
            current_time = df.index[i] if hasattr(df.index, 'to_pydatetime') else df['datetime_readable'].iloc[i]
            current_price = df['close'].iloc[i]

            # Get strategy signal
            signal, confidence = self.get_signal(df, strategy, i)

            # If not in position, check for entry signal
            if position is None and signal != 'HOLD' and confidence >= min_confidence:
                position = signal
                entry_price = current_price
                entry_time = current_time
                entry_confidence = confidence

                logger.debug(f"{current_time}: {signal} at {entry_price} (confidence: {confidence:.2f})")

            # If in position, check for exit (immediate exit for backtesting)
            elif position:
                # Use lot size from instrument
                if instrument and hasattr(instrument, 'lot_size'):
                    lot_size = instrument.lot_size
                else:
                    lot_size = 50  # Default fallback

                # Get future prices for trade simulation
                future_highs = df['high'].iloc[i+1:i+51]  # Next 50 bars
                future_lows = df['low'].iloc[i+1:i+51]

                # Simulate trade
                is_win, pnl_points, bars_held, exit_details = self.simulate_trade(
                    entry_price, position, future_highs, future_lows, entry_time, lot_size
                )

                # Calculate P&L
                pnl_currency = pnl_points * lot_size
                total_brokerage = self.brokerage_entry + self.brokerage_exit

                if is_win:
                    net_pnl = pnl_currency - total_brokerage
                else:
                    net_pnl = pnl_currency - total_brokerage

                capital += net_pnl

                exit_time = df.index[i + bars_held] if i + bars_held < len(df) else df.index[-1]
                trade = {
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'position': position,
                    'entry_price': round(entry_price, 2),
                    'target_price': exit_details['target_price'],
                    'stop_loss_price': exit_details['stop_loss_price'],
                    'lot_size': lot_size,
                    'pnl_points': round(pnl_points, 2),
                    'pnl_currency': round(net_pnl, 2),
                    'bars_held': bars_held,
                    'exit_reason': exit_details['exit_reason'],
                    'strategy': strategy,
                    'confidence': round(entry_confidence * 100, 1),
                    'capital': capital
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

                # Save trade to CSV
                self._save_trade_to_csv(trade)

                position = None
                entry_price = None
                entry_time = None
                entry_confidence = 0.0

        # Calculate final metrics
        results = self._calculate_metrics(self.initial_capital, total_trades, winning_trades, losing_trades)

        logger.info(f"Backtest completed - Total trades: {total_trades}, Win rate: {results['win_rate']:.2%}")
        logger.info(f"Total P&L: Rs.{results['total_pnl']:,.2f} ({results['total_pnl_percent']:.1f}%)")
        logger.info(f"Max Drawdown: Rs.{results['max_drawdown']:,.2f}")

        return results

    def optimize_parameters(self, csv_path: str, strategy: str = 'multi_indicator_combo',
                          instrument=None) -> Dict:
        """
        Optimize target/stop loss parameters for a strategy

        Args:
            csv_path: Path to processed data CSV with features
            strategy: Strategy name to optimize
            instrument: Instrument object

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing parameters for strategy: {strategy}")

        # Define parameter ranges to test
        target_range = [200, 300, 400, 500, 600, 800, 1000, 1200, 1500]
        stop_loss_range = [-100, -150, -200, -250, -300, -400, -500, -600]

        best_result = None
        best_sharpe = -float('inf')
        best_params = {}

        for target in target_range:
            for stop_loss in stop_loss_range:
                # Skip invalid combinations
                if abs(stop_loss) >= target:
                    continue

                # Reset backtester with new parameters
                self.trades = []
                self.equity_curve = []
                self.target_pnl = target
                self.stop_loss_pnl = stop_loss

                # Run backtest
                result = self.run_backtest(csv_path, strategy, min_confidence=0.5, instrument=instrument)

                # Check if this is better based on Sharpe ratio
                if result['sharpe_ratio'] > best_sharpe and result['total_trades'] > 10:
                    best_sharpe = result['sharpe_ratio']
                    best_result = result
                    best_params = {
                        'target_pnl': target,
                        'stop_loss_pnl': stop_loss,
                        'sharpe_ratio': result['sharpe_ratio'],
                        'total_trades': result['total_trades'],
                        'win_rate': result['win_rate'],
                        'total_pnl_percent': result['total_pnl_percent'],
                        'max_drawdown': result['max_drawdown'],
                        'profit_factor': result['profit_factor']
                    }

        logger.info(f"Best parameters: Target={best_params['target_pnl']}, SL={best_params['stop_loss_pnl']}")
        logger.info(f"Best Sharpe: {best_sharpe:.2f}")

        return best_params

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
                'avg_trade_pnl': 0,
                'profit_factor': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Total P&L
        total_pnl = trades_df['pnl_currency'].sum()
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Average trade P&L
        avg_trade_pnl = trades_df['pnl_currency'].mean()

        # Maximum drawdown
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = equity_df['capital'] - equity_df['peak']
            max_drawdown = equity_df['drawdown'].min()

            # Sharpe ratio
            equity_df['returns'] = equity_df['capital'].pct_change()
            sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()
                          * np.sqrt(252)) if equity_df['returns'].std() != 0 else 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0

        # Profit factor
        profits = trades_df[trades_df['pnl_currency'] > 0]['pnl_currency'].sum()
        losses = abs(trades_df[trades_df['pnl_currency'] < 0]['pnl_currency'].sum())
        profit_factor = profits / losses if losses != 0 else float('inf')

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
            'profit_factor': profit_factor
        }

    def _save_trade_to_csv(self, trade: Dict):
        """Save a single trade to CSV file"""
        try:
            trade_df = pd.DataFrame([trade])
            file_exists = os.path.exists(self.trade_log_path) if self.trade_log_path else False

            if self.trade_log_path:
                trade_df.to_csv(self.trade_log_path, mode='a', header=not file_exists, index=False)
        except Exception as e:
            logger.error(f"Failed to save trade to CSV: {e}")