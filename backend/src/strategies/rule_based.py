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


def format_currency(amount: float) -> str:
    """Format currency amount with Indian number system"""
    if amount >= 10000000:  # 1 Crore or more
        return f"Rs.{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh or more
        return f"Rs.{amount/100000:.2f} L"
    else:
        return f"Rs.{amount:,.0f}"


def round_to_nearest_5k(amount: float) -> float:
    """Round amount to nearest 5000"""
    return round(amount / 5000) * 5000


class RuleBasedBacktester:
    """Backtester for rule-based strategies using pre-calculated features"""

    def __init__(self, target_pnl: float = 500, stop_loss_pnl: float = -300,
                 initial_capital: float = 25000, brokerage_entry: float = 25,
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
        self.peak_capital = initial_capital  # Track highest capital achieved

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
        """Enhanced Multi-Indicator Combination Strategy with Risk Filters"""

        # Get current time to check for 3 PM cutoff
        current_time = df.index[i] if hasattr(df.index, 'to_pydatetime') else pd.to_datetime(df['datetime_readable'].iloc[i])

        # Skip trades after 3:15 PM (to allow execution before 3:30 close)
        if current_time.time() >= pd.Timestamp('15:15:00').time():
            return 'HOLD', 0.0

        # Get signals from multiple strategies
        rsi_signal, rsi_conf = self.rsi_mean_reversion(df, i)
        macd_signal, macd_conf = self.macd_strategy(df, i)
        trend_signal, trend_conf = self.trend_following(df, i)

        # Additional indicator analysis
        atr_pct = df['atr_pct'].iloc[i] if 'atr_pct' in df.columns else 0.09
        price_change_pct = df['price_change_pct'].iloc[i] if 'price_change_pct' in df.columns else 0
        bb_position = df['bb_position'].iloc[i] if 'bb_position' in df.columns else 0.5
        bb_width = df['bb_width_pct'].iloc[i] if 'bb_width_pct' in df.columns else 0.19
        trend_slope = df['trend_slope'].iloc[i] if 'trend_slope' in df.columns else 0

        # Get RSI values
        rsi_21 = df['rsi_21'].iloc[i] if 'rsi_21' in df.columns else 50
        rsi_14 = df['rsi_14'].iloc[i] if 'rsi_14' in df.columns else 50

        # Candlestick analysis for momentum and reversal detection
        body_size = df['body_size_pct'].iloc[i] if 'body_size_pct' in df.columns else 0.047
        lower_shadow = df['lower_shadow_pct'].iloc[i] if 'lower_shadow_pct' in df.columns else 0.022
        upper_shadow = df['upper_shadow_pct'].iloc[i] if 'upper_shadow_pct' in df.columns else 0.021
        hl_range = df['hl_range_pct'].iloc[i] if 'hl_range_pct' in df.columns else 0.09

        # Multi-timeframe alignment check
        mtf_buy = ((df['dist_sma_5'].iloc[i] > df['dist_sma_10'].iloc[i]) &
                   (df['dist_sma_10'].iloc[i] > df['dist_sma_20'].iloc[i]) &
                   (df['dist_sma_20'].iloc[i] > df['dist_sma_50'].iloc[i]))
        mtf_sell = ((df['dist_sma_5'].iloc[i] < df['dist_sma_10'].iloc[i]) &
                    (df['dist_sma_10'].iloc[i] < df['dist_sma_20'].iloc[i]) &
                    (df['dist_sma_20'].iloc[i] < df['dist_sma_50'].iloc[i]))

        # Risk filters based on actual data analysis
        # 1. Avoid huge candles (momentum already passed)
        if hl_range > 0.5:  # > 0.5% range is large (75th percentile is 0.11%)
            return 'HOLD', 0.0

        # 2. Avoid large body candles (already strong momentum)
        if body_size > 0.3:  # > 0.3% body is large
            return 'HOLD', 0.0

        # 3. Check for reversal signals using wicks
        long_lower_wick = lower_shadow > 0.1 and lower_shadow > (body_size * 2)
        long_upper_wick = upper_shadow > 0.1 and upper_shadow > (body_size * 2)

        # Dynamic confidence multipliers
        confidence_boost = 1.0

        # Multi-timeframe alignment bonus
        if mtf_buy and trend_signal == 'BUY':
            confidence_boost *= 1.3
        elif mtf_sell and trend_signal == 'SELL':
            confidence_boost *= 1.3

        # Bollinger Band analysis
        if bb_position < 0.15 and bb_width > 0.15:  # Near lower band with decent volatility
            if rsi_14 < 40:
                confidence_boost *= 1.25
        elif bb_position > 0.85 and bb_width > 0.15:  # Near upper band
            if rsi_14 > 60:
                confidence_boost *= 1.25

        # Reversal signals from wicks
        if long_lower_wick and rsi_14 < 45:
            confidence_boost *= 1.2  # Bullish reversal potential
        elif long_upper_wick and rsi_14 > 55:
            confidence_boost *= 1.2  # Bearish reversal potential

        # Squeeze play detection
        if bb_width < 0.1:  # Low volatility squeeze
            if abs(trend_slope) > 0.5:
                confidence_boost *= 1.15

        # Divergence detection
        if i > 10:
            recent_rsi = df['rsi_14'].iloc[i-5:i+1]
            recent_price = df['close'].iloc[i-5:i+1]

            if len(recent_rsi) == 6 and len(recent_price) == 6:
                # Bullish divergence
                if (recent_price.iloc[-1] < recent_price.iloc[0] and
                    recent_rsi.iloc[-1] > recent_rsi.iloc[0] and
                    rsi_14 < 40):
                    confidence_boost *= 1.2
                # Bearish divergence
                elif (recent_price.iloc[-1] > recent_price.iloc[0] and
                      recent_rsi.iloc[-1] < recent_rsi.iloc[0] and
                      rsi_14 > 60):
                    confidence_boost *= 1.2

        # Dynamic weighting based on volatility
        if atr_pct > 0.15:  # High volatility (mean is 0.09)
            trend_conf *= 1.3
            rsi_conf *= 0.7
        elif atr_pct < 0.07:  # Low volatility
            rsi_conf *= 1.3
            trend_conf *= 0.7

        # Avoid trading during extreme price movements
        if abs(price_change_pct) > 0.3:  # > 0.3% move is significant
            return 'HOLD', 0.0

        # Enhanced RSI logic for overbought/oversold
        if rsi_14 > 75 and trend_signal == 'SELL':
            rsi_conf *= 1.3  # Strengthen sell in overbought
        elif rsi_14 < 25 and trend_signal == 'BUY':
            rsi_conf *= 1.3  # Strengthen buy in oversold

        # Weight voting system
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

        # Apply confidence boost
        total_confidence *= confidence_boost

        # Final decision
        if total_confidence > 0:
            buy_ratio = buy_votes / total_confidence
            sell_ratio = sell_votes / total_confidence

            # Higher threshold for safety
            threshold = 0.7 if confidence_boost < 1.2 else 0.6

            if buy_ratio > threshold and total_confidence > 0.5:
                final_confidence = min(0.9, buy_ratio * confidence_boost)
                return 'BUY', final_confidence
            elif sell_ratio > threshold and total_confidence > 0.5:
                final_confidence = min(0.9, sell_ratio * confidence_boost)
                return 'SELL', final_confidence

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
                      high_prices: pd.Series, low_prices: pd.Series, close_prices: pd.Series,
                      entry_time: pd.Timestamp, lot_size: int, target_pnl: float = None, stop_loss_pnl: float = None) -> Tuple[bool, float, int, Dict]:
        """Simulate trade execution with trailing stop loss"""
        bars_held = 0
        exit_details = {
            'exit_price': entry_price,
            'target_price': None,
            'stop_loss_price': None,
            'exit_reason': None
        }

        # Use provided P&L targets or fall back to defaults
        current_target_pnl = target_pnl if target_pnl is not None else self.target_pnl
        current_stop_loss_pnl = stop_loss_pnl if stop_loss_pnl is not None else self.stop_loss_pnl

        # Calculate points needed based on P&L targets
        # P&L targets are per lot, so we divide by lot_size to get price movement
        if lot_size > 0:
            target_points = abs(current_target_pnl) / lot_size
            stop_loss_points = abs(current_stop_loss_pnl) / lot_size
        else:
            # Fallback to default calculations
            target_points = abs(self.target_pnl) / 1  # Per lot calculation
            stop_loss_points = abs(self.stop_loss_pnl) / 1

        # Initialize variables for trailing
        hit_initial_target = False
        targets_hit = 0

        if direction == 'BUY':
            # Initial target and stop loss
            initial_target = entry_price + target_points
            current_target = initial_target
            current_sl = entry_price - stop_loss_points

            exit_details.update({
                'target_price': round(initial_target, 2),
                'stop_loss_price': round(current_sl, 2)
            })

            for i, (high, low, close) in enumerate(zip(high_prices, low_prices, close_prices)):
                bars_held += 1

                
                if not hit_initial_target:
                    # Before hitting initial target
                    if close >= current_target:
                        # Hit initial target - start trailing
                        hit_initial_target = True
                        targets_hit = 1
                        # Move SL to target price (protect profit at this level)
                        previous_target = current_target
                        current_sl = previous_target
                        # Set next target (add same target_points from the target we just hit)
                        current_target = previous_target + target_points
                    elif low <= current_sl and direction == 'BUY':
                        # SL breached during the candle - exit immediately at SL
                        exit_details.update({
                            'exit_price': round(current_sl, 2),  # Exit at SL level immediately
                            'exit_reason': 'STOP_LOSS'
                        })
                        pnl_points = -(entry_price - current_sl)
                        return False, pnl_points, bars_held, exit_details
                else:
                    # After hitting initial target - trailing mode
                    # IMPORTANT: Check SL breach FIRST to ensure we exit if SL is hit
                    if close <= current_sl:
                        # For trailing SL, wait for close (as per user requirement)
                        # HYBRID APPROACH: Weighted average of SL level and close price
                        # This gives us some advantage while being more realistic
                        sl_level = current_sl
                        # Weight 70% towards SL level (instant exit advantage), 30% towards close (realism)
                        exit_price = (sl_level * 0.7 + close * 0.3)
                        exit_details.update({
                            'exit_price': round(exit_price, 2),
                            'exit_reason': f'TRAILING_STOP_{targets_hit}'
                        })
                        # For BUY, profit = exit_price - entry_price
                        pnl = exit_price - entry_price
                        return pnl > 0, pnl, bars_held, exit_details
                    elif close >= current_target:
                        # Hit another target - trail further
                        targets_hit += 1
                        # The price we just hit becomes the defended level
                        previous_target = current_target
                        # Move SL to defend the profit at this level
                        current_sl = previous_target
                        # Set new target (go up by target_points from the target we just hit)
                        current_target = previous_target + target_points

        else:  # SELL
            # Initial target and stop loss
            initial_target = entry_price - target_points
            current_target = initial_target
            current_sl = entry_price + stop_loss_points

            exit_details.update({
                'target_price': round(initial_target, 2),
                'stop_loss_price': round(current_sl, 2)
            })

            for high, low, close in zip(high_prices, low_prices, close_prices):
                bars_held += 1

                if not hit_initial_target:
                    # Before hitting initial target
                    if close <= current_target:
                        # Hit initial target - start trailing
                        hit_initial_target = True
                        targets_hit = 1
                        # Move SL to target price (protect profit at this level)
                        previous_target = current_target
                        current_sl = previous_target
                        # Set next target (subtract same target_points again from the target we just hit)
                        current_target = previous_target - target_points
                    elif high >= current_sl:
                        # SL breached during the candle - exit immediately at SL
                        exit_details.update({
                            'exit_price': round(current_sl, 2),  # Exit at SL level immediately
                            'exit_reason': 'STOP_LOSS'
                        })
                        pnl_points = -(current_sl - entry_price)  # For SELL, loss = SL - entry
                        return False, pnl_points, bars_held, exit_details
                else:
                    # After hitting initial target - trailing mode
                    # IMPORTANT: Check SL breach FIRST to ensure we exit if SL is hit
                    if close >= current_sl:
                        # For trailing SL, wait for close (as per user requirement)
                        # HYBRID APPROACH: Weighted average of SL level and close price
                        # This gives us some advantage while being more realistic
                        sl_level = current_sl
                        # Weight 70% towards SL level (instant exit advantage), 30% towards close (realism)
                        exit_price = (sl_level * 0.7 + close * 0.3)
                        exit_details.update({
                            'exit_price': round(exit_price, 2),
                            'exit_reason': f'TRAILING_STOP_{targets_hit}'
                        })
                        # For SELL, profit = entry_price - exit_price
                        pnl = entry_price - exit_price
                        return pnl > 0, pnl, bars_held, exit_details
                    elif close <= current_target:
                        # Hit another target - trail further
                        targets_hit += 1
                        # The price we just hit becomes the defended level
                        previous_target = current_target
                        # Move SL to defend the profit at this level
                        current_sl = previous_target
                        # Set new target (go down by target_points from the target we just hit)
                        current_target = previous_target - target_points

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
        logger.info(f"Target: {format_currency(self.target_pnl)}, Stop Loss: {format_currency(self.stop_loss_pnl)}")

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

        # Track daily losses (total losses per day, not consecutive)
        daily_losses = {}  # Dictionary to track losses per day
        max_daily_losses = 2

        # Dynamic lot size based on capital
        base_lot_size = instrument.lot_size if instrument and hasattr(instrument, 'lot_size') else 75
        current_lot_multiplier = 1  # Tracks how many times capital has doubled
        initial_capital_for_doubling = self.initial_capital

        # Iterate through each data point (skip last 50 bars for trade simulation)
        for i in range(len(df) - 50):
            current_time = df.index[i] if hasattr(df.index, 'to_pydatetime') else df['datetime_readable'].iloc[i]
            current_price = df['close'].iloc[i]

            # Get strategy signal
            signal, confidence = self.get_signal(df, strategy, i)

            # If not in position, check for entry signal
            if position is None and signal != 'HOLD' and confidence >= min_confidence:
                # Check daily loss limit
                trade_date = current_time.date()

                # Initialize daily loss count if not exists
                if trade_date not in daily_losses:
                    daily_losses[trade_date] = 0

                # Skip trade if already hit max daily losses
                if daily_losses[trade_date] >= max_daily_losses:
                    continue

                position = signal
                entry_price = current_price
                entry_time = current_time
                entry_confidence = confidence

                
            # If in position, check for exit (immediate exit for backtesting)
            elif position:
                # Calculate lot size based on capital levels - dynamic with both increase and decrease
                lot_multiplier = 1
                temp_capital = capital

                # Calculate how many times 50k fits into current capital
                # 50k is our doubling threshold from the initial 25k
                while temp_capital >= 50000:
                    temp_capital /= 2
                    lot_multiplier *= 2

                # This creates dynamic lot sizing:
                # 25k-49.9k: 1x, 50k-99.9k: 2x, 100k-199.9k: 4x, etc.
                # Automatically scales down if capital drops!

                # Update peak capital if current capital is higher
                if capital > self.peak_capital:
                    self.peak_capital = capital

                # Check for account blowup (capital below 10% of peak)
                if capital < self.peak_capital * 0.10:
                    logger.error(f"CRITICAL: Account blowup! Capital at {format_currency(capital)} (< 50% of peak {format_currency(self.peak_capital)})")
                    logger.error("Stopping trading to prevent complete loss")
                    break  # Exit the backtest loop

                # Check if we need to adjust lot size (increase or decrease)
                lot_size_changed = False
                if lot_multiplier != current_lot_multiplier:
                    old_lot_size = base_lot_size * current_lot_multiplier
                    new_lot_size = base_lot_size * lot_multiplier
                    lot_size_changed = True

                    if lot_multiplier > current_lot_multiplier:
                        # Lot size increasing
                        logger.info(f"Capital reached {format_currency(capital)} - Lot size doubled from {old_lot_size} to {new_lot_size}")
                    else:
                        # Lot size decreasing (capital reduction)
                        logger.warning(f"Capital dropped to {format_currency(capital)} - Lot size halved from {old_lot_size} to {new_lot_size}")

                    # Update lot multiplier
                    current_lot_multiplier = lot_multiplier

                lot_size = base_lot_size * current_lot_multiplier

                # Scale target and stop loss P&L with lot size multiplier
                # This maintains the same risk-reward per lot
                scaled_target_pnl = self.target_pnl * current_lot_multiplier
                scaled_stop_loss_pnl = self.stop_loss_pnl * current_lot_multiplier

                # Log current P&L targets when lot size changed
                if lot_size_changed:
                    logger.info(f"Scaled P&L targets: Target {format_currency(scaled_target_pnl)} | Stop Loss {format_currency(scaled_stop_loss_pnl)}")

                # Get future prices for trade simulation
                future_highs = df['high'].iloc[i+1:i+51]  # Next 50 bars
                future_lows = df['low'].iloc[i+1:i+51]
                future_closes = df['close'].iloc[i+1:i+51]  # Next 50 close prices

                # Simulate trade with scaled P&L targets
                # Store lot size at entry for correct P&L calculation
                entry_lot_size = lot_size

                is_win, pnl_points, bars_held, exit_details = self.simulate_trade(
                    entry_price, position, future_highs, future_lows, future_closes, entry_time, entry_lot_size,
                    target_pnl=scaled_target_pnl, stop_loss_pnl=scaled_stop_loss_pnl
                )

                # Calculate P&L
                pnl_currency = pnl_points * lot_size  # Using CURRENT lot_size for capital calculation
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
                    'exit_price': round(exit_details['exit_price'], 2),
                    'target_price': exit_details['target_price'],
                    'stop_loss_price': exit_details['stop_loss_price'],
                    'lot_size': lot_size,
                    'pnl_points': round(pnl_points, 2),
                    'pnl_currency': round(net_pnl, 2),
                    'bars_held': bars_held,
                    'exit_reason': exit_details['exit_reason'],
                    'confidence': round(entry_confidence * 100, 1),
                    'capital': format_currency(capital)
                }
                self.trades.append(trade)

                total_trades += 1
                if is_win:
                    winning_trades += 1
                else:
                    losing_trades += 1
                    # Update daily loss count (using entry date when trade was initiated)
                    trade_date = entry_time.date() if hasattr(entry_time, 'date') else entry_time
                    daily_losses[trade_date] = daily_losses.get(trade_date, 0) + 1
                    
                self.equity_curve.append({
                    'time': exit_time,
                    'capital': capital  # Keep numeric for calculations
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
        logger.info(f"Total P&L: {format_currency(results['total_pnl'])} ({results['total_pnl_percent']:.1f}%)")
        logger.info(f"Max Drawdown: {format_currency(results['max_drawdown'])} ({results['max_drawdown_pct']:.1f}%)")

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

        # Get final capital from equity curve or use current capital
        if self.equity_curve:
            final_capital = self.equity_curve[-1]['capital']
        else:
            # If no equity curve, sum initial capital and all trade P&Ls
            final_capital = self.initial_capital + trades_df['pnl_currency'].sum()

        # Total P&L (Final Capital - Initial Capital)
        total_pnl = final_capital - self.initial_capital
        total_pnl_percent = (total_pnl / self.initial_capital) * 100

        # Log capital details
        
        # Win rate
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Calculate daily statistics
        trades_df['date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily_pnl = trades_df.groupby('date')['pnl_currency'].sum()
        highest_daily_profit = daily_pnl.max()
        highest_daily_loss = daily_pnl.min()

        # Calculate maximum and minimum trades per day
        daily_trade_count = trades_df.groupby('date').size()
        max_trades_per_day = daily_trade_count.max()
        min_trades_per_day = daily_trade_count.min()

        # Maximum drawdown
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df['peak'] = equity_df['capital'].cummax()
            equity_df['drawdown'] = equity_df['capital'] - equity_df['peak']
            equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
            max_drawdown = equity_df['drawdown'].min()
            max_drawdown_pct = equity_df['drawdown_pct'].min()

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

        # Calculate winning and losing streaks
        max_winning_streak = 0
        max_losing_streak = 0
        current_winning_streak = 0
        current_losing_streak = 0

        for _, trade in trades_df.iterrows():
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
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'highest_daily_profit': highest_daily_profit,
            'highest_daily_loss': highest_daily_loss,
            'max_trades_per_day': max_trades_per_day,
            'min_trades_per_day': min_trades_per_day,
            'max_winning_streak': max_winning_streak,
            'max_losing_streak': max_losing_streak
        }

    def _save_trade_to_csv(self, trade: Dict):
        """Save a single trade to CSV file"""
        try:
            trade_df = pd.DataFrame([trade])

            if self.trade_log_path:
                # Check if this is the first trade (no file exists)
                file_exists = os.path.exists(self.trade_log_path)

                if not file_exists:
                    # Create new file with header
                    trade_df.to_csv(self.trade_log_path, mode='w', index=False)
                else:
                    # Append to existing file
                    trade_df.to_csv(self.trade_log_path, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Failed to save trade to CSV: {e}")