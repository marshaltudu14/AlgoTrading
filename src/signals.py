import pandas as pd
import numpy as np

def get_inside_candle_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generates trading signals based on the Inside Candle breakout strategy.

    Args:
        df: DataFrame with columns 'High', 'Low', 'Close', 'Open'.
            Assumes the index is a DatetimeIndex.

    Returns:
        A Pandas Series with signals: 1 for buy, -1 for sell, 0 for hold.
    """
    signals = pd.Series(0, index=df.index)

    # Need at least 3 bars for the pattern
    if len(df) < 3:
        return signals

    # Vectorized calculation for inside bar condition
    # Checks if the previous bar (-2) is inside the bar before that (-3)
    is_inside_bar_prev = (df['High'].shift(1) < df['High'].shift(2)) & \
                         (df['Low'].shift(1) > df['Low'].shift(2))

    # Vectorized calculation for breakout condition
    # Checks if the current bar (-1) breaks out of the mother candle (-3)
    # and the previous bar (-2) was an inside bar
    long_breakout = (df['High'] > df['High'].shift(2)) & is_inside_bar_prev
    short_breakout = (df['Low'] < df['Low'].shift(2)) & is_inside_bar_prev

    # Assign signals (1 for long, -1 for short) based on breakout of the *current* bar
    signals[long_breakout] = 1
    signals[short_breakout] = -1

    # Ensure signals are integers
    signals = signals.astype(int)

    # Prevent signals in the first few rows where pattern isn't possible
    signals.iloc[:2] = 0 # No signal possible on first two bars

    return signals

# --- Add other signal generation functions here in the future ---


# =============================================================================
# Real-time Signal Generation
# =============================================================================

import logging
from collections import deque
from datetime import datetime, timedelta
import pandas as pd # Keep pandas for potential candle aggregation/indicator calculation

logger = logging.getLogger(__name__)

class InsideCandleRealtimeSignalGenerator:
    """
    Generates Inside Candle breakout signals based on real-time data updates.
    Manages candle aggregation and maintains necessary historical data.
    """
    def __init__(self, timeframe_minutes: int, atr_period: int = 14, history_bars: int = 20):
        """
        Initializes the real-time signal generator.

        Args:
            timeframe_minutes (int): The candle timeframe in minutes (e.g., 5 for 5-min).
            atr_period (int): The period for ATR calculation.
            history_bars (int): Number of recent bars to keep for calculations (should be >= atr_period + pattern lookback).
        """
        if history_bars < atr_period + 3: # Need ATR period + 3 bars for inside candle pattern
            raise ValueError("history_bars must be at least atr_period + 3")

        self.timeframe = timedelta(minutes=timeframe_minutes)
        self.atr_period = atr_period
        self.history_bars = history_bars
        self.current_candle = None # Stores data for the candle being built
        self.history = deque(maxlen=self.history_bars) # Stores completed candles (as dicts or simple objects)
        self.last_tick_time = None
        logger.info(f"Initialized InsideCandleRealtimeSignalGenerator for {timeframe_minutes}-min timeframe.")

    def _initialize_candle(self, tick_data):
        """Starts a new candle based on the first tick."""
        # Prioritize 'last_traded_time', fallback to 'exch_feed_time'
        timestamp = tick_data.get('last_traded_time', tick_data.get('exch_feed_time'))
        price = tick_data.get('ltp')

        if timestamp is None or price is None:
            logger.warning(f"Tick data missing usable timestamp (last_traded_time/exch_feed_time) or ltp: {tick_data}")
            return

        dt_object = datetime.fromtimestamp(timestamp)
        candle_start_time = dt_object - (dt_object - datetime.min) % self.timeframe
        self.current_candle = {
            'timestamp': candle_start_time, # Start time of the candle
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': tick_data.get('volume', 0) # If volume is available
        }
        logger.debug(f"Initialized new candle at {candle_start_time} with price {price}")


    def _update_candle(self, tick_data):
        """Updates the current candle with new tick data."""
        price = tick_data.get('ltp')
        volume = tick_data.get('volume', 0)
        if price is None:
             logger.warning(f"Tick data missing ltp: {tick_data}")
             return

        if not self.current_candle:
            self._initialize_candle(tick_data)
            return # Initialized, wait for next tick

        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += volume # Accumulate volume if available


    def _finalize_candle(self):
        """Finalizes the current candle and adds it to history."""
        if self.current_candle:
            logger.debug(f"Finalizing candle: {self.current_candle}")
            # Convert to a simple structure or DataFrame row if needed for calculations
            # For now, keeping as dict
            self.history.append(self.current_candle.copy())
            self.current_candle = None # Reset for the next candle
            self._calculate_indicators() # Calculate indicators after adding new bar


    def _calculate_indicators(self):
        """Calculates necessary indicators (e.g., ATR) on the historical data."""
        if len(self.history) < self.atr_period + 1: # Need enough data for ATR
            return

        # Convert history deque to DataFrame for easier calculation with pandas-ta
        # This might be inefficient if called very frequently, consider alternatives if needed
        hist_df = pd.DataFrame(list(self.history))
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        hist_df = hist_df.set_index('timestamp')

        # Ensure columns are named correctly for pandas_ta (lowercase)
        hist_df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        hist_df.rename(columns=str.lower, inplace=True) # Convert all to lowercase

        try:
            # Use pandas_ta to calculate ATR
            import pandas_ta as ta
            hist_df.ta.atr(length=self.atr_period, append=True) # Appends 'ATRr_14' column

            # Store the calculated ATR back into our history deque (on the last element)
            # Note: pandas_ta might use uppercase ATR, adjust column name if needed
            atr_col_name = f'atr_{self.atr_period}' # Assuming lowercase from rename
            if atr_col_name in hist_df.columns and not pd.isna(hist_df[atr_col_name].iloc[-1]):
                self.history[-1][atr_col_name] = hist_df[atr_col_name].iloc[-1]
                # logger.debug(f"Calculated ATR for last candle: {self.history[-1][atr_col_name]}")
            else:
                 # logger.debug(f"ATR calculation skipped or resulted in NaN for last candle.")
                 # Ensure the key exists even if NaN
                 self.history[-1][atr_col_name] = np.nan

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}", exc_info=True)
            # Ensure the key exists even if calculation failed
            self.history[-1][f'atr_{self.atr_period}'] = np.nan


    def update(self, tick_data) -> int:
        """
        Processes a new tick data point, updates candles, and generates signals.

        Args:
            tick_data (dict): Dictionary containing tick information (e.g., {'timestamp': epoch, 'ltp': price, 'symbol': '...'})

        Returns:
            int: Signal (1 for buy, -1 for sell, 0 for hold).
        """
        # Prioritize 'last_traded_time', fallback to 'exch_feed_time'
        timestamp = tick_data.get('last_traded_time', tick_data.get('exch_feed_time'))

        if timestamp is None:
            logger.warning(f"Received tick without usable timestamp (last_traded_time/exch_feed_time): {tick_data}")
            return 0 # No signal

        current_dt = datetime.fromtimestamp(timestamp)

        if self.current_candle is None:
            # First tick received or starting fresh
            self._initialize_candle(tick_data)
            self.last_tick_time = current_dt
            return 0 # No signal yet

        # Check if the tick belongs to a new candle
        candle_end_time = self.current_candle['timestamp'] + self.timeframe
        if current_dt >= candle_end_time:
            # Finalize the previous candle
            self._finalize_candle()
            # Initialize the new candle with the current tick
            self._initialize_candle(tick_data)
        else:
            # Update the existing candle
            self._update_candle(tick_data)

        self.last_tick_time = current_dt

        # --- Signal Generation Logic ---
        # Generate signal based on the *last completed* candle in history
        if len(self.history) < 3: # Need 3 bars for the pattern
            return 0

        # Access the last 3 completed candles from the deque
        # history[-1] is the most recently completed candle
        # history[-2] is the one before that
        # history[-3] is the "mother" candle candidate
        try:
            current_bar = self.history[-1]
            prev_bar = self.history[-2]
            mother_bar = self.history[-3]

            # Check for inside bar condition (prev_bar inside mother_bar)
            is_inside_bar = (prev_bar['high'] < mother_bar['high']) and \
                            (prev_bar['low'] > mother_bar['low'])

            if is_inside_bar:
                # Check for breakout condition (current_bar breaks mother_bar)
                if current_bar['high'] > mother_bar['high']:
                    logger.info(f"LONG signal generated at {current_bar['timestamp']} (breakout of {mother_bar['timestamp']})")
                    return 1 # Buy signal
                elif current_bar['low'] < mother_bar['low']:
                    logger.info(f"SHORT signal generated at {current_bar['timestamp']} (breakdown of {mother_bar['timestamp']})")
                    return -1 # Sell signal

        except IndexError:
            # Should not happen if len check is correct, but safety first
            logger.warning("IndexError during signal generation, not enough history.")
            return 0
        except KeyError as e:
             logger.warning(f"KeyError during signal generation (missing data?): {e}")
             return 0

        return 0 # No signal

    def get_last_atr(self, symbol: str) -> float | None:
        """
        Returns the last calculated ATR value for the given symbol.
        Note: This implementation assumes ATR is calculated for the underlying symbol
              passed during the update method. It doesn't handle per-symbol ATR if
              multiple underlyings were tracked by the same instance (which isn't
              the current design).

        Args:
            symbol (str): The symbol (currently ignored, returns the latest ATR).

        Returns:
            float | None: The last calculated ATR value, or None if not available.
        """
        if self.history:
            last_candle = self.history[-1]
            atr_col_name = f'atr_{self.atr_period}'
            return last_candle.get(atr_col_name) # Returns None if key doesn't exist
        return None
