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
