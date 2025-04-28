import os
import json
import pandas as pd
import numpy as np
from data_processing.processor import process_df
from config import RR_RATIO, WINDOW_SIZE

def load_feature_cols(instrument: str, timeframe: int) -> list:
    """Load feature column names for a given instrument and timeframe."""
    fname = f"{instrument.replace(' ','_')}_{timeframe}.features.json"
    path = os.path.join(FEATURES_DIR, fname)
    with open(path, 'r') as f:
        return json.load(f)


def build_live_features(
    df: pd.DataFrame,
    instrument: str,
    timeframe: int,
    position: int = 0,
    position_duration: int = 0,
    entry_price: float = 0.0,
    unrealized_pnl: float = 0.0
) -> np.ndarray:
    """
    Process DataFrame and return observation with shape (WINDOW_SIZE, n_features).
    Includes position features for enhanced model.

    Args:
        df: Input DataFrame with OHLC data
        instrument: Instrument name
        timeframe: Timeframe in minutes
        position: Current position (0=none, 1=long)
        position_duration: Duration of current position in bars
        entry_price: Entry price of current position
        unrealized_pnl: Unrealized PnL of current position

    Returns:
        Observation array with shape (WINDOW_SIZE, n_features)
    """
    # Process the dataframe
    processed = process_df(df, rr_ratio=RR_RATIO)

    # Get market data features
    cols = ['open', 'high', 'low', 'close', 'ATR']
    window = processed[cols].iloc[-WINDOW_SIZE:].values

    # Create position features
    batch_size = window.shape[0]
    position_features = np.zeros((batch_size, 4), dtype=np.float32)

    # Fill the last row with current position features
    position_features[-1, 0] = position
    position_features[-1, 1] = position_duration
    position_features[-1, 2] = entry_price if position > 0 else 0.0
    position_features[-1, 3] = unrealized_pnl

    # Concatenate market data and position features
    return np.concatenate([window, position_features], axis=1)
