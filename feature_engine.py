import os
import json
import pandas as pd
import numpy as np
from data_processing.processor import process_df
from config import FEATURES_DIR, RR_RATIO, WINDOW_SIZE

def load_feature_cols(instrument: str, timeframe: int) -> list:
    """Load feature column names for a given instrument and timeframe."""
    fname = f"{instrument.replace(' ','_')}_{timeframe}.features.json"
    path = os.path.join(FEATURES_DIR, fname)
    with open(path, 'r') as f:
        return json.load(f)


def build_live_features(df: pd.DataFrame, instrument: str, timeframe: int) -> np.ndarray:
    """Process DataFrame and return 3D obs matching env: (1, WINDOW_SIZE, n_features)."""
    processed = process_df(df, rr_ratio=RR_RATIO)
    # Use base features matching TradingEnv.feature_cols
    cols = ['open', 'high', 'low', 'close', 'ATR']
    # Ensure we have enough rows
    window = processed[cols].iloc[-WINDOW_SIZE:]
    # Shape (WINDOW_SIZE, 5) then add batch dim
    return window.values[np.newaxis, :, :]
