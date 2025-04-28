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


def build_live_features(df: pd.DataFrame, instrument: str, timeframe: int) -> np.ndarray:
    """Process DataFrame and return flat feature vector: (1, WINDOW_SIZE * n_features)."""
    processed = process_df(df, rr_ratio=RR_RATIO)
    cols = ['open', 'high', 'low', 'close', 'ATR']
    window = processed[cols].iloc[-WINDOW_SIZE:]
    # Flatten window to 2D array
    return window.values.reshape(1, -1)
