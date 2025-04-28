"""
Feature engineering module for AlgoTrading.
Handles feature extraction and processing for model input.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from core.logging_setup import get_logger
from core.config import RR_RATIO, WINDOW_SIZE, FEATURES_DIR
from data.processor import process_df
from data.enhanced_processor import process_df as process_df_enhanced

logger = get_logger(__name__)


def load_feature_cols(
    instrument: str, 
    timeframe: int, 
    features_dir: str = FEATURES_DIR,
    enhanced: bool = False
) -> List[str]:
    """
    Load feature column names for a given instrument and timeframe.
    
    Args:
        instrument: Instrument name
        timeframe: Timeframe in minutes
        features_dir: Directory containing feature files
        enhanced: Whether to use enhanced features
        
    Returns:
        List of feature column names
    """
    prefix = "enhanced_" if enhanced else ""
    fname = f"{prefix}{instrument.replace(' ','_')}_{timeframe}.features.json"
    path = os.path.join(features_dir, fname)
    
    try:
        with open(path, 'r') as f:
            feature_cols = json.load(f)
            logger.info(f"Loaded {len(feature_cols)} features for {instrument} @ {timeframe}min")
            return feature_cols
    except FileNotFoundError:
        logger.warning(f"Feature file not found: {path}")
        # Try without enhanced prefix if enhanced=True
        if enhanced:
            logger.info("Trying to load non-enhanced features")
            return load_feature_cols(instrument, timeframe, features_dir, enhanced=False)
        # Return default features if file not found
        logger.warning("Using default features")
        return ['open', 'high', 'low', 'close', 'volume', 'ATR']
    except Exception as e:
        logger.error(f"Error loading feature columns: {e}")
        return ['open', 'high', 'low', 'close', 'volume', 'ATR']


def build_live_features(
    df: pd.DataFrame,
    instrument: str,
    timeframe: int,
    position: int = 0,
    position_duration: int = 0,
    entry_price: float = 0.0,
    unrealized_pnl: float = 0.0,
    use_enhanced_features: bool = True,
    window_size: int = WINDOW_SIZE
) -> np.ndarray:
    """
    Process DataFrame and return observation with shape (window_size, n_features).
    Includes position features for enhanced model.

    Args:
        df: Input DataFrame with OHLC data
        instrument: Instrument name
        timeframe: Timeframe in minutes
        position: Current position (0=none, 1=long)
        position_duration: Duration of current position in bars
        entry_price: Entry price of current position
        unrealized_pnl: Unrealized PnL of current position
        use_enhanced_features: Whether to use enhanced features
        window_size: Number of bars to include in the window

    Returns:
        Observation array with shape (window_size, n_features)
    """
    logger.info(f"Building features for {instrument} @ {timeframe}min")
    
    # Process the dataframe
    if use_enhanced_features:
        logger.info("Using enhanced feature processing")
        processed = process_df_enhanced(df, rr_ratio=RR_RATIO, normalize=True)
    else:
        logger.info("Using basic feature processing")
        processed = process_df(df, rr_ratio=RR_RATIO)
    
    # Ensure we have enough data
    if len(processed) < window_size:
        logger.warning(f"Not enough data: {len(processed)} rows, need {window_size}")
        # Pad with zeros if needed
        padding = window_size - len(processed)
        logger.info(f"Padding with {padding} rows of zeros")
        
        # Get all numeric columns
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        
        # Create padding dataframe
        pad_df = pd.DataFrame(0, index=range(padding), columns=numeric_cols)
        
        # Concatenate
        processed = pd.concat([pad_df, processed], ignore_index=True)
    
    # Get market data features
    base_cols = ['open', 'high', 'low', 'close', 'ATR']
    
    # Add additional features if available
    additional_cols = []
    for col in ['RSI_14', 'MACDh_12_26_9', 'BB_POSITION', 'NATR', 'VOL_REGIME']:
        if col in processed.columns:
            additional_cols.append(col)
    
    # Combine all feature columns
    feature_cols = base_cols + additional_cols
    
    # Get window of features
    window = processed[feature_cols].iloc[-window_size:].values
    
    # Create position features
    batch_size = window.shape[0]
    position_features = np.zeros((batch_size, 4), dtype=np.float32)
    
    # Fill the last row with current position features
    position_features[-1, 0] = position
    position_features[-1, 1] = position_duration
    position_features[-1, 2] = entry_price if position > 0 else 0.0
    position_features[-1, 3] = unrealized_pnl
    
    # Concatenate market data and position features
    features = np.concatenate([window, position_features], axis=1)
    
    logger.info(f"Built features with shape {features.shape}")
    return features


def verify_features(
    features: np.ndarray,
    min_window_size: int = 10,
    min_features: int = 5
) -> bool:
    """
    Verify that the features array is valid for model input.
    
    Args:
        features: Features array
        min_window_size: Minimum window size
        min_features: Minimum number of features
        
    Returns:
        True if features are valid, False otherwise
    """
    try:
        # Check if features is None
        if features is None:
            logger.warning("Features array is None")
            return False
        
        # Check if features is empty
        if features.size == 0:
            logger.warning("Features array is empty")
            return False
        
        # Check dimensions
        if len(features.shape) != 2:
            logger.warning(f"Features array has wrong dimensions: {features.shape}")
            return False
        
        # Check window size
        if features.shape[0] < min_window_size:
            logger.warning(f"Window size too small: {features.shape[0]}, minimum is {min_window_size}")
            return False
        
        # Check number of features
        if features.shape[1] < min_features:
            logger.warning(f"Not enough features: {features.shape[1]}, minimum is {min_features}")
            return False
        
        # Check for NaN or infinity
        if np.isnan(features).any() or np.isinf(features).any():
            logger.warning("Features array contains NaN or infinity values")
            return False
        
        logger.info(f"Features verification passed: shape {features.shape}")
        return True
    except Exception as e:
        logger.error(f"Error verifying features: {e}")
        return False
