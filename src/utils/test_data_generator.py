#!/usr/bin/env python3
"""
Test data generator for creating minimal, valid datasets for testing training pipeline.
Creates raw data (6 columns) and processes it through the feature generator to get final data (62 columns).
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import logging
from pathlib import Path

# Add project root to path for imports
from src.config.settings import get_settings

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def generate_raw_test_data(
    symbol: str = "Bank_Nifty_5",
    num_rows: int = 150,
    start_price: float = 46500.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Generate raw test data matching the structure of data/raw files (6 columns).

    Args:
        symbol: Trading symbol name
        num_rows: Number of data rows to generate
        start_price: Starting price for the data
        volatility: Price volatility (standard deviation as fraction of price)

    Returns:
        DataFrame with raw OHLCV data (6 columns: index, datetime, open, high, low, close, volume)
    """
    logger.info(f"Generating {num_rows} rows of raw test data for {symbol}")

    # Generate timestamps (5-minute intervals) as Unix timestamps
    start_time = datetime.now() - timedelta(minutes=5 * num_rows)
    timestamps = [int((start_time + timedelta(minutes=5 * i)).timestamp()) for i in range(num_rows)]

    # Generate price data with realistic patterns
    np.random.seed(42)  # For reproducible test data

    # Generate returns using random walk with slight upward bias
    returns = np.random.normal(0.0001, volatility, num_rows)  # Slight positive bias

    # Calculate prices
    prices = [start_price]
    for i in range(1, num_rows):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)

    # Generate OHLC from prices
    data = []
    for i, price in enumerate(prices):
        # Generate realistic OHLC around the price
        noise = np.random.normal(0, volatility * price * 0.5, 4)

        open_price = price + noise[0]
        close_price = price + noise[1]

        # Ensure high is highest and low is lowest
        high_price = max(open_price, close_price) + abs(noise[2])
        low_price = min(open_price, close_price) - abs(noise[3])

        # Ensure logical OHLC relationships
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        data.append({
            'datetime': timestamps[i],  # Unix timestamp like real data
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': 0  # Volume is 0 in the actual data
        })

    # Create DataFrame with index column like real data
    df = pd.DataFrame(data)
    df.reset_index(inplace=True)  # This creates the index column

    logger.info(f"Generated raw data: {len(df)} rows, price range {df['low'].min():.2f} - {df['high'].max():.2f}")
    return df


def process_raw_to_features_using_pipeline(raw_df: pd.DataFrame, symbol: str = "test_symbol") -> pd.DataFrame:
    """
    Process raw data through the actual feature generator pipeline to create features data.

    Args:
        raw_df: Raw DataFrame with 6 columns (index, datetime, open, high, low, close, volume)
        symbol: Symbol name for logging

    Returns:
        DataFrame with processed features using the actual pipeline
    """
    logger.info(f"Processing raw data through actual pipeline for {symbol}")

    try:
        # Import the feature generator
        from src.data_processing.feature_generator import DynamicFileProcessor

        # Create a copy of the raw data and prepare it for processing
        df = raw_df.copy()

        # Convert datetime from Unix timestamp to datetime string if needed
        if 'datetime' in df.columns and df['datetime'].dtype in ['int64', 'float64']:
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')

        # Initialize the feature processor
        processor = DynamicFileProcessor()

        # Process the DataFrame through the feature generator
        features_df = processor.process_dataframe(df)

        logger.info(f"Generated {len(features_df.columns)} features from raw data using actual pipeline")
        return features_df

    except Exception as e:
        logger.error(f"Error processing raw data to features using pipeline: {e}")
        # Fallback to simple feature generation if the processor fails
        return generate_simple_features_fallback(raw_df, symbol)


def generate_simple_features_fallback(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Fallback simple feature generation if the main processor fails.
    """
    logger.warning(f"Using simple feature generation fallback for {symbol}")

    # Extract OHLC columns
    df = raw_df[['open', 'high', 'low', 'close']].copy()

    # Add basic features to match expected structure
    df['sma_5'] = df['close'].rolling(5).mean()
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['rsi_14'] = 50.0  # Placeholder
    df['macd'] = 0.0  # Placeholder
    df['atr'] = df['high'] - df['low']

    # Fill NaN values
    df = df.bfill().fillna(0)

    return df


def generate_simple_features(raw_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Fallback simple feature generation if the main processor fails.
    """
    logger.warning(f"Using simple feature generation for {symbol}")

    # Extract OHLC columns
    df = raw_df[['open', 'high', 'low', 'close']].copy()

    # Add basic features to match expected structure
    df['sma_5'] = df['close'].rolling(5).mean()
    df['ema_5'] = df['close'].ewm(span=5).mean()
    df['rsi_14'] = 50.0  # Placeholder
    df['macd'] = 0.0  # Placeholder
    df['atr'] = df['high'] - df['low']

    # Fill NaN values
    df = df.bfill().fillna(0)

    return df


def generate_test_data_in_memory(
    symbol: str = "Bank_Nifty_5",
    num_rows: int = 150,
    start_price: float = 46500.0,
    volatility: float = 0.02
) -> Dict[str, pd.DataFrame]:
    """
    Generate test data in memory without saving files.

    Args:
        symbol: Trading symbol name
        num_rows: Number of data rows to generate
        start_price: Starting price for the data
        volatility: Price volatility

    Returns:
        Dictionary with 'raw' and 'features' DataFrames
    """
    logger.info(f"Generating in-memory test data for {symbol}")

    # Generate raw data (6 columns)
    raw_df = generate_raw_test_data(symbol, num_rows, start_price, volatility)

    # Process raw data to features using actual pipeline
    features_df = process_raw_to_features_using_pipeline(raw_df, symbol)

    return {
        'raw': raw_df,
        'features': features_df
    }


def generate_multiple_test_instruments(
    num_rows: int = 150
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate test data for multiple instruments in memory.

    Args:
        num_rows: Number of data rows to generate for each instrument

    Returns:
        Dictionary with instrument data: {symbol: {'raw': df, 'features': df}}
    """
    logger.info("Generating test data for multiple instruments")

    # Define instruments based on config/instruments.yaml with timeframes (matching real data format)
    instruments = [
        {"symbol": "RELIANCE_1", "type": "STOCK", "start_price": 2800.0, "volatility": 0.025},
        {"symbol": "Bank_Nifty_5", "type": "OPTION", "start_price": 46500.0, "volatility": 0.03}
    ]

    test_data = {}

    for instrument in instruments:
        symbol = instrument["symbol"]
        start_price = instrument["start_price"]
        volatility = instrument["volatility"]

        logger.info(f"Generating {instrument['type']} data for {symbol}")

        # Generate test data for this instrument
        instrument_data = generate_test_data_in_memory(
            symbol=symbol,
            num_rows=num_rows,
            start_price=start_price,
            volatility=volatility
        )

        test_data[symbol] = instrument_data

    logger.info(f"Generated test data for {len(test_data)} instruments")
    return test_data


def create_test_data_files(
    data_dir: str = None,
    symbol: str = "Bank_Nifty_5",
    num_rows: int = 150,
    create_both: bool = True,
    create_multiple_instruments: bool = True
) -> dict:
    settings = get_settings()
    paths_config = settings.get('paths', {})
    data_dir = data_dir or paths_config.get('test_data_dir', 'data/test')
    """
    Create test data files for training pipeline testing.

    Args:
        data_dir: Directory to create test data in
        symbol: Trading symbol name (used as base if create_multiple_instruments is False)
        num_rows: Number of data rows to generate
        create_both: Whether to create both raw and features files
        create_multiple_instruments: Whether to create both STOCK and OPTION test data

    Returns:
        Dictionary with paths to created files
    """
    logger.info(f"Creating test data files in {data_dir}")

    # Create directories
    raw_dir = os.path.join(data_dir, "raw")
    final_dir = os.path.join(data_dir, "final")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    created_files = {}

    # Define instruments to create (based on config/instruments.yaml with timeframes)
    if create_multiple_instruments:
        instruments = [
            {"symbol": "RELIANCE_1", "type": "STOCK", "start_price": 2800.0, "volatility": 0.025},
            {"symbol": "Bank_Nifty_5", "type": "OPTION", "start_price": 46500.0, "volatility": 0.03}
        ]
    else:
        instruments = [{"symbol": symbol, "type": "STOCK", "start_price": 46500.0, "volatility": 0.02}]

    for instrument in instruments:
        inst_symbol = instrument["symbol"]
        inst_type = instrument["type"]
        start_price = instrument["start_price"]
        volatility = instrument.get("volatility", 0.02)

        logger.info(f"Creating {inst_type} data for {inst_symbol} (start_price: {start_price})")

        # Generate raw data
        instrument_data = generate_test_data_in_memory(inst_symbol, num_rows, start_price, volatility)

        if create_both:
            # Save raw data
            raw_file = os.path.join(raw_dir, f"{inst_symbol}.csv")
            instrument_data['raw'].to_csv(raw_file, index=False)
            created_files[f'raw_{inst_symbol}'] = raw_file
            logger.info(f"Created raw {inst_type} data file: {raw_file}")

        # Save features data
        features_file = os.path.join(final_dir, f"features_{inst_symbol}.csv")
        instrument_data['features'].to_csv(features_file, index=False)
        created_files[f'features_{inst_symbol}'] = features_file
        logger.info(f"Created features {inst_type} data file: {features_file}")

    return created_files


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test in-memory generation
    print("Testing in-memory test data generation:")
    test_data = generate_multiple_test_instruments(num_rows=150)

    for symbol, data in test_data.items():
        print(f"\n{symbol}:")
        print(f"  Raw data: {data['raw'].shape} - columns: {list(data['raw'].columns)}")
        print(f"  Features data: {data['features'].shape} - columns: {len(data['features'].columns)} features")

    # Also test file creation
    print("\nTesting file creation:")
    settings = get_settings()
    paths_config = settings.get('paths', {})
    test_data_dir = paths_config.get('test_data_dir', 'data/test')
    files = create_test_data_files(
        data_dir=test_data_dir,
        symbol="Bank_Nifty_5",  # Base symbol (not used when create_multiple_instruments=True)
        num_rows=150,
        create_both=True,
        create_multiple_instruments=True  # Creates both RELIANCE (STOCK) and Bank_Nifty_5 (OPTION)
    )

    print("Created test data files:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")

