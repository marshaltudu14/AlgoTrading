import pandas as pd
import numpy as np
import time
import os
import gc
from datetime import date, timedelta
from pytz import timezone
import pandas_ta as ta
# Removed statsmodels and scipy imports as they are no longer needed

# Assuming fyersModel is initialized elsewhere and passed or accessed globally
# from fyers_apiv3 import fyersModel # Import if needed directly

# Import config and signal function from the same package
try:
    from . import config
    from .signals import label_signals_jit
except ImportError:
    import config # Fallback for running script directly
    from signals import label_signals_jit # Fallback


# --- Removed add_statistical_features, add_acf_pacf_features, and add_refined_indicators ---


# --- Main Pipeline Class ---

class FullFeaturePipeline:
    """
    Processes raw candle data (OHLCV) to add technical indicators,
    time features, adaptive targets/stops, and signal labels.
    """
    def __init__(self, df: pd.DataFrame):
        # df must have columns: [datetime, open, high, low, close, (volume optional)]
        if not all(col in df.columns for col in ['datetime', 'open', 'high', 'low', 'close']):
             raise ValueError("Input DataFrame missing required OHLC columns or 'datetime'.")
        self.df = df.copy()

    def preprocess_datetime(self):
        """Converts Unix timestamp to datetime, sets timezone, and sets as index."""
        # Convert Unix timestamp to datetime objects
        self.df['datetime'] = pd.to_datetime(self.df['datetime'], unit='s')

        # Localize to UTC first, then convert to IST
        self.df['datetime'] = self.df['datetime'].dt.tz_localize('UTC').dt.tz_convert(config.IST_TIMEZONE)

        # Remove timezone information after conversion if needed for indexing/compatibility
        self.df['datetime'] = self.df['datetime'].dt.tz_localize(None)

        # Check for duplicates or missing values *after* conversion
        if self.df['datetime'].duplicated().any():
            print("Warning: Duplicate datetime values found. Keeping first occurrence.")
            self.df.drop_duplicates(subset='datetime', keep='first', inplace=True)
        if self.df['datetime'].isnull().any():
            raise ValueError("The 'datetime' column contains missing values after conversion.")

        # Sort and set as index
        self.df.sort_values('datetime', inplace=True)
        self.df.set_index('datetime', inplace=True)
        return self

    def clean_data(self):
        """Removes or handles unnecessary columns like volume."""
        # Consistently handle volume: drop if present and null/zero, or just drop always if unused
        if 'volume' in self.df.columns:
            # Optional: Check for bad volume data before dropping
            # vol_condition = self.df['volume'].isnull() | (self.df['volume'] <= 0)
            # if vol_condition.any():
            #     print("Warning: Volume column contains null or zero values.")
            self.df.drop('volume', axis=1, inplace=True, errors='ignore') # Use errors='ignore'
        return self

    def add_indicator_features(self):
        """Adds various technical indicators to the DataFrame."""
        # ATR
        self.df['atr_14'] = ta.atr(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], length=14
        ).astype(np.float32).round(2)

        # RSI
        self.df['rsi_14'] = ta.rsi(
            close=self.df['close'], length=14
        ).astype(np.float32).round(2)

        # Stochastics %K and %D
        stoch = ta.stoch(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], k=14, d=3, smooth_k=3 # Standard params
        )
        if stoch is not None and not stoch.empty:
             self.df = self.df.join(stoch.astype(np.float32).round(2))
        else:
             print("Warning: Stochastic indicator calculation failed.")
             self.df[['STOCHk_14_3_3', 'STOCHd_14_3_3']] = np.nan # Add NaN columns if failed


        # MACD, Histogram, Signal Line
        macd = ta.macd(
            close=self.df['close'], fast=12, slow=26, signal=9
        )
        if macd is not None and not macd.empty:
             self.df = self.df.join(macd.astype(np.float32).round(2))
        else:
             print("Warning: MACD indicator calculation failed.")
             self.df[['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']] = np.nan


        # ADX, +DI, -DI
        adx = ta.adx(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], length=14
        )
        if adx is not None and not adx.empty:
             # Keep only ADX, rename for consistency
             self.df = self.df.join(adx[['ADX_14']].astype(np.float32).round(2))
             self.df.rename(columns={'ADX_14': 'adx_14'}, inplace=True)
        else:
             print("Warning: ADX indicator calculation failed.")
             self.df['adx_14'] = np.nan


        # EMAs
        self.df['ema_50'] = ta.ema(self.df['close'], length=50).astype(np.float32).round(2)
        self.df['ema_200'] = ta.ema(self.df['close'], length=200).astype(np.float32).round(2)

        # SMA
        self.df['sma_20'] = ta.sma(self.df['close'], length=20).astype(np.float32).round(2)

        # Keltner Channels (Upper, Basis, Lower)
        kc = ta.kc(
            high=self.df['high'], low=self.df['low'], close=self.df['close'], length=20, scalar=2.0 # Standard params
        )
        if kc is not None and not kc.empty:
             self.df = self.df.join(kc.astype(np.float32).round(2)) # e.g., KCUe_20_2.0, KCLe_20_2.0, KCe_20_2.0
        else:
             print("Warning: Keltner Channels indicator calculation failed.")
             # Add NaN columns manually if needed, names depend on ta library version
             self.df[['KCUe_20_2.0', 'KCLe_20_2.0', 'KCe_20_2.0']] = np.nan # Adjust names if needed


        # Price Action Features
        self.df['price_range'] = (self.df['high'] - self.df['low']).astype(np.float32).round(2)
        self.df['body_size'] = (self.df['close'] - self.df['open']).abs().astype(np.float32).round(2)
        self.df['upper_wick'] = (self.df['high'] - self.df[['open', 'close']].max(axis=1)).astype(np.float32).round(2)
        self.df['lower_wick'] = (self.df[['open', 'close']].min(axis=1) - self.df['low']).astype(np.float32).round(2)

        return self

    def add_time_features(self):
        """Adds time-based features (hour, day of week, month)."""
        if isinstance(self.df.index, pd.DatetimeIndex):
            self.df['hour'] = self.df.index.hour.astype(np.int32)
            self.df['day_of_week'] = self.df.index.dayofweek.astype(np.int32) # Monday=0, Sunday=6
            self.df['month'] = self.df.index.month.astype(np.int32)
        else:
            print("Warning: DataFrame index is not DatetimeIndex. Cannot add time features.")
        return self

    # --- REMOVED add_adaptive_targets_and_stops method ---

    # --- REMOVED label_signals method ---

    def run_pipeline(self):
        """Executes the simplified data processing pipeline for RL features."""
        print("Running simplified data processing pipeline for RL features...")
        self.preprocess_datetime()
        self.clean_data()
        self.add_indicator_features() # Existing basic indicators
        # --- Removed call to add_refined_indicators ---
        # Add time features last
        self.add_time_features()

        # --- Calls to removed features/methods are removed ---

        print("Simplified pipeline finished.")
        # Garbage collect after heavy processing
        gc.collect()
        return self # Return self to allow chaining if needed elsewhere

    def get_processed_df(self):
        """Returns the processed DataFrame, dropping NaNs."""
        print(f"Shape before dropping NaN: {self.df.shape}")
        # Drop rows with any NaN values resulting from indicator calculations (lookback periods)
        # This is crucial as RL environments typically cannot handle NaNs
        initial_len = len(self.df)
        self.df.dropna(axis=0, how='any', inplace=True)
        dropped_rows = initial_len - len(self.df)
        print(f"Shape after dropping {dropped_rows} NaN rows: {self.df.shape}")

        # Columns like 'Target', 'StopLoss', 'Signal' etc. are no longer added in run_pipeline
        # So no need to explicitly drop them here.

        print(f"Final columns for RL: {self.df.columns.tolist()}")
        return self.df


def get_dataframe_by_name(instrument_name, dynamic_config):
    """
    Loads a processed DataFrame based on the instrument name from the dynamic config.

    Args:
        instrument_name (str): The dynamic name, e.g., "BANK_NIFTY_5M".
        dynamic_config (dict): The loaded dynamic configuration dictionary.

    Returns:
        pd.DataFrame: The loaded DataFrame, or None if not found or error.
    """
    if not dynamic_config or 'instruments' not in dynamic_config:
        print("Error: Dynamic configuration is empty or invalid.")
        return None

    for instrument in dynamic_config["instruments"]:
        if instrument.get("name") == instrument_name:
            file_path = instrument.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    # Ensure datetime index is parsed correctly
                    return pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')
                except Exception as e:
                    print(f"Error reading processed file {file_path}: {e}")
                    return None
            else:
                print(f"Error: Processed file path not found or invalid for {instrument_name}: {file_path}")
                return None

    print(f"Warning: Instrument '{instrument_name}' not found in dynamic configuration.")
    return None

# Example usage (for testing within this module)
if __name__ == "__main__":
    # This part requires a sample raw CSV file named 'sample_raw.csv'
    # with columns ['datetime', 'open', 'high', 'low', 'close', 'volume']
    # where datetime is in Unix epoch format.
    sample_file = "sample_raw.csv"
    if os.path.exists(sample_file):
        print(f"Testing pipeline with {sample_file}...")
        raw_df = pd.read_csv(sample_file)
        pipeline = FullFeaturePipeline(raw_df)
        pipeline.run_pipeline()
        processed_df = pipeline.get_processed_df()
        print("\nSample Processed DataFrame Head:")
        print(processed_df.head())
        print("\nSample Processed DataFrame Info:")
        processed_df.info()
    else:
        print(f"Skipping FullFeaturePipeline test: {sample_file} not found.")

    # Test loading dynamic config (requires 'dynamic_config.json' to exist)
    # config.load_dynamic_config()
    # if config.DYNAMIC_INSTRUMENTS_CONFIG["instruments"]:
    #     test_instrument_name = config.DYNAMIC_INSTRUMENTS_CONFIG["instruments"][0]["name"]
    #     print(f"\nTesting get_dataframe_by_name for: {test_instrument_name}")
    #     loaded_df = get_dataframe_by_name(test_instrument_name, config.DYNAMIC_INSTRUMENTS_CONFIG)
    #     if loaded_df is not None:
    #         print("Successfully loaded DataFrame:")
    #         print(loaded_df.head())
    #     else:
    #         print("Failed to load DataFrame.")
    # else:
    #     print("\nSkipping get_dataframe_by_name test: No instruments in dynamic config.")
