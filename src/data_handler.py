import pandas as pd
import numpy as np
import time
import os
import gc
from datetime import date, timedelta
from pytz import timezone
import pandas_ta as ta
from statsmodels.tsa.stattools import acf, pacf
from scipy.stats import skew, kurtosis # Use scipy for rolling skew/kurtosis

# Assuming fyersModel is initialized elsewhere and passed or accessed globally
# from fyers_apiv3 import fyersModel # Import if needed directly

# Import config and signal function from the same package
try:
    from . import config
    from .signals import label_signals_jit
except ImportError:
    import config # Fallback for running script directly
    from signals import label_signals_jit # Fallback


# --- New Feature Calculation Functions ---

def add_statistical_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Adds rolling statistical features like skewness, kurtosis, and ROC."""
    print(f"  Adding statistical features (window={window})...")
    # Skewness and Kurtosis using rolling apply for potentially better NaN handling
    df[f'skew_close_{window}'] = df['close'].rolling(window=window, min_periods=window//2).apply(lambda x: skew(x.dropna()), raw=True).astype(np.float32).round(4)
    df[f'kurt_close_{window}'] = df['close'].rolling(window=window, min_periods=window//2).apply(lambda x: kurtosis(x.dropna()), raw=True).astype(np.float32).round(4) # Fisher's kurtosis (normal=0)

    # Rate of Change (ROC)
    for period in [5, 10, 20]: # Add a few ROC periods
        df[f'roc_close_{period}'] = ta.roc(df['close'], length=period).astype(np.float32).round(4)

    return df

def add_acf_pacf_features(df: pd.DataFrame, nlags: int = 10) -> pd.DataFrame:
    """Adds ACF and PACF features for the 'close' price."""
    print(f"  Adding ACF/PACF features (nlags={nlags})...")
    # Calculate ACF and PACF - Note: These are computationally more intensive
    # We calculate them once and then assign to avoid recalculating in rolling fashion
    try:
        close_series = df['close'].dropna()
        if len(close_series) > nlags * 2: # Need enough data points
            acf_values = acf(close_series, nlags=nlags, fft=True) # Use fft for speed
            pacf_values = pacf(close_series, nlags=nlags, method='ols') # Ordinary Least Squares method

            # Assign calculated values - these are not rolling features
            # They represent the overall series correlation up to that point
            # For RL, maybe rolling versions are better, but let's start simple
            # We'll add NaN for the first few rows where lags aren't available
            for i in range(1, nlags + 1):
                df[f'acf_close_lag{i}'] = acf_values[i] if i < len(acf_values) else np.nan
                df[f'pacf_close_lag{i}'] = pacf_values[i] if i < len(pacf_values) else np.nan
            # Fill initial NaNs if needed, or let dropna handle later
            # df[[f'acf_close_lag{i}' for i in range(1, nlags + 1)]] = df[[f'acf_close_lag{i}' for i in range(1, nlags + 1)]].fillna(method='bfill')
            # df[[f'pacf_close_lag{i}' for i in range(1, nlags + 1)]] = df[[f'pacf_close_lag{i}' for i in range(1, nlags + 1)]].fillna(method='bfill')
        else:
             print(f"  Warning: Not enough data points ({len(close_series)}) to calculate ACF/PACF with nlags={nlags}. Skipping.")
             for i in range(1, nlags + 1):
                 df[f'acf_close_lag{i}'] = np.nan
                 df[f'pacf_close_lag{i}'] = np.nan

    except Exception as e:
        print(f"  Error calculating ACF/PACF: {e}. Skipping.")
        for i in range(1, nlags + 1):
            df[f'acf_close_lag{i}'] = np.nan
            df[f'pacf_close_lag{i}'] = np.nan

    # Cast to float32 after calculations
    for i in range(1, nlags + 1):
        if f'acf_close_lag{i}' in df.columns:
            df[f'acf_close_lag{i}'] = df[f'acf_close_lag{i}'].astype(np.float32).round(4)
        if f'pacf_close_lag{i}' in df.columns:
            df[f'pacf_close_lag{i}'] = df[f'pacf_close_lag{i}'].astype(np.float32).round(4)

    return df


def add_refined_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds additional technical indicators like CCI and PSAR."""
    print("  Adding refined indicators (CCI, PSAR)...")
    # Commodity Channel Index (CCI)
    df['cci_20'] = ta.cci(high=df['high'], low=df['low'], close=df['close'], length=20).astype(np.float32).round(2)

    # Parabolic SAR (PSAR)
    psar = ta.psar(high=df['high'], low=df['low'], close=df['close']) # Default parameters
    if psar is not None and not psar.empty:
        # Select and rename relevant columns (names might vary slightly with pandas_ta versions)
        psar_cols = {col: col.lower().replace('.', '_') for col in psar.columns if 'psar' in col.lower()}
        df = df.join(psar[psar_cols.keys()].astype(np.float32).round(2))
        df.rename(columns=psar_cols, inplace=True)
    else:
        print("  Warning: PSAR calculation failed.")
        # Add NaN columns manually if needed
        df[['psarl_0.02_0.2', 'psars_0.02_0.2', 'psaraf_0.02_0.2', 'psarr_0.02_0.2']] = np.nan # Adjust names based on library version if needed

    return df


# --- Main Pipeline Class ---

class FullFeaturePipeline:
# from fyers_apiv3 import fyersModel # Import if needed directly

# Import config and signal function from the same package
try:
    from . import config
    from .signals import label_signals_jit
except ImportError:
    import config # Fallback for running script directly
    """
    Fetches historical candle data for a specified number of days back.

    Args:
        fyers_instance: An initialized fyersModel.FyersModel instance.
        number (int): Number of days back from today to start fetching.
        index_symbol (str): The symbol to fetch data for (e.g., "NSE:NIFTYBANK-INDEX").
        interval_minutes (str or int): The candle interval (e.g., "5", 15).

    Returns:
        pd.DataFrame: DataFrame with candle data, or None if an error occurs.
    """
    while True:
        try:
            today = date.today()
            # Ensure range_from is calculated correctly based on 'number' days ago
            range_from = today - timedelta(days=number)
            # Fetch up to today
            range_to = today

            data = {
                "symbol": index_symbol,
                "resolution": str(interval_minutes), # Ensure resolution is string
                "date_format": "1", # Use '1' for epoch format
                "range_from": range_from.strftime('%Y-%m-%d'), # Format as YYYY-MM-DD
                "range_to": range_to.strftime('%Y-%m-%d'),   # Format as YYYY-MM-DD
                "cont_flag": "1"
            }

            result = fyers_instance.history(data=data)

            if result.get("code") != 200 or not result.get("candles"):
                 print(f"Warning: No data received for {index_symbol} ({interval_minutes} min) from {range_from} to {range_to}. Response: {result.get('message', result)}")
                 # Decide if you want to return empty DF or None or retry
                 return pd.DataFrame(columns=['datetime', 'open', 'high', 'low', 'close', 'volume']) # Return empty DF

            # Process candles if data is present
            candles_df = pd.DataFrame(result['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            return candles_df

        except Exception as e:
            print(f"Error fetching Candle Data for {index_symbol}: {e}")
            print("Retrying after delay...")
            time.sleep(config.DEFAULT_TRADING_VARS.get("active_order_sleep", 1)) # Use sleep time from config


def fetch_train_candle_data(fyers_instance, days_count, index_symbol, interval_minutes):
    """
    Fetches historical candle data over multiple periods for training.

    Args:
        fyers_instance: An initialized fyersModel.FyersModel instance.
        days_count (int): Number of 100-day chunks to fetch back in time.
        index_symbol (str): The symbol to fetch data for.
        interval_minutes (str or int): The candle interval.

    Returns:
        pd.DataFrame: Concatenated DataFrame of historical data, or empty DataFrame if error.
    """
    all_candles_df = pd.DataFrame()
    date_offset = 0 # Start from today

    print(f"Fetching training data for {index_symbol} ({interval_minutes} min) over {days_count * 100} days...")

    for i in range(days_count):
        # Calculate date ranges for each 100-day chunk
        range_to = date.today() - timedelta(days=date_offset)
        range_from = range_to - timedelta(days=100) # Fetch 100 days per iteration

        print(f"  Fetching chunk {i+1}/{days_count}: {range_from} to {range_to}")

        data = {
            "symbol": index_symbol,
            "resolution": str(interval_minutes),
            "date_format": "1",
            "range_from": range_from.strftime('%Y-%m-%d'),
            "range_to": range_to.strftime('%Y-%m-%d'),
            "cont_flag": "1"
        }

        try:
            result = fyers_instance.history(data=data)

            if result.get("code") == 200 and result.get("candles"):
                temp_df = pd.DataFrame(result['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
                all_candles_df = pd.concat([temp_df, all_candles_df], ignore_index=True)
                print(f"    Fetched {len(temp_df)} candles for chunk {i+1}.")
            else:
                 print(f"    Warning: No data or error for chunk {i+1}. Response: {result.get('message', result)}")

            # Update offset for the next iteration
            date_offset += 100
            time.sleep(0.5) # Small delay between API calls

        except Exception as e:
            print(f"  Error fetching chunk {i+1} for {index_symbol}: {e}")
            print("  Retrying chunk after delay...")
            time.sleep(config.DEFAULT_TRADING_VARS.get("active_order_sleep", 2)) # Longer sleep on error
            # Optional: Implement retry logic for the specific chunk here
            # For simplicity, we continue to the next chunk for now

    if not all_candles_df.empty:
        # Sort by datetime and remove duplicates just in case
        all_candles_df.sort_values('datetime', inplace=True)
        all_candles_df.drop_duplicates(subset='datetime', keep='first', inplace=True)
        print(f"Total unique candles fetched for {index_symbol}: {len(all_candles_df)}")
    else:
        print(f"Failed to fetch any training data for {index_symbol}.")

    return all_candles_df


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

    def add_adaptive_targets_and_stops(self):
        """Calculates target and stoploss levels based on ATR."""
        if 'atr_14' not in self.df.columns or self.df['atr_14'].isnull().all():
             print("Warning: ATR not available or all NaN. Cannot calculate adaptive targets/stops.")
             self.df['Target'] = np.nan
             self.df['StopLoss'] = np.nan
        else:
             # Ensure ATR is positive before multiplication
             valid_atr = self.df['atr_14'].fillna(0).clip(lower=0.01) # Fill NaN, ensure minimum value
             self.df['Target'] = (4.0 * valid_atr).astype(np.float32).round(2)
             self.df['StopLoss'] = (2.0 * valid_atr).astype(np.float32).round(2)
        return self

    def label_signals(self):
        """Applies the JIT-compiled signal labeling function."""
        if not all(col in self.df.columns for col in ['close', 'high', 'low', 'Target', 'StopLoss']):
             print("Warning: Missing columns required for signal labeling (close, high, low, Target, StopLoss). Skipping.")
             self.df['Signal'] = 0.0
             self.df['Entry Price'] = np.nan
             self.df['Exit Price'] = np.nan
             self.df['candles_to_profit'] = 0.0
             self.df['candles_to_loss'] = 0.0
             return self

        # Drop rows where target/stoploss might be NaN before passing to Numba
        valid_rows = self.df.dropna(subset=['close', 'high', 'low', 'Target', 'StopLoss'])

        if valid_rows.empty:
             print("Warning: No valid rows with non-NaN Target/StopLoss for signal labeling. Skipping.")
             self.df['Signal'] = 0.0
             self.df['Entry Price'] = np.nan
             self.df['Exit Price'] = np.nan
             self.df['candles_to_profit'] = 0.0
             self.df['candles_to_loss'] = 0.0
             return self


        close_arr = valid_rows['close'].values
        high_arr = valid_rows['high'].values
        low_arr = valid_rows['low'].values
        target_arr = valid_rows['Target'].values
        stoploss_arr = valid_rows['StopLoss'].values

        # Call the imported JIT function
        signals, entry_prices, exit_prices, ctp, ctl = label_signals_jit(
            close_arr, high_arr, low_arr, target_arr, stoploss_arr
        )

        # Assign results back to the original DataFrame using the index of valid_rows
        self.df.loc[valid_rows.index, 'Signal'] = signals.astype(np.float32)
        self.df.loc[valid_rows.index, 'Entry Price'] = entry_prices.astype(np.float32).round(2)
        self.df.loc[valid_rows.index, 'Exit Price'] = exit_prices.astype(np.float32).round(2)
        self.df.loc[valid_rows.index, 'candles_to_profit'] = ctp.astype(np.float32)
        self.df.loc[valid_rows.index, 'candles_to_loss'] = ctl.astype(np.float32)

        # Fill NaN for rows that were initially dropped
        self.df['Signal'].fillna(0.0, inplace=True)
        self.df['candles_to_profit'].fillna(0.0, inplace=True)
        self.df['candles_to_loss'].fillna(0.0, inplace=True)


        return self

    def run_pipeline(self):
        """Executes the full data processing pipeline for RL features."""
        print("Running data processing pipeline for RL features...")
        self.preprocess_datetime()
        self.clean_data()
        self.add_indicator_features() # Existing basic indicators
        # Add new features
        add_statistical_features(self.df)
        add_acf_pacf_features(self.df)
        add_refined_indicators(self.df)
        # Add time features last
        self.add_time_features()

        # --- REMOVED ---
        # .add_adaptive_targets_and_stops()
        # .label_signals()
        # ---------------

        print("Pipeline finished.")
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
