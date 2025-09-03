#!/usr/bin/env python3
"""
Comprehensive Custom Technical Analysis Feature Generator
=========================================================

A professional-grade technical analysis system built from scratch for maximum
control, reliability, and customization. Generates 50+ technical indicators
and market structure features for algorithmic trading.

Author: AlgoTrading System
Version: 1.0
"""

import os
import glob
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import logging
from pytz import timezone
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class MarketStructureAnalyzer:
    """
    Advanced market structure analysis for support/resistance, trends, and price action.
    """
    
    @staticmethod
    def trend_strength(close: pd.Series, period: int = 20) -> Dict[str, pd.Series]:
        """Calculate trend strength and direction"""
        # Linear regression slope for trend direction
        def calculate_slope(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            return slope
        
        trend_slope = close.rolling(window=period).apply(calculate_slope)
        
        # Trend strength based on R-squared
        def calculate_r_squared(series):
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope, intercept = np.polyfit(x, series, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((series - y_pred) ** 2)
                ss_tot = np.sum((series - np.mean(series)) ** 2)
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            except:
                return 0
        
        trend_strength = close.rolling(window=period).apply(calculate_r_squared)
        
        return {
            'trend_slope': trend_slope,
            'trend_strength': trend_strength,
            'trend_direction': np.where(trend_slope > 0, 1, -1)
        }
        

class DynamicFileProcessor:
    """
    Handles dynamic processing of all CSV files in the historical_data folder.
    """

    def __init__(self, data_folder: str = None):
        # Load configuration
        self.config = get_settings()
        self.feature_config = self.config.get('feature_generation', {})

        # Set up folders
        data_processing_config = self.config.get('data_processing', {})
        self.data_folder = Path(data_folder or data_processing_config.get('input_folder'))
        self.processed_folder = Path(data_processing_config.get('output_folder'))
        self.processed_folder.mkdir(exist_ok=True)
        
        self.market_structure = MarketStructureAnalyzer()
    
    def scan_data_files(self) -> List[Path]:
        """Scan for all CSV files in the data folder"""
        csv_files = list(self.data_folder.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to process")
        return csv_files
    
    def load_and_validate_data(self, file_path: Path) -> pd.DataFrame:
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(file_path)

            # Remove any unnamed index columns that may exist in raw data
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Explicitly drop 'volume' column if it exists
            if 'volume' in df.columns:
                df = df.drop(columns=['volume'])
                logger.info(f"Dropped 'volume' column from {file_path.name}")

            # Validate required columns (removed volume as it's not always available)
            required_cols = ['datetime', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            

            # CRITICAL: Keep datetime as epoch for enhanced processing in process_dataframe
            # Do NOT convert to pandas datetime here - let process_dataframe handle it
            # This ensures consistent datetime processing for both pipeline and backtesting paths

            # Sort by datetime (epoch) and remove duplicates
            df = df.sort_values('datetime').reset_index(drop=True)
            df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)

            # Remove any rows with invalid OHLC data
            df = df.dropna(subset=['open', 'high', 'low', 'close'])
            df = df[(df['high'] >= df['low']) & (df['high'] >= df['open']) &
                   (df['high'] >= df['close']) & (df['low'] <= df['open']) &
                   (df['low'] <= df['close'])]

            # Do NOT set datetime as index here - process_dataframe will handle it

            logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def generate_all_features(self, open_prices: pd.Series, high_prices: pd.Series,
                             low_prices: pd.Series, close_prices: pd.Series) -> pd.DataFrame:
        """Generate comprehensive technical analysis features using pandas-ta"""

        # Create a temporary DataFrame for pandas-ta (it works better with DataFrames)
        temp_df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        })

        features = {}

        # === TREND INDICATORS ===
        # Moving Averages using pandas-ta
        for period in self.feature_config.get('sma_periods', [5, 10, 20, 50, 100, 200]):
            features[f'sma_{period}'] = ta.sma(close_prices, length=period)
        for period in self.feature_config.get('ema_periods', [5, 10, 20, 50, 100, 200]):
            features[f'ema_{period}'] = ta.ema(close_prices, length=period)

        # MACD using pandas-ta
        macd_fast = self.feature_config.get('macd_fast', 12)
        macd_slow = self.feature_config.get('macd_slow', 26)
        macd_signal = self.feature_config.get('macd_signal', 9)
        macd_data = ta.macd(close_prices, fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if macd_data is not None and not macd_data.empty:
            # Use dynamic column names based on configuration
            macd_col_name = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
            macd_signal_col_name = f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}'
            macd_hist_col_name = f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}'
            features.update({
                'macd': macd_data[macd_col_name],
                'macd_signal': macd_data[macd_signal_col_name],
                'macd_histogram': macd_data[macd_hist_col_name]
            })

        # === MOMENTUM INDICATORS ===
        # RSI using pandas-ta
        for period in self.feature_config.get('rsi_periods', [14, 21]):
            features[f'rsi_{period}'] = ta.rsi(close_prices, length=period)

        # Stochastic using pandas-ta
        stoch_k_period = self.feature_config.get('stoch_k_period', 14)
        stoch_d_period = self.feature_config.get('stoch_d_period', 3)
        stoch_data = ta.stoch(high_prices, low_prices, close_prices, k=stoch_k_period, d=stoch_d_period)
        if stoch_data is not None and not stoch_data.empty:
            # Use dynamic column names based on configuration
            stoch_k_col = f'STOCHk_{stoch_k_period}_{stoch_d_period}_3'
            stoch_d_col = f'STOCHd_{stoch_k_period}_{stoch_d_period}_3'
            # Reindex to match the original data length
            stoch_k = stoch_data[stoch_k_col].reindex(close_prices.index)
            stoch_d = stoch_data[stoch_d_col].reindex(close_prices.index)
            features.update({
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            })

        # Williams %R using pandas-ta
        williams_r_period = self.feature_config.get('williams_r_period', 14)
        features['williams_r'] = ta.willr(high_prices, low_prices, close_prices, length=williams_r_period)

        # CCI using pandas-ta
        cci_period = self.feature_config.get('cci_period', 20)
        features['cci'] = ta.cci(high_prices, low_prices, close_prices, length=cci_period)

        # ADX using pandas-ta
        adx_period = self.feature_config.get('adx_period', 14)
        adx_data = ta.adx(high_prices, low_prices, close_prices, length=adx_period)
        if adx_data is not None and not adx_data.empty:
            # Use dynamic column names based on configuration
            features.update({
                'adx': adx_data[f'ADX_{adx_period}'],
                'di_plus': adx_data[f'DMP_{adx_period}'],
                'di_minus': adx_data[f'DMN_{adx_period}']
            })

        # Momentum and ROC using pandas-ta
        momentum_period = self.feature_config.get('momentum_period', 10)
        roc_period = self.feature_config.get('roc_period', 10)
        features[f'momentum_{momentum_period}'] = ta.mom(close_prices, length=momentum_period)
        features[f'roc_{roc_period}'] = ta.roc(close_prices, length=roc_period)

        # TRIX using pandas-ta (returns DataFrame)
        trix_period = self.feature_config.get('trix_period', 14)
        trix_data = ta.trix(close_prices, length=trix_period)
        if trix_data is not None and not trix_data.empty:
            # TRIX column naming is period_9 format by default
            trix_col = f'TRIX_{trix_period}_9'
            if trix_col in trix_data.columns:
                features['trix'] = trix_data[trix_col]
            else:
                # Fallback to first available column
                features['trix'] = trix_data.iloc[:, 0]

        # === VOLATILITY INDICATORS ===
        # ATR using pandas-ta
        atr_period = self.feature_config.get('atr_period', 14)
        features['atr'] = ta.atr(high_prices, low_prices, close_prices, length=atr_period)

        # Bollinger Bands using pandas-ta
        bb_period = self.feature_config.get('bb_period', 20)
        bb_std_dev = self.feature_config.get('bb_std_dev', 2.0)
        bb_data = ta.bbands(close_prices, length=bb_period, std=bb_std_dev)
        if bb_data is not None and not bb_data.empty:
            # Use dynamic column names based on configuration
            bb_upper_col = f'BBU_{bb_period}_{bb_std_dev}'
            bb_middle_col = f'BBM_{bb_period}_{bb_std_dev}'
            bb_lower_col = f'BBL_{bb_period}_{bb_std_dev}'
            features.update({
                'bb_upper': bb_data[bb_upper_col],
                'bb_middle': bb_data[bb_middle_col],
                'bb_lower': bb_data[bb_lower_col],
                'bb_width': (bb_data[bb_upper_col] - bb_data[bb_lower_col]) / bb_data[bb_middle_col] * 100,
                'bb_position': (close_prices - bb_data[bb_lower_col]) / (bb_data[bb_upper_col] - bb_data[bb_lower_col]) * 100
            })

        # === MARKET STRUCTURE ===
        # Trend Analysis
        trend_data = self.market_structure.trend_strength(close_prices)
        features.update(trend_data)

        # === PRICE ACTION FEATURES ===
        # Price changes
        features['price_change'] = close_prices.pct_change() * 100
        features['price_change_abs'] = np.abs(features['price_change'])

        # High-Low range
        features['hl_range'] = (high_prices - low_prices) / close_prices * 100

        # Body and shadow analysis
        features['body_size'] = np.abs(close_prices - open_prices) / close_prices * 100
        features['upper_shadow'] = (high_prices - np.maximum(open_prices, close_prices)) / close_prices * 100
        features['lower_shadow'] = (np.minimum(open_prices, close_prices) - low_prices) / close_prices * 100

        # === DERIVED FEATURES ===
        # Moving average crossovers (only if the features exist and are not None)
        if ('sma_5' in features and 'sma_20' in features and
            features['sma_5'] is not None and features['sma_20'] is not None):
            features['sma_5_20_cross'] = np.where(features['sma_5'] > features['sma_20'], 1, -1)
        else:
            features['sma_5_20_cross'] = np.zeros(len(close_prices))

        if ('sma_10' in features and 'sma_50' in features and
            features['sma_10'] is not None and features['sma_50'] is not None):
            features['sma_10_50_cross'] = np.where(features['sma_10'] > features['sma_50'], 1, -1)
        else:
            features['sma_10_50_cross'] = np.zeros(len(close_prices))

        # Price position relative to moving averages (only if the features exist and are not None)
        if 'sma_20' in features and features['sma_20'] is not None:
            features['price_vs_sma_20'] = (close_prices - features['sma_20']) / features['sma_20'] * 100
        else:
            features['price_vs_sma_20'] = np.zeros(len(close_prices))

        if 'ema_20' in features and features['ema_20'] is not None:
            features['price_vs_ema_20'] = (close_prices - features['ema_20']) / features['ema_20'] * 100
        else:
            features['price_vs_ema_20'] = np.zeros(len(close_prices))

        # Volatility measures
        volatility_periods = self.feature_config.get('volatility_periods', [10, 20])
        for period in volatility_periods:
            features[f'volatility_{period}'] = close_prices.rolling(period).std() / close_prices.rolling(period).mean() * 100

        

        # Create DataFrame and ensure proper indexing
        features_df = pd.DataFrame(features)

        # Ensure the features DataFrame has the same index as the input data
        if len(features_df) != len(close_prices):
            logger.warning(f"Features length mismatch: {len(features_df)} vs {len(close_prices)}")
            # Trim to minimum length
            min_len = min(len(features_df), len(close_prices))
            features_df = features_df.iloc[:min_len]

        # Round all numeric columns to 2 decimal places
        for col in features_df.columns:
            if features_df[col].dtype in ['float64', 'float32']:
                features_df[col] = features_df[col].round(2)

        return features_df

    

    def process_single_file(self, file_path: Path) -> pd.DataFrame:
        """Process a single CSV file and generate all features"""
        logger.info(f"Processing {file_path.name}...")

        # Load data
        df = self.load_and_validate_data(file_path)

        # Extract OHLC data (volume removed as not always available)
        open_prices = df['open']
        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']

        # Generate all features
        features_df = self.generate_all_features(
            open_prices, high_prices, low_prices, close_prices
        )

        # CRITICAL: Preserve datetime index - do NOT reset it
        # Keep the datetime index that was set in the enhanced datetime processing
        df_reset = df.copy()

        # Ensure features_df has the same length as df_reset
        if len(features_df) != len(df_reset):
            min_length = min(len(features_df), len(df_reset))
            df_reset = df_reset.iloc[:min_length]
            features_df = features_df.iloc[:min_length]
            logger.warning(f"Trimmed data to {min_length} rows for alignment")

        features_df_reset = features_df.reset_index(drop=True)

        # Combine with original data
        result_df = pd.concat([df_reset, features_df_reset], axis=1)

        # Clean up the final dataset
        result_df = self.clean_final_dataset(result_df)

        logger.info(f"Generated {len(features_df.columns)} features for {file_path.name}")
        return result_df

    def clean_final_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the final dataset by removing rows with NaN values (but keep legitimate 0 values)"""
        initial_rows = len(df)

        # Remove rows with NaN values (but keep legitimate 0 values)
        df.dropna(inplace=True)
        # CRITICAL: Do NOT reset index - preserve datetime index
        # df.reset_index(drop=True, inplace=True)  # REMOVED to preserve datetime index

        removed_rows = initial_rows - len(df)
        if removed_rows > 0:
            logger.info(f"Removed {removed_rows} rows with NaN values. Final dataset: {len(df)} rows")

        return df


    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process an in-memory DataFrame and generate all features"""
        logger.info(f"Processing in-memory DataFrame with {len(df)} rows...")

        # Clean and prepare data first
        if 'datetime' in df.columns:
            # Sort by datetime and remove duplicates
            df = df.sort_values('datetime').reset_index(drop=True)
            df = df.drop_duplicates(subset=['datetime']).reset_index(drop=True)

            # ENHANCED: Keep epoch datetime as feature for temporal learning
            if df['datetime'].dtype in ['int64', 'float64']:
                # Already epoch format
                df['datetime_epoch'] = df['datetime']
                # Convert epoch to readable datetime for index (IST timezone for Indian markets)
                df['datetime_readable'] = pd.to_datetime(df['datetime'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
            else:
                # Convert to epoch for feature
                df['datetime_epoch'] = pd.to_datetime(df['datetime']).astype('int64') // 10**9
                # Keep readable datetime (assume already in IST)
                df['datetime_readable'] = pd.to_datetime(df['datetime'])

            # Set readable datetime as index, keep epoch as feature
            df = df.set_index('datetime_readable')
            df = df.drop(columns=['datetime'])  # Remove original datetime column
            logger.info("✅ Enhanced datetime processing: epoch as feature, readable as index")

        # Remove any remaining duplicate indices
        if df.index.duplicated().any():
            logger.warning(f"Found {df.index.duplicated().sum()} duplicate datetime indices, removing...")
            df = df[~df.index.duplicated(keep='first')]

        # Extract OHLC data
        open_prices = df['open']
        high_prices = df['high']
        low_prices = df['low']
        close_prices = df['close']

        # Generate all features
        features_df = self.generate_all_features(
            open_prices, high_prices, low_prices, close_prices
        )

        # Combine with original data
        result_df = df.join(features_df)

        # Clean up the final dataset
        result_df = self.clean_final_dataset(result_df)

        logger.info(f"Generated {len(features_df.columns)} features for DataFrame")
        return result_df

    def process_all_files(self) -> Dict[str, str]:
        """Process all files and return summary"""
        files = self.scan_data_files()
        results = {}

        if not files:
            logger.warning("No CSV files found in the historical_data folder")
            return results

        for file_path in files:
            try:
                # Load and validate data from file
                df = self.load_and_validate_data(file_path)

                # Process the dataframe
                processed_df = self.process_dataframe(df)

                # Save processed file as Parquet (replace existing)
                output_path = self.processed_folder / f"features_{file_path.stem}.parquet"
                # CRITICAL: Save with index=True to preserve datetime index
                processed_df.to_parquet(output_path, index=True)

                results[file_path.name] = f"Success: {len(processed_df)} rows, {len(processed_df.columns)} features"
                logger.info(f"Processed {file_path.name}: {len(processed_df)} rows, {len(processed_df.columns)} features")
                logger.info(f"Replaced existing file: {output_path}")

            except Exception as e:
                results[file_path.name] = f"Error: {str(e)}"
                logger.error(f"✗ Failed to process {file_path.name}: {str(e)}")

        return results


if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE TECHNICAL ANALYSIS FEATURE GENERATOR")
    print("=" * 80)
    print("Building custom technical indicators from scratch...")
    print("Features include: Trend, Momentum, Volatility, Patterns, Market Structure")
    print()

    # Initialize processor
    processor = DynamicFileProcessor()

    # Process all files
    results = processor.process_all_files()

    # Print summary
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)

    if results:
        for filename, status in results.items():
            print(f"{filename}: {status}")
    else:
        print("No files were processed.")

    print("\nFeature generation complete!")
    print(f"Processed files saved in: {processor.processed_folder}")
    print("=" * 80)
