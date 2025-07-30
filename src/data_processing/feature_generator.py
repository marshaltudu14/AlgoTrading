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
from config.config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class PandasTAIndicators:
    """
    Technical indicators using pandas-ta library for maximum reliability.
    All indicators use pandas-ta implementations with proper NaN handling.
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average using pandas-ta"""
        return ta.sma(data, length=period)

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average using pandas-ta"""
        return ta.ema(data, length=period)

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using pandas-ta"""
        return ta.rsi(data, length=period)
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence using pandas-ta"""
        macd_data = ta.macd(data, fast=fast, slow=slow, signal=signal)
        return {
            'macd': macd_data[f'MACD_{fast}_{slow}_{signal}'],
            'signal': macd_data[f'MACDs_{fast}_{slow}_{signal}'],
            'histogram': macd_data[f'MACDh_{fast}_{slow}_{signal}']
        }

    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands using pandas-ta"""
        bb_data = ta.bbands(data, length=period, std=std_dev)
        return {
            'upper': bb_data[f'BBU_{period}_{std_dev}'],
            'middle': bb_data[f'BBM_{period}_{std_dev}'],
            'lower': bb_data[f'BBL_{period}_{std_dev}']
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range using pandas-ta"""
        return ta.atr(high, low, close, length=period)

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator using pandas-ta"""
        stoch_data = ta.stoch(high, low, close, k=k_period, d=d_period)
        return {
            'k': stoch_data[f'STOCHk_{k_period}_{d_period}_{d_period}'],
            'd': stoch_data[f'STOCHd_{k_period}_{d_period}_{d_period}']
        }

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R using pandas-ta"""
        return ta.willr(high, low, close, length=period)

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index using pandas-ta"""
        return ta.cci(high, low, close, length=period)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index using pandas-ta"""
        adx_data = ta.adx(high, low, close, length=period)
        return {
            'adx': adx_data[f'ADX_{period}'],
            'di_plus': adx_data[f'DMP_{period}'],
            'di_minus': adx_data[f'DMN_{period}']
        }

    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum using pandas-ta"""
        return ta.mom(data, length=period)

    @staticmethod
    def roc(data: pd.Series, period: int = 10) -> pd.Series:
        """Rate of Change using pandas-ta"""
        return ta.roc(data, length=period)

    @staticmethod
    def trix(data: pd.Series, period: int = 14) -> pd.Series:
        """TRIX using pandas-ta"""
        return ta.trix(data, length=period)


class MarketStructureAnalyzer:
    """
    Advanced market structure analysis for support/resistance, trends, and price action.
    """
    
    @staticmethod
    def find_support_resistance(high: pd.Series, low: pd.Series, close: pd.Series, 
                               window: int = 20) -> Dict[str, pd.Series]:
        """Identify dynamic support and resistance levels"""
        # Rolling highs and lows for resistance and support
        resistance = high.rolling(window=window).max()
        support = low.rolling(window=window).min()
        
        # Distance from current price to support/resistance
        resistance_distance = (resistance - close) / close * 100
        support_distance = (close - support) / close * 100
        
        return {
            'resistance_level': resistance,
            'support_level': support,
            'resistance_distance': resistance_distance,
            'support_distance': support_distance
        }
    
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


class CandlestickPatterns:
    """
    Custom candlestick pattern recognition system.
    """

    @staticmethod
    def doji(open_prices: pd.Series, high_prices: pd.Series,
             low_prices: pd.Series, close_prices: pd.Series, threshold: float = 0.1) -> pd.Series:
        """Doji pattern detection"""
        body_size = np.abs(close_prices - open_prices) / (high_prices - low_prices)
        return (body_size <= threshold).astype(int)

    @staticmethod
    def hammer(open_prices: pd.Series, high_prices: pd.Series,
               low_prices: pd.Series, close_prices: pd.Series) -> pd.Series:
        """Hammer pattern detection"""
        body_size = np.abs(close_prices - open_prices)
        lower_shadow = np.minimum(open_prices, close_prices) - low_prices
        upper_shadow = high_prices - np.maximum(open_prices, close_prices)

        # Hammer conditions
        condition1 = lower_shadow >= 2 * body_size  # Long lower shadow
        condition2 = upper_shadow <= 0.1 * body_size  # Small upper shadow
        condition3 = body_size > 0  # Has a body

        return (condition1 & condition2 & condition3).astype(int)

    @staticmethod
    def engulfing(open_prices: pd.Series, high_prices: pd.Series,
                  low_prices: pd.Series, close_prices: pd.Series) -> Dict[str, pd.Series]:
        """Bullish and Bearish Engulfing patterns"""
        prev_open = open_prices.shift(1)
        prev_close = close_prices.shift(1)

        # Bullish engulfing
        bullish_condition1 = prev_close < prev_open  # Previous candle bearish
        bullish_condition2 = close_prices > open_prices  # Current candle bullish
        bullish_condition3 = open_prices < prev_close  # Current open below prev close
        bullish_condition4 = close_prices > prev_open  # Current close above prev open

        bullish_engulfing = (bullish_condition1 & bullish_condition2 &
                           bullish_condition3 & bullish_condition4).astype(int)

        # Bearish engulfing
        bearish_condition1 = prev_close > prev_open  # Previous candle bullish
        bearish_condition2 = close_prices < open_prices  # Current candle bearish
        bearish_condition3 = open_prices > prev_close  # Current open above prev close
        bearish_condition4 = close_prices < prev_open  # Current close below prev open

        bearish_engulfing = (bearish_condition1 & bearish_condition2 &
                           bearish_condition3 & bearish_condition4).astype(int)

        return {
            'bullish_engulfing': bullish_engulfing,
            'bearish_engulfing': bearish_engulfing
        }
        

class DynamicFileProcessor:
    """
    Handles dynamic processing of all CSV files in the historical_data folder.
    """

    def __init__(self, data_folder: str = None):
        # Load configuration
        self.config = get_config()

        # Set up folders
        self.data_folder = Path(data_folder or self.config['data']['input_folder'])
        self.processed_folder = Path("data/final")
        self.processed_folder.mkdir(exist_ok=True)

        # Initialize analyzers
        self.indicators = PandasTAIndicators()
        self.market_structure = MarketStructureAnalyzer()
        self.pattern_analyzer = CandlestickPatterns()
    
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
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = ta.sma(close_prices, length=period)
            features[f'ema_{period}'] = ta.ema(close_prices, length=period)

        # MACD using pandas-ta
        macd_data = ta.macd(close_prices)
        if macd_data is not None and not macd_data.empty:
            features.update({
                'macd': macd_data['MACD_12_26_9'],
                'macd_signal': macd_data['MACDs_12_26_9'],
                'macd_histogram': macd_data['MACDh_12_26_9']
            })

        # === MOMENTUM INDICATORS ===
        # RSI using pandas-ta
        for period in [14, 21]:
            features[f'rsi_{period}'] = ta.rsi(close_prices, length=period)

        # Stochastic using pandas-ta
        stoch_data = ta.stoch(high_prices, low_prices, close_prices)
        if stoch_data is not None and not stoch_data.empty:
            # Reindex to match the original data length
            stoch_k = stoch_data['STOCHk_14_3_3'].reindex(close_prices.index)
            stoch_d = stoch_data['STOCHd_14_3_3'].reindex(close_prices.index)
            features.update({
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            })

        # Williams %R using pandas-ta
        features['williams_r'] = ta.willr(high_prices, low_prices, close_prices)

        # CCI using pandas-ta
        features['cci'] = ta.cci(high_prices, low_prices, close_prices)

        # ADX using pandas-ta
        adx_data = ta.adx(high_prices, low_prices, close_prices)
        if adx_data is not None and not adx_data.empty:
            features.update({
                'adx': adx_data['ADX_14'],
                'di_plus': adx_data['DMP_14'],
                'di_minus': adx_data['DMN_14']
            })

        # Momentum and ROC using pandas-ta
        features['momentum_10'] = ta.mom(close_prices, length=10)
        features['roc_10'] = ta.roc(close_prices, length=10)

        # TRIX using pandas-ta (returns DataFrame)
        trix_data = ta.trix(close_prices)
        if trix_data is not None and not trix_data.empty:
            features['trix'] = trix_data['TRIX_30_9']

        # === VOLATILITY INDICATORS ===
        # ATR using pandas-ta
        features['atr'] = ta.atr(high_prices, low_prices, close_prices)

        # Bollinger Bands using pandas-ta
        bb_data = ta.bbands(close_prices, length=20)
        if bb_data is not None and not bb_data.empty:
            features.update({
                'bb_upper': bb_data['BBU_20_2.0'],
                'bb_middle': bb_data['BBM_20_2.0'],
                'bb_lower': bb_data['BBL_20_2.0'],
                'bb_width': (bb_data['BBU_20_2.0'] - bb_data['BBL_20_2.0']) / bb_data['BBM_20_2.0'] * 100,
                'bb_position': (close_prices - bb_data['BBL_20_2.0']) / (bb_data['BBU_20_2.0'] - bb_data['BBL_20_2.0']) * 100
            })

        # === MARKET STRUCTURE ===
        # Support/Resistance
        sr_data = self.market_structure.find_support_resistance(high_prices, low_prices, close_prices)
        features.update(sr_data)

        # Trend Analysis
        trend_data = self.market_structure.trend_strength(close_prices)
        features.update(trend_data)

        # === VOLUME INDICATORS REMOVED ===
        # Volume indicators removed as not all data sources have volume data
        # Note: No volume-based features are generated

        # === CANDLESTICK PATTERNS ===
        # Doji
        features['doji'] = self.pattern_analyzer.doji(open_prices, high_prices, low_prices, close_prices)

        # Hammer
        features['hammer'] = self.pattern_analyzer.hammer(open_prices, high_prices, low_prices, close_prices)

        # Engulfing patterns
        engulfing_data = self.pattern_analyzer.engulfing(open_prices, high_prices, low_prices, close_prices)
        features.update(engulfing_data)

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

        # Gap analysis
        features['gap_up'] = np.where(open_prices > close_prices.shift(1),
                                    (open_prices - close_prices.shift(1)) / close_prices.shift(1) * 100, 0)
        features['gap_down'] = np.where(open_prices < close_prices.shift(1),
                                      (close_prices.shift(1) - open_prices) / close_prices.shift(1) * 100, 0)

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
        features['volatility_10'] = close_prices.rolling(10).std() / close_prices.rolling(10).mean() * 100
        features['volatility_20'] = close_prices.rolling(20).std() / close_prices.rolling(20).mean() * 100

        

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

                # Save processed file (replace existing)
                output_path = self.processed_folder / f"features_{file_path.name}"
                # CRITICAL: Save with index=True to preserve datetime index
                processed_df.to_csv(output_path, index=True)

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
