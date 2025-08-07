#!/usr/bin/env python3
"""
Real-time Data Loader for Backtesting
Fetches live data from Fyers API and processes it through the existing pipeline
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.trading.fyers_client import FyersClient
from src.data_processing.feature_generator import DynamicFileProcessor
from src.utils.instrument_loader import load_instruments

logger = logging.getLogger(__name__)

class RealtimeDataLoader:
    """
    Loads real-time data from Fyers API and processes it for backtesting.
    Integrates with existing feature engineering pipeline.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, access_token: Optional[str] = None, app_id: Optional[str] = None):
        """
        Initialize the real-time data loader.

        Args:
            config (dict): Configuration dictionary with backtesting parameters
            access_token (str): Fyers access token for authentication
            app_id (str): Fyers app ID for authentication
        """
        self.config = config or {}
        self.fyers_client = FyersClient(access_token=access_token, app_id=app_id)
        self.feature_processor = DynamicFileProcessor()
        self.instruments = load_instruments('config/instruments.yaml')
        
        # Default configuration
        self.default_config = {
            'symbol': 'Nifty',
            'timeframe': '5',  # 5 minutes
            'days': 30,
            'min_data_points': 100,  # Minimum data points required
            'feature_columns_required': 60  # Expected number of features
        }
        
        # Merge with provided config
        self.config = {**self.default_config, **self.config}
        
    def fetch_and_process_data(self, symbol: str = None, timeframe: str = None, days: int = None) -> Optional[pd.DataFrame]:
        """
        Fetch real-time data and process it through the feature engineering pipeline.
        
        Args:
            symbol (str): Symbol to fetch (overrides config)
            timeframe (str): Timeframe in minutes (overrides config)
            days (int): Number of days to fetch (overrides config)
            
        Returns:
            pd.DataFrame: Processed data ready for backtesting, or None if failed
        """
        # Use provided parameters or fall back to config
        symbol = symbol or self.config['symbol']
        timeframe = timeframe or self.config['timeframe']
        days = days or self.config['days']
        
        logger.info(f"🔄 Starting real-time data fetch and processing...")
        logger.info(f"   Symbol: {symbol}, Timeframe: {timeframe}min, Days: {days}")
        
        try:
            # Step 1: Fetch raw data from Fyers API
            raw_data = self._fetch_raw_data(symbol, timeframe, days)
            if raw_data is None or raw_data.empty:
                logger.error("Failed to fetch raw data from Fyers API")
                return None
                
            # Step 2: Validate raw data
            if not self._validate_raw_data(raw_data):
                logger.error("Raw data validation failed")
                return None
                
            # Step 3: Process through feature engineering pipeline
            processed_data = self._process_features(raw_data, symbol)
            if processed_data is None or processed_data.empty:
                logger.error("Feature processing failed")
                return None
                
            # Step 4: Final validation
            if not self._validate_processed_data(processed_data):
                logger.error("Processed data validation failed")
                return None
                
            logger.info(f"✅ Successfully processed {len(processed_data)} data points")
            logger.info(f"   Features: {processed_data.shape[1]} columns")
            logger.info(f"   Date range: {processed_data.index[0]} to {processed_data.index[-1]}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error in fetch_and_process_data: {e}", exc_info=True)
            return None
    
    def _fetch_raw_data(self, symbol: str, timeframe: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch raw OHLCV data from Fyers API."""
        try:
            logger.info(f"📡 Fetching raw data from Fyers API...")

            # Map frontend symbol to Fyers symbol format
            fyers_symbol = self._get_fyers_symbol(symbol)
            if not fyers_symbol:
                logger.error(f"No Fyers symbol mapping found for {symbol}")
                return None

            logger.info(f"Using Fyers symbol: {fyers_symbol}")

            raw_data = self.fyers_client.get_backtesting_data(
                symbol=fyers_symbol,
                timeframe=timeframe,
                days=days
            )
            
            if raw_data.empty:
                logger.error("No data received from Fyers API")
                return None
                
            logger.info(f"📊 Received {len(raw_data)} raw candles")
            return raw_data
            
        except Exception as e:
            logger.error(f"Error fetching raw data: {e}")
            return None

    def _get_fyers_symbol(self, symbol: str) -> Optional[str]:
        """Map frontend symbol to Fyers symbol format."""
        try:
            # Load instruments config directly to get exchange-symbol
            import yaml
            from pathlib import Path

            config_path = Path(__file__).parent.parent.parent / "config" / "instruments.yaml"
            if config_path.exists():
                with open(config_path, 'r') as file:
                    config_data = yaml.safe_load(file)

                # Find instrument by symbol
                for instrument in config_data.get('instruments', []):
                    if instrument.get('symbol') == symbol:
                        return instrument.get('exchange-symbol', symbol)

            # Fallback mapping for common symbols
            symbol_map = {
                "Bank_Nifty": "NSE:NIFTYBANK-INDEX",
                "Nifty": "NSE:NIFTY50-INDEX",
                "Bankex": "NSE:BANKEX-INDEX",
                "Finnifty": "NSE:FINNIFTY-INDEX",
                "Sensex": "BSE:SENSEX-INDEX",
                "Reliance": "NSE:RELIANCE-EQ",
                "TCS": "NSE:TCS-EQ",
                "HDFC": "NSE:HDFCBANK-EQ"
            }

            return symbol_map.get(symbol, symbol)

        except Exception as e:
            logger.error(f"Error mapping symbol {symbol}: {e}")
            return symbol
    
    def _validate_raw_data(self, data: pd.DataFrame) -> bool:
        """Validate raw OHLCV data."""
        try:
            # Check minimum data points
            if len(data) < self.config['min_data_points']:
                logger.error(f"Insufficient data points: {len(data)} < {self.config['min_data_points']}")
                return False
                
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Check for null values
            null_counts = data.isnull().sum()
            if null_counts.any():
                logger.warning(f"Found null values: {null_counts[null_counts > 0].to_dict()}")
                
            # Check for invalid prices
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                logger.error("Found invalid (zero or negative) prices")
                return False
                
            logger.info("✅ Raw data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating raw data: {e}")
            return False
    
    def _process_features(self, raw_data: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Process raw data through feature engineering pipeline."""
        try:
            logger.info(f"⚙️ Processing features through pipeline...")

            # Process through feature generator directly with DataFrame
            processed_data = self.feature_processor.process_dataframe(raw_data)

            if processed_data is None or processed_data.empty:
                logger.error("Feature processing returned empty data")
                return None

            logger.info(f"✅ Feature processing completed: {processed_data.shape}")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing features: {e}", exc_info=True)
            return None
    
    def _get_instrument_config(self, symbol: str) -> Dict[str, Any]:
        """Get instrument configuration for the symbol."""
        # Map common symbols to generic data configurations
        symbol_lower = symbol.lower()

        if 'banknifty' in symbol_lower or 'nifty' in symbol_lower:
            return {
                'symbol': symbol,
                'lot_size': 25,    # Bank Nifty lot size
                'tick_size': 0.05
            }
        else:
            return {
                'symbol': symbol,
                'lot_size': 1,
                'tick_size': 0.05
            }
    
    def _validate_processed_data(self, data: pd.DataFrame) -> bool:
        """Validate processed feature data."""
        try:
            # Check minimum data points after processing
            if len(data) < self.config['min_data_points'] // 2:  # Allow for some data loss during processing
                logger.error(f"Insufficient processed data points: {len(data)}")
                return False
                
            # Check for excessive null values
            null_percentage = (data.isnull().sum() / len(data) * 100)
            high_null_columns = null_percentage[null_percentage > 50]
            if not high_null_columns.empty:
                logger.warning(f"Columns with >50% null values: {high_null_columns.to_dict()}")
                
            # Check if we have reasonable number of features
            if data.shape[1] < 20:  # Minimum expected features
                logger.warning(f"Low feature count: {data.shape[1]} columns")
                
            logger.info("✅ Processed data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating processed data: {e}")
            return False
    
    def get_data_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Convenience method to get processed data for a specific symbol.
        Uses default configuration.
        """
        return self.fetch_and_process_data(symbol=symbol)
    
    def get_default_backtesting_data(self) -> Optional[pd.DataFrame]:
        """
        Get default backtesting data (Bank Nifty, 5min, 30 days).
        """
        return self.fetch_and_process_data()
