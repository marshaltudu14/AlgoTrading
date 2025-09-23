"""
Script to fetch training data from Fyers API
Fetches last 1000 candle data for instruments and timeframes specified in config
"""
import os
import sys
import yaml
import asyncio
import pandas as pd
import logging
import argparse
import time
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from src.auth.fyers_auth_service import get_access_token, create_fyers_model, FyersAuthenticationError
from src.config.fyers_config import (
    APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP
)


# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fix numpy compatibility issue for pandas_ta
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Import for candlestick chart generation
import mplfinance as mpf

# Try to import the feature generator, but handle compatibility issues
try:
    from src.data_processing.feature_generator import DynamicFileProcessor
    FEATURE_GENERATOR_AVAILABLE = True
    logger.info("Feature generator loaded successfully")
except ImportError as e:
    logger.warning(f"Feature generator not available due to compatibility issues: {e}")
    FEATURE_GENERATOR_AVAILABLE = False

# Fyers credentials are now imported from config module

def load_config(config_path="config/instruments.yaml"):
    """Load instruments and timeframes from config file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['instruments'], config['timeframes']

def process_data_with_features(df):
    """Process raw data using the existing feature generator"""
    if df.empty:
        return df

    if not FEATURE_GENERATOR_AVAILABLE:
        logger.warning("Feature generator not available, returning original data")
        return df

    try:
        # Use the existing DynamicFileProcessor
        processor = DynamicFileProcessor(data_folder="temp")

        # Process the dataframe directly
        result = processor.process_dataframe(df)

        if result is not None:
            logger.info(f"Successfully processed data using feature generator: {len(result)} rows, {len(result.columns)} features")
            return result
        else:
            logger.warning("Feature generator returned None, returning original data")
            return df

    except Exception as e:
        logger.error(f"Error processing data with feature generator: {e}")
        logger.info("Returning original data without features")
        return df


def create_candlestick_chart(df, filepath):
    """Create and save a candlestick chart from OHLC data"""
    try:
        if df.empty:
            logger.warning("Cannot create chart: Empty dataframe")
            return

        # Create a copy of the dataframe with proper column names for mplfinance
        chart_data = df[['datetime', 'open', 'high', 'low', 'close']].copy()
        
        # Convert epoch timestamp to datetime
        # Fyers API returns timestamp in seconds, so we need to specify unit='s'
        chart_data['datetime'] = pd.to_datetime(chart_data['datetime'], unit='s')
        
        # Set datetime as index
        chart_data.set_index('datetime', inplace=True)
        
        # Create the chart
        mpf.plot(
            chart_data,
            type='candle',
            style='charles',
            title='Candlestick Chart',
            ylabel='Price',
            volume=False,
            figsize=(12, 8),
            savefig=filepath
        )
        
        logger.info(f"Saved candlestick chart to {filepath}")
    except Exception as e:
        logger.error(f"Error creating candlestick chart: {e}")


def fetch_candles(fyers, symbol, timeframe, start_date=None, end_date=None):
    """Fetch candle data for specified date range, handling API limits"""
    if end_date is None:
        end_date = datetime.now()

    if start_date is None:
        # Default to 30 days if no start date specified
        start_date = end_date - timedelta(days=30)

    all_data = pd.DataFrame()
    current_start = start_date
    current_end = end_date

    # For timeframes with 100-day limit, we need to chunk the requests
    limited_timeframes = ["1", "2", "3", "5", "10", "15", "20", "30", "45", "60", "120", "180", "240"]

    if timeframe in limited_timeframes:
        max_days = 90  # Stay under the 100-day limit
        chunk_size = timedelta(days=max_days)

        while current_start <= current_end:
            chunk_end = min(current_start + chunk_size, current_end)

            start_str = current_start.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"Fetching chunk from {start_str} to {end_str}")

            df = fetch_candle_data(fyers, symbol, timeframe, start_str, end_str)

            if not df.empty:
                all_data = pd.concat([all_data, df], ignore_index=True)
            else:
                logger.warning(f"No data for chunk {start_str} to {end_str}")

            # Move to next chunk
            current_start = chunk_end + timedelta(days=1)

            # Small delay to avoid rate limiting
            time.sleep(0.1)
    else:
        # For other timeframes, fetch all at once
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching data from {start_str} to {end_str}")

        df = fetch_candle_data(fyers, symbol, timeframe, start_str, end_str)
        all_data = df

    if all_data.empty:
        logger.warning(f"No data available for {symbol} {timeframe}")
        return pd.DataFrame()

    # Remove duplicates and sort by timestamp
    all_data = all_data.drop_duplicates().sort_values(0).reset_index(drop=True)

    logger.info(f"Fetched total of {len(all_data)} candles for {symbol} {timeframe}")
    return all_data

def fetch_candle_data(fyers, symbol, timeframe, start_date, end_date):
    """Fetch candle data from Fyers API"""
    try:
        data = {
            "symbol": symbol,
            "resolution": timeframe,
            "date_format": "1",  # DD-MM-YYYY format
            "range_from": start_date,
            "range_to": end_date,
            "cont_flag": "1"  # Continuous data
        }

        response = fyers.history(data)

        if response.get("s") == "ok":
            candles = response.get("candles", [])
            if candles:
                # Convert to DataFrame - let Fyers API provide the column structure
                df = pd.DataFrame(candles)
                return df
            else:
                print(f"No data returned for {symbol} timeframe {timeframe}")
                return pd.DataFrame()
        else:
            print(f"Error fetching data for {symbol} timeframe {timeframe}: {response}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Exception while fetching data for {symbol} timeframe {timeframe}: {str(e)}")
        return pd.DataFrame()

async def main():
    """Main function to fetch training data"""
    parser = argparse.ArgumentParser(description="Fetch training data from Fyers API")
    parser.add_argument("--symbol", required=True, help="Exchange symbol (e.g., NSE:NIFTY50-INDEX)")
    parser.add_argument("--timeframe", required=True, help="Timeframe (e.g., 1, 5, 15, 60)")
    parser.add_argument("--start_date", help="Start date (YYYY-MM-DD format)")
    parser.add_argument("--end_date", help="End date (YYYY-MM-DD format)")

    args = parser.parse_args()

    logger.info("Starting training data fetch...")

    try:
        # Set the environment variable for the create_fyers_model function
        os.environ["FYERS_APP_ID"] = APP_ID

        # Get access token (cached or fresh)
        logger.info("Getting access token...")
        access_token = await get_access_token(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )
        logger.info("Authentication successful!")

        # Create Fyers model instance
        fyers = create_fyers_model(access_token)

        # Create data directory if it doesn't exist
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data directory ready: {data_dir}")

        # Parse date arguments if provided
        start_date = None
        end_date = None

        if args.start_date:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

        if args.end_date:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

        logger.info(f"Fetching data for symbol: {args.symbol}, timeframe: {args.timeframe}")
        if start_date:
            logger.info(f"Start date: {start_date.strftime('%Y-%m-%d')}")
        if end_date:
            logger.info(f"End date: {end_date.strftime('%Y-%m-%d')}")

        # Fetch data
        df = fetch_candles(fyers, args.symbol, args.timeframe, start_date, end_date)

        if not df.empty:
            # Set proper column names for the feature generator
            # Fyers API returns: [timestamp, open, high, low, close, volume]
            df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

            # Create candlestick chart before processing
            chart_filepath = os.path.join(data_dir, "candlestick_chart.png")
            logger.info("Creating candlestick chart...")
            create_candlestick_chart(df, chart_filepath)

            logger.info("Processing data with technical indicators...")
            processed_df = process_data_with_features(df)

            # Use default hardcoded filename
            filename = "candle_data.csv"
            filepath = os.path.join(data_dir, filename)
            processed_df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(processed_df)} processed candles to {filename}")
            logger.info(f"Total features: {len(processed_df.columns)}")
        else:
            logger.warning("No data available for the specified parameters")

    except FyersAuthenticationError as e:
        logger.error(f"Authentication error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error during data fetch: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())