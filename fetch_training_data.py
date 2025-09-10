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
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from src.auth.fyers_auth_service import authenticate_fyers_user, create_fyers_model, FyersAuthenticationError

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Hardcoded Fyers credentials
APP_ID = "TS79V3NXK1-100"
SECRET_KEY = "KQCPB0FJ74"
REDIRECT_URI = "https://google.com"
FYERS_USER = "XM22383"
FYERS_PIN = "4628"
FYERS_TOTP = "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"

def load_config(config_path="config/instruments.yaml"):
    """Load instruments and timeframes from config file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['instruments'], config['timeframes']

def fetch_candles(fyers, symbol, timeframe, target_count=1000):
    """Fetch target number of candles by progressively expanding date range until target is reached or no more data available"""
    end_date = datetime.now()
    all_data = pd.DataFrame()
    
    # Start with a reasonable initial range based on timeframe
    initial_days = {
        "1": 3,      # 1 minute
        "2": 5,      # 2 minutes  
        "3": 7,      # 3 minutes
        "5": 10,     # 5 minutes
        "10": 15,    # 10 minutes
        "15": 20,    # 15 minutes
        "20": 25,    # 20 minutes
        "30": 35,    # 30 minutes
        "45": 50,    # 45 minutes
        "60": 60,    # 1 hour
        "120": 120,  # 2 hours
        "180": 180,  # 3 hours
        "240": 240   # 4 hours
    }
    
    days = initial_days.get(timeframe, 30)
    attempt = 0
    last_data_count = 0
    
    while len(all_data) < target_count:
        attempt += 1
        start_date = end_date - timedelta(days=days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        df = fetch_candle_data(fyers, symbol, timeframe, start_str, end_str)
        
        if df.empty:
            logger.warning(f"No data for {symbol} {timeframe} with {days} days range")
            break
            
        current_count = len(df)
        
        # Check if we're getting new data or if we've hit the limit
        if current_count <= last_data_count and attempt > 1:
            logger.warning(f"No more data available for {symbol} {timeframe}. Reached maximum available: {current_count} candles")
            all_data = df
            break
            
        all_data = df
        last_data_count = current_count
        
        if current_count >= target_count:
            # Return exactly target_count most recent candles (tail keeps latest data, removes older data)
            return all_data.tail(target_count).reset_index(drop=True)
        else:
            # Need more data, double the range
            days = min(days * 2, 2000)  # Cap at ~5.5 years of data
    
    # Return whatever we managed to get
    final_count = len(all_data)
    if final_count > 0:
        logger.info(f"Fetched {final_count} candles for {symbol} {timeframe} (target was {target_count})")
        return all_data.reset_index(drop=True)
    else:
        logger.warning(f"No data available for {symbol} {timeframe}")
        return pd.DataFrame()

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
    logger.info("Starting training data fetch...")
    
    try:
        # Load instruments and timeframes
        logger.info("Loading configuration...")
        instruments, timeframes = load_config()
        logger.info(f"Loaded {len(instruments)} instruments and {len(timeframes)} timeframes")
        
        # Set the environment variable for the create_fyers_model function
        os.environ["FYERS_APP_ID"] = APP_ID
        
        # Authenticate with Fyers
        logger.info("Authenticating with Fyers...")
        access_token = await authenticate_fyers_user(
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
        data_dir = "data/raw"
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data directory ready: {data_dir}")
        
        # Fetch data for each instrument and timeframe
        total_saved = 0
        for instrument in instruments:
            symbol = instrument["exchange-symbol"]
            instrument_name = instrument["symbol"]
            logger.info(f"Processing instrument: {instrument_name} ({symbol})")
            
            for timeframe_info in timeframes:
                timeframe = timeframe_info["name"]
                timeframe_desc = timeframe_info["description"]
                logger.info(f"  Fetching {timeframe_desc} data...")
                
                # Fetch exactly target number of candles (or as close as possible)  
                df = fetch_candles(fyers, symbol, timeframe, target_count=2500)
                
                if not df.empty:
                    # Set proper column names for the feature generator
                    # Fyers API returns: [timestamp, open, high, low, close, volume]
                    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                    
                    # Save to CSV file with the specified naming convention
                    filename = f"{instrument_name}_{timeframe}.csv"
                    filepath = os.path.join(data_dir, filename)
                    df.to_csv(filepath, index=False)
                    logger.info(f"    Saved {len(df)} candles to {filename}")
                    total_saved += len(df)
                else:
                    logger.warning(f"    No data available for {timeframe_desc}")
                    
        logger.info(f"Data fetch completed! Total candles saved: {total_saved}")
        
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