"""
Script to fetch training data from Fyers API
Fetches last 1000 candle data for instruments and timeframes specified in config
"""
import os
import sys
import yaml
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from src.auth.fyers_auth_service import authenticate_fyers_user, create_fyers_model

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

def get_date_range(timeframe_name):
    """Calculate date range based on timeframe to get approximately 1000 candles"""
    # Map timeframe names to approximate days needed for 1000 candles
    timeframe_days = {
        "1": 1,      # 1 minute - ~1 day (600 candles in 10 hours)
        "2": 2,      # 2 minutes - ~2 days
        "3": 3,      # 3 minutes - ~3 days
        "5": 5,      # 5 minutes - ~5 days
        "10": 8,     # 10 minutes - ~8 days
        "15": 12,    # 15 minutes - ~12 days
        "20": 16,    # 20 minutes - ~16 days
        "30": 24,    # 30 minutes - ~24 days
        "45": 36,    # 45 minutes - ~36 days
        "60": 48,    # 1 hour - ~48 days
        "120": 96,   # 2 hours - ~96 days
        "180": 144,  # 3 hours - ~144 days
        "240": 192   # 4 hours - ~192 days
    }
    
    days = timeframe_days.get(timeframe_name, 30)  # Default to 30 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

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
                # Convert to DataFrame
                df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
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
    print("Starting training data fetch...")
    
    # Load instruments and timeframes
    instruments, timeframes = load_config()
    
    try:
        # Authenticate with Fyers
        print("Authenticating with Fyers...")
        access_token = await authenticate_fyers_user(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )
        
        # Create Fyers model instance
        # Set the environment variable for the create_fyers_model function
        os.environ["FYERS_APP_ID"] = APP_ID
        fyers = create_fyers_model(access_token)
        
        # Create data directory if it doesn't exist
        data_dir = "data/raw"
        os.makedirs(data_dir, exist_ok=True)
        
        # Fetch data for each instrument and timeframe
        total_saved = 0
        for instrument in instruments:
            symbol = instrument["exchange-symbol"]
            instrument_name = instrument["symbol"]
            print(f"\nProcessing instrument: {instrument_name} ({symbol})")
            
            for timeframe_info in timeframes:
                timeframe = timeframe_info["name"]
                timeframe_desc = timeframe_info["description"]
                print(f"  Fetching {timeframe_desc} data...")
                
                # Get date range for this timeframe
                start_date, end_date = get_date_range(timeframe)
                
                # Fetch candle data
                df = fetch_candle_data(fyers, symbol, timeframe, start_date, end_date)
                
                if not df.empty:
                    # Save to CSV file with the specified naming convention
                    filename = f"{instrument_name}_{timeframe}.csv"
                    filepath = os.path.join(data_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"    Saved {len(df)} candles to {filename}")
                    total_saved += len(df)
                else:
                    print(f"    No data available for {timeframe_desc}")
                    
        print(f"\nData fetch completed! Total candles saved: {total_saved}")
        
    except Exception as e:
        print(f"Error during data fetch: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())