from broker.broker_api import authenticate_fyers
from historical_data_fetcher import fetch_historical_data
from envs.data_fetcher import fetch_candle_data
from envs.preprocessor import DataProcessor
from data_processing.processor import process_all
from config import HIST_DIR, PROCESSED_DIR, INSTRUMENTS, TIMEFRAMES, DAYS, RR_RATIO
import os
import pandas as pd

# Load credentials from environment or config
APP_ID = os.getenv("FY_APP_ID", "TS79V3NXK1-100")
SECRET_KEY = os.getenv("FY_SECRET_KEY", "KQCPB0FJ74")
REDIRECT_URI = os.getenv("FY_REDIRECT_URI", "https://google.com")
FYERS_USER = os.getenv("FYERS_USER", "XM22383")
FYERS_PIN = os.getenv("FYERS_PIN", "4628")
FYERS_TOTP = os.getenv("FYERS_TOTP", "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW")

if __name__ == "__main__":
    fyers, fyers_socket = authenticate_fyers(
        APP_ID, SECRET_KEY, REDIRECT_URI,
        FYERS_USER, FYERS_PIN, FYERS_TOTP
    )
    print("Authenticated with Fyers. Fyers object:", fyers)
    # Test profile fetch
    profile = fyers.get_profile()
    print("Profile:", profile)
    # Fetch and process historical data
    fetch_historical_data()
    process_all()
    # Test fetching candle data
    raw_df = fetch_candle_data(fyers, days=5, index_symbol="NSE:NIFTY50-INDEX", interval_minutes=1)
    processor = DataProcessor(raw_df)
    processed_df = processor.preprocess_datetime().clean_data().df
    print("Processed candle data sample:\n", processed_df.head())
