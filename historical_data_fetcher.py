import os
import pandas as pd
from broker.broker_api import authenticate_fyers
from envs.data_fetcher import fetch_train_candle_data
from config import HIST_DIR as OUTPUT_DIR, INSTRUMENTS, TIMEFRAMES, DAYS, APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP

def fetch_historical_data():
    """Fetch and save historical candle data for all instruments and timeframes"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fyers, _ = authenticate_fyers(
        APP_ID, SECRET_KEY, REDIRECT_URI,
        FYERS_USER, FYERS_PIN, FYERS_TOTP
    )
    for name, symbol in INSTRUMENTS.items():
        for tf in TIMEFRAMES:
            fname = f"{name.replace(' ', '_')}_{tf}.csv"
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                print(f"{fname} exists, skipping fetch.")
                continue
            print(f"Fetching {name} @ {tf}m -> {fname}")
            df = fetch_train_candle_data(
                fyers, total_days=DAYS,
                index_symbol=symbol,
                interval_minutes=tf
            )
            df.to_csv(path, index=False)
    print("All historical files saved in", OUTPUT_DIR)

if __name__ == "__main__":
    fetch_historical_data()
