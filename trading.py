#!/usr/bin/env python3
"""
Minimal Algo Trading Script
============================

A minimal script for algorithmic trading with Fyers API.
Steps:
1. Authentication
2. Instrument configuration
3. Fetch historical data
4. Preprocess data
"""

import sys
import os
import asyncio
import logging
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path

# Add project root and backend src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'backend', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
access_token = None
fyers_model = None
profile = None
instrument = None
historical_data = None
processed_data = None

# Configuration
HISTORICAL_DAYS = 365  # Can be changed to fetch different number of days
ENABLE_ML_PREDICTION = True  # Enable/disable ML prediction step


def save_token_to_cache(token_data):
    """Save token and profile to cache file"""
    cache_file = "backend/data/token_cache.json"
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    cache_data = {
        "access_token": token_data.get("access_token"),
        "profile": token_data.get("profile"),
        "timestamp": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(hours=23, minutes=50)).isoformat()  # Token is valid for 24 hours
    }

    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    print(f"[OK] Token cached for {cache_data['expires_at']}")


def load_token_from_cache():
    """Load token from cache if valid"""
    cache_file = "backend/data/token_cache.json"

    try:
        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        # Check if token is still valid
        expires_at = datetime.fromisoformat(cache_data['expires_at'])
        if datetime.now() > expires_at:
            print("[INFO] Cached token expired")
            os.remove(cache_file)
            return None

        print("[OK] Using cached token")
        return cache_data

    except Exception as e:
        print(f"[WARNING] Failed to load cached token: {e}")
        return None


async def authenticate():
    """Authenticate with Fyers API with token caching"""
    global access_token, fyers_model, profile

    try:
        # Import configs at the top
        from config.fyers_config import APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP

        # Try to load from cache first
        cached_data = load_token_from_cache()
        if cached_data:
            access_token = cached_data['access_token']
            profile = cached_data['profile']
            from auth.fyers_auth_service import create_fyers_model
            fyers_model = create_fyers_model(access_token, APP_ID)

            print(f"[OK] Authenticated (cached): {profile['name']}")
            print(f"[OK] Capital: Rs.{profile['capital']:,.2f}")
            return True

        # If no valid cache, authenticate fresh
        from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile, create_fyers_model

        print("Authenticating with Fyers...")

        # Get access token
        access_token = await authenticate_fyers_user(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )

        if access_token:
            # Get user profile
            profile = await get_user_profile(access_token, APP_ID)
            fyers_model = create_fyers_model(access_token, APP_ID)

            # Save to cache
            save_token_to_cache({
                "access_token": access_token,
                "profile": profile
            })

            print(f"[OK] Authenticated: {profile['name']}")
            print(f"[OK] Capital: Rs.{profile['capital']:,.2f}")
            return True
        return False

    except Exception as e:
        print(f"[ERROR] Authentication failed: {e}")
        return False


def configure_instrument():
    """Configure trading instrument"""
    global instrument

    try:
        from config.instrument import Instrument
        from config.fyers_config import INDEX_SYMBOLS, DEFAULT_LOT_SIZE

        # Define NIFTY instrument
        instrument = Instrument(
            symbol=INDEX_SYMBOLS["NIFTY"],
            lot_size=DEFAULT_LOT_SIZE,
            tick_size=0.05,
            instrument_type="index",
            option_premium_range=[0.5, 5.0]
        )

        # Calculate position size
        if profile:
            available_capital = profile['available_balance']
            trade_capital = available_capital * 0.10  # 10% of capital
            position_size = max(1, int(trade_capital / (instrument.lot_size * 200)))

            print(f"[OK] Instrument: NIFTY")
            print(f"[OK] Lot Size: {instrument.lot_size}")
            print(f"[OK] Trade Capital: Rs.{trade_capital:,.2f}")
            print(f"[OK] Position Size: {position_size} lots")

        return True

    except Exception as e:
        print(f"[ERROR] Instrument configuration failed: {e}")
        return False


def fetch_historical_data():
    """Fetch 365 days of historical data using existing fetch_candles function"""
    global historical_data

    try:
        if not fyers_model or not instrument:
            print("[ERROR] Please authenticate and configure instrument first")
            return False

        print(f"Fetching {HISTORICAL_DAYS} days of historical data...")

        # Import the existing fetch_candles function
        from backend.fetch_candle_data import fetch_candles

        # Fetch data using the existing function
        historical_data = fetch_candles(
            fyers=fyers_model,
            symbol=instrument.symbol,
            timeframe="D",
            days=HISTORICAL_DAYS
        )

        if historical_data.empty:
            print("[ERROR] No data received")
            return False

        print(f"[OK] Fetched {len(historical_data)} days of data")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_data():
    """Preprocess data using the existing data processing pipeline"""
    global processed_data

    try:
        if historical_data is None or historical_data.empty:
            print("[ERROR] No historical data to process")
            return False

        print("Using data processing pipeline...")

        # Save historical data to CSV for processing
        input_dir = "backend/data/raw"
        output_dir = "backend/data/processed"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{input_dir}/nifty_{HISTORICAL_DAYS}d.csv"
        historical_data.to_csv(csv_path)

        # Use the existing pipeline class to process the data
        from backend.src.data_processing.pipeline import DataProcessingPipeline

        pipeline = DataProcessingPipeline()

        # Run the feature generation step
        results = pipeline.run_feature_generation(
            input_dir=input_dir,
            output_dir=output_dir,
            parallel=False  # Process single file
        )

        if results.get('success'):
            # Find the processed file
            output_data_path = Path(output_dir)
            feature_files = list(output_data_path.glob("features_*.csv"))

            if feature_files:
                # Load the processed data
                processed_data = pd.read_csv(feature_files[0])
                print(f"[OK] Processed data shape: {processed_data.shape}")

                # Save processed data as CSV
                csv_output_path = "backend/data/processed_nifty_data.csv"
                processed_data.to_csv(csv_output_path)
                print(f"[OK] Data saved to {csv_output_path}")

                return True
            else:
                print("[ERROR] No processed files found")
                return False
        else:
            print(f"[ERROR] Pipeline failed: {results.get('error')}")
            return False

    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_ml_models():
    """Train ML models for direction and volatility prediction"""
    global processed_data

    try:
        if not ENABLE_ML_PREDICTION:
            print("[INFO] ML prediction is disabled")
            return True

        if processed_data is None or processed_data.empty:
            print("[ERROR] No processed data available for ML training")
            return False

        print("\n--- Step 5: ML Model Training ---")

        # Import the ML predictor
        from backend.src.ml.predictor import train_and_evaluate

        print("Training Random Forest models...")

        # Train models
        predictor = train_and_evaluate(
            csv_path="backend/data/processed_nifty_data.csv",
            save_path="backend/data/trading_models.pkl"
        )

        print("[OK] ML models trained and saved successfully!")

        # Make prediction for the most recent data
        latest_prediction = predictor.predict_latest(processed_data)

        print(f"\nLatest Prediction:")
        print(f"  Current Price: Rs.{latest_prediction['current_price']:.2f}")
        print(f"  Direction: {'UP' if latest_prediction['direction_signal'] == 1 else 'DOWN'}")
        print(f"  Expected Move: {latest_prediction['direction_prediction']*100:.2f}%")
        print(f"  Volatility: {latest_prediction['volatility_prediction']:.2f}%")
        print(f"  Confidence: {latest_prediction['prediction_confidence']*100:.1f}%")

        return True

    except Exception as e:
        print(f"[ERROR] ML training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main execution function"""
    print("\n=== Algo Trading Pipeline ===\n")

    # Step 1: Authentication
    if not await authenticate():
        return

    # Step 2: Configure instrument
    print("\n--- Step 2: Instrument Configuration ---")
    if not configure_instrument():
        return

    # Step 3: Fetch historical data
    print("\n--- Step 3: Fetch Historical Data ---")
    if not fetch_historical_data():
        return

    # Step 4: Preprocess data
    print("\n--- Step 4: Data Preprocessing ---")
    if not preprocess_data():
        return

    # Step 5: ML Model Training (optional)
    if ENABLE_ML_PREDICTION:
        if not train_ml_models():
            return

    print("\n[OK] All steps completed successfully!")
    print("\nReady for algorithmic trading!")


if __name__ == "__main__":
    asyncio.run(main())