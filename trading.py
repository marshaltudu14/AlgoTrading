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
import argparse
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
# For regular/backtest mode - using larger dataset for better testing
BACKTEST_DAYS = 365

# For ML training (commented out - using indicator strategies instead)
# TRAINING_DAYS = 365

# Configuration
TIMEFRAME = "5"  # 5-minute timeframe - better Sharpe ratio, more stable

# Trading strategies configuration
STRATEGY = "multi_indicator_combo"  # Default strategy - best performer
MIN_CONFIDENCE = 0.5  # Optimal confidence threshold

# Instrument configuration
DEFAULT_INSTRUMENT_NAME = "Nifty"  # Can be "Bank_Nifty", "Nifty", "Bankex", "Finnifty", "Sensex", etc.

# Trading parameters - P&L based (instrument agnostic) - Fixed configuration
TARGET_PNL = 500  # Target profit in Rs (per position) - FIXED
STOP_LOSS_PNL = -250  # Stop loss in Rs (per position) - FIXED

# Generate unique identifier for filenames
DATA_ID = f"{BACKTEST_DAYS}d_{TIMEFRAME}min"


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
        from backend.config.instruments import INSTRUMENTS
        from backend.src.config.instrument import Instrument

        # Find instrument by name
        instrument_config = None
        for inst in INSTRUMENTS:
            if inst.symbol == DEFAULT_INSTRUMENT_NAME:
                instrument_config = inst
                break

        if instrument_config is None:
            print(f"[ERROR] Instrument '{DEFAULT_INSTRUMENT_NAME}' not found!")
            print(f"Available instruments: {[inst.symbol for inst in INSTRUMENTS]}")
            return False

        # Create Instrument object with correct parameters
        instrument = Instrument(
            symbol=instrument_config.exchangeSymbol,
            lot_size=instrument_config.lotSize,
            tick_size=instrument_config.tickSize,
            instrument_type=instrument_config.type,
            option_premium_range=[0.5, 5.0]
        )

        # Calculate position size
        if profile:
            available_capital = profile['available_balance']
            trade_capital = available_capital * 0.10  # 10% of capital
            position_size = max(1, int(trade_capital / (instrument.lot_size * 200)))

            print(f"[OK] Instrument: {instrument_config.name}")
            print(f"[OK] Lot Size: {instrument.lot_size}")
            print(f"[OK] Trade Capital: Rs.{trade_capital:,.2f}")
            print(f"[OK] Position Size: {position_size} lots")

        return True

    except Exception as e:
        print(f"[ERROR] Instrument configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def fetch_historical_data(days=None):
    global historical_data

    try:
        if not fyers_model or not instrument:
            print("[ERROR] Please authenticate and configure instrument first")
            return False

        # Use provided days or default to BACKTEST_DAYS
        days_to_fetch = days if days is not None else BACKTEST_DAYS

        print(f"Fetching {days_to_fetch} days of historical data with {TIMEFRAME}-minute timeframe...")

        from backend.fetch_candle_data import fetch_candles

        historical_data = fetch_candles(
            fyers=fyers_model,
            symbol=instrument.symbol,
            timeframe=TIMEFRAME,
            days=days_to_fetch
        )

        if historical_data.empty:
            print("[ERROR] No data received")
            return False

        print(f"[OK] Fetched {len(historical_data)} candles")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        import traceback
        traceback.print_exc()
        return False


def preprocess_data():
    global processed_data

    try:
        if historical_data is None or historical_data.empty:
            print("[ERROR] No historical data to process")
            return False

        print("Using data processing pipeline...")

        input_dir = "backend/data/raw"
        output_dir = "backend/data/processed"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        csv_path = f"{input_dir}/nifty_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"
        historical_data.to_csv(csv_path, index=False)

        from backend.src.data_processing.pipeline import DataProcessingPipeline

        pipeline = DataProcessingPipeline()

        results = pipeline.run_feature_generation(
            input_dir=input_dir,
            output_dir=output_dir,
            parallel=False
        )

        if results.get('success'):
            output_data_path = Path(output_dir)
            feature_files = list(output_data_path.glob("features_*.csv"))

            # Get the most recent feature file that matches our data ID
            matching_files = [f for f in feature_files if f"nifty_{BACKTEST_DAYS}d_{TIMEFRAME}min" in f.name]

            if matching_files:
                # Use the file that matches our historical days
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                processed_data = pd.read_csv(latest_file)
                print(f"[OK] Processed data shape: {processed_data.shape}")
                print(f"[OK] Using data from {latest_file}")

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
    """ML Model Training - DISABLED - Using indicator strategies instead"""
    print("\n--- Step 5: ML Model Training ---")
    print("[INFO] ML model training is disabled - Using indicator-based strategies")
    print("[INFO] Strategies available: rsi_mean_reversion, macd_crossover, bollinger_bands, ema_crossover,")
    print("                     supertrend, stochastic_oversold, volume_price_trend, atr_breakout,")
    print("                     vwap_reversal, multi_indicator_combo")
    print(f"[INFO] Current strategy: {STRATEGY}")
    return True


def parse_arguments():
    parser = argparse.ArgumentParser(description='Algo Trading Pipeline')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode')
    parser.add_argument('--train', action='store_true',
                        help='Run ML training only')
    return parser.parse_args()


async def real_trading():
    print("\n=== REAL TRADING MODE ===")
    print("\n[INFO] Real trading is currently disabled for safety")
    print(f"[INFO] Current Price: NIFTY - Check live prices")
    print(f"[INFO] Target: {TARGET_POINTS} points")
    print(f"[INFO] Stop Loss: {STOP_LOSS_POINTS} points")
    print("\n[WARNING] Please implement real trading logic with proper risk management")
    print("[WARNING] Ensure to test thoroughly with paper trading first!")
    return True


async def backtest():
    print(f"\n=== BACKTESTING MODE ===")
    print(f"[INFO] Using Strategy: {STRATEGY}")
    print(f"[INFO] Target P&L: Rs.{TARGET_PNL}")
    print(f"[INFO] Stop Loss P&L: Rs.{STOP_LOSS_PNL}")
    print(f"[INFO] Minimum Confidence: {MIN_CONFIDENCE}")

    from backend.src.strategies.rule_based import RuleBasedBacktester

    try:
        # Use backtest data - get the processed file directly
        output_dir = "backend/data/processed"
        feature_files = list(Path(output_dir).glob("features_*.csv"))
        matching_files = [f for f in feature_files if f"nifty_{BACKTEST_DAYS}d_{TIMEFRAME}min" in f.name]

        if matching_files:
            csv_path = max(matching_files, key=lambda x: x.stat().st_mtime)
        else:
            print(f"[ERROR] No processed backtest data found")
            return False

        # Create rule-based backtester
        backtester = RuleBasedBacktester(
            target_pnl=TARGET_PNL,
            stop_loss_pnl=STOP_LOSS_PNL,
            initial_capital=20000,
            brokerage_entry=25,
            brokerage_exit=25
        )

        # Set trade log path for real-time saving
        trade_log_path = f"backend/data/backtest_trades_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"
        backtester.trade_log_path = trade_log_path

        results = backtester.run_backtest(
            csv_path,
            strategy=STRATEGY,
            min_confidence=MIN_CONFIDENCE,
            instrument=instrument
        )

        if results:
            print(f"[OK] Trade log saved to {trade_log_path}")

            print("\n[BACKTEST RESULTS]")
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Winning Trades: {results.get('winning_trades', 0)}")
            print(f"Losing Trades: {results.get('losing_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0):.2%}")
            print(f"Total P&L (after brokerage): Rs.{results.get('total_pnl', 0):,.2f}")
            print(f"Total P&L %: {results.get('total_pnl_percent', 0):.1f}%")
            print(f"Max Drawdown: Rs.{results.get('max_drawdown', 0):,.2f}")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Average Trade P&L: Rs.{results.get('avg_trade_pnl', 0):,.2f}")
            print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")

            return True
        else:
            print("[ERROR] Backtest failed to produce results")
            return False

    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    args = parse_arguments()

    mode = 'TRAIN' if args.train else ('BACKTEST' if args.backtest else 'REAL')
    print("\n=== Algo Trading Pipeline ===\n")
    print(f"Mode: {mode}")
    print(f"Timeframe: {TIMEFRAME}-minute")
    print(f"Historical Days: {BACKTEST_DAYS}")
    print(f"Target P&L: Rs.{TARGET_PNL}")
    print(f"Stop Loss P&L: Rs.{STOP_LOSS_PNL}")

    # For backtest mode, check if processed data exists to skip authentication
    if args.backtest:
        output_dir = "backend/data/processed"
        feature_files = list(Path(output_dir).glob("features_*.csv"))
        matching_files = [f for f in feature_files if f"nifty_{BACKTEST_DAYS}d_{TIMEFRAME}min" in f.name]

        if matching_files:
            print("\n[OK] Found existing processed data - skipping authentication and data fetch")
            print("\n--- Step 4: Data Preprocessing ---")
            print(f"[OK] Using existing processed data: {matching_files[0].name}")

            # Configure instrument with defaults since we can't get capital without auth
            print("\n--- Step 2: Instrument Configuration (Default) ---")
            if not configure_instrument():
                print("[WARNING] Using default instrument configuration")

            # Skip ML training for backtest
            print("\n--- Step 6: Execution ---")
            await backtest()
            return

    # If no existing data or not backtest mode, proceed with full pipeline
    if not await authenticate():
        return

    print("\n--- Step 2: Instrument Configuration ---")
    if not configure_instrument():
        return

    if args.train:
        # Train mode - only train ML models
        print("\n--- Step 3: ML Model Training ---")
        if not train_ml_models():
            return
        print("\n[OK] Training completed successfully!")
        return

    print("\n--- Step 3: Fetch Historical Data ---")
    if not fetch_historical_data():
        return

    print("\n--- Step 4: Data Preprocessing ---")
    # Check if processed data already exists
    output_dir = "backend/data/processed"
    feature_files = list(Path(output_dir).glob("features_*.csv"))
    matching_files = [f for f in feature_files if f"nifty_{BACKTEST_DAYS}d_{TIMEFRAME}min" in f.name]

    if not matching_files:
        print("[INFO] No processed data found, running preprocessing...")
        if not preprocess_data():
            return
    else:
        print(f"[OK] Using existing processed data: {matching_files[0].name}")

    # ML models are disabled - using indicator strategies instead
    # Only show training info if explicitly requested
    if not args.backtest:
        if not train_ml_models():
            return

    print("\n--- Step 6: Execution ---")
    if args.backtest:
        await backtest()
    else:
        await real_trading()

    print("\n[OK] Pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())