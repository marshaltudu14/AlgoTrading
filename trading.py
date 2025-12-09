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
BACKTEST_DAYS = 150

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


def format_currency(amount: float) -> str:
    """Format currency amount with Indian number system"""
    if amount >= 10000000:  # 1 Crore or more
        return f"Rs.{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh or more
        return f"Rs.{amount/100000:.2f} L"
    else:
        return f"Rs.{amount:,.0f}"


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

        # Use instrument-specific filenames
        instrument_name = DEFAULT_INSTRUMENT_NAME.lower().replace(" ", "_").replace("-", "_")  # Clean name for file
        input_dir = "backend/data/raw"
        output_dir = "backend/data/processed"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Specific file paths for our instrument/days/timeframe
        raw_csv_path = f"{input_dir}/{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"
        processed_csv_path = f"{output_dir}/features_{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"

        # Save raw data
        historical_data.to_csv(raw_csv_path, index=False)
        print(f"[INFO] Saved raw data to {raw_csv_path}")

        # Check if processed file already exists
        if os.path.exists(processed_csv_path):
            print(f"[OK] Using existing processed data: {processed_csv_path}")
            processed_data = pd.read_csv(processed_csv_path)
            print(f"[OK] Processed data shape: {processed_data.shape}")
            return True

        # Create a temporary directory with only our file
        temp_input_dir = f"{input_dir}/temp_{instrument_name}_{BACKTEST_DAYS}d"
        os.makedirs(temp_input_dir, exist_ok=True)

        # Copy only our file to the temp directory
        import shutil
        temp_csv_path = f"{temp_input_dir}/{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"
        shutil.copy2(raw_csv_path, temp_csv_path)

        # Process only the files in temp directory
        from backend.src.data_processing.pipeline import DataProcessingPipeline
        pipeline = DataProcessingPipeline()

        # Run feature generation on temp directory
        results = pipeline.run_feature_generation(
            input_dir=temp_input_dir,
            output_dir=output_dir,
            parallel=False
        )

        # Clean up temp directory
        shutil.rmtree(temp_input_dir)

        if results.get('success'):
            # Read the processed data
            processed_data = pd.read_csv(processed_csv_path)
            print(f"[OK] Processed data shape: {processed_data.shape}")
            print(f"[OK] Saved processed data to {processed_csv_path}")
            return True
        else:
            print(f"[ERROR] Pipeline failed: {results.get('error')}")
            return False

    except Exception as e:
        print(f"[ERROR] Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False






def parse_arguments():
    parser = argparse.ArgumentParser(description='Algo Trading Pipeline')
    parser.add_argument('--backtest', action='store_true',
                        help='Run in backtest mode')
    return parser.parse_args()


async def real_trading():
    print("\n=== REAL TRADING MODE ===")
    print("\n[INFO] Real trading is currently disabled for safety")
    print(f"[INFO] Current Price: NIFTY - Check live prices")
    print(f"[INFO] Target P&L: Rs.{TARGET_PNL}")
    print(f"[INFO] Stop Loss P&L: Rs.{STOP_LOSS_PNL}")
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
        # Use backtest data - get the specific processed file
        instrument_name = DEFAULT_INSTRUMENT_NAME.lower().replace(" ", "_").replace("-", "_")
        output_dir = "backend/data/processed"
        csv_path = f"{output_dir}/features_{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"

        if not os.path.exists(csv_path):
            print(f"[ERROR] No processed backtest data found at {csv_path}")
            print(f"[INFO] Please run without --backtest flag first to fetch and process data")
            return False

        print(f"[OK] Using processed data: {csv_path}")

        # Create rule-based backtester
        backtester = RuleBasedBacktester(
            target_pnl=TARGET_PNL,
            stop_loss_pnl=STOP_LOSS_PNL,
            initial_capital=25000,
            brokerage_entry=25,
            brokerage_exit=25
        )

        # Set trade log path for real-time saving
        instrument_symbol = DEFAULT_INSTRUMENT_NAME.lower()
        trade_log_path = f"backend/data/backtest_trades_{instrument_symbol}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"

        # Delete existing CSV file if it exists
        if os.path.exists(trade_log_path):
            os.remove(trade_log_path)
            print(f"[INFO] Deleted existing trade log: {trade_log_path}")

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
            print(f"Total P&L (after brokerage): {format_currency(results.get('total_pnl', 0))}")
            print(f"Total P&L %: {results.get('total_pnl_percent', 0):.1f}%")
            print(f"Max Drawdown: {format_currency(results.get('max_drawdown', 0))} ({results.get('max_drawdown_pct', 0):.1f}%)")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
            print(f"Highest Daily Profit: {format_currency(results.get('highest_daily_profit', 0))}")
            print(f"Highest Daily Loss: {format_currency(results.get('highest_daily_loss', 0))}")
            print(f"Max Trades in a Day: {results.get('max_trades_per_day', 0)}")
            print(f"Min Trades in a Day: {results.get('min_trades_per_day', 0)}")
            print(f"Max Winning Streak: {results.get('max_winning_streak', 0)} trades")
            print(f"Max Losing Streak: {results.get('max_losing_streak', 0)} trades")

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

    mode = 'BACKTEST' if args.backtest else 'REAL'
    print("\n=== Algo Trading Pipeline ===\n")
    print(f"Mode: {mode}")
    print(f"Timeframe: {TIMEFRAME}-minute")
    print(f"Historical Days: {BACKTEST_DAYS}")
    print(f"Target P&L: {format_currency(TARGET_PNL)}")
    print(f"Stop Loss P&L: {format_currency(STOP_LOSS_PNL)}")
    print(f"Strategy: {STRATEGY}")

    # For backtest mode, check if processed data exists to skip authentication
    if args.backtest:
        instrument_name = DEFAULT_INSTRUMENT_NAME.lower().replace(" ", "_").replace("-", "_")
        output_dir = "backend/data/processed"

        # Specific file we're looking for
        processed_csv_path = f"{output_dir}/features_{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv"

        if os.path.exists(processed_csv_path):
            print("\n[OK] Found existing processed data - skipping authentication and data fetch")
            print("\n--- Data Preprocessing ---")
            print(f"[OK] Using existing processed data: features_{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min.csv")

            # Configure instrument with defaults since we can't get capital without auth
            print("\n--- Instrument Configuration (Default) ---")
            if not configure_instrument():
                print("[WARNING] Using default instrument configuration")

            # Execute backtest
            print("\n--- Execution ---")
            await backtest()
            return

    # If no existing data or not backtest mode, proceed with full pipeline
    if not await authenticate():
        return

    print("\n--- Instrument Configuration ---")
    if not configure_instrument():
        return

    print("\n--- Fetch Historical Data ---")
    if not fetch_historical_data():
        return

    print("\n--- Data Preprocessing ---")
    # Check if processed data already exists
    instrument_name = DEFAULT_INSTRUMENT_NAME.lower()
    output_dir = "backend/data/processed"
    feature_files = list(Path(output_dir).glob("features_*.csv"))
    matching_files = [f for f in feature_files if f"{instrument_name}_{BACKTEST_DAYS}d_{TIMEFRAME}min" in f.name]

    if not matching_files:
        print("[INFO] No processed data found, running preprocessing...")
        if not preprocess_data():
            return
    else:
        print(f"[OK] Using existing processed data: {matching_files[0].name}")

    print("\n--- Execution ---")
    if args.backtest:
        await backtest()
    else:
        await real_trading()

    print("\n[OK] Pipeline completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())