#!/usr/bin/env python3
"""
Test script for the live trading bot.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.trading.live_trader import LiveTrader
from src.config.config import get_config

def run_test():
    print("Initializing test...")
    config = get_config()
    # Add trading specific config for testing
    config['trading'] = {
        "instrument": "NSE:NIFTY50-INDEX",
        "timeframe": "5"
    }

    try:
        print("Instantiating LiveTrader...")
        trader = LiveTrader(config)
        print("LiveTrader instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating LiveTrader: {e}")
        return

    try:
        print("Attempting to run a single trade decision cycle...")
        # We will mock the data fetching part to avoid real API calls
        # and focus on testing the processing and inference logic.
        print("Creating mock historical data...")
        # Generate more realistic mock data for feature generation
        num_candles = 500 # Increased number of candles
        base_price = 1000
        np.random.seed(42)
        prices = base_price + np.cumsum(np.random.randn(num_candles) * 0.5)
        
        mock_data = {
            'datetime': pd.to_datetime(pd.date_range(start='1/1/2024', periods=num_candles, freq='5min')),
            'open': prices,
            'high': prices + np.random.rand(num_candles) * 2,
            'low': prices - np.random.rand(num_candles) * 2,
            'close': prices + np.random.randn(num_candles) * 0.5,
        }
        mock_df = pd.DataFrame(mock_data)

        print("Mocking FyersClient to return mock data...")
        def mock_get_historical_data(*args, **kwargs):
            return mock_df
        
        trader.fyers_client.get_historical_data = mock_get_historical_data

        print("Calling get_trade_decision()...")
        # This will test the integration of LiveDataProcessor and InferenceEngine
        # It is expected to fail at the predict stage if models are not trained,
        # but it will validate the data processing flow.
        decision = trader.get_trade_decision()
        print(f"Trade decision cycle completed. Predicted signal: {decision}")

    except Exception as e:
        print(f"Error during trade decision cycle: {e}")

if __name__ == "__main__":
    run_test()
