#!/usr/bin/env python3
"""
Main entry point for the live trading bot.
"""

from src.trading.live_trader import LiveTrader
from src.config.config import get_config

def main():
    config = get_config()
    # Add trading specific config
    config['trading'] = {
        "instrument": "NSE:NIFTY50-INDEX", # Hardcoded for MVP
        "timeframe": "5", # Options: "1", "2", "5", "15"
        "check_interval_seconds": 60, # How often to check for new signals (e.g., 60 seconds for 1-min chart)
        "trade_quantity": 1, # Quantity to trade for MVP
        "risk_multiplier": 1.0, # ATR multiplier for stop loss
        "reward_multiplier": 2.0 # ATR multiplier for take profit
    }

    trader = LiveTrader(config)
    trader.run()

if __name__ == "__main__":
    main()
