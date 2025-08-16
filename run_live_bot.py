#!/usr/bin/env python3
"""
Main entry point for the live trading bot.
"""

from src.trading.live_trader import LiveTrader
from src.config.settings import get_settings

def main():
    config = get_settings()
    trader = LiveTrader(config)
    trader.run()

if __name__ == "__main__":
    main()
