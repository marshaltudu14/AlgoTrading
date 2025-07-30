#!/usr/bin/env python3
"""
Test Bank Nifty Index Data
"""

from src.trading.fyers_client import FyersClient
import logging

logging.basicConfig(level=logging.INFO)

def test_banknifty_index():
    client = FyersClient()
    print('Testing Bank Nifty index data...')
    print(f'Symbol: {client.default_symbols["banknifty"]}')
    
    data = client.get_backtesting_data(symbol='banknifty', timeframe='5', days=7)
    print(f'Data shape: {data.shape}')
    
    if not data.empty:
        print(f'Date range: {data.index[0]} to {data.index[-1]}')
        print(f'Sample: Open={data.iloc[0]["open"]:.2f}, Close={data.iloc[0]["close"]:.2f}')
        print('✅ Bank Nifty index data test successful')
        return True
    else:
        print('❌ No data received')
        return False

if __name__ == "__main__":
    test_banknifty_index()
