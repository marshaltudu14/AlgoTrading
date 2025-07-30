#!/usr/bin/env python3
"""
Test Fyers data fetching
"""

from src.trading.fyers_client import FyersClient
import logging

logging.basicConfig(level=logging.INFO)

def test_fyers_data():
    client = FyersClient()
    print('Testing Bank Nifty futures data fetch...')
    print(f'Bank Nifty symbol: {client.default_symbols["banknifty"]}')
    
    data = client.get_backtesting_data(symbol='banknifty', timeframe='5', days=7)
    print(f'Data shape: {data.shape}')
    
    if not data.empty:
        print(f'Columns: {list(data.columns)}')
        print(f'Date range: {data.index[0]} to {data.index[-1]}')
        print('Sample data:')
        print(data.head(2))
    else:
        print('No data received')
        
    print('✅ Real-time data fetch test completed')

if __name__ == "__main__":
    test_fyers_data()
