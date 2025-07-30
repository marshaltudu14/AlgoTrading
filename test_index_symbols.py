#!/usr/bin/env python3
"""
Test Index Symbol Formats
"""

from src.trading.fyers_client import FyersClient
import logging

logging.basicConfig(level=logging.INFO)

def test_index_symbols():
    client = FyersClient()
    
    symbols_to_test = [
        'NSE:NIFTY_BANK-INDEX',
        'NSE:NIFTY_50-INDEX', 
        'BSE:SENSEX-INDEX'
    ]
    
    for symbol in symbols_to_test:
        print(f"\n🔍 Testing symbol: {symbol}")
        try:
            data = client.get_historical_data(symbol=symbol, timeframe='5', days=7)
            if not data.empty:
                print(f"✅ Success: {len(data)} candles")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                print(f"   Sample: Open={data.iloc[0]['open']:.2f}, Close={data.iloc[0]['close']:.2f}")
            else:
                print(f"❌ No data received")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🔍 Testing client default symbols:")
    for name, symbol in client.default_symbols.items():
        print(f"   {name}: {symbol}")

if __name__ == "__main__":
    test_index_symbols()
