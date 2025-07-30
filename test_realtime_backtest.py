#!/usr/bin/env python3
"""
Test Real-time Backtesting
"""

import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_realtime_backtest():
    print('🔄 Testing Real-time Backtesting...')
    
    # Test with a simple command
    cmd = 'python run_backtest.py --realtime --symbol banknifty'
    print(f'Running: {cmd}')
    
    result = os.system(cmd)
    
    if result == 0:
        print('✅ Real-time backtesting test completed successfully')
    else:
        print(f'❌ Real-time backtesting test failed with exit code: {result}')
    
    return result == 0

if __name__ == "__main__":
    test_realtime_backtest()
