#!/usr/bin/env python3
"""
Test Real-time Data Loader
"""

from src.utils.realtime_data_loader import RealtimeDataLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_realtime_loader():
    print('Testing Real-time Data Loader...')
    
    # Create loader with test configuration
    config = {
        'symbol': 'banknifty',
        'timeframe': '5',
        'days': 7,  # Use fewer days for testing
        'min_data_points': 50
    }
    
    loader = RealtimeDataLoader(config=config)
    
    # Test data fetching and processing
    processed_data = loader.fetch_and_process_data()
    
    if processed_data is not None and not processed_data.empty:
        print(f'✅ Processed data shape: {processed_data.shape}')
        print(f'   Features: {processed_data.shape[1]} columns')
        print(f'   Date range: {processed_data.index[0]} to {processed_data.index[-1]}')
        print(f'   Sample columns: {list(processed_data.columns[:10])}...')
        print('✅ Real-time data loader test completed successfully')
        return True
    else:
        print('❌ Real-time data loader test failed')
        return False

if __name__ == "__main__":
    test_realtime_loader()
