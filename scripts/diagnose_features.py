import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing.feature_generator import DynamicFileProcessor

def test_feature_generation():
    print("Testing DynamicFileProcessor.generate_all_features...")
    num_candles = 500
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

    processor = DynamicFileProcessor()
    features_df = processor.generate_all_features(
        open_prices=mock_df['open'],
        high_prices=mock_df['high'],
        low_prices=mock_df['low'],
        close_prices=mock_df['close']
    )

    print(f"Number of features generated: {len(features_df.columns)}")
    print(f"Features generated: {features_df.columns.tolist()}")

if __name__ == "__main__":
    test_feature_generation()
