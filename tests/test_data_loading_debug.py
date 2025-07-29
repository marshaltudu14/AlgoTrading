"""
Debug test to understand data loading issues.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.data_processing.feature_generator import DynamicFileProcessor
from src.utils.data_loader import DataLoader
from src.backtesting.environment import TradingEnv

class TestDataLoadingDebug:
    """Debug test for data loading issues."""
    
    def create_robust_test_data(self, symbol="Bank_Nifty", num_points=500):
        """Create robust test data with enough points for feature generation."""
        np.random.seed(42)
        base_price = 45000
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        returns = np.random.normal(0.0001, 0.02, num_points)
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.008)))
            low = price * (1 - abs(np.random.normal(0, 0.008)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 100000)
            
            data.append({
                'datetime': int(date.timestamp()),
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_feature_generation_step_by_step(self):
        """Test feature generation step by step to identify issues."""
        # Create test data
        df = self.create_robust_test_data("Bank_Nifty", 500)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            
            # Save test data
            csv_path = os.path.join(raw_dir, "Bank_Nifty.csv")
            df.to_csv(csv_path, index=False)
            
            print(f"Created test data: {len(df)} rows")
            print(f"Columns: {list(df.columns)}")
            print(f"Sample data:\n{df.head()}")
            
            # Test feature generation
            processor = DynamicFileProcessor()
            
            try:
                # Step 1: Load and validate data
                loaded_df = processor.load_and_validate_data(Path(csv_path))
                print(f"\nLoaded data: {len(loaded_df)} rows")
                print(f"Columns after loading: {list(loaded_df.columns)}")
                
                # Step 2: Generate features
                features_df = processor.generate_all_features(
                    loaded_df['open'], loaded_df['high'], loaded_df['low'], loaded_df['close']
                )
                print(f"\nGenerated features: {len(features_df)} rows, {len(features_df.columns)} columns")
                print(f"Feature columns: {list(features_df.columns)[:10]}...")  # First 10
                
                # Step 3: Combine with original data
                loaded_df_reset = loaded_df.reset_index()
                features_df_reset = features_df.reset_index(drop=True)
                
                print(f"\nBefore combination:")
                print(f"Original data shape: {loaded_df_reset.shape}")
                print(f"Features shape: {features_df_reset.shape}")
                
                # Ensure same length
                min_length = min(len(loaded_df_reset), len(features_df_reset))
                loaded_df_reset = loaded_df_reset.iloc[:min_length]
                features_df_reset = features_df_reset.iloc[:min_length]
                
                result_df = pd.concat([loaded_df_reset, features_df_reset], axis=1)
                print(f"\nCombined data shape: {result_df.shape}")
                print(f"Combined columns: {list(result_df.columns)[:15]}...")  # First 15
                
                # Step 4: Clean final dataset
                cleaned_df = processor.clean_final_dataset(result_df)
                print(f"\nCleaned data shape: {cleaned_df.shape}")
                print(f"Final columns: {list(cleaned_df.columns)[:15]}...")
                
                # Check if close column exists
                assert 'close' in cleaned_df.columns, f"'close' column missing! Available: {list(cleaned_df.columns)}"
                assert len(cleaned_df) > 0, "Final dataset is empty!"
                
                print(f"\n✅ Feature generation successful!")
                print(f"Final dataset: {len(cleaned_df)} rows, {len(cleaned_df.columns)} columns")
                
            except Exception as e:
                print(f"\n❌ Feature generation failed: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    def test_data_loader_functionality(self):
        """Test DataLoader functionality."""
        # Create test data
        df = self.create_robust_test_data("Bank_Nifty", 500)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save raw data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Generate features
            processor = DynamicFileProcessor()
            features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
            
            if len(features_df) > 0:
                # Save processed data
                features_df.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
                print(f"Saved features: {len(features_df)} rows, {len(features_df.columns)} columns")
                
                # Test DataLoader
                loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
                
                # Test loading final data
                loaded_data = loader.load_final_data_for_symbol("Bank_Nifty")
                print(f"DataLoader loaded: {len(loaded_data)} rows, {len(loaded_data.columns)} columns")
                print(f"Columns: {list(loaded_data.columns)[:15]}...")
                
                # Check if close column exists
                assert 'close' in loaded_data.columns, f"'close' column missing! Available: {list(loaded_data.columns)}"
                assert len(loaded_data) > 0, "Loaded data is empty!"
                
                print(f"✅ DataLoader test successful!")
                
            else:
                print(f"❌ No features generated!")
                raise AssertionError("Feature generation returned empty DataFrame")
    
    def test_trading_env_data_loading(self):
        """Test TradingEnv data loading."""
        # Create test data
        df = self.create_robust_test_data("Bank_Nifty", 500)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save raw data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Generate features
            processor = DynamicFileProcessor()
            features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
            
            if len(features_df) > 0:
                # Save processed data
                features_df.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
                
                # Test TradingEnv
                loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
                
                try:
                    env = TradingEnv(
                        data_loader=loader,
                        symbol="Bank_Nifty",
                        initial_capital=100000,
                        lookback_window=10,
                        episode_length=100,
                        use_streaming=False
                    )
                    
                    print(f"TradingEnv created successfully")
                    
                    # Check data loading
                    print(f"Environment data shape: {env.data.shape if env.data is not None else 'None'}")
                    if env.data is not None:
                        print(f"Environment data columns: {list(env.data.columns)[:15]}...")
                        
                        # Check if close column exists
                        if 'close' in env.data.columns:
                            print(f"✅ 'close' column found in environment data!")
                        else:
                            print(f"❌ 'close' column missing! Available: {list(env.data.columns)}")
                            raise AssertionError("'close' column missing from environment data")
                    
                    # Test reset
                    obs = env.reset()
                    print(f"Environment reset successful, observation shape: {obs.shape}")
                    
                    # Test step
                    action = (4, 0)  # HOLD
                    next_obs, reward, done, info = env.step(action)
                    print(f"Environment step successful, reward: {reward}")
                    
                    print(f"✅ TradingEnv test successful!")
                    
                except Exception as e:
                    print(f"❌ TradingEnv test failed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                    
            else:
                print(f"❌ No features generated for TradingEnv test!")
                raise AssertionError("Feature generation returned empty DataFrame")
    
    def test_minimal_working_example(self):
        """Test minimal working example to isolate the issue."""
        # Create sufficient test data for technical indicators (need 200+ rows for proper calculation)
        np.random.seed(42)
        num_points = 300
        base_price = 45000

        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        returns = np.random.normal(0.0001, 0.02, num_points)

        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        simple_data = {
            'datetime': [int(date.timestamp()) for date in dates],
            'open': [prices[i-1] if i > 0 else prices[i] for i in range(num_points)],
            'high': [price * (1 + abs(np.random.normal(0, 0.005))) for price in prices],
            'low': [price * (1 - abs(np.random.normal(0, 0.005))) for price in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, num_points)
        }

        # Ensure OHLC relationships are valid
        for i in range(num_points):
            simple_data['high'][i] = max(simple_data['open'][i], simple_data['high'][i], simple_data['close'][i])
            simple_data['low'][i] = min(simple_data['open'][i], simple_data['low'][i], simple_data['close'][i])

        df = pd.DataFrame(simple_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save simple data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            print(f"Created test data: {len(df)} rows")
            print(f"Sample data:\n{df.head()}")
            
            # Test if this works
            try:
                processor = DynamicFileProcessor()
                features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
                
                print(f"Simple feature generation result: {len(features_df)} rows")
                if len(features_df) > 0:
                    print(f"Columns: {list(features_df.columns)}")
                    print(f"✅ Simple test successful!")
                else:
                    print(f"❌ Simple test failed - no features generated")
                    
            except Exception as e:
                print(f"❌ Simple test failed with error: {e}")
                import traceback
                traceback.print_exc()
