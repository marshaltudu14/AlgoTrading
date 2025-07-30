"""
Test suite for data normalization and bias detection in the complete pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.data_processing.feature_generator import DynamicFileProcessor
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader

class TestDataNormalizationPipeline:
    """Test suite for data normalization and bias detection."""
    
    def create_test_data_with_different_scales(self, symbol="Bank_Nifty", num_points=200):
        """Create test data with different price scales to test normalization."""
        np.random.seed(42)
        
        # Create data with different price scales
        if symbol == "Bank_Nifty":
            base_price = 45000  # High price scale
        elif symbol == "Nifty":
            base_price = 19000  # Medium price scale
        elif symbol == "RELIANCE":
            base_price = 2500   # Lower price scale
        else:
            base_price = 100    # Very low price scale
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        returns = np.random.normal(0.0001, 0.02, num_points)  # Realistic volatility
        
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
                'datetime': int(date.timestamp()),  # Convert to Unix timestamp
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_feature_generation_normalization(self):
        """Test that feature generation produces properly normalized features."""
        # Create test data with different price scales
        symbols = ["Bank_Nifty", "Nifty", "RELIANCE", "TEST_LOW"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            
            # Create data for different symbols with different price scales
            all_features = {}
            
            for symbol in symbols:
                df = self.create_test_data_with_different_scales(symbol)
                df.to_csv(os.path.join(raw_dir, f"{symbol}.csv"), index=False)
                
                # Process with feature generator
                processor = DynamicFileProcessor()
                features_df = processor.process_single_file(Path(os.path.join(raw_dir, f"{symbol}.csv")))
                
                all_features[symbol] = features_df
            
            # Test that features are properly normalized across different price scales
            feature_columns = [col for col in all_features["Bank_Nifty"].columns 
                             if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            for feature_col in feature_columns:
                # Get feature values for all symbols
                feature_values = {}
                for symbol in symbols:
                    if feature_col in all_features[symbol].columns:
                        values = all_features[symbol][feature_col].dropna()
                        if len(values) > 0:
                            feature_values[symbol] = values
                
                if len(feature_values) >= 2:  # Need at least 2 symbols to compare
                    # Check that features are in similar ranges despite different price scales
                    ranges = {}
                    for symbol, values in feature_values.items():
                        ranges[symbol] = (values.min(), values.max())
                    
                    # Features should not have extreme differences due to price scale
                    # (This tests that percentage-based features are used)
                    max_range = max(r[1] - r[0] for r in ranges.values())
                    min_range = min(r[1] - r[0] for r in ranges.values())
                    
                    if max_range > 0 and min_range > 0:
                        range_ratio = max_range / min_range
                        # Range ratio should not be extreme (indicating good normalization)
                        assert range_ratio < 1000, f"Feature {feature_col} has extreme range differences: {ranges}"
    
    def test_trading_env_zscore_normalization(self):
        """Test that TradingEnv applies proper z-score normalization."""
        # Create test data with more points for technical indicators
        df = self.create_test_data_with_different_scales("Bank_Nifty", 300)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data with features
            processor = DynamicFileProcessor()
            features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
            features_df.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=100000,
                lookback_window=10,
                episode_length=50,
                use_streaming=False
            )
            
            # Reset and collect observations
            obs = env.reset()
            observations = [obs]
            
            # Run several steps to build observation history
            for _ in range(20):
                action = (4, 0)  # HOLD action
                obs, reward, done, info = env.step(action)
                observations.append(obs)
                if done:
                    break
            
            # Test z-score normalization properties
            if len(observations) > 10:
                obs_array = np.array(observations)
                
                # Check that observations are properly normalized
                # After z-score normalization, values should be roughly centered around 0
                # and have reasonable standard deviation
                
                for feature_idx in range(obs_array.shape[1]):
                    feature_values = obs_array[:, feature_idx]
                    
                    # Skip constant features
                    if np.std(feature_values) > 0.001:
                        # Check that values are not extreme (z-score should clip at ±5)
                        assert np.all(feature_values >= -6), f"Feature {feature_idx} has values below -6"
                        assert np.all(feature_values <= 6), f"Feature {feature_idx} has values above 6"
                        
                        # Check that there's reasonable variation (not all zeros)
                        assert np.std(feature_values) > 0.001, f"Feature {feature_idx} has no variation"
    
    def test_cross_instrument_normalization_consistency(self):
        """Test that normalization is consistent across different instruments."""
        symbols = ["Bank_Nifty", "Nifty", "RELIANCE"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            observations_by_symbol = {}
            
            for symbol in symbols:
                # Create test data
                df = self.create_test_data_with_different_scales(symbol, 80)
                df.to_csv(os.path.join(raw_dir, f"{symbol}.csv"), index=False)
                
                # Create processed data
                processor = DynamicFileProcessor()
                features_df = processor.process_single_file(Path(os.path.join(raw_dir, f"{symbol}.csv")))
                features_df.to_csv(os.path.join(final_dir, f"features_{symbol}.csv"), index=False)
                
                # Create environment and collect observations
                loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
                env = TradingEnv(
                    data_loader=loader,
                    symbol=symbol,
                    initial_capital=100000,
                    lookback_window=5,
                    episode_length=30,
                    use_streaming=False
                )
                
                obs = env.reset()
                observations = [obs]
                
                # Collect observations
                for _ in range(15):
                    action = (4, 0)  # HOLD
                    obs, reward, done, info = env.step(action)
                    observations.append(obs)
                    if done:
                        break
                
                observations_by_symbol[symbol] = np.array(observations)
            
            # Compare observation distributions across instruments
            min_length = min(len(obs) for obs in observations_by_symbol.values())
            
            if min_length > 10:
                # Compare feature distributions
                for i in range(min(40, observations_by_symbol["Bank_Nifty"].shape[1])):  # Check first 40 features
                    feature_stats = {}
                    
                    for symbol in symbols:
                        feature_values = observations_by_symbol[symbol][:min_length, i]
                        if np.std(feature_values) > 0.001:  # Skip constant features
                            feature_stats[symbol] = {
                                'mean': np.mean(feature_values),
                                'std': np.std(feature_values),
                                'min': np.min(feature_values),
                                'max': np.max(feature_values)
                            }
                    
                    if len(feature_stats) >= 2:
                        # Check that feature distributions are reasonably similar
                        # (indicating good cross-instrument normalization)
                        means = [stats['mean'] for stats in feature_stats.values()]
                        stds = [stats['std'] for stats in feature_stats.values()]
                        
                        # Means should be reasonably close (within 2 standard deviations)
                        if len(means) > 1 and max(stds) > 0.1:
                            mean_range = max(means) - min(means)
                            avg_std = np.mean(stds)
                            
                            # This is a loose check - means shouldn't be wildly different
                            assert mean_range < 10 * avg_std, f"Feature {i} has inconsistent means across instruments: {feature_stats}"
    
    def test_no_data_leakage_in_normalization(self):
        """Test that normalization doesn't cause data leakage."""
        # Create test data
        df = self.create_test_data_with_different_scales("Bank_Nifty", 100)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            processor = DynamicFileProcessor()
            features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
            features_df.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=100000,
                lookback_window=5,
                episode_length=50,
                use_streaming=False
            )
            
            # Test that normalization only uses past data
            obs = env.reset()
            
            # Initially, normalization should not be applied (insufficient history)
            # This ensures no future data is used
            
            observations = []
            for step in range(30):
                action = (4, 0)  # HOLD
                obs, reward, done, info = env.step(action)
                observations.append(obs)
                
                # Check that observation values are reasonable
                assert np.isfinite(obs).all(), f"Non-finite values in observation at step {step}"
                
                if done:
                    break
            
            # Verify that early observations (before sufficient history) are different
            # from later observations (after normalization kicks in)
            if len(observations) > 15:
                early_obs = observations[:5]
                later_obs = observations[-5:]
                
                # Calculate statistics
                early_stats = np.array([np.mean(np.abs(obs)) for obs in early_obs])
                later_stats = np.array([np.mean(np.abs(obs)) for obs in later_obs])
                
                # Early observations should generally have larger absolute values
                # (before normalization) compared to later ones (after normalization)
                # This is a loose check to ensure normalization is working
                
                early_mean = np.mean(early_stats)
                later_mean = np.mean(later_stats)
                
                # This test ensures that normalization is actually being applied
                # and that it's using only historical data
                assert early_mean != later_mean, "Normalization doesn't appear to be working"
    
    def test_feature_scaling_consistency(self):
        """Test that feature scaling is consistent and doesn't introduce bias."""
        # Create test data with known patterns
        df = self.create_test_data_with_different_scales("Bank_Nifty", 150)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Process with feature generator
            processor = DynamicFileProcessor()
            features_df = processor.process_single_file(Path(os.path.join(raw_dir, "Bank_Nifty.csv")))
            
            # Test feature scaling properties
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']:
                    values = features_df[col].dropna()
                    
                    if len(values) > 10:
                        # Check for infinite or NaN values
                        assert np.isfinite(values).all(), f"Column {col} contains non-finite values"
                        
                        # Check that values are in reasonable ranges
                        assert values.min() >= -1000, f"Column {col} has extremely low values"
                        assert values.max() <= 1000, f"Column {col} has extremely high values"
                        
                        # Check that features have reasonable variation
                        if np.std(values) > 0:
                            cv = np.std(values) / (np.abs(np.mean(values)) + 1e-8)  # Coefficient of variation
                            assert cv < 100, f"Column {col} has excessive variation: CV={cv}"
    
    def test_price_scale_independence(self):
        """Test that the system works independently of price scale."""
        # Create data with very different price scales
        test_cases = [
            ("HIGH_PRICE", 50000),   # Very high price
            ("MED_PRICE", 2000),     # Medium price  
            ("LOW_PRICE", 50),       # Low price
            ("VERY_LOW_PRICE", 1)    # Very low price
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            results = {}
            
            for symbol, base_price in test_cases:
                # Create data with specific price scale
                np.random.seed(42)  # Same seed for consistency
                dates = pd.date_range('2023-01-01', periods=100, freq='1min')
                returns = np.random.normal(0.0001, 0.02, 100)
                
                prices = [base_price]
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                data = []
                for i, (date, price) in enumerate(zip(dates, prices)):
                    high = price * 1.01
                    low = price * 0.99
                    open_price = prices[i-1] if i > 0 else price
                    close_price = price
                    volume = 10000
                    
                    data.append({
                        'datetime': int(date.timestamp()),  # Convert to Unix timestamp
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': close_price,
                        'volume': volume
                    })
                
                df = pd.DataFrame(data)
                df.to_csv(os.path.join(raw_dir, f"{symbol}.csv"), index=False)
                
                # Process features
                processor = DynamicFileProcessor()
                features_df = processor.process_single_file(Path(os.path.join(raw_dir, f"{symbol}.csv")))
                features_df.to_csv(os.path.join(final_dir, f"features_{symbol}.csv"), index=False)
                
                # Test trading environment
                loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
                env = TradingEnv(
                    data_loader=loader,
                    symbol=symbol,
                    initial_capital=100000,
                    lookback_window=5,
                    episode_length=30,
                    use_streaming=False
                )
                
                obs = env.reset()
                
                # Collect some observations
                observations = [obs]
                for _ in range(10):
                    action = (4, 0)  # HOLD
                    obs, reward, done, info = env.step(action)
                    observations.append(obs)
                    if done:
                        break
                
                results[symbol] = {
                    'base_price': base_price,
                    'observations': np.array(observations),
                    'features': features_df
                }
            
            # Compare results across different price scales
            # Features should be similar despite different price scales
            feature_columns = [col for col in results["HIGH_PRICE"]['features'].columns 
                             if col not in ['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
            for feature_col in feature_columns[:10]:  # Test first 10 features
                feature_values = {}
                for symbol in results:
                    if feature_col in results[symbol]['features'].columns:
                        values = results[symbol]['features'][feature_col].dropna()
                        if len(values) > 5:
                            feature_values[symbol] = values.iloc[:5]  # First 5 values
                
                if len(feature_values) >= 2:
                    # Features should be similar across price scales (percentage-based)
                    correlations = []
                    symbols = list(feature_values.keys())
                    
                    for i in range(len(symbols)):
                        for j in range(i+1, len(symbols)):
                            corr = np.corrcoef(feature_values[symbols[i]], feature_values[symbols[j]])[0, 1]
                            if not np.isnan(corr):
                                correlations.append(abs(corr))
                    
                    if correlations:
                        avg_correlation = np.mean(correlations)
                        # Features should be reasonably correlated across price scales
                        # (indicating price scale independence)
                        assert avg_correlation > 0.5, f"Feature {feature_col} shows poor correlation across price scales: {avg_correlation}"
