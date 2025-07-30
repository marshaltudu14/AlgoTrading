"""
Comprehensive tests for data processing components.
Tests DataLoader, data validation, feature engineering, and data pipeline integrity.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from unittest.mock import patch, Mock
import logging

from src.utils.data_loader import DataLoader
from tests.conftest import assert_valid_observation

class TestDataLoader:
    """Test suite for DataLoader functionality."""
    
    def test_initialization(self, test_data_dir):
        """Test DataLoader initialization with valid directories."""
        final_dir = os.path.join(test_data_dir, "final")
        raw_dir = os.path.join(test_data_dir, "raw")

        loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)

        assert loader.final_data_dir == final_dir
        assert loader.raw_data_dir == raw_dir
        assert loader.use_parquet == False
    
    def test_initialization_with_parquet(self, test_data_dir):
        """Test DataLoader initialization with parquet option."""
        final_dir = os.path.join(test_data_dir, "final")
        
        loader = DataLoader(final_data_dir=final_dir, use_parquet=True)
        
        assert loader.use_parquet == True
    
    def test_load_all_processed_data_success(self, mock_data_loader):
        """Test successful loading of all processed data."""
        df = mock_data_loader.load_all_processed_data()
        
        assert not df.empty
        assert 'datetime' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        
        # Check for feature columns
        assert 'returns' in df.columns
        assert 'sma_10' in df.columns
        assert 'sma_20' in df.columns
        assert 'rsi' in df.columns
        assert 'volatility' in df.columns
    
    def test_load_all_processed_data_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = DataLoader(final_data_dir=temp_dir)
            df = loader.load_all_processed_data()
            
            assert df.empty
    
    def test_load_raw_data_for_symbol_success(self, mock_data_loader):
        """Test successful loading of raw data for a specific symbol."""
        df = mock_data_loader.load_raw_data_for_symbol("Bank_Nifty")
        
        assert not df.empty
        assert 'datetime' in df.columns
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        
        # Validate OHLC consistency
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
    
    def test_load_raw_data_for_symbol_not_found(self, mock_data_loader):
        """Test loading raw data for non-existent symbol."""
        df = mock_data_loader.load_raw_data_for_symbol("NONEXISTENT")
        
        assert df.empty
    
    def test_get_available_tasks(self, mock_data_loader):
        """Test getting available tasks from processed data."""
        tasks = mock_data_loader.get_available_tasks()

        assert len(tasks) > 0
        assert all(isinstance(task, tuple) and len(task) == 2 for task in tasks)

        # Check that we have our test symbols (they are stored as features_SYMBOL.csv)
        symbols = [task[0] for task in tasks]
        assert 'features' in symbols[0]  # Files are named features_SYMBOL.csv
    
    def test_sample_tasks_valid_count(self, mock_data_loader):
        """Test sampling valid number of tasks."""
        available_tasks = mock_data_loader.get_available_tasks()
        sample_size = min(2, len(available_tasks))
        
        sampled_tasks = mock_data_loader.sample_tasks(sample_size)
        
        assert len(sampled_tasks) == sample_size
        assert all(task in available_tasks for task in sampled_tasks)
    
    def test_sample_tasks_excessive_count(self, mock_data_loader):
        """Test sampling more tasks than available."""
        available_tasks = mock_data_loader.get_available_tasks()
        excessive_count = len(available_tasks) + 10
        
        sampled_tasks = mock_data_loader.sample_tasks(excessive_count)
        
        # Should return all available tasks
        assert len(sampled_tasks) == len(available_tasks)
    
    def test_get_task_data_success(self, mock_data_loader):
        """Test getting data for a specific task."""
        # Get available tasks first to use a valid task
        tasks = mock_data_loader.get_available_tasks()
        if tasks:
            instrument, timeframe = tasks[0]
            df = mock_data_loader.get_task_data(instrument, timeframe)

            assert not df.empty
            assert 'datetime' in df.columns
            assert 'close' in df.columns
    
    def test_get_task_data_not_found(self, mock_data_loader):
        """Test getting data for non-existent task."""
        df = mock_data_loader.get_task_data('NONEXISTENT', '1')
        
        assert df.empty

class TestDataValidation:
    """Test suite for data validation functionality."""
    
    def test_ohlc_validation_valid_data(self, sample_market_data):
        """Test OHLC validation with valid data."""
        # The sample_market_data fixture ensures OHLC consistency
        assert (sample_market_data['high'] >= sample_market_data['open']).all()
        assert (sample_market_data['high'] >= sample_market_data['close']).all()
        assert (sample_market_data['low'] <= sample_market_data['open']).all()
        assert (sample_market_data['low'] <= sample_market_data['close']).all()
    
    def test_ohlc_validation_invalid_data(self):
        """Test OHLC validation with invalid data."""
        invalid_data = pd.DataFrame({
            'datetime': ['2023-01-01'],
            'open': [100],
            'high': [90],  # High < Open (invalid)
            'low': [110],  # Low > Open (invalid)
            'close': [105],
            'volume': [1000]
        })
        
        # Check for OHLC inconsistencies
        assert not (invalid_data['high'] >= invalid_data['open']).all()
        assert not (invalid_data['low'] <= invalid_data['open']).all()
    
    def test_missing_data_detection(self):
        """Test detection of missing data."""
        data_with_nans = pd.DataFrame({
            'datetime': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'open': [100, np.nan, 102],
            'high': [105, 106, np.nan],
            'low': [99, 100, 101],
            'close': [103, 104, 105],
            'volume': [1000, 2000, 3000]
        })
        
        assert data_with_nans.isnull().any().any()
        
        # Test cleaning
        cleaned_data = data_with_nans.dropna()
        assert not cleaned_data.isnull().any().any()
        assert len(cleaned_data) == 1  # Only first row is complete
    
    def test_data_type_validation(self, sample_market_data):
        """Test data type validation."""
        # Check that numeric columns are numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(sample_market_data[col])
        
        # Check datetime column
        if 'datetime' in sample_market_data.columns:
            # Should be convertible to datetime
            pd.to_datetime(sample_market_data['datetime'])

class TestFeatureEngineering:
    """Test suite for feature engineering functionality."""
    
    def test_technical_indicators_calculation(self, mock_data_loader):
        """Test calculation of technical indicators."""
        df = mock_data_loader.load_all_processed_data()
        
        # Check that technical indicators are present
        assert 'returns' in df.columns
        assert 'sma_10' in df.columns
        assert 'sma_20' in df.columns
        assert 'rsi' in df.columns
        assert 'volatility' in df.columns
        
        # Validate RSI range (should be between 0 and 100)
        rsi_values = df['rsi'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Validate returns calculation
        returns = df['returns'].dropna()
        assert not returns.isnull().all()
    
    def test_lag_features_creation(self, mock_data_loader):
        """Test creation of lag features."""
        df = mock_data_loader.load_all_processed_data()
        
        # Check for lag features
        for i in range(1, 6):
            assert f'close_lag_{i}' in df.columns
            assert f'volume_lag_{i}' in df.columns
        
        # Validate lag relationships
        non_null_rows = df.dropna()
        if len(non_null_rows) > 5:
            # Check that lag_1 is the previous value
            for i in range(1, min(5, len(non_null_rows))):
                current_close = non_null_rows.iloc[i]['close']
                lag_1_close = non_null_rows.iloc[i]['close_lag_1']
                previous_close = non_null_rows.iloc[i-1]['close']
                
                # lag_1 should equal the previous close
                assert abs(lag_1_close - previous_close) < 1e-6
    
    def test_feature_scaling_normalization(self, mock_data_loader):
        """Test feature scaling and normalization."""
        df = mock_data_loader.load_all_processed_data()
        
        # Check that features don't have extreme values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            values = df[col].dropna()
            if len(values) > 0:
                # Check for infinite values
                assert not np.isinf(values).any()
                
                # Check for reasonable ranges (no extreme outliers)
                q99 = values.quantile(0.99)
                q01 = values.quantile(0.01)
                
                # Values should be within reasonable bounds
                assert not (values > q99 * 100).any()  # No extreme outliers

class TestDataPipelineIntegrity:
    """Test suite for data pipeline integrity."""
    
    def test_data_consistency_across_symbols(self, mock_data_loader):
        """Test data consistency across different symbols."""
        symbols = ['Bank_Nifty', 'Nifty']
        
        for symbol in symbols:
            df = mock_data_loader.load_raw_data_for_symbol(symbol)
            
            if not df.empty:
                # Check basic structure
                required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                assert all(col in df.columns for col in required_columns)
                
                # Check data types
                assert pd.api.types.is_numeric_dtype(df['open'])
                assert pd.api.types.is_numeric_dtype(df['high'])
                assert pd.api.types.is_numeric_dtype(df['low'])
                assert pd.api.types.is_numeric_dtype(df['close'])
                assert pd.api.types.is_numeric_dtype(df['volume'])
    
    def test_temporal_consistency(self, mock_data_loader):
        """Test temporal consistency of data."""
        df = mock_data_loader.load_raw_data_for_symbol('Bank_Nifty')
        
        if not df.empty and 'datetime' in df.columns:
            # Convert to datetime if not already
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Check that timestamps are in ascending order
            assert df['datetime'].is_monotonic_increasing
            
            # Check for reasonable time intervals
            if len(df) > 1:
                time_diffs = df['datetime'].diff().dropna()
                # All time differences should be positive
                assert (time_diffs > pd.Timedelta(0)).all()
    
    def test_data_completeness(self, mock_data_loader):
        """Test data completeness and coverage."""
        df = mock_data_loader.load_all_processed_data()
        
        if not df.empty:
            # Check that we have sufficient data points
            assert len(df) >= 100  # Minimum data points for meaningful analysis
            
            # Check feature completeness
            feature_columns = [col for col in df.columns if col not in ['datetime']]
            
            for col in feature_columns:
                non_null_ratio = df[col].count() / len(df)
                # At least 80% of data should be non-null for each feature
                assert non_null_ratio >= 0.8, f"Column {col} has too many null values: {non_null_ratio}"
    
    def test_memory_efficiency(self, mock_data_loader):
        """Test memory efficiency of data loading."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Load data
        df = mock_data_loader.load_all_processed_data()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test data)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # Clean up
        del df
    
    def test_error_handling_corrupted_files(self, test_data_dir):
        """Test error handling with corrupted files."""
        final_dir = os.path.join(test_data_dir, "final")
        
        # Create a corrupted file
        corrupted_file = os.path.join(final_dir, "corrupted.csv")
        with open(corrupted_file, 'w') as f:
            f.write("invalid,csv,content\n1,2,3,4,5,6,7,8,9,10")  # Wrong number of columns
        
        loader = DataLoader(final_data_dir=final_dir)
        
        # Should handle corrupted files gracefully
        try:
            df = loader.load_all_processed_data()
            # Should either skip corrupted file or handle it gracefully
            assert True  # If we get here, error was handled
        except Exception as e:
            # If an exception is raised, it should be informative
            assert "corrupted" in str(e).lower() or "invalid" in str(e).lower()
        
        # Clean up
        os.remove(corrupted_file)

class TestDataLoaderPerformance:
    """Test suite for DataLoader performance."""
    
    def test_loading_speed(self, mock_data_loader):
        """Test data loading speed."""
        import time
        
        start_time = time.time()
        df = mock_data_loader.load_all_processed_data()
        end_time = time.time()
        
        loading_time = end_time - start_time
        
        # Loading should complete within reasonable time (5 seconds for test data)
        assert loading_time < 5.0
        assert not df.empty
    
    def test_concurrent_access(self, mock_data_loader):
        """Test concurrent access to data loader."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def load_data():
            try:
                df = mock_data_loader.load_raw_data_for_symbol('Bank_Nifty')
                results.put(len(df))
            except Exception as e:
                errors.put(e)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=load_data)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        assert results.qsize() == 3
        
        # All results should be the same (same data loaded)
        result_list = list(results.queue)
        assert all(r == result_list[0] for r in result_list)
