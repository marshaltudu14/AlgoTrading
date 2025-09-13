#!/usr/bin/env python3
"""
Tests for Preprocessing Pipeline Integration
==========================================

Comprehensive tests for the complete preprocessing pipeline including:
- End-to-end historical data processing
- Real-time data processing
- Performance monitoring
- State management
- Integration with existing feature generation
- Validation and error handling

Author: AlgoTrading System
Version: 1.0
"""

import pytest
import numpy as np
import pandas as pd
import time
import tempfile
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.preprocessing_pipeline import (
    PreprocessingPipeline, PreprocessingConfig, ProcessingMode
)
from src.data_processing.feature_generator import DynamicFileProcessor


class TestPreprocessingPipeline:
    """Test cases for PreprocessingPipeline class."""

    def test_initialization(self):
        """Test pipeline initialization with default and custom configurations."""
        # Test default initialization
        pipeline = PreprocessingPipeline()
        assert pipeline.config.processing_mode == ProcessingMode.BATCH
        assert pipeline.config.batch_size == 10000
        assert pipeline.config.enable_caching == True
        assert pipeline.config.enable_performance_monitoring == True

        # Test custom configuration
        custom_config = PreprocessingConfig(
            processing_mode=ProcessingMode.STREAMING,
            batch_size=5000,
            enable_gpu=False,
            enable_caching=False
        )
        pipeline_custom = PreprocessingPipeline(custom_config)
        assert pipeline_custom.config.processing_mode == ProcessingMode.STREAMING
        assert pipeline_custom.config.batch_size == 5000
        assert pipeline_custom.config.enable_gpu == False
        assert pipeline_custom.config.enable_caching == False

    def test_historical_data_processing_mock(self):
        """Test historical data processing with mock data."""
        # Create mock input directory with test data
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create test CSV files
            self._create_test_csv_files(input_dir)

            pipeline = PreprocessingPipeline()
            result = pipeline.process_historical_data(str(input_dir), str(output_dir))

            # Check results
            assert result['success'] == True
            assert 'total_time' in result
            assert 'feature_generation' in result
            assert 'data_shape' in result
            assert 'performance_stats' in result

            # Check performance stats
            perf_stats = result['performance_stats']
            assert perf_stats['total_processing_time'] > 0
            assert perf_stats['data_points_processed'] > 0
            assert perf_stats['features_generated'] > 0

    def test_real_time_data_processing(self):
        """Test real-time data processing."""
        # Create pipeline and fit on historical data first
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create training data
            self._create_test_csv_files(input_dir)

            pipeline = PreprocessingPipeline()
            pipeline.process_historical_data(str(input_dir), str(output_dir))

            # Create new real-time data
            new_data = pd.DataFrame({
                'datetime': pd.date_range('2023-12-01', periods=100, freq='1H'),
                'open': np.random.uniform(100, 110, 100),
                'high': np.random.uniform(110, 115, 100),
                'low': np.random.uniform(95, 100, 100),
                'close': np.random.uniform(105, 110, 100)
            })

            # Process real-time data
            processed_data = pipeline.process_real_time_data(new_data)

            # Check results
            assert isinstance(processed_data, pd.DataFrame)
            assert len(processed_data) == len(new_data)
            assert not processed_data.isna().any().any()

    def test_state_management(self):
        """Test state management (save/load preprocessing parameters)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            state_dir = Path(temp_dir) / "state"
            input_dir.mkdir()
            output_dir.mkdir()
            state_dir.mkdir()

            # Create test data and process
            self._create_test_csv_files(input_dir)

            pipeline1 = PreprocessingPipeline()
            pipeline1.process_historical_data(str(input_dir), str(output_dir))

            # Copy state files to state directory
            for file in Path(output_dir).glob("*.json"):
                import shutil
                shutil.copy2(file, state_dir)

            # Create new pipeline and load state
            pipeline2 = PreprocessingPipeline()
            pipeline2.load_preprocessing_state(str(state_dir))

            # Check that state is loaded
            assert len(pipeline2.feature_normalizer.normalization_params) > 0
            assert len(pipeline2.feature_specific_normalizer.feature_configs) > 0

            # Test that loaded pipeline can process new data
            new_data = pd.DataFrame({
                'datetime': pd.date_range('2023-12-01', periods=50, freq='1H'),
                'open': np.random.uniform(100, 110, 50),
                'high': np.random.uniform(110, 115, 50),
                'low': np.random.uniform(95, 100, 50),
                'close': np.random.uniform(105, 110, 50)
            })

            processed_data = pipeline2.process_real_time_data(new_data)
            assert isinstance(processed_data, pd.DataFrame)

    def test_performance_monitoring(self):
        """Test performance monitoring and statistics."""
        pipeline = PreprocessingPipeline()

        # Check initial performance stats
        stats = pipeline.performance_stats
        assert 'total_processing_time' in stats
        assert 'feature_generation_time' in stats
        assert 'normalization_time' in stats
        assert stats['total_processing_time'] == 0  # Initially zero

        # Create mock data and process
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            self._create_test_csv_files(input_dir)

            result = pipeline.process_historical_data(str(input_dir), str(output_dir))

            # Check that performance stats are updated
            assert pipeline.performance_stats['total_processing_time'] > 0
            assert pipeline.performance_stats['feature_generation_time'] > 0
            assert pipeline.performance_stats['normalization_time'] > 0
            assert pipeline.performance_stats['data_points_processed'] > 0

    def test_preprocessing_summary(self):
        """Test preprocessing summary generation."""
        pipeline = PreprocessingPipeline()
        summary = pipeline.get_preprocessing_summary()

        # Check summary structure
        assert 'configuration' in summary
        assert 'performance_stats' in summary
        assert 'feature_normalization' in summary
        assert 'feature_specific_configs' in summary

        # Check configuration details
        config = summary['configuration']
        assert 'processing_mode' in config
        assert 'enable_gpu' in config
        assert 'batch_size' in config
        assert 'max_workers' in config

    def test_pipeline_validation(self):
        """Test pipeline validation with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create training data
            self._create_test_csv_files(input_dir)

            pipeline = PreprocessingPipeline()
            pipeline.process_historical_data(str(input_dir), str(output_dir))

            # Create test data for validation
            test_data = pd.DataFrame({
                'datetime': pd.date_range('2023-12-01', periods=100, freq='1H'),
                'open': np.random.uniform(100, 110, 100),
                'high': np.random.uniform(110, 115, 100),
                'low': np.random.uniform(95, 100, 100),
                'close': np.random.uniform(105, 110, 100)
            })

            validation_results = pipeline.validate_preprocessing_pipeline(test_data)

            # Check validation results
            assert validation_results['success'] == True
            assert 'input_shape' in validation_results
            assert 'output_shape' in validation_results
            assert 'missing_values' in validation_results
            assert 'value_ranges' in validation_results
            assert 'processing_time' in validation_results

            # Check that output shape is reasonable
            assert validation_results['output_shape'][0] == validation_results['input_shape'][0]
            assert validation_results['output_shape'][1] >= validation_results['input_shape'][1]  # Should have features added

            # Check that there are no missing values
            assert validation_results['missing_values'] == 0

    def test_error_handling(self):
        """Test error handling in various scenarios."""
        pipeline = PreprocessingPipeline()

        # Test with invalid input directory
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "nonexistent"
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            result = pipeline.process_historical_data(str(input_dir), str(output_dir))
            assert result['success'] == False
            assert 'error' in result

        # Test real-time processing without fitted normalizer
        new_data = pd.DataFrame({
            'datetime': pd.date_range('2023-12-01', periods=10, freq='1H'),
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(110, 115, 10),
            'low': np.random.uniform(95, 100, 10),
            'close': np.random.uniform(105, 110, 10)
        })

        with pytest.raises(ValueError, match="Normalizer must be fitted"):
            pipeline.process_real_time_data(new_data)

    def test_different_processing_modes(self):
        """Test different processing modes."""
        # Test batch mode
        batch_config = PreprocessingConfig(processing_mode=ProcessingMode.BATCH)
        batch_pipeline = PreprocessingPipeline(batch_config)
        assert batch_pipeline.config.processing_mode == ProcessingMode.BATCH

        # Test streaming mode
        streaming_config = PreprocessingConfig(processing_mode=ProcessingMode.STREAMING)
        streaming_pipeline = PreprocessingPipeline(streaming_config)
        assert streaming_pipeline.config.processing_mode == ProcessingMode.STREAMING

        # Test incremental mode
        incremental_config = PreprocessingConfig(processing_mode=ProcessingMode.INCREMENTAL)
        incremental_pipeline = PreprocessingPipeline(incremental_config)
        assert incremental_pipeline.config.processing_mode == ProcessingMode.INCREMENTAL

    def test_caching_functionality(self):
        """Test caching functionality."""
        cache_config = PreprocessingConfig(enable_caching=True)
        pipeline = PreprocessingPipeline(cache_config)

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            self._create_test_csv_files(input_dir)

            # Process data (should populate cache)
            result = pipeline.process_historical_data(str(input_dir), str(output_dir))

            # Check that cache directory was created
            assert pipeline.cache_dir.exists()
            assert pipeline.cache_dir.is_dir()

    def test_large_dataset_processing(self):
        """Test processing of larger datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Create larger test dataset
            self._create_large_test_csv_files(input_dir)

            pipeline = PreprocessingPipeline()
            start_time = time.time()
            result = pipeline.process_historical_data(str(input_dir), str(output_dir))
            processing_time = time.time() - start_time

            # Check that processing completes successfully
            assert result['success'] == True
            assert result['performance_stats']['data_points_processed'] > 1000

            # Check that processing time is reasonable (adjust threshold as needed)
            assert processing_time < 30.0, f"Processing took {processing_time:.2f}s, expected <30s"

    def _create_test_csv_files(self, input_dir: Path):
        """Create test CSV files for testing."""
        np.random.seed(42)

        # Create OHLCV data
        dates = pd.date_range('2023-01-01', periods=500, freq='1H')
        base_price = 100

        for i in range(3):  # Create 3 test files
            prices = base_price + np.random.normal(0, 2, len(dates)).cumsum()
            data = pd.DataFrame({
                'datetime': dates,
                'open': prices + np.random.normal(0, 0.5, len(dates)),
                'high': prices + np.abs(np.random.normal(1, 0.5, len(dates))),
                'low': prices - np.abs(np.random.normal(1, 0.5, len(dates))),
                'close': prices + np.random.normal(0, 0.5, len(dates)),
                'volume': np.random.lognormal(10, 1, len(dates)).astype(int)
            })

            # Ensure high >= low and high >= open/close
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)

            file_path = input_dir / f"test_data_{i}.csv"
            data.to_csv(file_path, index=False)

    def _create_large_test_csv_files(self, input_dir: Path):
        """Create larger test CSV files for performance testing."""
        np.random.seed(42)

        # Create larger OHLCV dataset
        dates = pd.date_range('2022-01-01', periods=10000, freq='1H')
        base_price = 100

        for i in range(2):  # Create 2 larger test files
            prices = base_price + np.random.normal(0, 0.1, len(dates)).cumsum()
            data = pd.DataFrame({
                'datetime': dates,
                'open': prices + np.random.normal(0, 0.1, len(dates)),
                'high': prices + np.abs(np.random.normal(0.2, 0.1, len(dates))),
                'low': prices - np.abs(np.random.normal(0.2, 0.1, len(dates))),
                'close': prices + np.random.normal(0, 0.1, len(dates)),
                'volume': np.random.lognormal(8, 1, len(dates)).astype(int)
            })

            # Ensure high >= low and high >= open/close
            data['high'] = data[['open', 'high', 'close']].max(axis=1)
            data['low'] = data[['open', 'low', 'close']].min(axis=1)

            file_path = input_dir / f"large_test_data_{i}.csv"
            data.to_csv(file_path, index=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])