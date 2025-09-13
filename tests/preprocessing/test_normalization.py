#!/usr/bin/env python3
"""
Tests for Feature Normalization Module
=====================================

Comprehensive tests for the normalization pipeline including:
- Min-max scaling to 0-100 range
- Feature-specific normalization strategies
- Parameter storage and retrieval
- Outlier detection and handling
- Missing value imputation
- Performance requirements

Author: AlgoTrading System
Version: 1.0
"""

import pytest
import numpy as np
import pandas as pd
import time
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.normalization import (
    FeatureNormalizer, NormalizationParams, FeatureType,
    OutlierHandlingStrategy, ImputationStrategy
)


class TestFeatureNormalizer:
    """Test cases for FeatureNormalizer class."""

    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = FeatureNormalizer()
        assert normalizer.normalization_range == (0.0, 100.0)
        assert len(normalizer.normalization_params) == 0
        assert len(normalizer.fitted_scalers) == 0
        assert len(normalizer.imputers) == 0

    def test_basic_normalization(self):
        """Test basic min-max normalization to 0-100 range."""
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(50, 10, 1000),  # Normal distribution
            'feature2': np.random.exponential(2, 1000),  # Exponential distribution
            'feature3': np.random.uniform(0, 200, 1000)  # Uniform distribution
        })

        normalizer = FeatureNormalizer()
        normalized_data = normalizer.fit_transform(data)

        # Test that all values are in 0-100 range
        for col in normalized_data.columns:
            assert normalized_data[col].min() >= 0
            assert normalized_data[col].max() <= 100

        # Test that normalization parameters are stored
        assert len(normalizer.normalization_params) == 3
        assert 'feature1' in normalizer.normalization_params
        assert 'feature2' in normalizer.normalization_params
        assert 'feature3' in normalizer.normalization_params

    def test_feature_type_detection(self):
        """Test automatic feature type detection."""
        data = pd.DataFrame({
            'continuous_feature': np.random.normal(100, 20, 100),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
            'temporal_feature': pd.date_range('2023-01-01', periods=100, freq='H'),
            'indicator_feature': np.random.choice([0, 1], 100, p=[0.8, 0.2])
        })

        normalizer = FeatureNormalizer()
        normalizer.fit(data)

        # Check that feature types are detected
        assert normalizer.feature_metadata['continuous_feature'] == FeatureType.CONTINUOUS
        assert normalizer.feature_metadata['categorical_feature'] == FeatureType.CATEGORICAL
        assert normalizer.feature_metadata['temporal_feature'] == FeatureType.TEMPORAL
        assert normalizer.feature_metadata['indicator_feature'] == FeatureType.INDICATOR

    def test_missing_value_handling(self):
        """Test missing value imputation strategies."""
        # Create data with missing values
        np.random.seed(42)
        data = pd.DataFrame({
            'normal_data': np.random.normal(50, 10, 100),
            'missing_data': np.random.normal(50, 10, 100)
        })

        # Introduce missing values
        data.loc[10:20, 'missing_data'] = np.nan
        data.loc[50:55, 'normal_data'] = np.nan

        normalizer = FeatureNormalizer()
        normalized_data = normalizer.fit_transform(data)

        # Check that missing values are handled
        assert not normalized_data.isna().any().any()

        # Check that imputers are created
        assert len(normalizer.imputers) > 0

    def test_outlier_detection_and_handling(self):
        """Test outlier detection and handling."""
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 950)
        outliers = np.array([200, -100, 150, 180, -50])
        data_with_outliers = np.concatenate([normal_data, outliers])

        data = pd.DataFrame({'outlier_feature': data_with_outliers})

        normalizer = FeatureNormalizer()
        normalized_data = normalizer.fit_transform(data)

        # Check that outliers are handled (should be clipped to reasonable range)
        assert normalized_data['outlier_feature'].min() >= 0
        assert normalized_data['outlier_feature'].max() <= 100

        # Check that outlier parameters are calculated
        params = normalizer.normalization_params['outlier_feature']
        assert params.outlier_threshold > 0

    def test_inverse_transformation(self):
        """Test inverse transformation functionality."""
        # Create test data
        np.random.seed(42)
        original_data = pd.DataFrame({
            'test_feature': np.random.normal(100, 20, 1000)
        })

        normalizer = FeatureNormalizer()
        normalized_data = normalizer.fit_transform(original_data)
        reconstructed_data = normalizer.inverse_transform(normalized_data)

        # Check that inverse transformation is accurate (within tolerance)
        np.testing.assert_allclose(
            original_data['test_feature'].values,
            reconstructed_data['test_feature'].values,
            rtol=1e-10,
            atol=1e-10
        )

    def test_parameter_storage_and_retrieval(self):
        """Test saving and loading normalization parameters."""
        # Create test data
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.normal(50, 10, 100),
            'feature2': np.random.uniform(0, 100, 100)
        })

        with tempfile.TemporaryDirectory() as temp_dir:
            # Train normalizer and save parameters
            normalizer1 = FeatureNormalizer()
            normalizer1.fit_transform(data)

            params_file = Path(temp_dir) / "normalization_params.json"
            normalizer1.save_parameters(str(params_file))

            # Create new normalizer and load parameters
            normalizer2 = FeatureNormalizer()
            normalizer2.load_parameters(str(params_file))

            # Check that parameters are loaded correctly
            assert len(normalizer2.normalization_params) == 2
            assert 'feature1' in normalizer2.normalization_params
            assert 'feature2' in normalizer2.normalization_params

            # Test transformation with loaded parameters
            new_data = pd.DataFrame({
                'feature1': np.random.normal(50, 10, 50),
                'feature2': np.random.uniform(0, 100, 50)
            })

            normalized_new = normalizer2.transform(new_data)
            assert normalized_new.shape == new_data.shape

    def test_performance_requirements(self):
        """Test performance requirements (<50ms per batch)."""
        # Create larger test dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            f'feature_{i}': np.random.normal(100, 20, 10000)
            for i in range(50)  # 50 features
        })

        normalizer = FeatureNormalizer()

        # Test fitting performance
        start_time = time.time()
        normalizer.fit(large_data)
        fit_time = time.time() - start_time
        assert fit_time < 0.05, f"Fitting took {fit_time:.3f}s, expected <0.05s"

        # Test transformation performance
        start_time = time.time()
        normalized_data = normalizer.transform(large_data)
        transform_time = time.time() - start_time
        assert transform_time < 0.05, f"Transformation took {transform_time:.3f}s, expected <0.05s"

        # Test that all features are processed
        assert len(normalized_data.columns) == 50

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty DataFrame
        empty_data = pd.DataFrame()
        normalizer = FeatureNormalizer()

        with pytest.raises(ValueError):
            normalizer.fit(empty_data)

        # Test with single feature
        single_feature = pd.DataFrame({'single': [1, 2, 3, 4, 5]})
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(single_feature)
        assert result.shape == single_feature.shape

        # Test with all identical values
        identical_data = pd.DataFrame({'identical': [5, 5, 5, 5, 5]})
        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(identical_data)
        # Should handle identical values gracefully
        assert result.shape == identical_data.shape

    def test_different_imputation_strategies(self):
        """Test different imputation strategies."""
        # Create data with missing values
        data = pd.DataFrame({
            'mean_test': [1, 2, np.nan, 4, 5],
            'median_test': [1, np.nan, 3, np.nan, 5],
            'ffill_test': [1, np.nan, np.nan, 4, 5]
        })

        normalizer = FeatureNormalizer()
        result = normalizer.fit_transform(data)

        # Check that all missing values are handled
        assert not result.isna().any().any()

    def test_feature_statistics(self):
        """Test feature statistics generation."""
        np.random.seed(42)
        data = pd.DataFrame({
            'test_feature': np.random.normal(100, 20, 1000)
        })

        normalizer = FeatureNormalizer()
        normalizer.fit(data)

        stats = normalizer.get_feature_statistics()

        assert 'test_feature' in stats
        feature_stats = stats['test_feature']
        assert 'min' in feature_stats
        assert 'max' in feature_stats
        assert 'mean' in feature_stats
        assert 'std' in feature_stats
        assert 'outlier_threshold' in feature_stats

    def test_performance_stats(self):
        """Test performance statistics tracking."""
        np.random.seed(42)
        data = pd.DataFrame({
            'test_feature': np.random.normal(100, 20, 1000)
        })

        normalizer = FeatureNormalizer()
        normalizer.fit_transform(data)

        perf_stats = normalizer.get_performance_stats()

        assert 'fit_time' in perf_stats
        assert 'transform_time' in perf_stats
        assert perf_stats['fit_time'] > 0
        assert perf_stats['transform_time'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])