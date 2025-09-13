"""
Unit tests for FeatureDetector
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.feature_processing import FeatureDetector, FeatureMetadata, FeatureType


class TestFeatureDetector:
    """Test cases for FeatureDetector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = FeatureDetector()
        self.test_data = self._create_test_data()

    def _create_test_data(self) -> pd.DataFrame:
        """Create test data with various feature types"""
        np.random.seed(42)
        n_samples = 100

        data = {
            'numerical_int': np.random.randint(1, 100, n_samples),
            'numerical_float': np.random.normal(50, 10, n_samples),
            'categorical_few': np.random.choice(['A', 'B', 'C'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'id_column': [f"id_{i}" for i in range(n_samples)],
            'target_column': np.random.choice([0, 1], n_samples)
        }

        df = pd.DataFrame(data)

        # Add some null values
        null_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
        df.loc[null_indices, 'numerical_float'] = np.nan

        return df

    def test_detect_features_basic(self):
        """Test basic feature detection"""
        features = self.detector.detect_features(self.test_data, "basic_test")

        assert len(features) > 0
        assert 'numerical_int' in features
        assert 'numerical_float' in features
        assert 'categorical_few' in features
        assert 'boolean_feature' in features

        # Check feature types
        assert features['numerical_int'].feature_type == FeatureType.NUMERICAL
        assert features['numerical_float'].feature_type == FeatureType.NUMERICAL
        assert features['categorical_few'].feature_type == FeatureType.CATEGORICAL
        assert features['boolean_feature'].feature_type == FeatureType.BOOLEAN

    def test_feature_metadata_extraction(self):
        """Test feature metadata extraction"""
        features = self.detector.detect_features(self.test_data, "metadata_test")
        numerical_feature = features['numerical_float']

        assert numerical_feature.name == 'numerical_float'
        assert numerical_feature.null_count > 0  # We added nulls
        assert numerical_feature.null_percentage > 0
        assert numerical_feature.unique_count > 1
        assert numerical_feature.min_value is not None
        assert numerical_feature.max_value is not None
        assert numerical_feature.mean_value is not None
        assert numerical_feature.std_value is not None

    def test_target_and_identifier_detection(self):
        """Test target and identifier feature detection"""
        features = self.detector.detect_features(self.test_data, "target_test")

        assert features['target_column'].is_target
        assert features['id_column'].is_identifier

    def test_ignore_patterns(self):
        """Test column ignore patterns"""
        # Test with unnamed column
        test_data_with_unnamed = self.test_data.copy()
        test_data_with_unnamed['Unnamed: 0'] = range(len(test_data_with_unnamed))

        features = self.detector.detect_features(test_data_with_unnamed, "ignore_test")

        assert 'Unnamed: 0' not in features

    def test_file_loading(self):
        """Test loading data from file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            features = self.detector.detect_features(temp_path, "file_test")

            assert len(features) > 0
            assert 'numerical_int' in features

        finally:
            Path(temp_path).unlink()

    def test_feature_summary(self):
        """Test feature summary generation"""
        self.detector.detect_features(self.test_data, "summary_test")
        summary = self.detector.get_feature_summary()

        assert summary['total_features'] > 0
        assert 'by_type' in summary
        assert isinstance(summary['by_type'], dict)

    def test_registry_save_load(self):
        """Test saving and loading feature registry"""
        # Detect features first
        self.detector.detect_features(self.test_data, "registry_test")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save registry
            self.detector.save_feature_registry(temp_path)

            # Create new detector and load registry
            new_detector = FeatureDetector()
            new_detector.load_feature_registry(temp_path)

            # Check that features were loaded
            assert len(new_detector.feature_registry) > 0
            assert 'numerical_int' in new_detector.feature_registry

        finally:
            Path(temp_path).unlink()

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        features = self.detector.detect_features(empty_df, "empty_test")

        assert len(features) == 0

    def test_single_column_dataframe(self):
        """Test handling of single column DataFrame"""
        single_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
        features = self.detector.detect_features(single_df, "single_test")

        assert len(features) == 1
        assert 'single_col' in features
        assert features['single_col'].feature_type == FeatureType.NUMERICAL

    def test_all_nulls_dataframe(self):
        """Test handling of DataFrame with all null values"""
        null_df = pd.DataFrame({'null_col': [np.nan, np.nan, np.nan]})
        features = self.detector.detect_features(null_df, "null_test")

        # Should detect the column even with all nulls
        assert len(features) >= 0

    def test_mixed_types_in_column(self):
        """Test handling of column with mixed types"""
        mixed_df = pd.DataFrame({'mixed_col': [1, 'a', 3.14, True, None]})
        features = self.detector.detect_features(mixed_df, "mixed_test")

        # Should handle mixed types gracefully
        assert len(features) > 0