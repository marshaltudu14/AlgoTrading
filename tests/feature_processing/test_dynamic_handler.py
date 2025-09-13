"""
Unit tests for DynamicFeatureHandler
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.feature_processing import (
    DynamicFeatureHandler, FeatureRegistry, FeatureMetadata, FeatureType,
    ValidationLevel, CompatibilityStatus
)


class TestDynamicFeatureHandler:
    """Test cases for DynamicFeatureHandler"""

    def setup_method(self):
        """Setup test fixtures"""
        self.feature_registry = FeatureRegistry()
        self.handler = DynamicFeatureHandler(self.feature_registry)
        self.test_data = self._create_test_data()
        self._setup_registry()

    def _create_test_data(self) -> pd.DataFrame:
        """Create test data"""
        np.random.seed(42)
        n_samples = 100

        data = {
            'numerical_feature': np.random.normal(50, 10, n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples),
            'id_feature': [f"id_{i}" for i in range(n_samples)]
        }

        df = pd.DataFrame(data)

        # Add some null values
        null_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
        df.loc[null_indices, 'numerical_feature'] = np.nan

        return df

    def _setup_registry(self):
        """Setup feature registry with test features"""
        detector_config = {'max_categories': 10}
        detector = type('MockDetector', (), {'config': detector_config})()

        # Create and register test features
        features = [
            FeatureMetadata(
                name='numerical_feature',
                feature_type=FeatureType.NUMERICAL,
                dtype='float64',
                null_count=5,
                null_percentage=0.05,
                unique_count=95,
                min_value=20.0,
                max_value=80.0,
                mean_value=50.0,
                std_value=10.0
            ),
            FeatureMetadata(
                name='categorical_feature',
                feature_type=FeatureType.CATEGORICAL,
                dtype='object',
                null_count=0,
                null_percentage=0.0,
                unique_count=3,
                categories=['A', 'B', 'C']
            ),
            FeatureMetadata(
                name='boolean_feature',
                feature_type=FeatureType.BOOLEAN,
                dtype='bool',
                null_count=0,
                null_percentage=0.0,
                unique_count=2
            )
        ]

        for feature in features:
            self.feature_registry.register_feature(feature)

    def test_create_schema(self):
        """Test schema creation"""
        schema = self.handler.create_schema(self.test_data)

        assert len(schema) > 0
        assert 'numerical_feature' in schema
        assert 'categorical_feature' in schema

        # Check schema properties
        numerical_schema = schema['numerical_feature']
        assert numerical_schema.feature_type == FeatureType.NUMERICAL
        assert numerical_schema.min_value is not None
        assert numerical_schema.max_value is not None

    def test_validate_features_strict(self):
        """Test strict validation"""
        result = self.handler.validate_features(
            self.test_data,
            validation_level=ValidationLevel.STRICT
        )

        assert isinstance(result.is_valid, bool)
        assert isinstance(result.status, CompatibilityStatus)
        assert isinstance(result.issues, list)
        assert isinstance(result.warnings, list)

    def test_validate_features_permissive(self):
        """Test permissive validation"""
        # Test with additional features
        modified_data = self.test_data.copy()
        modified_data['new_feature'] = np.random.normal(0, 1, len(modified_data))

        result = self.handler.validate_features(
            modified_data,
            validation_level=ValidationLevel.PERMISSIVE
        )

        # Permissive validation should accept new features
        assert len(result.new_features) > 0
        assert 'new_feature' in result.new_features

    def test_validate_features_missing_required(self):
        """Test validation with missing required features"""
        # Remove a feature
        incomplete_data = self.test_data.drop(columns=['numerical_feature'])

        result = self.handler.validate_features(incomplete_data)

        assert 'numerical_feature' in result.missing_features
        assert not result.is_valid

    def test_type_conversion(self):
        """Test automatic type conversion"""
        # Create data with compatible but different types, but within expected range
        test_data_int = self.test_data.copy()
        # Fill NaN values and ensure values are within reasonable range
        test_data_int['numerical_feature'] = test_data_int['numerical_feature'].fillna(50).astype(int)

        result = self.handler.validate_features(test_data_int)

        # Should handle type conversion gracefully
        assert result.is_valid or result.status == CompatibilityStatus.PARTIALLY_COMPATIBLE

    def test_high_null_percentage(self):
        """Test handling of high null percentage"""
        # Create data with high null percentage
        high_null_data = self.test_data.copy()
        high_null_data.loc[high_null_data.index[:90], 'numerical_feature'] = np.nan  # 90% nulls

        result = self.handler.validate_features(high_null_data)

        # Should detect high null percentage
        # Note: With permissive validation, this might still pass but with warnings
        assert any('null' in warning.lower() for warning in result.warnings) or \
               any('null' in issue.lower() for issue in result.issues)

    def test_resolve_feature_dependencies(self):
        """Test feature dependency resolution"""
        feature_names = ['id_feature', 'numerical_feature', 'categorical_feature']

        result = self.handler.resolve_feature_dependencies(feature_names)

        assert 'resolved_order' in result
        assert 'missing_features' in result
        assert 'dependency_graph' in result

        # The resolved order should contain all features if they exist in registry
        if not result['missing_features']:
            assert len(result['resolved_order']) == len(feature_names)

    def test_create_feature_configuration(self):
        """Test feature configuration creation"""
        feature_names = ['numerical_feature', 'categorical_feature']

        config = self.handler.create_feature_configuration(feature_names)

        assert 'features' in config
        assert 'processing_order' in config
        assert 'metadata' in config

        # Check individual feature config
        assert 'numerical_feature' in config['features']
        assert config['features']['numerical_feature']['type'] == 'numerical'

    def test_validation_history(self):
        """Test validation history tracking"""
        # Run multiple validations
        for i in range(3):
            self.handler.validate_features(self.test_data)

        summary = self.handler.get_validation_summary()

        assert summary['total_validations'] == 3
        assert 'successful_validations' in summary
        assert 'failed_validations' in summary

    def test_save_load_configuration(self):
        """Test saving and loading handler configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save configuration
            self.handler.save_configuration(temp_path)

            # Create new handler and load configuration
            new_handler = DynamicFeatureHandler(self.feature_registry)
            new_handler.load_configuration(temp_path)

            # Check that configuration was loaded
            assert new_handler.config == self.handler.config

        finally:
            Path(temp_path).unlink()

    def test_validation_level_comparison(self):
        """Test different validation levels"""
        modified_data = self.test_data.copy()
        modified_data['new_feature'] = np.random.normal(0, 1, len(modified_data))

        # Test different validation levels
        strict_result = self.handler.validate_features(
            modified_data, validation_level=ValidationLevel.STRICT
        )
        lenient_result = self.handler.validate_features(
            modified_data, validation_level=ValidationLevel.LENIENT
        )
        permissive_result = self.handler.validate_features(
            modified_data, validation_level=ValidationLevel.PERMISSIVE
        )

        # Strict should be most restrictive
        assert not strict_result.is_valid
        assert len(strict_result.new_features) > 0

        # Permissive should be most lenient
        assert permissive_result.is_valid or permissive_result.status != CompatibilityStatus.INCOMPATIBLE

    def test_empty_dataframe_validation(self):
        """Test validation of empty DataFrame"""
        empty_df = pd.DataFrame()

        result = self.handler.validate_features(empty_df)

        # Should handle empty DataFrame gracefully
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.status, CompatibilityStatus)