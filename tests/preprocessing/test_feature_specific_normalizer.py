#!/usr/bin/env python3
"""
Tests for Feature-Specific Normalization Module
===============================================

Comprehensive tests for feature-specific normalization strategies including:
- Technical indicator type detection
- Custom normalization ranges
- Feature-specific outlier handling
- Imputation strategy selection
- Configuration validation

Author: AlgoTrading System
Version: 1.0
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.feature_specific_normalizer import (
    FeatureSpecificNormalizer, FeatureNormalizationConfig,
    TechnicalIndicatorType, FeatureType
)


class TestFeatureSpecificNormalizer:
    """Test cases for FeatureSpecificNormalizer class."""

    def test_initialization(self):
        """Test normalizer initialization."""
        normalizer = FeatureSpecificNormalizer()
        assert len(normalizer.feature_configs) == 0
        assert len(normalizer.feature_type_map) > 0  # Should have default mappings

    def test_technical_indicator_type_detection(self):
        """Test automatic detection of technical indicator types."""
        normalizer = FeatureSpecificNormalizer()

        test_features = [
            ('rsi_14', TechnicalIndicatorType.MOMENTUM),
            ('macd_12_26_9', TechnicalIndicatorType.TREND),
            ('atr', TechnicalIndicatorType.VOLATILITY),
            ('sma_20', TechnicalIndicatorType.OVERLAY),
            ('stoch_k', TechnicalIndicatorType.MOMENTUM),
            ('adx', TechnicalIndicatorType.TREND),
            ('bb_upper', TechnicalIndicatorType.VOLATILITY),
            ('williams_r', TechnicalIndicatorType.MOMENTUM),
            ('cci', TechnicalIndicatorType.MOMENTUM),
            ('price_change', TechnicalIndicatorType.TREND)
        ]

        for feature_name, expected_type in test_features:
            detected_type = normalizer._detect_indicator_type(feature_name)
            assert detected_type == expected_type, \
                f"Feature {feature_name}: expected {expected_type}, got {detected_type}"

    def test_custom_range_detection(self):
        """Test detection of custom normalization ranges."""
        normalizer = FeatureSpecificNormalizer()

        test_cases = [
            ('rsi_14', (0, 100)),
            ('stoch_k', (0, 100)),
            ('williams_r_14', (0, 100)),
            ('macd', (-50, 50)),
            ('adx', (0, 100)),
            ('cci', (-50, 50)),
            ('unknown_feature', None)
        ]

        for feature_name, expected_range in test_cases:
            detected_range = normalizer._get_custom_range(
                feature_name,
                normalizer._detect_indicator_type(feature_name)
            )
            assert detected_range == expected_range, \
                f"Feature {feature_name}: expected {expected_range}, got {detected_range}"

    def test_outlier_multiplier_selection(self):
        """Test selection of outlier multipliers based on indicator type."""
        normalizer = FeatureSpecificNormalizer()

        multipliers = {
            TechnicalIndicatorType.MOMENTUM: 2.0,
            TechnicalIndicatorType.VOLATILITY: 3.0,
            TechnicalIndicatorType.TREND: 1.5,
            TechnicalIndicatorType.OVERLAY: 1.5,
            TechnicalIndicatorType.VOLUME: 2.5,
        }

        for indicator_type, expected_multiplier in multipliers.items():
            actual_multiplier = normalizer._get_outlier_multiplier(indicator_type)
            assert actual_multiplier == expected_multiplier, \
                f"Indicator type {indicator_type}: expected {expected_multiplier}, got {actual_multiplier}"

    def test_imputation_strategy_selection(self):
        """Test selection of imputation strategies based on indicator type."""
        from src.data_processing.normalization import ImputationStrategy

        normalizer = FeatureSpecificNormalizer()

        strategies = {
            TechnicalIndicatorType.MOMENTUM: ImputationStrategy.MEDIAN,
            TechnicalIndicatorType.VOLATILITY: ImputationStrategy.MEDIAN,
            TechnicalIndicatorType.TREND: ImputationStrategy.FORWARD_FILL,
            TechnicalIndicatorType.OVERLAY: ImputationStrategy.FORWARD_FILL,
            TechnicalIndicatorType.VOLUME: ImputationStrategy.MEDIAN,
        }

        for indicator_type, expected_strategy in strategies.items():
            actual_strategy = normalizer._get_imputation_strategy(indicator_type)
            assert actual_strategy == expected_strategy, \
                f"Indicator type {indicator_type}: expected {expected_strategy}, got {actual_strategy}"

    def test_auto_configuration(self):
        """Test automatic configuration of multiple features."""
        feature_names = [
            'rsi_14', 'macd_12_26_9', 'atr', 'sma_20', 'stoch_k',
            'adx', 'bb_width', 'williams_r', 'cci', 'price_change',
            'volatility_10', 'ema_50', 'di_plus', 'momentum_10'
        ]

        normalizer = FeatureSpecificNormalizer()
        configs = normalizer.auto_configure_features(feature_names)

        # Check that all features are configured
        assert len(configs) == len(feature_names)
        for feature_name in feature_names:
            assert feature_name in configs
            assert configs[feature_name].feature_name == feature_name

        # Check that configurations are stored
        assert len(normalizer.feature_configs) == len(feature_names)

    def test_feature_normalization(self):
        """Test normalization of features with specific configurations."""
        # Create test data for different indicator types
        np.random.seed(42)
        test_data = pd.DataFrame({
            'rsi_14': np.random.uniform(0, 100, 1000),  # Should be 0-100 naturally
            'macd': np.random.normal(0, 2, 1000),      # Should be centered around 0
            'atr': np.random.exponential(1, 1000),     # Should be positive with outliers
            'sma_20': np.random.normal(100, 20, 1000)   # Should follow price movements
        })

        normalizer = FeatureSpecificNormalizer()
        configs = normalizer.auto_configure_features(test_data.columns.tolist())

        for feature in test_data.columns:
            config = configs[feature]
            normalized_feature = normalizer.normalize_feature(test_data[feature], config)

            # Check that normalization is applied
            assert len(normalized_feature) == len(test_data[feature])
            assert not normalized_feature.isna().any()

            # Check that values are in reasonable range
            if config.custom_range:
                min_val, max_val = config.custom_range
                assert normalized_feature.min() >= min_val - 1  # Small tolerance
                assert normalized_feature.max() <= max_val + 1   # Small tolerance
            else:
                # Default range should be 0-100
                assert normalized_feature.min() >= -1   # Small tolerance
                assert normalized_feature.max() <= 101  # Small tolerance

    def test_outlier_handling(self):
        """Test outlier handling for different strategies."""
        from src.data_processing.normalization import OutlierHandlingStrategy

        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 950)
        outliers = np.array([200, -100, 150])
        data_with_outliers = np.concatenate([normal_data, outliers])

        test_data = pd.DataFrame({'test_feature': data_with_outliers})

        normalizer = FeatureSpecificNormalizer()

        # Test clipping strategy
        config_clip = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            outlier_handling=OutlierHandlingStrategy.CLIP
        )
        clipped_data = normalizer._handle_outliers(test_data['test_feature'], config_clip)

        # Check that outliers are clipped
        assert clipped_data.min() > -100  # Should be clipped
        assert clipped_data.max() < 150   # Should be clipped

        # Test flagging strategy
        config_flag = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            outlier_handling=OutlierHandlingStrategy.FLAG
        )
        flagged_data = normalizer._handle_outliers(test_data['test_feature'], config_flag)

        # Flagging should not modify the data
        pd.testing.assert_series_equal(test_data['test_feature'], flagged_data)

    def test_missing_value_handling(self):
        """Test missing value handling with different strategies."""
        from src.data_processing.normalization import ImputationStrategy

        # Create data with missing values
        test_data = pd.Series([1, 2, np.nan, 4, np.nan, 6, np.nan, 8])

        normalizer = FeatureSpecificNormalizer()

        # Test median imputation
        config_median = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            imputation_strategy=ImputationStrategy.MEDIAN
        )
        median_filled = normalizer._handle_missing_values(test_data, config_median)
        assert not median_filled.isna().any()
        assert median_filled.iloc[2] == test_data.median()

        # Test forward fill
        config_ffill = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            imputation_strategy=ImputationStrategy.FORWARD_FILL
        )
        ffill_data = normalizer._handle_missing_values(test_data, config_ffill)
        assert not ffill_data.isna().any()
        assert ffill_data.iloc[2] == 2  # Forward filled from previous value

    def test_scaler_creation(self):
        """Test creation of appropriate scalers for different configurations."""
        from src.data_processing.normalization import ImputationStrategy, OutlierHandlingStrategy

        normalizer = FeatureSpecificNormalizer()

        # Test standard scaler
        config_standard = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            preserve_distribution=True
        )
        scaler_standard = normalizer.create_scaler_for_feature(config_standard)
        assert scaler_standard.__class__.__name__ == 'StandardScaler'

        # Test robust scaler
        config_robust = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            use_robust_scaling=True
        )
        scaler_robust = normalizer.create_scaler_for_feature(config_robust)
        assert scaler_robust.__class__.__name__ == 'RobustScaler'

        # Test min-max scaler with custom range
        config_custom = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            custom_range=(-50, 50)
        )
        scaler_custom = normalizer.create_scaler_for_feature(config_custom)
        assert scaler_custom.__class__.__name__ == 'MinMaxScaler'
        assert scaler_custom.feature_range == (-50, 50)

    def test_configuration_validation(self):
        """Test validation of normalization configurations."""
        from src.data_processing.normalization import ImputationStrategy, OutlierHandlingStrategy

        normalizer = FeatureSpecificNormalizer()

        # Test valid configuration
        valid_config = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            custom_range=(0, 100),
            outlier_multiplier=1.5
        )
        assert normalizer.validate_normalization_config(valid_config) == True

        # Test invalid custom range
        invalid_range_config = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            custom_range=(100, 0),  # Invalid: min > max
            outlier_multiplier=1.5
        )
        assert normalizer.validate_normalization_config(invalid_range_config) == False

        # Test invalid outlier multiplier
        invalid_multiplier_config = FeatureNormalizationConfig(
            feature_name='test_feature',
            feature_type=FeatureType.CONTINUOUS,
            outlier_multiplier=-1.0  # Invalid: negative
        )
        assert normalizer.validate_normalization_config(invalid_multiplier_config) == False

    def test_feature_summary_generation(self):
        """Test generation of feature normalization summary."""
        feature_names = ['rsi_14', 'macd_12_26_9', 'atr', 'sma_20']

        normalizer = FeatureSpecificNormalizer()
        configs = normalizer.auto_configure_features(feature_names)
        summary = normalizer.get_feature_summary(feature_names)

        # Check that summary contains all features
        assert len(summary) == len(feature_names)
        for feature in feature_names:
            assert feature in summary

        # Check summary structure
        for feature_summary in summary.values():
            assert 'feature_type' in feature_summary
            assert 'indicator_type' in feature_summary
            assert 'custom_range' in feature_summary
            assert 'outlier_multiplier' in feature_summary
            assert 'imputation_strategy' in feature_summary
            assert 'outlier_handling' in feature_summary
            assert 'use_robust_scaling' in feature_summary
            assert 'preserve_distribution' in feature_summary

    def test_trading_specific_features(self):
        """Test normalization of trading-specific features."""
        # Create realistic trading feature data
        np.random.seed(42)
        trading_data = pd.DataFrame({
            'rsi_14': np.random.uniform(20, 80, 1000),  # RSI typically 20-80
            'macd_12_26_9': np.random.normal(0, 1, 1000),  # MACD centered at 0
            'atr': np.random.exponential(0.5, 1000),      # ATR positive and skewed
            'bb_width': np.random.uniform(1, 10, 1000),   # BB width positive
            'sma_20_cross': np.random.choice([-1, 0, 1], 1000),  # Crossover signals
            'price_vs_sma_20': np.random.normal(0, 2, 1000),  # Price deviation
            'volatility_10': np.random.exponential(1, 1000),     # Volatility
        })

        normalizer = FeatureSpecificNormalizer()
        configs = normalizer.auto_configure_features(trading_data.columns.tolist())

        # Normalize all features
        normalized_data = trading_data.copy()
        for feature in trading_data.columns:
            config = configs[feature]
            normalized_data[feature] = normalizer.normalize_feature(trading_data[feature], config)

        # Check that all features are properly normalized
        for feature in normalized_data.columns:
            assert not normalized_data[feature].isna().any()
            assert normalized_data[feature].min() >= -10  # Reasonable lower bound
            assert normalized_data[feature].max() <= 110   # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v"])