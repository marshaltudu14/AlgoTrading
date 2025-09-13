#!/usr/bin/env python3
"""
Feature-Specific Normalization Strategies
========================================

Specialized normalization strategies for different types of trading features:
- Technical indicators (RSI, MACD, etc.)
- Price and volume data
- Volatility measures
- Market structure features
- Temporal features

Each feature type has optimal normalization approaches based on its
statistical properties and trading domain knowledge.

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
from dataclasses import dataclass
from enum import Enum
import re
from .normalization import FeatureType, OutlierHandlingStrategy, ImputationStrategy

logger = logging.getLogger(__name__)


class TechnicalIndicatorType(Enum):
    """Types of technical indicators with specific normalization needs."""
    MOMENTUM = "momentum"  # RSI, Stochastic, Williams %R
    TREND = "trend"        # MACD, ADX, Trend Strength
    VOLATILITY = "volatility"  # ATR, Bollinger Bands, Standard Deviation
    VOLUME = "volume"      # Volume-based indicators
    OVERLAY = "overlay"    # Moving averages, Bollinger Bands


@dataclass
class FeatureNormalizationConfig:
    """Configuration for feature-specific normalization."""
    feature_name: str
    feature_type: FeatureType
    indicator_type: Optional[TechnicalIndicatorType] = None
    custom_range: Optional[Tuple[float, float]] = None
    outlier_multiplier: float = 1.5
    imputation_strategy: ImputationStrategy = ImputationStrategy.MEDIAN
    outlier_handling: OutlierHandlingStrategy = OutlierHandlingStrategy.CLIP
    use_robust_scaling: bool = False
    preserve_distribution: bool = False


class FeatureSpecificNormalizer:
    """
    Advanced normalizer with feature-specific strategies for trading data.
    """

    def __init__(self):
        self.feature_configs: Dict[str, FeatureNormalizationConfig] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_type_map: Dict[str, TechnicalIndicatorType] = {}
        self._setup_default_feature_mappings()

    def _setup_default_feature_mappings(self):
        """Setup automatic feature type detection based on naming patterns."""
        self.feature_type_map.update({
            # Momentum indicators
            r'rsi_\d+': TechnicalIndicatorType.MOMENTUM,
            r'stoch.*': TechnicalIndicatorType.MOMENTUM,
            r'williams.*r': TechnicalIndicatorType.MOMENTUM,
            r'cci': TechnicalIndicatorType.MOMENTUM,

            # Trend indicators
            r'macd.*': TechnicalIndicatorType.TREND,
            r'adx': TechnicalIndicatorType.TREND,
            r'di_.*': TechnicalIndicatorType.TREND,
            r'trend_.*': TechnicalIndicatorType.TREND,

            # Volatility indicators
            r'atr': TechnicalIndicatorType.VOLATILITY,
            r'bb_.*': TechnicalIndicatorType.VOLATILITY,
            r'volatility_\d+': TechnicalIndicatorType.VOLATILITY,

            # Overlay indicators
            r'sma_\d+': TechnicalIndicatorType.OVERLAY,
            r'ema_\d+': TechnicalIndicatorType.OVERLAY,

            # Price and volume
            r'price_.*': TechnicalIndicatorType.TREND,
            r'hl_range': TechnicalIndicatorType.VOLATILITY,
            r'body_size': TechnicalIndicatorType.VOLATILITY,
            r'.*_shadow': TechnicalIndicatorType.VOLATILITY,

            # Market structure
            r'.*_cross': TechnicalIndicatorType.TREND,
            r'.*_vs_.*': TechnicalIndicatorType.TREND,
        })

    def auto_configure_features(self, feature_names: List[str]) -> Dict[str, FeatureNormalizationConfig]:
        """
        Automatically configure normalization parameters for all features.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary of feature configurations
        """
        configs = {}

        for feature in feature_names:
            config = self._auto_detect_feature_config(feature)
            configs[feature] = config

        self.feature_configs = configs
        logger.info(f"Auto-configured {len(configs)} features")
        return configs

    def _auto_detect_feature_config(self, feature_name: str) -> FeatureNormalizationConfig:
        """Detect the optimal configuration for a feature based on its name."""
        indicator_type = self._detect_indicator_type(feature_name)
        feature_type = self._detect_feature_type(feature_name)
        custom_range = self._get_custom_range(feature_name, indicator_type)

        return FeatureNormalizationConfig(
            feature_name=feature_name,
            feature_type=feature_type,
            indicator_type=indicator_type,
            custom_range=custom_range,
            outlier_multiplier=self._get_outlier_multiplier(indicator_type),
            imputation_strategy=self._get_imputation_strategy(indicator_type),
            outlier_handling=self._get_outlier_handling(indicator_type),
            use_robust_scaling=self._should_use_robust_scaling(indicator_type),
            preserve_distribution=self._should_preserve_distribution(indicator_type)
        )

    def _detect_indicator_type(self, feature_name: str) -> TechnicalIndicatorType:
        """Detect indicator type based on feature name patterns."""
        feature_lower = feature_name.lower()

        for pattern, indicator_type in self.feature_type_map.items():
            if re.match(pattern, feature_lower):
                return indicator_type

        # Default based on feature name characteristics
        if any(keyword in feature_lower for keyword in ['cross', 'trend', 'slope']):
            return TechnicalIndicatorType.TREND
        elif any(keyword in feature_lower for keyword in ['volatility', 'range', 'shadow']):
            return TechnicalIndicatorType.VOLATILITY
        elif any(keyword in feature_lower for keyword in ['rsi', 'stoch', 'williams']):
            return TechnicalIndicatorType.MOMENTUM
        else:
            return TechnicalIndicatorType.OVERLAY

    def _detect_feature_type(self, feature_name: str) -> FeatureType:
        """Detect feature type based on name and characteristics."""
        feature_lower = feature_name.lower()

        if any(keyword in feature_lower for keyword in ['time', 'hour', 'minute']):
            return FeatureType.TEMPORAL
        elif re.match(r'.*_\d+$', feature_lower) and 'sma' not in feature_lower and 'ema' not in feature_lower:
            return FeatureType.INDICATOR
        elif any(keyword in feature_lower for keyword in ['cross', 'signal']):
            return FeatureType.INDICATOR
        else:
            return FeatureType.CONTINUOUS

    def _get_custom_range(self, feature_name: str, indicator_type: TechnicalIndicatorType) -> Optional[Tuple[float, float]]:
        """Get custom normalization range for specific indicators."""
        feature_lower = feature_name.lower()

        # RSI is naturally 0-100
        if feature_lower.startswith('rsi_'):
            return (0, 100)

        # Stochastic is naturally 0-100
        if 'stoch' in feature_lower:
            return (0, 100)

        # Williams %R is naturally -100 to 0, we'll shift to 0-100
        if 'williams' in feature_lower and 'r' in feature_lower:
            return (0, 100)

        # MACD and derivatives centered around 0
        if 'macd' in feature_lower:
            return (-50, 50)

        # ADX is 0-100
        if 'adx' in feature_lower:
            return (0, 100)

        # CCI is typically -100 to +100
        if feature_lower == 'cci':
            return (-50, 50)

        return None

    def _get_outlier_multiplier(self, indicator_type: TechnicalIndicatorType) -> float:
        """Get outlier detection multiplier based on indicator type."""
        multipliers = {
            TechnicalIndicatorType.MOMENTUM: 2.0,  # Momentum indicators have natural extremes
            TechnicalIndicatorType.VOLATILITY: 3.0,  # Volatility can have legitimate spikes
            TechnicalIndicatorType.TREND: 1.5,  # Trend indicators should be more stable
            TechnicalIndicatorType.OVERLAY: 1.5,  # Overlay indicators follow price
            TechnicalIndicatorType.VOLUME: 2.5,  # Volume can have legitimate spikes
        }
        return multipliers.get(indicator_type, 1.5)

    def _get_imputation_strategy(self, indicator_type: TechnicalIndicatorType) -> ImputationStrategy:
        """Get imputation strategy based on indicator type."""
        strategies = {
            TechnicalIndicatorType.MOMENTUM: ImputationStrategy.MEDIAN,
            TechnicalIndicatorType.VOLATILITY: ImputationStrategy.MEDIAN,
            TechnicalIndicatorType.TREND: ImputationStrategy.FORWARD_FILL,
            TechnicalIndicatorType.OVERLAY: ImputationStrategy.FORWARD_FILL,
            TechnicalIndicatorType.VOLUME: ImputationStrategy.MEDIAN,
        }
        return strategies.get(indicator_type, ImputationStrategy.MEDIAN)

    def _get_outlier_handling(self, indicator_type: TechnicalIndicatorType) -> OutlierHandlingStrategy:
        """Get outlier handling strategy based on indicator type."""
        strategies = {
            TechnicalIndicatorType.MOMENTUM: OutlierHandlingStrategy.CLIP,
            TechnicalIndicatorType.VOLATILITY: OutlierHandlingStrategy.CLIP,
            TechnicalIndicatorType.TREND: OutlierHandlingStrategy.CLIP,
            TechnicalIndicatorType.OVERLAY: OutlierHandlingStrategy.CLIP,
            TechnicalIndicatorType.VOLUME: OutlierHandlingStrategy.CLIP,
        }
        return strategies.get(indicator_type, OutlierHandlingStrategy.CLIP)

    def _should_use_robust_scaling(self, indicator_type: TechnicalIndicatorType) -> bool:
        """Determine if robust scaling should be used."""
        robust_types = [TechnicalIndicatorType.VOLATILITY, TechnicalIndicatorType.VOLUME]
        return indicator_type in robust_types

    def _should_preserve_distribution(self, indicator_type: TechnicalIndicatorType) -> bool:
        """Determine if distribution should be preserved."""
        preserve_types = [TechnicalIndicatorType.MOMENTUM, TechnicalIndicatorType.TREND]
        return indicator_type in preserve_types

    def create_scaler_for_feature(self, config: FeatureNormalizationConfig) -> Any:
        """
        Create the appropriate scaler for a feature based on its configuration.

        Args:
            config: Feature normalization configuration

        Returns:
            Configured scaler
        """
        if config.use_robust_scaling:
            return RobustScaler()
        elif config.preserve_distribution:
            return StandardScaler()
        else:
            feature_range = config.custom_range or (0, 100)
            return MinMaxScaler(feature_range=feature_range)

    def normalize_feature(self, feature_data: pd.Series, config: FeatureNormalizationConfig) -> pd.Series:
        """
        Normalize a single feature using its specific configuration.

        Args:
            feature_data: Feature data to normalize
            config: Feature configuration

        Returns:
            Normalized feature data
        """
        # Handle missing values
        if feature_data.isna().any():
            feature_data = self._handle_missing_values(feature_data, config)

        # Handle outliers
        feature_data = self._handle_outliers(feature_data, config)

        # Apply normalization
        scaler = self.create_scaler_for_feature(config)
        normalized_values = scaler.fit_transform(feature_data.values.reshape(-1, 1))

        return pd.Series(normalized_values.flatten(), index=feature_data.index)

    def _handle_missing_values(self, feature_data: pd.Series, config: FeatureNormalizationConfig) -> pd.Series:
        """Handle missing values based on feature type."""
        if config.imputation_strategy == ImputationStrategy.FORWARD_FILL:
            return feature_data.fillna(method='ffill').fillna(method='bfill')
        elif config.imputation_strategy == ImputationStrategy.BACKWARD_FILL:
            return feature_data.fillna(method='bfill').fillna(method='ffill')
        elif config.imputation_strategy == ImputationStrategy.MEDIAN:
            return feature_data.fillna(feature_data.median())
        elif config.imputation_strategy == ImputationStrategy.MEAN:
            return feature_data.fillna(feature_data.mean())
        elif config.imputation_strategy == ImputationStrategy.MODE:
            return feature_data.fillna(feature_data.mode().iloc[0] if not feature_data.mode().empty else 0)
        else:
            return feature_data.fillna(feature_data.median())

    def _handle_outliers(self, feature_data: pd.Series, config: FeatureNormalizationConfig) -> pd.Series:
        """Handle outliers based on feature type and configuration."""
        if config.outlier_handling == OutlierHandlingStrategy.CLIP:
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.outlier_multiplier * IQR
            upper_bound = Q3 + config.outlier_multiplier * IQR
            return feature_data.clip(lower_bound, upper_bound)

        elif config.outlier_handling == OutlierHandlingStrategy.FLAG:
            # Add outlier flags as additional features
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - config.outlier_multiplier * IQR
            upper_bound = Q3 + config.outlier_multiplier * IQR
            outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
            return feature_data  # Return original with flags to be handled separately

        return feature_data

    def get_feature_summary(self, feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of normalization strategies for all features.

        Args:
            feature_names: List of feature names

        Returns:
            Summary dictionary
        """
        if not self.feature_configs:
            self.auto_configure_features(feature_names)

        summary = {}
        for feature in feature_names:
            if feature in self.feature_configs:
                config = self.feature_configs[feature]
                summary[feature] = {
                    'feature_type': config.feature_type.value,
                    'indicator_type': config.indicator_type.value if config.indicator_type else None,
                    'custom_range': config.custom_range,
                    'outlier_multiplier': config.outlier_multiplier,
                    'imputation_strategy': config.imputation_strategy.value,
                    'outlier_handling': config.outlier_handling.value,
                    'use_robust_scaling': config.use_robust_scaling,
                    'preserve_distribution': config.preserve_distribution
                }

        return summary

    def validate_normalization_config(self, config: FeatureNormalizationConfig) -> bool:
        """
        Validate a normalization configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        # Check for logical consistency
        if config.custom_range and config.custom_range[0] >= config.custom_range[1]:
            logger.error(f"Invalid custom range for {config.feature_name}: {config.custom_range}")
            return False

        if config.outlier_multiplier <= 0:
            logger.error(f"Invalid outlier multiplier for {config.feature_name}: {config.outlier_multiplier}")
            return False

        return True