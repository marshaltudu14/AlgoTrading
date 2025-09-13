"""
Instrument-Specific Feature Normalization

This module provides normalization capabilities tailored to specific trading instruments,
accounting for their unique characteristics and market behaviors.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

from .instrument_embeddings import InstrumentRegistry, InstrumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for instrument-specific normalization"""
    method: str  # 'standard', 'minmax', 'robust', 'quantile'
    feature_range: Tuple[float, float] = (0.0, 1.0)  # For minmax scaling
    quantile_range: Tuple[float, float] = (25.0, 75.0)  # For robust scaling
    handle_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    instrument_aware: bool = True  # Whether to use instrument-specific parameters
    timeframe_aware: bool = True  # Whether to use timeframe-specific parameters


@dataclass
class InstrumentNormalizerStats:
    """Statistics for instrument-specific normalization"""
    symbol: str
    feature_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    median: float
    q25: float
    q75: float
    outlier_count: int
    sample_count: int
    timeframe: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class InstrumentFeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Feature normalizer that accounts for instrument-specific characteristics
    """

    def __init__(self, config: NormalizationConfig):
        """
        Initialize normalizer

        Args:
            config: Normalization configuration
        """
        self.config = config
        self.scalers: Dict[str, Dict[str, BaseEstimator]] = {}
        self.stats: Dict[str, Dict[str, InstrumentNormalizerStats]] = {}
        self.instrument_registry: Optional[InstrumentRegistry] = None
        self.fitted = False

    def set_instrument_registry(self, registry: InstrumentRegistry):
        """
        Set instrument registry for instrument-aware normalization

        Args:
            registry: Instrument registry
        """
        self.instrument_registry = registry

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None,
            symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None,
            feature_names: Optional[List[str]] = None) -> 'InstrumentFeatureNormalizer':
        """
        Fit normalizer to data

        Args:
            X: Input data
            y: Target values (not used for normalization)
            symbols: List of instrument symbols for each sample
            timeframes: List of timeframes for each sample
            feature_names: Names of features

        Returns:
            Self
        """
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            feature_names = feature_names or [f'feature_{i}' for i in range(X_array.shape[1])]

        if symbols is None:
            symbols = ['default'] * len(X_array)

        if timeframes is None:
            timeframes = ['default'] * len(X_array)

        # Group data by symbol and timeframe
        data_groups = self._group_data_by_instrument_timeframe(
            X_array, symbols, timeframes, feature_names
        )

        # Fit scalers for each group
        for group_key, group_data in data_groups.items():
            symbol, timeframe = group_key

            if symbol not in self.scalers:
                self.scalers[symbol] = {}

            if timeframe not in self.scalers[symbol]:
                self.scalers[symbol][timeframe] = {}

            if symbol not in self.stats:
                self.stats[symbol] = {}

            if timeframe not in self.stats[symbol]:
                self.stats[symbol][timeframe] = {}

            # Fit scaler for each feature
            for i, feature_name in enumerate(feature_names):
                feature_data = group_data[:, i]

                # Handle outliers if configured
                if self.config.handle_outliers:
                    feature_data, outlier_count = self._handle_outliers(feature_data)
                else:
                    outlier_count = 0

                # Compute statistics
                stats = self._compute_statistics(feature_data, symbol, feature_name, timeframe, outlier_count)
                self.stats[symbol][timeframe][feature_name] = stats

                # Create and fit scaler
                scaler = self._create_scaler()
                scaler.fit(feature_data.reshape(-1, 1))
                self.scalers[symbol][timeframe][feature_name] = scaler

        self.fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray],
                  symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None,
                  feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Transform data using fitted normalizers

        Args:
            X: Input data
            symbols: List of instrument symbols for each sample
            timeframes: List of timeframes for each sample
            feature_names: Names of features

        Returns:
            Normalized data
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transforming data")

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            feature_names = feature_names or [f'feature_{i}' for i in range(X_array.shape[1])]

        if symbols is None:
            symbols = ['default'] * len(X_array)

        if timeframes is None:
            timeframes = ['default'] * len(X_array)

        # Initialize output array
        X_normalized = np.zeros_like(X_array)

        # Transform each sample
        for i, (sample, symbol, timeframe) in enumerate(zip(X_array, symbols, timeframes)):
            for j, feature_name in enumerate(feature_names):
                feature_value = sample[j]

                # Get appropriate scaler
                scaler = self._get_scaler(symbol, timeframe, feature_name)
                normalized_value = scaler.transform([[feature_value]])[0, 0]

                # Apply instrument-specific adjustments if registry is available
                if self.instrument_registry and self.config.instrument_aware:
                    normalized_value = self._apply_instrument_adjustments(
                        normalized_value, symbol, feature_name
                    )

                X_normalized[i, j] = normalized_value

        return X_normalized

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y=None,
                     symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None,
                     feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Fit normalizer and transform data

        Args:
            X: Input data
            y: Target values (not used for normalization)
            symbols: List of instrument symbols for each sample
            timeframes: List of timeframes for each sample
            feature_names: Names of features

        Returns:
            Normalized data
        """
        return self.fit(X, y, symbols, timeframes, feature_names).transform(
            X, symbols, timeframes, feature_names
        )

    def inverse_transform(self, X: Union[pd.DataFrame, np.ndarray],
                         symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None,
                         feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale

        Args:
            X: Normalized data
            symbols: List of instrument symbols for each sample
            timeframes: List of timeframes for each sample
            feature_names: Names of features

        Returns:
            Data in original scale
        """
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming data")

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            feature_names = feature_names or [f'feature_{i}' for i in range(X_array.shape[1])]

        if symbols is None:
            symbols = ['default'] * len(X_array)

        if timeframes is None:
            timeframes = ['default'] * len(X_array)

        # Initialize output array
        X_original = np.zeros_like(X_array)

        # Inverse transform each sample
        for i, (sample, symbol, timeframe) in enumerate(zip(X_array, symbols, timeframes)):
            for j, feature_name in enumerate(feature_names):
                normalized_value = sample[j]

                # Remove instrument-specific adjustments if applied
                if self.instrument_registry and self.config.instrument_aware:
                    normalized_value = self._remove_instrument_adjustments(
                        normalized_value, symbol, feature_name
                    )

                # Get appropriate scaler
                scaler = self._get_scaler(symbol, timeframe, feature_name)
                original_value = scaler.inverse_transform([[normalized_value]])[0, 0]

                X_original[i, j] = original_value

        return X_original

    def get_statistics(self, symbol: Optional[str] = None, timeframe: Optional[str] = None,
                      feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get normalization statistics

        Args:
            symbol: Instrument symbol (None for all)
            timeframe: Timeframe (None for all)
            feature_name: Feature name (None for all)

        Returns:
            Statistics dictionary
        """
        if symbol is None:
            return self.stats

        if symbol not in self.stats:
            return {}

        if timeframe is None:
            return self.stats[symbol]

        if timeframe not in self.stats[symbol]:
            return {}

        if feature_name is None:
            return self.stats[symbol][timeframe]

        return self.stats[symbol][timeframe].get(feature_name, {})

    def update_statistics(self, new_data: np.ndarray, symbol: str, timeframe: str,
                          feature_names: List[str]):
        """
        Update normalization statistics with new data

        Args:
            new_data: New data samples
            symbol: Instrument symbol
            timeframe: Timeframe
            feature_names: Feature names
        """
        if symbol not in self.stats:
            self.stats[symbol] = {}

        if timeframe not in self.stats[symbol]:
            self.stats[symbol][timeframe] = {}

        for i, feature_name in enumerate(feature_names):
            feature_data = new_data[:, i]

            # Handle outliers
            if self.config.handle_outliers:
                feature_data, outlier_count = self._handle_outliers(feature_data)
            else:
                outlier_count = 0

            # Compute new statistics
            new_stats = self._compute_statistics(feature_data, symbol, feature_name, timeframe, outlier_count)

            # Update existing statistics
            if feature_name in self.stats[symbol][timeframe]:
                existing_stats = self.stats[symbol][timeframe][feature_name]
                # Combine statistics (simple approach: use weighted average)
                weight_existing = existing_stats.sample_count
                weight_new = new_stats.sample_count
                total_weight = weight_existing + weight_new

                if total_weight > 0:
                    new_stats.mean = (existing_stats.mean * weight_existing + new_stats.mean * weight_new) / total_weight
                    new_stats.sample_count = total_weight

            self.stats[symbol][timeframe][feature_name] = new_stats

    def save_normalizer(self, path: str):
        """
        Save normalizer to file

        Args:
            path: File path
        """
        normalizer_data = {
            'config': {
                'method': self.config.method,
                'feature_range': self.config.feature_range,
                'quantile_range': self.config.quantile_range,
                'handle_outliers': self.config.handle_outliers,
                'outlier_threshold': self.config.outlier_threshold,
                'instrument_aware': self.config.instrument_aware,
                'timeframe_aware': self.config.timeframe_aware
            },
            'stats': self._serialize_stats(),
            'fitted': self.fitted
        }

        with open(path, 'wb') as f:
            pickle.dump(normalizer_data, f)

        logger.info(f"Normalizer saved to {path}")

    def load_normalizer(self, path: str):
        """
        Load normalizer from file

        Args:
            path: File path
        """
        with open(path, 'rb') as f:
            normalizer_data = pickle.load(f)

        # Restore configuration
        self.config = NormalizationConfig(**normalizer_data['config'])
        self.stats = self._deserialize_stats(normalizer_data['stats'])
        self.fitted = normalizer_data['fitted']

        logger.info(f"Normalizer loaded from {path}")

    def _group_data_by_instrument_timeframe(self, X: np.ndarray, symbols: List[str],
                                          timeframes: List[str], feature_names: List[str]) -> Dict[Tuple[str, str], np.ndarray]:
        """Group data by instrument and timeframe"""
        groups = {}

        for i, (sample, symbol, timeframe) in enumerate(zip(X, symbols, timeframes)):
            group_key = (symbol, timeframe)

            if group_key not in groups:
                groups[group_key] = []

            groups[group_key].append(sample)

        # Convert lists to arrays
        return {key: np.array(data) for key, data in groups.items()}

    def _create_scaler(self) -> BaseEstimator:
        """Create scaler based on configuration"""
        if self.config.method == 'standard':
            return StandardScaler()
        elif self.config.method == 'minmax':
            return MinMaxScaler(feature_range=self.config.feature_range)
        elif self.config.method == 'robust':
            return RobustScaler(quantile_range=self.config.quantile_range)
        else:
            raise ValueError(f"Unknown normalization method: {self.config.method}")

    def _handle_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Handle outliers in data"""
        if len(data) == 0:
            return data, 0

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return data, 0

        # Identify outliers
        outlier_mask = np.abs(data - mean) > self.config.outlier_threshold * std
        outlier_count = np.sum(outlier_mask)

        # Cap outliers
        data_capped = data.copy()
        data_capped[outlier_mask] = mean + np.sign(data[outlier_mask] - mean) * self.config.outlier_threshold * std

        return data_capped, outlier_count

    def _compute_statistics(self, data: np.ndarray, symbol: str, feature_name: str,
                          timeframe: str, outlier_count: int) -> InstrumentNormalizerStats:
        """Compute statistics for normalization"""
        if len(data) == 0:
            raise ValueError("Cannot compute statistics for empty data")

        return InstrumentNormalizerStats(
            symbol=symbol,
            feature_name=feature_name,
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            median=float(np.median(data)),
            q25=float(np.percentile(data, 25)),
            q75=float(np.percentile(data, 75)),
            outlier_count=outlier_count,
            sample_count=len(data),
            timeframe=timeframe
        )

    def _get_scaler(self, symbol: str, timeframe: str, feature_name: str) -> BaseEstimator:
        """Get scaler for specific instrument, timeframe, and feature"""
        # Try to get specific scaler
        if symbol in self.scalers and timeframe in self.scalers[symbol]:
            if feature_name in self.scalers[symbol][timeframe]:
                return self.scalers[symbol][timeframe][feature_name]

        # Fall back to default scalers
        if 'default' in self.scalers and 'default' in self.scalers['default']:
            if feature_name in self.scalers['default']['default']:
                return self.scalers['default']['default'][feature_name]

        # Create and fit default scaler if not exists
        if 'default' not in self.scalers:
            self.scalers['default'] = {}
        if 'default' not in self.scalers['default']:
            self.scalers['default']['default'] = {}

        if feature_name not in self.scalers['default']['default']:
            scaler = self._create_scaler()
            # Fit with default data
            default_data = np.array([[0.0]])
            scaler.fit(default_data)
            self.scalers['default']['default'][feature_name] = scaler

        return self.scalers['default']['default'][feature_name]

    def _apply_instrument_adjustments(self, normalized_value: float, symbol: str, feature_name: str) -> float:
        """Apply instrument-specific adjustments to normalized values"""
        if not self.instrument_registry:
            return normalized_value

        # Get instrument metadata
        metadata = self.instrument_registry.get_instrument_metadata(symbol)
        if not metadata:
            return normalized_value

        # Apply adjustments based on instrument characteristics
        # This is a simple example - in practice, you might use learned embeddings
        # or more sophisticated adjustments

        # Adjust for volatility (high volatility instruments might need different scaling)
        if metadata.asset_class == 'crypto':
            # Crypto instruments often have higher volatility
            adjusted_value = normalized_value * 0.8  # Scale down slightly
        elif metadata.asset_class == 'forex':
            # Forex pairs typically have lower volatility
            adjusted_value = normalized_value * 1.1  # Scale up slightly
        else:
            adjusted_value = normalized_value

        return adjusted_value

    def _remove_instrument_adjustments(self, normalized_value: float, symbol: str, feature_name: str) -> float:
        """Remove instrument-specific adjustments"""
        if not self.instrument_registry:
            return normalized_value

        # Get instrument metadata
        metadata = self.instrument_registry.get_instrument_metadata(symbol)
        if not metadata:
            return normalized_value

        # Reverse the adjustments
        if metadata.asset_class == 'crypto':
            original_value = normalized_value / 0.8
        elif metadata.asset_class == 'forex':
            original_value = normalized_value / 1.1
        else:
            original_value = normalized_value

        return original_value

    def _serialize_stats(self) -> Dict[str, Any]:
        """Serialize statistics for saving"""
        serialized = {}

        for symbol, timeframe_stats in self.stats.items():
            serialized[symbol] = {}
            for timeframe, feature_stats in timeframe_stats.items():
                serialized[symbol][timeframe] = {}
                for feature_name, stats in feature_stats.items():
                    serialized[symbol][timeframe][feature_name] = {
                        'symbol': stats.symbol,
                        'feature_name': stats.feature_name,
                        'mean': stats.mean,
                        'std': stats.std,
                        'min_val': stats.min_val,
                        'max_val': stats.max_val,
                        'median': stats.median,
                        'q25': stats.q25,
                        'q75': stats.q75,
                        'outlier_count': stats.outlier_count,
                        'sample_count': stats.sample_count,
                        'timeframe': stats.timeframe,
                        'created_at': stats.created_at.isoformat(),
                        'updated_at': stats.updated_at.isoformat()
                    }

        return serialized

    def _deserialize_stats(self, serialized_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize statistics after loading"""
        deserialized = {}

        for symbol, timeframe_stats in serialized_stats.items():
            deserialized[symbol] = {}
            for timeframe, feature_stats in timeframe_stats.items():
                deserialized[symbol][timeframe] = {}
                for feature_name, stats_data in feature_stats.items():
                    deserialized[symbol][timeframe][feature_name] = InstrumentNormalizerStats(
                        symbol=stats_data['symbol'],
                        feature_name=stats_data['feature_name'],
                        mean=stats_data['mean'],
                        std=stats_data['std'],
                        min_val=stats_data['min_val'],
                        max_val=stats_data['max_val'],
                        median=stats_data['median'],
                        q25=stats_data['q25'],
                        q75=stats_data['q75'],
                        outlier_count=stats_data['outlier_count'],
                        sample_count=stats_data['sample_count'],
                        timeframe=stats_data['timeframe'],
                        created_at=datetime.fromisoformat(stats_data['created_at']),
                        updated_at=datetime.fromisoformat(stats_data['updated_at'])
                    )

        return deserialized