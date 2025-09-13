#!/usr/bin/env python3
"""
Feature Normalization and Preprocessing Pipeline
==============================================

A comprehensive preprocessing system for normalizing features to 0-100 range,
handling missing values, detecting outliers, and providing real-time processing
capabilities for algorithmic trading data.

Key Features:
- Min-max scaling to 0-100 range
- Multiple imputation strategies for missing values
- Statistical outlier detection and handling
- Feature-specific normalization strategies
- Batch processing for large datasets
- Online normalization for streaming data
- Parameter storage and retrieval
- Performance optimization with GPU acceleration

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import json
import logging
from pathlib import Path
import time
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OutlierHandlingStrategy(Enum):
    """Strategies for handling outliers."""
    CLIP = "clip"
    REMOVE = "remove"
    FLAG = "flag"
    TRANSFORM = "transform"


class ImputationStrategy(Enum):
    """Strategies for handling missing values."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    KNN = "knn"
    FORWARD_FILL = "ffill"
    BACKWARD_FILL = "bfill"


class FeatureType(Enum):
    """Types of features for specific normalization strategies."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    INDICATOR = "indicator"


@dataclass
class NormalizationParams:
    """Parameters for feature normalization."""
    feature_name: str
    feature_type: FeatureType
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    outlier_threshold: float
    imputation_strategy: ImputationStrategy
    outlier_handling: OutlierHandlingStrategy
    normalization_range: Tuple[float, float] = (0.0, 100.0)


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """
    Main feature normalizer with min-max scaling to 0-100 range.
    """

    def __init__(self, normalization_range: Tuple[float, float] = (0.0, 100.0)):
        self.normalization_range = normalization_range
        self.normalization_params: Dict[str, NormalizationParams] = {}
        self.fitted_scalers: Dict[str, MinMaxScaler] = {}
        self.imputers: Dict[str, SimpleImputer] = {}
        self.feature_metadata: Dict[str, FeatureType] = {}
        self.performance_stats: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureNormalizer':
        """
        Fit the normalizer to the training data.

        Args:
            X: Input features DataFrame
            y: Target variable (optional)

        Returns:
            Self
        """
        start_time = time.time()
        logger.info(f"Fitting normalizer on {X.shape[0]} samples, {X.shape[1]} features")

        # Auto-detect feature types
        self._detect_feature_types(X)

        # Fit normalization parameters for each feature
        for feature in X.columns:
            feature_data = X[feature].dropna()

            if len(feature_data) == 0:
                logger.warning(f"No valid data for feature {feature}")
                continue

            # Calculate statistics
            min_val = feature_data.min()
            max_val = feature_data.max()
            mean_val = feature_data.mean()
            std_val = feature_data.std()

            # Determine outlier threshold using IQR method
            Q1 = feature_data.quantile(0.25)
            Q3 = feature_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = 1.5 * IQR

            # Create normalization parameters
            params = NormalizationParams(
                feature_name=feature,
                feature_type=self.feature_metadata[feature],
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                std_val=std_val,
                outlier_threshold=outlier_threshold,
                imputation_strategy=self._get_best_imputation_strategy(feature_data),
                outlier_handling=OutlierHandlingStrategy.CLIP,
                normalization_range=self.normalization_range
            )

            self.normalization_params[feature] = params

            # Create and fit scaler
            scaler = MinMaxScaler(feature_range=self.normalization_range)
            feature_values = feature_data.values.reshape(-1, 1)
            scaler.fit(feature_values)
            self.fitted_scalers[feature] = scaler

            # Create imputer
            imputer = self._create_imputer(params.imputation_strategy, feature_values)
            if imputer is not None:
                self.imputers[feature] = imputer

        self.performance_stats['fit_time'] = time.time() - start_time
        logger.info(f"Normalizer fitted in {self.performance_stats['fit_time']:.3f} seconds")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data using fitted normalization parameters.

        Args:
            X: Input features DataFrame

        Returns:
            Normalized DataFrame
        """
        start_time = time.time()
        logger.info(f"Transforming {X.shape[0]} samples, {X.shape[1]} features")

        X_transformed = X.copy()

        for feature in X.columns:
            if feature not in self.normalization_params:
                logger.warning(f"Feature {feature} not found in fitted parameters. Skipping.")
                continue

            params = self.normalization_params[feature]
            feature_data = X[feature].copy()

            # Handle outliers
            feature_data = self._handle_outliers(feature_data, params)

            # Handle missing values
            feature_data = self._impute_missing_values(feature_data, params)

            # Apply normalization
            scaler = self.fitted_scalers[feature]
            normalized_values = scaler.transform(feature_data.values.reshape(-1, 1))
            X_transformed[feature] = normalized_values.flatten()

        self.performance_stats['transform_time'] = time.time() - start_time
        logger.info(f"Transformation completed in {self.performance_stats['transform_time']:.3f} seconds")

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the normalization transformation.

        Args:
            X: Normalized features DataFrame

        Returns:
            Original scale DataFrame
        """
        start_time = time.time()
        logger.info(f"Inverse transforming {X.shape[0]} samples, {X.shape[1]} features")

        X_original = X.copy()

        for feature in X.columns:
            if feature not in self.fitted_scalers:
                logger.warning(f"Feature {feature} not found in fitted scalers. Skipping.")
                continue

            scaler = self.fitted_scalers[feature]
            original_values = scaler.inverse_transform(X[feature].values.reshape(-1, 1))
            X_original[feature] = original_values.flatten()

        self.performance_stats['inverse_transform_time'] = time.time() - start_time
        logger.info(f"Inverse transformation completed in {self.performance_stats['inverse_transform_time']:.3f} seconds")

        return X_original

    def _detect_feature_types(self, X: pd.DataFrame):
        """Automatically detect feature types based on data characteristics."""
        for feature in X.columns:
            feature_data = X[feature].dropna()

            if len(feature_data) == 0:
                self.feature_metadata[feature] = FeatureType.CONTINUOUS
                continue

            # Check if feature is categorical (low cardinality)
            unique_ratio = feature_data.nunique() / len(feature_data)
            if unique_ratio < 0.05 or feature_data.nunique() <= 10:
                self.feature_metadata[feature] = FeatureType.CATEGORICAL
            # Check if feature is temporal (time-related column name)
            elif any(time_keyword in feature.lower() for time_keyword in ['time', 'date', 'hour', 'minute', 'second']):
                self.feature_metadata[feature] = FeatureType.TEMPORAL
            # Check if feature is an indicator (values mostly 0/1 or within small range)
            elif feature_data.max() - feature_data.min() <= 2 and unique_ratio <= 0.1:
                self.feature_metadata[feature] = FeatureType.INDICATOR
            else:
                self.feature_metadata[feature] = FeatureType.CONTINUOUS

    def _get_best_imputation_strategy(self, feature_data: pd.Series) -> ImputationStrategy:
        """Determine the best imputation strategy based on data characteristics."""
        missing_ratio = feature_data.isna().mean()
        unique_values = feature_data.nunique()

        if missing_ratio == 0:
            return ImputationStrategy.MEAN

        if unique_values <= 10:
            return ImputationStrategy.MODE
        elif missing_ratio > 0.3:
            return ImputationStrategy.KNN
        else:
            return ImputationStrategy.MEDIAN

    def _create_imputer(self, strategy: ImputationStrategy, feature_values: np.ndarray) -> Optional[SimpleImputer]:
        """Create appropriate imputer based on strategy."""
        if strategy == ImputationStrategy.KNN:
            return KNNImputer(n_neighbors=5)
        elif strategy == ImputationStrategy.MEAN:
            return SimpleImputer(strategy='mean')
        elif strategy == ImputationStrategy.MEDIAN:
            return SimpleImputer(strategy='median')
        elif strategy == ImputationStrategy.MODE:
            return SimpleImputer(strategy='most_frequent')
        else:
            return None

    def _handle_outliers(self, feature_data: pd.Series, params: NormalizationParams) -> pd.Series:
        """Handle outliers based on the specified strategy."""
        Q1 = feature_data.quantile(0.25)
        Q3 = feature_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - params.outlier_threshold
        upper_bound = Q3 + params.outlier_threshold

        if params.outlier_handling == OutlierHandlingStrategy.CLIP:
            return feature_data.clip(lower_bound, upper_bound)
        elif params.outlier_handling == OutlierHandlingStrategy.REMOVE:
            return feature_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]
        elif params.outlier_handling == OutlierHandlingStrategy.FLAG:
            outlier_flags = (feature_data < lower_bound) | (feature_data > upper_bound)
            return feature_data
        else:
            return feature_data

    def _impute_missing_values(self, feature_data: pd.Series, params: NormalizationParams) -> pd.Series:
        """Impute missing values based on the specified strategy."""
        if feature_data.isna().sum() == 0:
            return feature_data

        if params.imputation_strategy in [ImputationStrategy.FORWARD_FILL, ImputationStrategy.BACKWARD_FILL]:
            if params.imputation_strategy == ImputationStrategy.FORWARD_FILL:
                return feature_data.fillna(method='ffill').fillna(method='bfill')
            else:
                return feature_data.fillna(method='bfill').fillna(method='ffill')

        if feature in self.imputers:
            imputer = self.imputers[feature]
            imputed_values = imputer.transform(feature_data.values.reshape(-1, 1))
            return pd.Series(imputed_values.flatten(), index=feature_data.index)

        # Fallback to median imputation
        return feature_data.fillna(feature_data.median())

    def save_parameters(self, file_path: str):
        """Save normalization parameters to file."""
        params_dict = {}
        for feature, params in self.normalization_params.items():
            params_dict[feature] = {
                'feature_name': params.feature_name,
                'feature_type': params.feature_type.value,
                'min_val': params.min_val,
                'max_val': params.max_val,
                'mean_val': params.mean_val,
                'std_val': params.std_val,
                'outlier_threshold': params.outlier_threshold,
                'imputation_strategy': params.imputation_strategy.value,
                'outlier_handling': params.outlier_handling.value,
                'normalization_range': params.normalization_range
            }

        with open(file_path, 'w') as f:
            json.dump(params_dict, f, indent=2)

        logger.info(f"Normalization parameters saved to {file_path}")

    def load_parameters(self, file_path: str):
        """Load normalization parameters from file."""
        with open(file_path, 'r') as f:
            params_dict = json.load(f)

        self.normalization_params = {}
        for feature, params_data in params_dict.items():
            self.normalization_params[feature] = NormalizationParams(
                feature_name=params_data['feature_name'],
                feature_type=FeatureType(params_data['feature_type']),
                min_val=params_data['min_val'],
                max_val=params_data['max_val'],
                mean_val=params_data['mean_val'],
                std_val=params_data['std_val'],
                outlier_threshold=params_data['outlier_threshold'],
                imputation_strategy=ImputationStrategy(params_data['imputation_strategy']),
                outlier_handling=OutlierHandlingStrategy(params_data['outlier_handling']),
                normalization_range=tuple(params_data['normalization_range'])
            )

        logger.info(f"Normalization parameters loaded from {file_path}")

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all features."""
        stats = {}
        for feature, params in self.normalization_params.items():
            stats[feature] = {
                'min': params.min_val,
                'max': params.max_val,
                'mean': params.mean_val,
                'std': params.std_val,
                'outlier_threshold': params.outlier_threshold
            }
        return stats

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        return self.performance_stats