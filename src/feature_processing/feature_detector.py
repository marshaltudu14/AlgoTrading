"""
Feature Detection Engine for Dynamic Feature Processing

This module provides automatic detection and handling of varying numbers of input features
for the transformer trading prediction system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureType(Enum):
    """Feature type enumeration"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"


@dataclass
class FeatureMetadata:
    """Metadata for a detected feature"""
    name: str
    feature_type: FeatureType
    dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None
    categories: Optional[List[str]] = None
    is_target: bool = False
    is_identifier: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class FeatureDetector:
    """
    Automatic feature detection and metadata extraction engine
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature detector

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_registry: Dict[str, FeatureMetadata] = {}
        self.feature_history: List[Dict[str, Any]] = []

        # Default configuration
        self.default_config = {
            'max_categories': 50,  # Max unique values for categorical
            'null_threshold': 0.8,  # Max null percentage to keep feature
            'identifier_patterns': ['id', 'identifier', 'key'],
            'target_patterns': ['target', 'label', 'y'],
            'datetime_patterns': ['date', 'time', 'timestamp'],
            'ignore_patterns': ['Unnamed', 'index'],
        }

        # Merge with provided config
        self.config = {**self.default_config, **self.config}

    def detect_features(self, data: Union[pd.DataFrame, str, Path],
                       source_name: Optional[str] = None) -> Dict[str, FeatureMetadata]:
        """
        Detect features from input data

        Args:
            data: Input data (DataFrame or file path)
            source_name: Name of the data source

        Returns:
            Dictionary of feature metadata
        """
        logger.info(f"Starting feature detection for {source_name or 'unknown source'}")

        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = self._load_data_from_file(data)

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame or file path")

        detected_features = {}

        for column in data.columns:
            # Skip ignored columns
            if self._should_ignore_column(column):
                continue

            try:
                feature_metadata = self._extract_feature_metadata(data[column], column)
                detected_features[column] = feature_metadata

                # Add to registry
                self.feature_registry[column] = feature_metadata

                logger.debug(f"Detected feature: {column} ({feature_metadata.feature_type.value})")

            except Exception as e:
                logger.warning(f"Failed to process column {column}: {e}")
                continue

        # Record detection event
        self._record_detection_event(detected_features, source_name)

        logger.info(f"Detected {len(detected_features)} features from {source_name or 'unknown source'}")
        return detected_features

    def _load_data_from_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data from file with automatic format detection"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = file_path.suffix.lower()

        if file_ext == '.csv':
            return pd.read_csv(file_path)
        elif file_ext in ['.parquet', '.pq']:
            return pd.read_parquet(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_ext == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    def _should_ignore_column(self, column_name: str) -> bool:
        """Check if column should be ignored"""
        return any(pattern.lower() in column_name.lower()
                  for pattern in self.config['ignore_patterns'])

    def _extract_feature_metadata(self, series: pd.Series, column_name: str) -> FeatureMetadata:
        """Extract metadata for a single feature"""
        # Basic statistics
        null_count = series.isnull().sum()
        null_percentage = null_count / len(series)
        unique_count = series.nunique()

        # Determine feature type
        feature_type = self._determine_feature_type(series, column_name)

        # Create base metadata
        metadata = FeatureMetadata(
            name=column_name,
            feature_type=feature_type,
            dtype=str(series.dtype),
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            is_target=self._is_target_column(column_name),
            is_identifier=self._is_identifier_column(column_name)
        )

        # Add type-specific metadata
        if feature_type == FeatureType.NUMERICAL:
            self._add_numerical_metadata(metadata, series)
        elif feature_type == FeatureType.CATEGORICAL:
            self._add_categorical_metadata(metadata, series)
        elif feature_type == FeatureType.DATETIME:
            self._add_datetime_metadata(metadata, series)

        return metadata

    def _determine_feature_type(self, series: pd.Series, column_name: str) -> FeatureType:
        """Determine the type of a feature"""
        # Check for boolean first
        if series.dtype == 'bool':
            return FeatureType.BOOLEAN
        # Check for boolean-like values
        unique_values = set(series.dropna().unique())
        if unique_values.issubset({0, 1, True, False}) and len(unique_values) <= 2:
            return FeatureType.BOOLEAN

        # Check for datetime patterns
        if self._is_datetime_column(series, column_name):
            return FeatureType.DATETIME

        # Check for explicit numeric types
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_dtype(series):
            return FeatureType.NUMERICAL

        # Check for categorical
        if series.dtype == 'object' or series.nunique() <= self.config['max_categories']:
            return FeatureType.CATEGORICAL

        # Default to numerical
        return FeatureType.NUMERICAL

    def _is_datetime_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains datetime data"""
        if series.dtype == 'datetime64[ns]':
            return True

        # Skip datetime detection for clearly numeric data
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_datetime64_dtype(series):
            return False

        # Check column name patterns
        if any(pattern in column_name.lower() for pattern in self.config['datetime_patterns']):
            return True

        # Try to convert to datetime, but be more conservative
        try:
            # Only try conversion if it's object dtype or could potentially be datetime
            if series.dtype == 'object':
                sample = series.dropna().head(10)
                if len(sample) > 0:
                    # Check if values look like dates before attempting conversion
                    first_val = str(sample.iloc[0])
                    if any(char in first_val for char in ['-', '/', ':']) and len(first_val) > 6:
                        pd.to_datetime(sample)
                        return True
        except:
            pass

        return False

    def _is_target_column(self, column_name: str) -> bool:
        """Check if column is a target variable"""
        return any(pattern in column_name.lower() for pattern in self.config['target_patterns'])

    def _is_identifier_column(self, column_name: str) -> bool:
        """Check if column is an identifier"""
        return any(pattern in column_name.lower() for pattern in self.config['identifier_patterns'])

    def _add_numerical_metadata(self, metadata: FeatureMetadata, series: pd.Series):
        """Add numerical-specific metadata"""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()

        if len(numeric_series) > 0:
            metadata.min_value = float(numeric_series.min())
            metadata.max_value = float(numeric_series.max())
            metadata.mean_value = float(numeric_series.mean())
            metadata.std_value = float(numeric_series.std())

    def _add_categorical_metadata(self, metadata: FeatureMetadata, series: pd.Series):
        """Add categorical-specific metadata"""
        categories = series.dropna().astype(str).unique().tolist()
        metadata.categories = categories[:self.config['max_categories']]  # Limit categories

    def _add_datetime_metadata(self, metadata: FeatureMetadata, series: pd.Series):
        """Add datetime-specific metadata"""
        datetime_series = pd.to_datetime(series, errors='coerce').dropna()

        if len(datetime_series) > 0:
            metadata.min_value = datetime_series.min().timestamp()
            metadata.max_value = datetime_series.max().timestamp()

    def _record_detection_event(self, features: Dict[str, FeatureMetadata], source_name: str):
        """Record feature detection event in history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'source': source_name or 'unknown',
            'feature_count': len(features),
            'features': {name: {
                'type': meta.feature_type.value,
                'null_percentage': meta.null_percentage,
                'unique_count': meta.unique_count
            } for name, meta in features.items()}
        }

        self.feature_history.append(event)

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all detected features"""
        if not self.feature_registry:
            return {'total_features': 0, 'by_type': {}}

        type_counts = {}
        for metadata in self.feature_registry.values():
            feature_type = metadata.feature_type.value
            type_counts[feature_type] = type_counts.get(feature_type, 0) + 1

        return {
            'total_features': len(self.feature_registry),
            'by_type': type_counts,
            'high_null_features': [
                name for name, meta in self.feature_registry.items()
                if meta.null_percentage > 0.5
            ],
            'identifier_features': [
                name for name, meta in self.feature_registry.items()
                if meta.is_identifier
            ],
            'target_features': [
                name for name, meta in self.feature_registry.items()
                if meta.is_target
            ]
        }

    def save_feature_registry(self, file_path: Union[str, Path]):
        """Save feature registry to file"""
        file_path = Path(file_path)

        # Convert metadata to serializable format
        serializable_registry = {}
        for name, metadata in self.feature_registry.items():
            serializable_registry[name] = {
                'name': metadata.name,
                'feature_type': metadata.feature_type.value,
                'dtype': metadata.dtype,
                'null_count': int(metadata.null_count),
                'null_percentage': float(metadata.null_percentage),
                'unique_count': int(metadata.unique_count),
                'min_value': float(metadata.min_value) if metadata.min_value is not None else None,
                'max_value': float(metadata.max_value) if metadata.max_value is not None else None,
                'mean_value': float(metadata.mean_value) if metadata.mean_value is not None else None,
                'std_value': float(metadata.std_value) if metadata.std_value is not None else None,
                'categories': metadata.categories,
                'is_target': metadata.is_target,
                'is_identifier': metadata.is_identifier,
                'created_at': metadata.created_at.isoformat(),
                'updated_at': metadata.updated_at.isoformat()
            }

        with open(file_path, 'w') as f:
            json.dump(serializable_registry, f, indent=2)

    def load_feature_registry(self, file_path: Union[str, Path]):
        """Load feature registry from file"""
        file_path = Path(file_path)

        with open(file_path, 'r') as f:
            serializable_registry = json.load(f)

        self.feature_registry = {}
        for name, data in serializable_registry.items():
            self.feature_registry[name] = FeatureMetadata(
                name=data['name'],
                feature_type=FeatureType(data['feature_type']),
                dtype=data['dtype'],
                null_count=data['null_count'],
                null_percentage=data['null_percentage'],
                unique_count=data['unique_count'],
                min_value=data['min_value'],
                max_value=data['max_value'],
                mean_value=data['mean_value'],
                std_value=data['std_value'],
                categories=data['categories'],
                is_target=data['is_target'],
                is_identifier=data['is_identifier'],
                created_at=datetime.fromisoformat(data['created_at']),
                updated_at=datetime.fromisoformat(data['updated_at'])
            )