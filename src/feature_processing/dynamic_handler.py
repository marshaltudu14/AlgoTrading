"""
Dynamic Feature Handler for schema validation and compatibility checking

This module provides dynamic feature handling capabilities including schema validation,
compatibility checking, and feature configuration management.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json
from pathlib import Path

from .feature_detector import FeatureMetadata, FeatureType, FeatureDetector
from .feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # All features must match exactly
    LENIENT = "lenient"    # Allow minor variations (new features, type changes)
    PERMISSIVE = "permissive"  # Allow any features, only validate basic structure


class CompatibilityStatus(Enum):
    """Feature compatibility status"""
    COMPATIBLE = "compatible"
    PARTIALLY_COMPATIBLE = "partially_compatible"
    INCOMPATIBLE = "incompatible"


@dataclass
class ValidationResult:
    """Result of feature validation"""
    is_valid: bool
    status: CompatibilityStatus
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    feature_mapping: Dict[str, str] = field(default_factory=dict)  # old_name -> new_name
    missing_features: List[str] = field(default_factory=list)
    new_features: List[str] = field(default_factory=list)
    changed_features: List[str] = field(default_factory=list)


@dataclass
class FeatureSchema:
    """Feature schema definition"""
    name: str
    feature_type: FeatureType
    dtype: str
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_categories: Optional[List[str]] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class DynamicFeatureHandler:
    """
    Dynamic feature handling with schema validation and compatibility checking
    """

    def __init__(self, feature_registry: Optional[FeatureRegistry] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dynamic feature handler

        Args:
            feature_registry: Feature registry instance
            config: Configuration dictionary
        """
        self.feature_registry = feature_registry or FeatureRegistry()
        self.config = config or {}
        self.schemas: Dict[str, FeatureSchema] = {}
        self.validation_history: List[Dict[str, Any]] = []

        # Default configuration
        self.default_config = {
            'validation_level': ValidationLevel.LENIENT,
            'auto_type_conversion': True,
            'handle_missing_values': True,
            'allow_new_features': True,
            'strict_type_checking': False,
            'max_null_percentage': 0.8
        }

        self.config = {**self.default_config, **self.config}

    def create_schema(self, features: Union[List[FeatureMetadata], Dict[str, FeatureMetadata], pd.DataFrame],
                     schema_name: Optional[str] = None) -> Dict[str, FeatureSchema]:
        """
        Create feature schema from feature metadata

        Args:
            features: List or dict of feature metadata, or DataFrame
            schema_name: Name for the schema

        Returns:
            Dictionary of feature schemas
        """
        # If DataFrame is passed, detect features first
        if isinstance(features, pd.DataFrame):
            detector = FeatureDetector()
            detected_features = detector.detect_features(features, "schema_creation")
            features = list(detected_features.values())
        elif isinstance(features, dict):
            features = list(features.values())

        schemas = {}

        for metadata in features:
            schema = FeatureSchema(
                name=metadata.name,
                feature_type=metadata.feature_type,
                dtype=metadata.dtype,
                validation_rules={}
            )

            # Add type-specific validation rules
            if metadata.feature_type == FeatureType.NUMERICAL:
                if metadata.min_value is not None:
                    schema.min_value = metadata.min_value
                if metadata.max_value is not None:
                    schema.max_value = metadata.max_value
                schema.validation_rules.update({
                    'allow_nulls': metadata.null_percentage < self.config['max_null_percentage']
                })

            elif metadata.feature_type == FeatureType.CATEGORICAL:
                if metadata.categories:
                    schema.allowed_categories = metadata.categories
                schema.validation_rules.update({
                    'max_categories': 50,
                    'allow_nulls': metadata.null_percentage < self.config['max_null_percentage']
                })

            schemas[metadata.name] = schema

        # Store schema if name provided
        if schema_name:
            self.schemas[schema_name] = schemas

        return schemas

    def validate_features(self, data: pd.DataFrame,
                         reference_schema: Optional[Dict[str, FeatureSchema]] = None,
                         validation_level: Optional[ValidationLevel] = None) -> ValidationResult:
        """
        Validate features against schema or existing registry

        Args:
            data: DataFrame to validate
            reference_schema: Reference schema (if None, uses registry)
            validation_level: Validation strictness level

        Returns:
            Validation result
        """
        validation_level = validation_level or self.config['validation_level']
        result = ValidationResult(
            is_valid=True,
            status=CompatibilityStatus.COMPATIBLE
        )

        # Get reference features
        if reference_schema is None:
            # Use all current features from registry as reference
            current_features = self.feature_registry.get_all_features()
            reference_features = {name: info['current_metadata'] for name, info in current_features.items()}
            reference_schema = self.create_schema(reference_features)
        else:
            reference_features = {name: None for name in reference_schema.keys()}  # Schema only

        # Get current features
        current_columns = set(data.columns)
        reference_columns = set(reference_schema.keys())

        # Check for missing features
        if validation_level in [ValidationLevel.STRICT, ValidationLevel.LENIENT]:
            missing_required = []
            missing_optional = []

            for col_name, schema in reference_schema.items():
                if col_name not in current_columns:
                    if schema.required:
                        missing_required.append(col_name)
                    else:
                        missing_optional.append(col_name)

            if missing_required:
                result.missing_features.extend(missing_required)
                result.issues.append(f"Missing required features: {missing_required}")
                result.is_valid = False
                result.status = CompatibilityStatus.INCOMPATIBLE

            if missing_optional:
                result.warnings.append(f"Missing optional features: {missing_optional}")

        # Check for new features
        new_features = current_columns - reference_columns
        if new_features:
            result.new_features.extend(new_features)

            if validation_level == ValidationLevel.STRICT:
                result.issues.append(f"Unexpected new features: {list(new_features)}")
                result.is_valid = False
                result.status = CompatibilityStatus.INCOMPATIBLE
            elif validation_level == ValidationLevel.LENIENT:
                result.warnings.append(f"New features detected: {list(new_features)}")
                result.status = CompatibilityStatus.PARTIALLY_COMPATIBLE
            else:  # PERMISSIVE
                result.warnings.append(f"New features accepted: {list(new_features)}")

        # Validate individual features
        for col_name in current_columns:
            if col_name in reference_schema:
                validation_result = self._validate_single_feature(
                    data[col_name], reference_schema[col_name]
                )

                if not validation_result['valid']:
                    result.changed_features.append(col_name)
                    result.issues.extend(validation_result['issues'])
                    result.is_valid = False
                    result.status = CompatibilityStatus.INCOMPATIBLE

                if validation_result['warnings']:
                    result.warnings.extend(validation_result['warnings'])
                    if result.status == CompatibilityStatus.COMPATIBLE:
                        result.status = CompatibilityStatus.PARTIALLY_COMPATIBLE

        # Create feature mapping
        for col_name in current_columns:
            if col_name in reference_columns:
                result.feature_mapping[col_name] = col_name
            elif validation_level != ValidationLevel.STRICT:
                # New features map to themselves
                result.feature_mapping[col_name] = col_name

        # Record validation event
        self._record_validation_event(result, len(current_columns), len(reference_columns))

        return result

    def _validate_single_feature(self, series: pd.Series, schema: FeatureSchema) -> Dict[str, Any]:
        """Validate a single feature against its schema"""
        result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }

        # Check data type compatibility
        if not self._is_dtype_compatible(series.dtype, schema.dtype):
            if self.config['auto_type_conversion']:
                try:
                    converted_series = self._convert_series_type(series, schema.feature_type)
                    result['warnings'].append(
                        f"Auto-converted {schema.name} from {series.dtype} to compatible type"
                    )
                except Exception as e:
                    result['valid'] = False
                    result['issues'].append(
                        f"Type conversion failed for {schema.name}: {e}"
                    )
            else:
                result['valid'] = False
                result['issues'].append(
                    f"Type mismatch for {schema.name}: expected {schema.dtype}, got {series.dtype}"
                )

        # Check for null values
        null_percentage = series.isnull().sum() / len(series)
        if null_percentage > self.config['max_null_percentage']:
            result['valid'] = False
            result['issues'].append(
                f"High null percentage in {schema.name}: {null_percentage:.2%}"
            )

        # Type-specific validation
        if schema.feature_type == FeatureType.NUMERICAL:
            self._validate_numerical_feature(series, schema, result)
        elif schema.feature_type == FeatureType.CATEGORICAL:
            self._validate_categorical_feature(series, schema, result)

        return result

    def _is_dtype_compatible(self, actual_dtype: str, expected_dtype: str) -> bool:
        """Check if data types are compatible"""
        # Convert dtype strings for comparison
        actual_clean = str(actual_dtype).lower()
        expected_clean = str(expected_dtype).lower()

        # Basic numeric compatibility
        numeric_types = ['int', 'float', 'int64', 'float64', 'int32', 'float32']
        if actual_clean in numeric_types and expected_clean in numeric_types:
            return True

        # Object/string compatibility
        string_types = ['object', 'str', 'string']
        if actual_clean in string_types and expected_clean in string_types:
            return True

        # Boolean compatibility
        bool_types = ['bool', 'boolean']
        if actual_clean in bool_types and expected_clean in bool_types:
            return True

        return actual_clean == expected_clean

    def _convert_series_type(self, series: pd.Series, target_type: FeatureType) -> pd.Series:
        """Convert series to target type"""
        if target_type == FeatureType.NUMERICAL:
            return pd.to_numeric(series, errors='coerce')
        elif target_type == FeatureType.CATEGORICAL:
            return series.astype('category')
        elif target_type == FeatureType.BOOLEAN:
            return series.astype('bool')
        elif target_type == FeatureType.DATETIME:
            return pd.to_datetime(series, errors='coerce')
        else:
            return series

    def _validate_numerical_feature(self, series: pd.Series, schema: FeatureSchema, result: Dict[str, Any]):
        """Validate numerical feature constraints"""
        numeric_series = pd.to_numeric(series, errors='coerce').dropna()

        if len(numeric_series) == 0:
            result['valid'] = False
            result['issues'].append(f"No valid numeric values in {schema.name}")
            return

        # Check range constraints
        if schema.min_value is not None:
            if (numeric_series < schema.min_value).any():
                result['valid'] = False
                result['issues'].append(
                    f"Values below minimum ({schema.min_value}) in {schema.name}"
                )

        if schema.max_value is not None:
            if (numeric_series > schema.max_value).any():
                result['valid'] = False
                result['issues'].append(
                    f"Values above maximum ({schema.max_value}) in {schema.name}"
                )

    def _validate_categorical_feature(self, series: pd.Series, schema: FeatureSchema, result: Dict[str, Any]):
        """Validate categorical feature constraints"""
        # Check number of unique values
        unique_count = series.nunique()
        max_categories = schema.validation_rules.get('max_categories', 50)

        if unique_count > max_categories:
            result['warnings'].append(
                f"High cardinality in {schema.name}: {unique_count} unique values"
            )

        # Check allowed categories
        if schema.allowed_categories is not None:
            invalid_categories = set(series.dropna().unique()) - set(schema.allowed_categories)
            if invalid_categories:
                result['valid'] = False
                result['issues'].append(
                    f"Invalid categories in {schema.name}: {list(invalid_categories)}"
                )

    def resolve_feature_dependencies(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Resolve dependencies between features

        Args:
            feature_names: List of feature names to resolve

        Returns:
            Dependency resolution result
        """
        # Simple dependency resolution based on feature types and relationships
        # This is a basic implementation that can be extended

        result = {
            'resolved_order': [],
            'circular_dependencies': [],
            'missing_features': [],
            'dependency_graph': {}
        }

        # Build dependency graph
        for feature_name in feature_names:
            metadata = self.feature_registry.get_feature(feature_name)
            if metadata is None:
                result['missing_features'].append(feature_name)
                continue

            # Simple dependencies: identifiers depend on nothing, others may depend on identifiers
            dependencies = []
            if not metadata.is_identifier:
                # Add identifier features as dependencies
                for other_name in feature_names:
                    if other_name != feature_name:
                        other_metadata = self.feature_registry.get_feature(other_name)
                        if other_metadata and other_metadata.is_identifier:
                            dependencies.append(other_name)

            result['dependency_graph'][feature_name] = dependencies

        # Topological sort for dependency resolution
        if result['missing_features']:
            return result

        try:
            resolved = self._topological_sort(result['dependency_graph'])
            result['resolved_order'] = resolved
        except ValueError as e:
            result['circular_dependencies'] = [str(e)]

        return result

    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph"""
        # Simple implementation of topological sort
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] = in_degree.get(neighbor, 0) + 1

        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(graph):
            raise ValueError("Circular dependency detected")

        return result

    def create_feature_configuration(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Create configuration for selected features

        Args:
            feature_names: List of feature names

        Returns:
            Feature configuration dictionary
        """
        config = {
            'features': {},
            'processing_order': [],
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'feature_count': len(feature_names)
            }
        }

        # Resolve dependencies for processing order
        dependency_result = self.resolve_feature_dependencies(feature_names)
        if dependency_result['resolved_order']:
            config['processing_order'] = dependency_result['resolved_order']
        else:
            config['processing_order'] = feature_names

        # Add individual feature configurations
        for feature_name in feature_names:
            metadata = self.feature_registry.get_feature(feature_name)
            if metadata:
                config['features'][feature_name] = {
                    'type': metadata.feature_type.value,
                    'dtype': metadata.dtype,
                    'required': True,
                    'processing': self._get_feature_processing_config(metadata)
                }

        return config

    def _get_feature_processing_config(self, metadata: FeatureMetadata) -> Dict[str, Any]:
        """Get processing configuration for a feature based on its type"""
        config = {
            'handle_missing': self.config['handle_missing_values'],
            'normalization': None
        }

        if metadata.feature_type == FeatureType.NUMERICAL:
            config.update({
                'normalization': 'min_max',  # 0-100 range as per requirements
                'outlier_detection': True,
                'scaling': True
            })
        elif metadata.feature_type == FeatureType.CATEGORICAL:
            config.update({
                'encoding': 'one_hot',
                'handle_unknown': 'ignore'
            })
        elif metadata.feature_type == FeatureType.BOOLEAN:
            config.update({
                'encoding': 'boolean'
            })

        return config

    def _record_validation_event(self, result: ValidationResult, current_count: int, reference_count: int):
        """Record validation event in history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'validation_result': {
                'is_valid': result.is_valid,
                'status': result.status.value,
                'issues_count': len(result.issues),
                'warnings_count': len(result.warnings),
                'missing_features': len(result.missing_features),
                'new_features': len(result.new_features),
                'changed_features': len(result.changed_features)
            },
            'feature_counts': {
                'current': current_count,
                'reference': reference_count
            }
        }

        self.validation_history.append(event)

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history"""
        if not self.validation_history:
            return {'total_validations': 0}

        return {
            'total_validations': len(self.validation_history),
            'successful_validations': len([v for v in self.validation_history if v['validation_result']['is_valid']]),
            'failed_validations': len([v for v in self.validation_history if not v['validation_result']['is_valid']]),
            'average_issues': np.mean([v['validation_result']['issues_count'] for v in self.validation_history]),
            'recent_validation': self.validation_history[-1]
        }

    def save_configuration(self, file_path: Union[str, Path]):
        """Save handler configuration to file"""
        config = {
            'config': self._serialize_config(self.config),
            'schemas': {name: {
                'name': schema.name,
                'feature_type': schema.feature_type.value,
                'dtype': schema.dtype,
                'required': schema.required,
                'min_value': schema.min_value,
                'max_value': schema.max_value,
                'allowed_categories': schema.allowed_categories,
                'validation_rules': schema.validation_rules
            } for name, schema in self.schemas.items()},
            'validation_history': self.validation_history[-10:],  # Last 10 events
            'saved_at': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)

    def _serialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration for JSON storage"""
        serialized = {}
        for key, value in config_dict.items():
            if hasattr(value, 'value'):  # Handle enums
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    def load_configuration(self, file_path: Union[str, Path]):
        """Load handler configuration from file"""
        with open(file_path, 'r') as f:
            config = json.load(f)

        self.config = self._deserialize_config(config.get('config', {}))
        self.validation_history = config.get('validation_history', [])

        # Load schemas
        self.schemas = {}
        for name, schema_data in config.get('schemas', {}).items():
            self.schemas[name] = FeatureSchema(
                name=schema_data['name'],
                feature_type=FeatureType(schema_data['feature_type']),
                dtype=schema_data['dtype'],
                required=schema_data['required'],
                min_value=schema_data['min_value'],
                max_value=schema_data['max_value'],
                allowed_categories=schema_data['allowed_categories'],
                validation_rules=schema_data['validation_rules']
            )

    def _deserialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize configuration from JSON storage"""
        deserialized = {}
        for key, value in config_dict.items():
            if key == 'validation_level' and isinstance(value, str):
                deserialized[key] = ValidationLevel(value)
            else:
                deserialized[key] = value
        return deserialized