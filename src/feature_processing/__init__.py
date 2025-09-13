"""
Feature Processing Module

This module provides dynamic feature detection, processing, monitoring,
and feature importance tracking capabilities for the transformer trading prediction system.
"""

from .feature_detector import FeatureDetector, FeatureMetadata, FeatureType
from .feature_registry import FeatureRegistry, FeatureVersion
from .dynamic_handler import (
    DynamicFeatureHandler, FeatureSchema, ValidationResult,
    ValidationLevel, CompatibilityStatus
)
from .feature_monitor import (
    FeatureMonitor, DriftResult, HealthMetrics, FeatureSnapshot,
    DriftType, HealthStatus
)
from .testing_utils import FeatureTestingUtils, TestResult, TestSuiteResult

# Feature importance tracking components
from .importance_engine import (
    FeatureImportanceEngine, ImportanceResult, ImportanceScore,
    ImportanceMethod, ImportanceType
)
from .importance_tracker import (
    ImportanceTracker, ImportanceSnapshot, DriftAlert,
    ImportanceTrend, DriftDetectionMethod, DriftSeverity
)
from .importance_visualization import ImportanceVisualizer
from .model_monitor import (
    ModelMonitor, ModelHealth, ModelMetrics, MonitoringAlert,
    ModelHealthStatus, PerformanceMetric, ModelPrediction
)

__all__ = [
    # Core feature processing
    'FeatureDetector', 'FeatureMetadata', 'FeatureType',
    'FeatureRegistry', 'FeatureVersion',
    'DynamicFeatureHandler', 'FeatureSchema', 'ValidationResult',
    'ValidationLevel', 'CompatibilityStatus',
    'FeatureMonitor', 'DriftResult', 'HealthMetrics', 'FeatureSnapshot',
    'DriftType', 'HealthStatus',
    'FeatureTestingUtils', 'TestResult', 'TestSuiteResult',

    # Feature importance tracking
    'FeatureImportanceEngine', 'ImportanceResult', 'ImportanceScore',
    'ImportanceMethod', 'ImportanceType',
    'ImportanceTracker', 'ImportanceSnapshot', 'DriftAlert',
    'ImportanceTrend', 'DriftDetectionMethod', 'DriftSeverity',
    'ImportanceVisualizer',
    'ModelMonitor', 'ModelHealth', 'ModelMetrics', 'MonitoringAlert',
    'ModelHealthStatus', 'PerformanceMetric', 'ModelPrediction'
]