"""
Feature Monitoring System for drift detection and health monitoring

This module provides comprehensive feature monitoring capabilities including drift detection,
health monitoring, performance tracking, and alerting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings

from .feature_detector import FeatureMetadata, FeatureType
from .feature_registry import FeatureRegistry

logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class DriftType(Enum):
    """Types of feature drift"""
    DISTRIBUTION = "distribution"
    STATISTICAL = "statistical"
    MISSING_VALUES = "missing_values"
    CARDINALITY = "cardinality"
    RANGE = "range"


class HealthStatus(Enum):
    """Feature health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class DriftResult:
    """Result of drift analysis"""
    drift_detected: bool
    drift_score: float
    drift_type: DriftType
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Convert numpy types to Python types"""
        self.drift_detected = bool(self.drift_detected)
        self.drift_score = float(self.drift_score)
        self.confidence = float(self.confidence)


@dataclass
class HealthMetrics:
    """Health metrics for a feature"""
    status: HealthStatus
    drift_score: float
    null_percentage: float
    uniqueness_score: float
    stability_score: float
    last_updated: datetime
    alerts: List[str] = field(default_factory=list)


@dataclass
class FeatureSnapshot:
    """Snapshot of feature statistics at a point in time"""
    timestamp: datetime
    feature_name: str
    statistics: Dict[str, Any]
    sample_size: int


class FeatureMonitor:
    """
    Comprehensive feature monitoring system
    """

    def __init__(self, feature_registry: FeatureRegistry, config: Optional[Dict[str, Any]] = None):
        """
        Initialize feature monitor

        Args:
            feature_registry: Feature registry instance
            config: Configuration dictionary
        """
        self.feature_registry = feature_registry
        self.config = config or {}
        self.baseline_snapshots: Dict[str, FeatureSnapshot] = {}
        self.current_snapshots: Dict[str, FeatureSnapshot] = {}
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.monitoring_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []

        # Default configuration
        self.default_config = {
            'drift_threshold': 0.1,  # Threshold for drift detection
            'statistical_test_alpha': 0.05,  # Significance level for statistical tests
            'min_sample_size': 100,  # Minimum sample size for monitoring
            'monitoring_window_days': 30,  # Days to keep monitoring data
            'alert_cooldown_minutes': 60,  # Minimum time between alerts for same feature
            'health_check_interval_hours': 24,  # How often to run health checks
            'baseline_sample_size': 1000  # Sample size for baseline statistics
        }

        self.config = {**self.default_config, **self.config}

    def establish_baseline(self, data: pd.DataFrame, feature_names: Optional[List[str]] = None):
        """
        Establish baseline statistics for features

        Args:
            data: Baseline data
            feature_names: Specific features to baseline (if None, uses all columns)
        """
        if feature_names is None:
            feature_names = list(data.columns)

        logger.info(f"Establishing baseline for {len(feature_names)} features")

        for feature_name in feature_names:
            if feature_name not in data.columns:
                logger.warning(f"Feature {feature_name} not found in data")
                continue

            try:
                snapshot = self._create_snapshot(data[feature_name], feature_name)
                self.baseline_snapshots[feature_name] = snapshot
                logger.debug(f"Baseline established for {feature_name}")
            except Exception as e:
                logger.error(f"Failed to create baseline for {feature_name}: {e}")

    def detect_drift(self, data: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Dict[str, DriftResult]:
        """
        Detect drift in features compared to baseline

        Args:
            data: Current data to check for drift
            feature_names: Specific features to check (if None, uses all columns)

        Returns:
            Dictionary of drift results by feature
        """
        if feature_names is None:
            feature_names = list(data.columns)

        drift_results = {}

        for feature_name in feature_names:
            if feature_name not in data.columns:
                continue

            if feature_name not in self.baseline_snapshots:
                logger.warning(f"No baseline available for {feature_name}")
                continue

            try:
                drift_result = self._analyze_drift(data[feature_name], feature_name)
                drift_results[feature_name] = drift_result

                # Check for alert conditions
                if drift_result.drift_detected:
                    self._create_drift_alert(feature_name, drift_result)

            except Exception as e:
                logger.error(f"Failed to detect drift for {feature_name}: {e}")
                continue

        logger.info(f"Drift detection completed for {len(drift_results)} features")
        return drift_results

    def _analyze_drift(self, current_data: pd.Series, feature_name: str) -> DriftResult:
        """Analyze drift for a single feature"""
        baseline = self.baseline_snapshots[feature_name]
        metadata = self.feature_registry.get_feature(feature_name)

        # If metadata is not found, create it on-the-fly for testing
        if metadata is None:
            metadata = self._create_metadata_from_data(current_data, feature_name)
            self.feature_registry.register_feature(metadata)

        drift_results = []

        # Distribution drift
        if metadata.feature_type in [FeatureType.NUMERICAL, FeatureType.CATEGORICAL]:
            distribution_drift = self._detect_distribution_drift(current_data, baseline, metadata)
            drift_results.append(distribution_drift)

        # Statistical drift
        if metadata.feature_type == FeatureType.NUMERICAL:
            statistical_drift = self._detect_statistical_drift(current_data, baseline)
            drift_results.append(statistical_drift)

        # Missing value drift
        missing_drift = self._detect_missing_value_drift(current_data, baseline)
        drift_results.append(missing_drift)

        # Cardinality drift (for categorical features)
        if metadata.feature_type == FeatureType.CATEGORICAL:
            cardinality_drift = self._detect_cardinality_drift(current_data, baseline)
            drift_results.append(cardinality_drift)

        # Range drift (for numerical features)
        if metadata.feature_type == FeatureType.NUMERICAL:
            range_drift = self._detect_range_drift(current_data, baseline)
            drift_results.append(range_drift)

        # Find the most significant drift
        max_drift = max(drift_results, key=lambda x: x.drift_score)

        return max_drift

    def _create_metadata_from_data(self, data: pd.Series, feature_name: str) -> FeatureMetadata:
        """Create feature metadata from data series"""
        from .feature_detector import FeatureDetector, FeatureType

        # Determine feature type
        if pd.api.types.is_numeric_dtype(data):
            feature_type = FeatureType.NUMERICAL
        elif pd.api.types.is_datetime64_dtype(data):
            feature_type = FeatureType.DATETIME
        elif data.nunique() <= 10:
            feature_type = FeatureType.CATEGORICAL
        else:
            feature_type = FeatureType.NUMERICAL

        # Calculate statistics
        null_count = data.isnull().sum()
        null_percentage = null_count / len(data)
        unique_count = data.nunique()

        metadata = FeatureMetadata(
            name=feature_name,
            feature_type=feature_type,
            dtype=str(data.dtype),
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            is_target=False,
            is_identifier=False
        )

        # Add additional statistics for numerical features
        if feature_type == FeatureType.NUMERICAL and not data.empty:
            clean_data = data.dropna()
            if len(clean_data) > 0:
                metadata.min_value = float(clean_data.min())
                metadata.max_value = float(clean_data.max())
                metadata.mean_value = float(clean_data.mean())
                metadata.std_value = float(clean_data.std())

        # Add categories for categorical features
        if feature_type == FeatureType.CATEGORICAL:
            metadata.categories = data.dropna().unique().tolist()

        return metadata

    def _detect_distribution_drift(self, current_data: pd.Series, baseline: FeatureSnapshot,
                                   metadata: FeatureMetadata) -> DriftResult:
        """Detect distribution drift using Jensen-Shannon divergence"""
        try:
            if metadata.feature_type == FeatureType.NUMERICAL:
                # For numerical data, create histograms
                baseline_hist = self._create_histogram(baseline.statistics['values'])
                current_hist = self._create_histogram(current_data.dropna())

                # Calculate Jensen-Shannon divergence
                js_distance = jensenshannon(baseline_hist, current_hist)
                drift_score = float(js_distance)

            else:  # Categorical
                # For categorical data, compare value distributions
                baseline_dist = baseline.statistics.get('value_counts', {})
                current_dist = current_data.value_counts().to_dict()

                # Align categories
                all_categories = set(baseline_dist.keys()) | set(current_dist.keys())
                baseline_probs = np.array([baseline_dist.get(cat, 0) for cat in all_categories])
                current_probs = np.array([current_dist.get(cat, 0) for cat in all_categories])

                # Normalize to probabilities
                baseline_probs = baseline_probs / baseline_probs.sum()
                current_probs = current_probs / current_probs.sum()

                js_distance = jensenshannon(baseline_probs, current_probs)
                drift_score = float(js_distance)

            return DriftResult(
                drift_detected=drift_score > self.config['drift_threshold'],
                drift_score=drift_score,
                drift_type=DriftType.DISTRIBUTION,
                confidence=min(drift_score * 10, 1.0),
                details={
                    'method': 'jensen_shannon_divergence',
                    'baseline_sample_size': baseline.sample_size,
                    'current_sample_size': len(current_data)
                }
            )

        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.DISTRIBUTION,
                confidence=0.0,
                details={'error': str(e)}
            )

    def _detect_statistical_drift(self, current_data: pd.Series, baseline: FeatureSnapshot) -> DriftResult:
        """Detect statistical drift using hypothesis tests"""
        try:
            current_clean = current_data.dropna()
            baseline_values = baseline.statistics['values']

            if len(current_clean) < self.config['min_sample_size']:
                return DriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    drift_type=DriftType.STATISTICAL,
                    confidence=0.0,
                    details={'error': 'Insufficient sample size'}
                )

            # Kolmogorov-Smirnov test for distribution comparison
            ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_clean)

            # T-test for mean comparison
            t_stat, t_pvalue = stats.ttest_ind(baseline_values, current_clean, equal_var=False)

            # Combined drift score
            drift_score = float((1 - ks_pvalue) + (1 - t_pvalue)) / 2

            return DriftResult(
                drift_detected=ks_pvalue < self.config['statistical_test_alpha'] or
                              t_pvalue < self.config['statistical_test_alpha'],
                drift_score=drift_score,
                drift_type=DriftType.STATISTICAL,
                confidence=drift_score,
                details={
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    't_statistic': t_stat,
                    't_pvalue': t_pvalue
                }
            )

        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.STATISTICAL,
                confidence=0.0,
                details={'error': str(e)}
            )

    def _detect_missing_value_drift(self, current_data: pd.Series, baseline: FeatureSnapshot) -> DriftResult:
        """Detect drift in missing value patterns"""
        try:
            current_null_pct = current_data.isnull().sum() / len(current_data)
            baseline_null_pct = baseline.statistics.get('null_percentage', 0.0)

            drift_score = abs(current_null_pct - baseline_null_pct)

            return DriftResult(
                drift_detected=drift_score > self.config['drift_threshold'],
                drift_score=drift_score,
                drift_type=DriftType.MISSING_VALUES,
                confidence=drift_score,
                details={
                    'baseline_null_percentage': baseline_null_pct,
                    'current_null_percentage': current_null_pct
                }
            )

        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.MISSING_VALUES,
                confidence=0.0,
                details={'error': str(e)}
            )

    def _detect_cardinality_drift(self, current_data: pd.Series, baseline: FeatureSnapshot) -> DriftResult:
        """Detect cardinality drift for categorical features"""
        try:
            current_unique = current_data.nunique()
            baseline_unique = baseline.statistics.get('unique_count', 0)

            if baseline_unique == 0:
                return DriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    drift_type=DriftType.CARDINALITY,
                    confidence=0.0,
                    details={'error': 'No baseline cardinality'}
                )

            # Calculate relative change
            relative_change = abs(current_unique - baseline_unique) / baseline_unique
            drift_score = min(relative_change, 1.0)

            return DriftResult(
                drift_detected=relative_change > self.config['drift_threshold'],
                drift_score=drift_score,
                drift_type=DriftType.CARDINALITY,
                confidence=drift_score,
                details={
                    'baseline_cardinality': baseline_unique,
                    'current_cardinality': current_unique,
                    'relative_change': relative_change
                }
            )

        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.CARDINALITY,
                confidence=0.0,
                details={'error': str(e)}
            )

    def _detect_range_drift(self, current_data: pd.Series, baseline: FeatureSnapshot) -> DriftResult:
        """Detect range drift for numerical features"""
        try:
            current_clean = current_data.dropna()
            if len(current_clean) == 0:
                return DriftResult(
                    drift_detected=False,
                    drift_score=0.0,
                    drift_type=DriftType.RANGE,
                    confidence=0.0,
                    details={'error': 'No valid current data'}
                )

            current_min, current_max = current_clean.min(), current_clean.max()
            baseline_min = baseline.statistics.get('min_value', current_min)
            baseline_max = baseline.statistics.get('max_value', current_max)

            # Calculate range expansion
            range_expansion = 0.0
            if current_min < baseline_min:
                range_expansion += (baseline_min - current_min) / (baseline_max - baseline_min + 1e-8)
            if current_max > baseline_max:
                range_expansion += (current_max - baseline_max) / (baseline_max - baseline_min + 1e-8)

            drift_score = min(range_expansion, 1.0)

            return DriftResult(
                drift_detected=range_expansion > self.config['drift_threshold'],
                drift_score=drift_score,
                drift_type=DriftType.RANGE,
                confidence=drift_score,
                details={
                    'baseline_range': (baseline_min, baseline_max),
                    'current_range': (current_min, current_max),
                    'range_expansion': range_expansion
                }
            )

        except Exception as e:
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type=DriftType.RANGE,
                confidence=0.0,
                details={'error': str(e)}
            )

    def _create_histogram(self, data: pd.Series, bins: int = 50) -> np.ndarray:
        """Create histogram for numerical data"""
        counts, _ = np.histogram(data, bins=bins, density=True)
        return counts

    def _create_snapshot(self, data: pd.Series, feature_name: str) -> FeatureSnapshot:
        """Create snapshot of feature statistics"""
        clean_data = data.dropna()
        statistics = {
            'null_percentage': data.isnull().sum() / len(data),
            'unique_count': data.nunique(),
            'values': clean_data.tolist()[:self.config['baseline_sample_size']]  # Sample for efficiency
        }

        # Add type-specific statistics
        if pd.api.types.is_numeric_dtype(data) and data.dtype != 'bool':
            statistics.update({
                'min_value': clean_data.min(),
                'max_value': clean_data.max(),
                'mean_value': clean_data.mean(),
                'std_value': clean_data.std(),
                'median_value': clean_data.median(),
                'percentiles': {
                    '25%': clean_data.quantile(0.25),
                    '75%': clean_data.quantile(0.75)
                }
            })

            if len(clean_data) > 0:
                statistics['value_counts'] = clean_data.value_counts().head(20).to_dict()
        elif data.dtype == 'bool':
            # For boolean data, just store value counts
            if len(clean_data) > 0:
                statistics['value_counts'] = clean_data.value_counts().to_dict()
                statistics['true_count'] = int(clean_data.sum())
                statistics['false_count'] = int(len(clean_data) - clean_data.sum())
        else:
            # For categorical/object data
            if len(clean_data) > 0:
                statistics['value_counts'] = clean_data.value_counts().head(20).to_dict()

        return FeatureSnapshot(
            timestamp=datetime.now(),
            feature_name=feature_name,
            statistics=statistics,
            sample_size=len(data)
        )

    def _create_drift_alert(self, feature_name: str, drift_result: DriftResult):
        """Create alert for detected drift"""
        # Check cooldown period
        recent_alerts = [a for a in self.alerts
                        if a['feature_name'] == feature_name and
                        (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < self.config['alert_cooldown_minutes'] * 60]

        if recent_alerts:
            return  # Skip due to cooldown

        alert = {
            'timestamp': datetime.now().isoformat(),
            'feature_name': feature_name,
            'alert_type': 'drift_detected',
            'severity': 'high' if drift_result.drift_score > 0.5 else 'medium',
            'drift_score': drift_result.drift_score,
            'drift_type': drift_result.drift_type.value,
            'confidence': drift_result.confidence,
            'details': drift_result.details
        }

        self.alerts.append(alert)
        logger.warning(f"Drift alert created for {feature_name}: {drift_result.drift_type.value} (score: {drift_result.drift_score:.3f})")

    def calculate_health_metrics(self, drift_results: Dict[str, DriftResult]) -> Dict[str, HealthMetrics]:
        """Calculate health metrics for all features"""
        health_metrics = {}

        for feature_name, drift_result in drift_results.items():
            # Determine health status
            if drift_result.drift_score > 0.5:
                status = HealthStatus.CRITICAL
            elif drift_result.drift_score > self.config['drift_threshold']:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY

            # Get feature metadata for additional metrics
            metadata = self.feature_registry.get_feature(feature_name)
            uniqueness_score = 0.0
            stability_score = 1.0 - drift_result.drift_score

            if metadata:
                uniqueness_score = min(metadata.unique_count / 100, 1.0)  # Normalize uniqueness

            alerts = []
            if status != HealthStatus.HEALTHY:
                alerts.append(f"Drift detected: {drift_result.drift_type.value}")

            health_metrics[feature_name] = HealthMetrics(
                status=status,
                drift_score=drift_result.drift_score,
                null_percentage=drift_result.details.get('current_null_percentage', 0.0),
                uniqueness_score=uniqueness_score,
                stability_score=stability_score,
                last_updated=datetime.now(),
                alerts=alerts
            )

        self.health_metrics.update(health_metrics)
        return health_metrics

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring status"""
        health_counts = {}
        if self.health_metrics:
            for metrics in self.health_metrics.values():
                status = metrics.status.value
                health_counts[status] = health_counts.get(status, 0) + 1

        return {
            'total_features': len(self.health_metrics),
            'health_distribution': health_counts,
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:],  # Last 5 alerts
            'average_drift_score': np.mean([m.drift_score for m in self.health_metrics.values()]) if self.health_metrics else 0.0,
            'critical_features': [
                name for name, metrics in self.health_metrics.items()
                if metrics.status == HealthStatus.CRITICAL
            ]
        }

    def save_monitoring_data(self, file_path: Union[str, Path]):
        """Save monitoring data to file"""
        monitoring_data = {
            'config': self._serialize_config(self.config),
            'baseline_snapshots': {
                name: {
                    'timestamp': snapshot.timestamp.isoformat(),
                    'feature_name': snapshot.feature_name,
                    'statistics': self._serialize_stats(snapshot.statistics),
                    'sample_size': int(snapshot.sample_size)
                } for name, snapshot in self.baseline_snapshots.items()
            },
            'health_metrics': {
                name: {
                    'status': metrics.status.value,
                    'drift_score': float(metrics.drift_score),
                    'null_percentage': float(metrics.null_percentage),
                    'uniqueness_score': float(metrics.uniqueness_score),
                    'stability_score': float(metrics.stability_score),
                    'last_updated': metrics.last_updated.isoformat(),
                    'alerts': metrics.alerts
                } for name, metrics in self.health_metrics.items()
            },
            'alerts': self.alerts[-100:],  # Last 100 alerts
            'monitoring_history': self.monitoring_history[-50:],  # Last 50 events
            'saved_at': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(monitoring_data, f, indent=2)

    def _serialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize configuration for JSON storage"""
        serialized = {}
        for key, value in config_dict.items():
            if hasattr(value, 'value'):  # Handle enums
                serialized[key] = value.value
            else:
                serialized[key] = value
        return serialized

    def _serialize_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize statistics for JSON storage"""
        serialized = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                if isinstance(value, (int, float)):
                    serialized[key] = float(value) if isinstance(value, float) else int(value)
                else:
                    serialized[key] = value
            elif value is None:
                serialized[key] = None
            else:
                # Handle numpy arrays or other complex types
                try:
                    serialized[key] = float(value)
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        return serialized

        logger.info(f"Monitoring data saved to {file_path}")

    def load_monitoring_data(self, file_path: Union[str, Path]):
        """Load monitoring data from file"""
        with open(file_path, 'r') as f:
            monitoring_data = json.load(f)

        self.config = self._deserialize_config(monitoring_data.get('config', {}))
        self.alerts = monitoring_data.get('alerts', [])
        self.monitoring_history = monitoring_data.get('monitoring_history', [])

        # Load baseline snapshots
        self.baseline_snapshots = {}
        for name, snapshot_data in monitoring_data.get('baseline_snapshots', {}).items():
            self.baseline_snapshots[name] = FeatureSnapshot(
                timestamp=datetime.fromisoformat(snapshot_data['timestamp']),
                feature_name=snapshot_data['feature_name'],
                statistics=snapshot_data['statistics'],
                sample_size=int(snapshot_data['sample_size'])
            )

        # Load health metrics
        self.health_metrics = {}
        for name, metrics_data in monitoring_data.get('health_metrics', {}).items():
            self.health_metrics[name] = HealthMetrics(
                status=HealthStatus(metrics_data['status']),
                drift_score=float(metrics_data['drift_score']),
                null_percentage=float(metrics_data['null_percentage']),
                uniqueness_score=float(metrics_data['uniqueness_score']),
                stability_score=float(metrics_data['stability_score']),
                last_updated=datetime.fromisoformat(metrics_data['last_updated']),
                alerts=metrics_data['alerts']
            )

        logger.info(f"Monitoring data loaded from {file_path}")

    def _deserialize_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize configuration from JSON storage"""
        deserialized = {}
        for key, value in config_dict.items():
            if key == 'drift_threshold' or isinstance(value, str):
                deserialized[key] = value
            else:
                deserialized[key] = value
        return deserialized