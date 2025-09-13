#!/usr/bin/env python3
"""
Model Monitoring with Feature Importance Integration
==================================================

A comprehensive model monitoring system that integrates feature importance
tracking to provide real-time insights into model performance and feature behavior.

This system provides:
- Real-time model performance monitoring
- Feature importance integration for model diagnostics
- Automated model health assessment
- Performance degradation detection
- Feature impact analysis on model predictions

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
from pathlib import Path
import sqlite3
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

from .importance_engine import FeatureImportanceEngine, ImportanceResult, ImportanceScore, ImportanceMethod, ImportanceType
from .importance_tracker import ImportanceTracker, DriftAlert, ImportanceTrend
from .feature_monitor import FeatureMonitor, DriftResult, HealthMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelHealthStatus(Enum):
    """Model health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class PerformanceMetric(Enum):
    """Performance metrics for monitoring."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MSE = "mse"
    MAE = "mae"
    R2 = "r2"
    CUSTOM = "custom"


@dataclass
class ModelPrediction:
    """Single model prediction with metadata."""
    timestamp: datetime
    prediction: Union[float, np.ndarray, torch.Tensor]
    actual: Optional[Union[float, np.ndarray, torch.Tensor]]
    features: Union[np.ndarray, pd.DataFrame, torch.Tensor]
    confidence: Optional[float]
    prediction_id: Optional[str]


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    timestamp: datetime
    metric: PerformanceMetric
    value: float
    window_size: int
    model_version: str
    metadata: Dict[str, Any] = None


@dataclass
class ModelHealth:
    """Overall model health assessment."""
    timestamp: datetime
    status: ModelHealthStatus
    overall_score: float
    performance_metrics: Dict[PerformanceMetric, float]
    feature_importance_stability: float
    drift_detected: bool
    recommendations: List[str]
    critical_issues: List[str]


@dataclass
class MonitoringAlert:
    """Alert for model monitoring issues."""
    timestamp: datetime
    alert_type: str
    severity: ModelHealthStatus
    description: str
    feature_name: Optional[str]
    metric_name: Optional[str]
    current_value: Optional[float]
    threshold: Optional[float]
    recommendations: List[str]


class ModelMonitor:
    """Main class for monitoring model performance with feature importance integration."""

    def __init__(self,
                 model: nn.Module,
                 importance_engine: FeatureImportanceEngine,
                 feature_monitor: FeatureMonitor,
                 importance_tracker: ImportanceTracker,
                 storage_path: str = "model_monitor.db",
                 monitoring_config: Dict[str, Any] = None):
        """
        Initialize model monitor.

        Args:
            model: PyTorch model to monitor
            importance_engine: Feature importance calculation engine
            feature_monitor: Feature drift and health monitor
            importance_tracker: Feature importance tracking system
            storage_path: Path to database file
            monitoring_config: Configuration for monitoring thresholds and settings
        """
        self.model = model
        self.importance_engine = importance_engine
        self.feature_monitor = feature_monitor
        self.importance_tracker = importance_tracker
        self.storage_path = Path(storage_path)

        # Default monitoring configuration
        self.config = monitoring_config or {
            'performance_thresholds': {
                'accuracy': {'warning': 0.05, 'degraded': 0.1, 'critical': 0.2},
                'mse': {'warning': 0.1, 'degraded': 0.2, 'critical': 0.5},
                'mae': {'warning': 0.05, 'degraded': 0.1, 'critical': 0.2}
            },
            'importance_stability_threshold': 0.15,
            'min_predictions_for_metrics': 50,
            'monitoring_window_size': 100,
            'health_check_interval': 3600,  # 1 hour
            'alert_cooldown': 300  # 5 minutes
        }

        self.predictions = deque(maxlen=self.config['monitoring_window_size'])
        self.metrics_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=500)
        self.last_health_check = None
        self.last_alert_times = defaultdict(lambda: 0)

        self._initialize_database()

    def _initialize_database(self):
        """Initialize SQLite database for storing monitoring data."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    prediction_id TEXT,
                    prediction_value REAL,
                    actual_value REAL,
                    features TEXT,
                    confidence REAL,
                    computation_time REAL
                )
            ''')

            # Create metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    window_size INTEGER,
                    model_version TEXT,
                    metadata TEXT
                )
            ''')

            # Create health assessments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    status TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    importance_stability REAL,
                    drift_detected BOOLEAN,
                    recommendations TEXT,
                    critical_issues TEXT
                )
            ''')

            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    feature_name TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold REAL,
                    recommendations TEXT
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON model_predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON model_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_health_timestamp ON model_health(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON monitoring_alerts(timestamp)')

            conn.commit()

        logger.info(f"Model monitoring database initialized at {self.storage_path}")

    def record_prediction(self,
                         prediction: Union[float, np.ndarray, torch.Tensor],
                         features: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                         actual: Optional[Union[float, np.ndarray, torch.Tensor]] = None,
                         confidence: Optional[float] = None,
                         prediction_id: Optional[str] = None,
                         calculate_importance: bool = True) -> Dict[str, Any]:
        """
        Record a model prediction and optionally calculate feature importance.

        Args:
            prediction: Model prediction
            features: Input features
            actual: Actual value (if available)
            confidence: Prediction confidence score
            prediction_id: Unique identifier for the prediction
            calculate_importance: Whether to calculate feature importance

        Returns:
            Dictionary with prediction record and importance results
        """
        start_time = time.time()
        timestamp = datetime.now()

        # Prepare prediction data
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.detach().cpu().numpy()
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()

        # Store prediction
        pred_record = ModelPrediction(
            timestamp=timestamp,
            prediction=prediction,
            actual=actual,
            features=features,
            confidence=confidence,
            prediction_id=prediction_id
        )
        self.predictions.append(pred_record)

        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_predictions
                (timestamp, prediction_id, prediction_value, actual_value, features, confidence, computation_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(),
                prediction_id,
                float(prediction) if np.isscalar(prediction) else None,
                float(actual) if actual is not None and np.isscalar(actual) else None,
                json.dumps(self._serialize_features(features)),
                confidence,
                time.time() - start_time
            ))
            conn.commit()

        result = {
            'prediction_record': pred_record,
            'importance_results': None,
            'alerts_generated': []
        }

        # Calculate feature importance if requested
        if calculate_importance and len(self.predictions) >= 5:
            try:
                importance_results = self._calculate_prediction_importance(features, prediction)
                result['importance_results'] = importance_results

                # Record importance in tracker
                for method, imp_result in importance_results.items():
                    self.importance_tracker.record_importance(imp_result, timestamp)

            except Exception as e:
                logger.warning(f"Failed to calculate importance for prediction: {e}")

        # Check for immediate alerts
        alerts = self._check_immediate_alerts(pred_record)
        result['alerts_generated'] = alerts
        self.alerts.extend(alerts)

        for alert in alerts:
            self._store_alert(alert)

        # Update metrics if we have actual values
        if actual is not None:
            self._update_performance_metrics()

        logger.info(f"Recorded prediction {prediction_id or 'unnamed'}")
        return result

    def _calculate_prediction_importance(self,
                                       features: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                                       prediction: Union[float, np.ndarray]) -> Dict[ImportanceMethod, ImportanceResult]:
        """Calculate feature importance for a specific prediction."""
        # Use local importance methods for individual predictions
        importance_results = self.importance_engine.calculate_local_importance(
            self.model, features, **{'prediction_context': True}
        )

        return importance_results

    def _serialize_features(self, features: Union[np.ndarray, pd.DataFrame, torch.Tensor]) -> Any:
        """Serialize features for database storage."""
        if isinstance(features, pd.DataFrame):
            return {
                'type': 'dataframe',
                'columns': features.columns.tolist(),
                'shape': features.shape,
                'sample_values': features.head(5).values.tolist()
            }
        elif isinstance(features, torch.Tensor):
            return {
                'type': 'tensor',
                'shape': tuple(features.shape),
                'dtype': str(features.dtype),
                'sample_values': features[:5].detach().cpu().numpy().tolist()
            }
        elif isinstance(features, np.ndarray):
            return {
                'type': 'array',
                'shape': features.shape,
                'dtype': str(features.dtype),
                'sample_values': features[:5].tolist()
            }
        else:
            return {'type': 'unknown', 'data': str(features)}

    def _check_immediate_alerts(self, prediction: ModelPrediction) -> List[MonitoringAlert]:
        """Check for immediate alerts based on prediction data."""
        alerts = []
        current_time = time.time()

        # Check confidence scores
        if prediction.confidence is not None:
            if prediction.confidence < 0.3:
                if current_time - self.last_alert_times['low_confidence'] > self.config['alert_cooldown']:
                    alert = MonitoringAlert(
                        timestamp=datetime.now(),
                        alert_type="low_confidence",
                        severity=ModelHealthStatus.WARNING,
                        description=f"Low confidence prediction: {prediction.confidence:.3f}",
                        current_value=prediction.confidence,
                        threshold=0.3,
                        recommendations=[
                            "Review model input features",
                            "Check for data quality issues",
                            "Consider model retraining"
                        ]
                    )
                    alerts.append(alert)
                    self.last_alert_times['low_confidence'] = current_time

        # Check for unusual feature patterns using feature monitor
        if hasattr(self.feature_monitor, 'check_feature_health'):
            try:
                feature_health = self.feature_monitor.check_feature_health(prediction.features)
                if feature_health.get('drift_detected', False):
                    if current_time - self.last_alert_times['feature_drift'] > self.config['alert_cooldown']:
                        alert = MonitoringAlert(
                            timestamp=datetime.now(),
                            alert_type="feature_drift",
                            severity=ModelHealthStatus.WARNING,
                            description="Feature drift detected in recent prediction",
                            recommendations=[
                                "Investigate feature distribution changes",
                                "Update feature preprocessing if needed",
                                "Monitor model performance impact"
                            ]
                        )
                        alerts.append(alert)
                        self.last_alert_times['feature_drift'] = current_time
            except Exception as e:
                logger.warning(f"Failed to check feature health: {e}")

        return alerts

    def _update_performance_metrics(self):
        """Update performance metrics based on recent predictions with actual values."""
        if len(self.predictions) < self.config['min_predictions_for_metrics']:
            return

        # Get predictions with actual values
        valid_predictions = [p for p in self.predictions if p.actual is not None]
        if len(valid_predictions) < self.config['min_predictions_for_metrics']:
            return

        predictions = np.array([p.prediction for p in valid_predictions])
        actuals = np.array([p.actual for p in valid_predictions])

        # Calculate various metrics
        timestamp = datetime.now()

        # MSE
        mse = np.mean((predictions - actuals) ** 2)
        self._record_metric(PerformanceMetric.MSE, mse, len(valid_predictions), timestamp)

        # MAE
        mae = np.mean(np.abs(predictions - actuals))
        self._record_metric(PerformanceMetric.MAE, mae, len(valid_predictions), timestamp)

        # R2 score
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        self._record_metric(PerformanceMetric.R2, r2, len(valid_predictions), timestamp)

        logger.info(f"Updated performance metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    def _record_metric(self, metric: PerformanceMetric, value: float, window_size: int, timestamp: datetime):
        """Record a performance metric."""
        metric_record = ModelMetrics(
            timestamp=timestamp,
            metric=metric,
            value=value,
            window_size=window_size,
            model_version="v1.0",  # Could be dynamic
            metadata={'computed_from_predictions': True}
        )
        self.metrics_history.append(metric_record)

        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_metrics
                (timestamp, metric_name, metric_value, window_size, model_version, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(),
                metric.value,
                value,
                window_size,
                metric_record.model_version,
                json.dumps(metric_record.metadata)
            ))
            conn.commit()

    def assess_model_health(self, force: bool = False) -> ModelHealth:
        """
        Assess overall model health using performance metrics and feature importance.

        Args:
            force: Force health check even if not scheduled

        Returns:
            Model health assessment
        """
        current_time = datetime.now()

        # Check if health check is due
        if not force and self.last_health_check:
            time_since_last = (current_time - self.last_health_check).total_seconds()
            if time_since_last < self.config['health_check_interval']:
                logger.info("Health check not due yet")
                return None

        self.last_health_check = current_time

        # Get recent performance metrics
        recent_metrics = self._get_recent_metrics()
        performance_issues = []
        performance_scores = {}

        # Analyze each metric
        for metric in [PerformanceMetric.MSE, PerformanceMetric.MAE, PerformanceMetric.R2]:
            if metric in recent_metrics:
                metric_value = recent_metrics[metric]
                threshold_config = self.config['performance_thresholds'].get(metric.value, {})

                # Check against thresholds
                if 'critical' in threshold_config and metric_value > threshold_config['critical']:
                    performance_issues.append(f"Critical {metric.value} degradation: {metric_value:.4f}")
                    performance_scores[metric] = 0.2
                elif 'degraded' in threshold_config and metric_value > threshold_config['degraded']:
                    performance_issues.append(f"Degraded {metric.value}: {metric_value:.4f}")
                    performance_scores[metric] = 0.5
                elif 'warning' in threshold_config and metric_value > threshold_config['warning']:
                    performance_issues.append(f"Warning: {metric.value} at {metric_value:.4f}")
                    performance_scores[metric] = 0.8
                else:
                    performance_scores[metric] = 1.0

        # Check feature importance stability
        importance_stability = self._assess_importance_stability()

        # Check for drift alerts
        recent_drift_alerts = self.importance_tracker.detect_drift()
        drift_detected = len(recent_drift_alerts) > 0

        # Calculate overall health score
        if performance_scores:
            performance_score = np.mean(list(performance_scores.values()))
        else:
            performance_score = 0.8  # Default if no metrics available

        overall_score = (performance_score * 0.6) + (importance_stability * 0.4)

        # Determine health status
        if overall_score >= 0.8 and not performance_issues and not drift_detected:
            status = ModelHealthStatus.HEALTHY
        elif overall_score >= 0.6 and len(performance_issues) <= 1:
            status = ModelHealthStatus.WARNING
        elif overall_score >= 0.4:
            status = ModelHealthStatus.DEGRADED
        else:
            status = ModelHealthStatus.CRITICAL

        # Generate recommendations
        recommendations = self._generate_health_recommendations(
            status, performance_issues, importance_stability, drift_detected
        )

        health = ModelHealth(
            timestamp=current_time,
            status=status,
            overall_score=overall_score,
            performance_metrics=performance_scores,
            feature_importance_stability=importance_stability,
            drift_detected=drift_detected,
            recommendations=recommendations,
            critical_issues=[issue for issue in performance_issues if "Critical" in issue]
        )

        # Store health assessment
        self._store_health_assessment(health)

        logger.info(f"Model health assessment: {status.value} (score: {overall_score:.3f})")
        return health

    def _get_recent_metrics(self) -> Dict[PerformanceMetric, float]:
        """Get recent performance metrics."""
        if not self.metrics_history:
            return {}

        recent_metrics = {}
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour

        recent = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        if not recent:
            return {}

        # Get latest value for each metric
        for metric in PerformanceMetric:
            metric_records = [m for m in recent if m.metric == metric]
            if metric_records:
                latest = max(metric_records, key=lambda x: x.timestamp)
                recent_metrics[metric] = latest.value

        return recent_metrics

    def _assess_importance_stability(self) -> float:
        """Assess feature importance stability."""
        try:
            # Get recent importance trends
            trends = self.importance_tracker.analyze_trends(lookback_days=7)

            if not trends:
                return 0.8  # Default stability if no trends available

            # Calculate stability score based on trend consistency
            stability_scores = []
            for trend in trends.values():
                # Stable trends get higher scores
                if trend.trend_direction == "stable":
                    stability_scores.append(1.0)
                elif trend.trend_strength < 0.1:  # Weak trend
                    stability_scores.append(0.8)
                elif trend.trend_strength < 0.2:  # Moderate trend
                    stability_scores.append(0.6)
                else:  # Strong trend (potentially unstable)
                    stability_scores.append(0.4)

            return np.mean(stability_scores) if stability_scores else 0.8

        except Exception as e:
            logger.warning(f"Failed to assess importance stability: {e}")
            return 0.7  # Conservative default

    def _generate_health_recommendations(self,
                                        status: ModelHealthStatus,
                                        performance_issues: List[str],
                                        importance_stability: float,
                                        drift_detected: bool) -> List[str]:
        """Generate recommendations based on health assessment."""
        recommendations = []

        if status == ModelHealthStatus.CRITICAL:
            recommendations.extend([
                "Immediate model retraining recommended",
                "Investigate data quality and distribution",
                "Review feature engineering pipeline",
                "Consider model architecture changes"
            ])
        elif status == ModelHealthStatus.DEGRADED:
            recommendations.extend([
                "Schedule model retraining",
                "Monitor performance closely",
                "Check for data drift"
            ])
        elif status == ModelHealthStatus.WARNING:
            recommendations.extend([
                "Continue monitoring",
                "Investigate performance metrics",
                "Check feature quality"
            ])

        if importance_stability < 0.6:
            recommendations.append("Investigate feature importance instability")

        if drift_detected:
            recommendations.append("Address detected feature drift")

        return recommendations

    def _store_health_assessment(self, health: ModelHealth):
        """Store health assessment in database."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO model_health
                (timestamp, status, overall_score, importance_stability, drift_detected, recommendations, critical_issues)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                health.timestamp.isoformat(),
                health.status.value,
                health.overall_score,
                health.feature_importance_stability,
                health.drift_detected,
                json.dumps(health.recommendations),
                json.dumps(health.critical_issues)
            ))
            conn.commit()

    def _store_alert(self, alert: MonitoringAlert):
        """Store alert in database."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO monitoring_alerts
                (timestamp, alert_type, severity, description, feature_name, metric_name, current_value, threshold, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.alert_type,
                alert.severity.value,
                alert.description,
                alert.feature_name,
                alert.metric_name,
                alert.current_value,
                alert.threshold,
                json.dumps(alert.recommendations)
            ))
            conn.commit()

    def get_monitoring_summary(self, lookback_days: int = 7) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Args:
            lookback_days: Number of days to look back for summary

        Returns:
            Monitoring summary dictionary
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Get prediction statistics
            cursor.execute('''
                SELECT COUNT(*) as total_predictions,
                       AVG(confidence) as avg_confidence,
                       COUNT(CASE WHEN actual_value IS NOT NULL THEN 1 END) as predictions_with_actual
                FROM model_predictions
                WHERE timestamp >= ? AND timestamp <= ?
            ''', (start_time.isoformat(), end_time.isoformat()))
            pred_stats = cursor.fetchone()

            # Get metrics summary
            cursor.execute('''
                SELECT metric_name, AVG(metric_value), COUNT(*)
                FROM model_metrics
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY metric_name
            ''', (start_time.isoformat(), end_time.isoformat()))
            metrics_summary = cursor.fetchall()

            # Get health assessments
            cursor.execute('''
                SELECT status, COUNT(*) as count,
                       AVG(overall_score) as avg_score
                FROM model_health
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY status
            ''', (start_time.isoformat(), end_time.isoformat()))
            health_summary = cursor.fetchall()

            # Get recent alerts
            cursor.execute('''
                SELECT alert_type, severity, COUNT(*) as count
                FROM monitoring_alerts
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY alert_type, severity
                ORDER BY count DESC
            ''', (start_time.isoformat(), end_time.isoformat()))
            alerts_summary = cursor.fetchall()

        return {
            'summary_period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': lookback_days
            },
            'predictions': {
                'total_count': pred_stats[0] if pred_stats else 0,
                'avg_confidence': pred_stats[1] if pred_stats and pred_stats[1] else 0,
                'with_actual_values': pred_stats[2] if pred_stats else 0
            },
            'performance_metrics': [
                {
                    'metric': row[0],
                    'average_value': row[1],
                    'measurement_count': row[2]
                }
                for row in metrics_summary
            ],
            'health_assessments': [
                {
                    'status': row[0],
                    'count': row[1],
                    'average_score': row[2]
                }
                for row in health_summary
            ],
            'alerts': [
                {
                    'type': row[0],
                    'severity': row[1],
                    'count': row[2]
                }
                for row in alerts_summary
            ]
        }

    def generate_monitoring_report(self, export_path: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.

        Args:
            export_path: Path to save report
            lookback_days: Number of days to include in report

        Returns:
            Complete monitoring report
        """
        # Get current model health
        current_health = self.assess_model_health(force=True)

        # Get monitoring summary
        summary = self.get_monitoring_summary(lookback_days)

        # Get feature importance summary
        importance_summary = self.importance_tracker.get_feature_importance_summary(lookback_days=lookback_days)

        # Get recent drift alerts
        recent_drift_alerts = self.importance_tracker.detect_drift()

        report = {
            'report_timestamp': datetime.now().isoformat(),
            'report_period_days': lookback_days,
            'current_model_health': {
                'status': current_health.status.value if current_health else 'unknown',
                'overall_score': current_health.overall_score if current_health else 0,
                'recommendations': current_health.recommendations if current_health else [],
                'critical_issues': current_health.critical_issues if current_health else []
            },
            'monitoring_summary': summary,
            'feature_importance_summary': importance_summary,
            'recent_drift_alerts': [
                {
                    'feature_name': alert.feature_name,
                    'severity': alert.severity.value,
                    'drift_score': alert.drift_score,
                    'description': alert.description
                }
                for alert in recent_drift_alerts
            ],
            'system_integration_status': {
                'importance_engine': 'connected',
                'feature_monitor': 'connected',
                'importance_tracker': 'connected',
                'database_status': 'active'
            }
        }

        # Save report
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Monitoring report exported to {export_path}")
        return report

    def cleanup_old_data(self, max_days: int = 90):
        """Clean up old monitoring data."""
        cutoff_date = datetime.now() - timedelta(days=max_days)

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Clean up old predictions (keep last 30 days)
            pred_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute('''
                DELETE FROM model_predictions
                WHERE timestamp < ?
            ''', (pred_cutoff.isoformat(),))

            # Clean up old metrics (keep last 90 days)
            cursor.execute('''
                DELETE FROM model_metrics
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))

            # Clean up old health assessments (keep last 90 days)
            cursor.execute('''
                DELETE FROM model_health
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))

            # Clean up old alerts (keep last 30 days)
            alert_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute('''
                DELETE FROM monitoring_alerts
                WHERE timestamp < ?
            ''', (alert_cutoff.isoformat(),))

            conn.commit()

        logger.info(f"Cleaned up monitoring data older than {cutoff_date}")