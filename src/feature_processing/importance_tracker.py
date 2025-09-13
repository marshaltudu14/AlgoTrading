#!/usr/bin/env python3
"""
Feature Importance Tracking System
=================================

A comprehensive system for tracking feature importance over time,
detecting importance drift, and maintaining historical records of
feature importance metrics.

This system provides:
- Time-series importance tracking
- Importance trend analysis
- Importance drift detection
- Importance aggregation across predictions
- Historical importance data management

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
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

from .importance_engine import ImportanceResult, ImportanceScore, ImportanceMethod, ImportanceType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DriftDetectionMethod(Enum):
    """Methods for detecting importance drift."""
    STATISTICAL = "statistical"
    THRESHOLD_BASED = "threshold_based"
    TREND_BASED = "trend_based"
    DISTRIBUTION_BASED = "distribution_based"


class DriftSeverity(Enum):
    """Severity levels for importance drift."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ImportanceSnapshot:
    """Snapshot of importance scores at a specific time."""
    timestamp: datetime
    feature_importance: Dict[str, float]
    method: ImportanceMethod
    importance_type: ImportanceType
    model_info: Dict[str, Any]
    prediction_count: int
    metadata: Dict[str, Any] = None


@dataclass
class DriftAlert:
    """Alert for importance drift detection."""
    feature_name: str
    timestamp: datetime
    severity: DriftSeverity
    drift_score: float
    method: DriftDetectionMethod
    baseline_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    description: str
    recommendations: List[str]


@dataclass
class ImportanceTrend:
    """Trend analysis for feature importance over time."""
    feature_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable', 'volatile'
    trend_strength: float
    correlation_with_performance: float
    seasonality_detected: bool
    change_points: List[datetime]


class ImportanceTracker:
    """Main class for tracking feature importance over time."""

    def __init__(self,
                 storage_path: str = "importance_tracker.db",
                 max_history_days: int = 365,
                 drift_detection_methods: List[DriftDetectionMethod] = None,
                 drift_thresholds: Dict[DriftSeverity, float] = None):
        """
        Initialize importance tracker.

        Args:
            storage_path: Path to database file
            max_history_days: Maximum days to keep history
            drift_detection_methods: Methods for drift detection
            drift_thresholds: Thresholds for drift severity levels
        """
        self.storage_path = Path(storage_path)
        self.max_history_days = max_history_days
        self.drift_detection_methods = drift_detection_methods or [
            DriftDetectionMethod.STATISTICAL,
            DriftDetectionMethod.THRESHOLD_BASED
        ]

        self.drift_thresholds = drift_thresholds or {
            DriftSeverity.LOW: 0.1,
            DriftSeverity.MEDIUM: 0.2,
            DriftSeverity.HIGH: 0.3,
            DriftSeverity.CRITICAL: 0.5
        }

        self._initialize_database()
        self.alerts = deque(maxlen=1000)
        self.trends_cache = {}

    def _initialize_database(self):
        """Initialize SQLite database for storing importance data."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Create snapshots table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS importance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    importance_score REAL NOT NULL,
                    method TEXT NOT NULL,
                    importance_type TEXT NOT NULL,
                    model_info TEXT,
                    prediction_count INTEGER,
                    metadata TEXT
                )
            ''')

            # Create drift alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    feature_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    method TEXT NOT NULL,
                    baseline_start DATETIME,
                    baseline_end DATETIME,
                    current_start DATETIME,
                    current_end DATETIME,
                    description TEXT,
                    recommendations TEXT
                )
            ''')

            # Create trends table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS importance_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feature_name TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    trend_direction TEXT NOT NULL,
                    trend_strength REAL NOT NULL,
                    correlation_with_performance REAL,
                    seasonality_detected BOOLEAN,
                    change_points TEXT
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON importance_snapshots(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_feature ON importance_snapshots(feature_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON drift_alerts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_feature ON drift_alerts(feature_name)')

            conn.commit()

        logger.info(f"Database initialized at {self.storage_path}")

    def record_importance(self,
                         result: ImportanceResult,
                         timestamp: Optional[datetime] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Record importance scores from a calculation result.

        Args:
            result: Importance calculation result
            timestamp: Custom timestamp (uses current time if None)
            metadata: Additional metadata to store
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Prepare feature importance dictionary
        feature_importance = {}
        for score in result.scores:
            if score.feature_name not in feature_importance:
                feature_importance[score.feature_name] = []
            feature_importance[score.feature_name].append(score.score)

        # Average scores for each feature
        avg_importance = {
            feat: np.mean(scores) for feat, scores in feature_importance.items()
        }

        # Store in database
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            for feature_name, score in avg_importance.items():
                cursor.execute('''
                    INSERT INTO importance_snapshots
                    (timestamp, feature_name, importance_score, method, importance_type, model_info, prediction_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp.isoformat(),
                    feature_name,
                    float(score),
                    result.method.value,
                    result.importance_type.value,
                    json.dumps(result.model_info),
                    len(result.scores),
                    json.dumps(metadata or {})
                ))

            conn.commit()

        logger.info(f"Recorded importance for {len(avg_importance)} features at {timestamp}")

    def get_importance_history(self,
                              feature_names: Optional[List[str]] = None,
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None,
                              methods: Optional[List[ImportanceMethod]] = None) -> pd.DataFrame:
        """
        Get historical importance data.

        Args:
            feature_names: Filter by feature names
            start_time: Start time for history
            end_time: End time for history
            methods: Filter by calculation methods

        Returns:
            DataFrame with historical importance data
        """
        query = '''
            SELECT timestamp, feature_name, importance_score, method, importance_type,
                   model_info, prediction_count, metadata
            FROM importance_snapshots
            WHERE 1=1
        '''

        params = []

        if feature_names:
            placeholders = ','.join(['?' for _ in feature_names])
            query += f' AND feature_name IN ({placeholders})'
            params.extend(feature_names)

        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time.isoformat())

        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time.isoformat())

        if methods:
            placeholders = ','.join(['?' for _ in methods])
            query += f' AND method IN ({placeholders})'
            params.extend([method.value for method in methods])

        query += ' ORDER BY timestamp ASC'

        with sqlite3.connect(self.storage_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['importance_score'] = pd.to_numeric(df['importance_score'])

        return df

    def detect_drift(self,
                    feature_names: Optional[List[str]] = None,
                    baseline_period: Tuple[datetime, datetime] = None,
                    current_period: Tuple[datetime, datetime] = None,
                    methods: Optional[List[DriftDetectionMethod]] = None) -> List[DriftAlert]:
        """
        Detect importance drift for specified features.

        Args:
            feature_names: Features to check for drift
            baseline_period: Baseline time period
            current_period: Current time period
            methods: Drift detection methods to use

        Returns:
            List of drift alerts
        """
        if methods is None:
            methods = self.drift_detection_methods

        if baseline_period is None:
            # Use last 30 days as baseline
            baseline_end = datetime.now()
            baseline_start = baseline_end - timedelta(days=30)
            baseline_period = (baseline_start, baseline_end)

        if current_period is None:
            # Use last 7 days as current period
            current_end = datetime.now()
            current_start = current_end - timedelta(days=7)
            current_period = (current_start, current_end)

        # Get importance data for both periods
        baseline_data = self.get_importance_history(
            feature_names=feature_names,
            start_time=baseline_period[0],
            end_time=baseline_period[1]
        )

        current_data = self.get_importance_history(
            feature_names=feature_names,
            start_time=current_period[0],
            end_time=current_period[1]
        )

        if baseline_data.empty or current_data.empty:
            logger.warning("Insufficient data for drift detection")
            return []

        alerts = []
        all_features = set(baseline_data['feature_name'].unique()) | set(current_data['feature_name'].unique())

        for feature in all_features:
            baseline_scores = baseline_data[baseline_data['feature_name'] == feature]['importance_score'].values
            current_scores = current_data[current_data['feature_name'] == feature]['importance_score'].values

            if len(baseline_scores) == 0 or len(current_scores) == 0:
                continue

            for method in methods:
                drift_score, severity = self._calculate_drift_score(
                    baseline_scores, current_scores, method
                )

                if severity != DriftSeverity.NONE:
                    alert = DriftAlert(
                        feature_name=feature,
                        timestamp=datetime.now(),
                        severity=severity,
                        drift_score=drift_score,
                        method=method,
                        baseline_period=baseline_period,
                        current_period=current_period,
                        description=self._generate_drift_description(feature, drift_score, method),
                        recommendations=self._generate_drift_recommendations(feature, severity, method)
                    )
                    alerts.append(alert)

                    # Store alert in database
                    self._store_drift_alert(alert)

        self.alerts.extend(alerts)
        logger.info(f"Detected {len(alerts)} drift alerts")

        return alerts

    def _calculate_drift_score(self,
                             baseline_scores: np.ndarray,
                             current_scores: np.ndarray,
                             method: DriftDetectionMethod) -> Tuple[float, DriftSeverity]:
        """Calculate drift score and severity."""
        if method == DriftDetectionMethod.STATISTICAL:
            # Use Kolmogorov-Smirnov test
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(baseline_scores, current_scores)
            drift_score = ks_stat

        elif method == DriftDetectionMethod.THRESHOLD_BASED:
            # Use difference in means normalized by baseline standard deviation
            baseline_mean, baseline_std = np.mean(baseline_scores), np.std(baseline_scores)
            current_mean = np.mean(current_scores)

            if baseline_std > 0:
                drift_score = abs(current_mean - baseline_mean) / baseline_std
            else:
                drift_score = 0.0

        elif method == DriftDetectionMethod.TREND_BASED:
            # Use linear trend coefficient
            if len(current_scores) > 1:
                x = np.arange(len(current_scores))
                slope, _ = np.polyfit(x, current_scores, 1)
                drift_score = abs(slope)
            else:
                drift_score = 0.0

        elif method == DriftDetectionMethod.DISTRIBUTION_BASED:
            # Use Wasserstein distance
            try:
                from scipy import stats
                drift_score = stats.wasserstein_distance(baseline_scores, current_scores)
            except ImportError:
                # Fallback to mean difference
                drift_score = abs(np.mean(current_scores) - np.mean(baseline_scores))

        else:
            drift_score = 0.0

        # Determine severity
        severity = DriftSeverity.NONE
        for sev_level, threshold in sorted(self.drift_thresholds.items(), key=lambda x: x[1]):
            if drift_score >= threshold:
                severity = sev_level

        return drift_score, severity

    def _generate_drift_description(self,
                                   feature_name: str,
                                   drift_score: float,
                                   method: DriftDetectionMethod) -> str:
        """Generate description for drift alert."""
        base_desc = f"Importance drift detected for feature '{feature_name}'"

        if method == DriftDetectionMethod.STATISTICAL:
            return f"{base_desc} using statistical testing (KS statistic: {drift_score:.3f})"
        elif method == DriftDetectionMethod.THRESHOLD_BASED:
            return f"{base_desc} using threshold-based detection (normalized difference: {drift_score:.3f})"
        elif method == DriftDetectionMethod.TREND_BASED:
            return f"{base_desc} using trend analysis (trend coefficient: {drift_score:.3f})"
        elif method == DriftDetectionMethod.DISTRIBUTION_BASED:
            return f"{base_desc} using distribution comparison (Wasserstein distance: {drift_score:.3f})"
        else:
            return f"{base_desc} (drift score: {drift_score:.3f})"

    def _generate_drift_recommendations(self,
                                       feature_name: str,
                                       severity: DriftSeverity,
                                       method: DriftDetectionMethod) -> List[str]:
        """Generate recommendations for drift alert."""
        recommendations = []

        if severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
            recommendations.append("Investigate recent changes in data distribution or model behavior")
            recommendations.append("Consider retraining model with recent data")
            recommendations.append("Review feature engineering pipeline for this feature")

        if method == DriftDetectionMethod.STATISTICAL:
            recommendations.append("Check for statistical significance of distribution changes")
        elif method == DriftDetectionMethod.TREND_BASED:
            recommendations.append("Monitor trend direction and consider feature stabilization")
        elif method == DriftDetectionMethod.DISTRIBUTION_BASED:
            recommendations.append("Analyze distribution shifts and potential causes")

        recommendations.append(f"Continue monitoring feature '{feature_name}' importance")

        return recommendations

    def _store_drift_alert(self, alert: DriftAlert):
        """Store drift alert in database."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO drift_alerts
                (timestamp, feature_name, severity, drift_score, method,
                 baseline_start, baseline_end, current_start, current_end,
                 description, recommendations)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.feature_name,
                alert.severity.value,
                alert.drift_score,
                alert.method.value,
                alert.baseline_period[0].isoformat(),
                alert.baseline_period[1].isoformat(),
                alert.current_period[0].isoformat(),
                alert.current_period[1].isoformat(),
                alert.description,
                json.dumps(alert.recommendations)
            ))
            conn.commit()

    def analyze_trends(self,
                       feature_names: Optional[List[str]] = None,
                       lookback_days: int = 90,
                       min_data_points: int = 10) -> Dict[str, ImportanceTrend]:
        """
        Analyze importance trends for features.

        Args:
            feature_names: Features to analyze
            lookback_days: Number of days to look back
            min_data_points: Minimum data points required for analysis

        Returns:
            Dictionary mapping feature names to trend analysis
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        # Get historical data
        history_df = self.get_importance_history(
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time
        )

        if history_df.empty:
            logger.warning("No historical data available for trend analysis")
            return {}

        trends = {}
        for feature in history_df['feature_name'].unique():
            feature_data = history_df[history_df['feature_name'] == feature]

            if len(feature_data) < min_data_points:
                continue

            trend = self._analyze_single_feature_trend(feature, feature_data)
            trends[feature] = trend

            # Store trend in database
            self._store_trend(trend)

        # Update cache
        self.trends_cache.update(trends)

        logger.info(f"Analyzed trends for {len(trends)} features")
        return trends

    def _analyze_single_feature_trend(self,
                                    feature_name: str,
                                    feature_data: pd.DataFrame) -> ImportanceTrend:
        """Analyze trend for a single feature."""
        scores = feature_data['importance_score'].values
        timestamps = feature_data['timestamp'].values

        # Linear trend analysis
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        r_squared = np.corrcoef(x, scores)[0, 1] ** 2

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"

        trend_strength = abs(slope)

        # Detect change points (simplified approach)
        change_points = self._detect_change_points(scores, timestamps)

        # Simple seasonality detection (check for periodic patterns)
        seasonality_detected = self._detect_seasonality(scores)

        return ImportanceTrend(
            feature_name=feature_name,
            trend_direction=trend_direction,
            trend_strength=float(trend_strength),
            correlation_with_performance=float(r_squared),
            seasonality_detected=seasonality_detected,
            change_points=change_points
        )

    def _detect_change_points(self, scores: np.ndarray, timestamps: np.ndarray) -> List[datetime]:
        """Detect change points in importance scores."""
        change_points = []

        # Simple change point detection using rolling statistics
        window_size = min(10, len(scores) // 4)
        if window_size < 3:
            return change_points

        rolling_mean = pd.Series(scores).rolling(window=window_size).mean()
        rolling_std = pd.Series(scores).rolling(window=window_size).std()

        for i in range(window_size, len(scores) - window_size):
            current_val = scores[i]
            prev_mean = rolling_mean.iloc[i-1]
            prev_std = rolling_std.iloc[i-1]

            if prev_std > 0:
                z_score = abs(current_val - prev_mean) / prev_std
                if z_score > 2.5:  # Significant change
                    change_points.append(timestamps[i])

        return [pd.to_datetime(cp) for cp in change_points]

    def _detect_seasonality(self, scores: np.ndarray) -> bool:
        """Detect seasonality in importance scores."""
        if len(scores) < 20:
            return False

        # Simple autocorrelation test
        try:
            from statsmodels.tsa.stattools import acf
            autocorr = acf(scores, nlags=min(20, len(scores)//2), fft=True)

            # Check for significant autocorrelation at any lag
            max_autocorr = np.max(np.abs(autocorr[1:]))
            return max_autocorr > 0.3
        except ImportError:
            # Fallback: check for periodic patterns using variance ratios
            return False

    def _store_trend(self, trend: ImportanceTrend):
        """Store trend analysis in database."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO importance_trends
                (feature_name, timestamp, trend_direction, trend_strength,
                 correlation_with_performance, seasonality_detected, change_points)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trend.feature_name,
                datetime.now().isoformat(),
                trend.trend_direction,
                trend.trend_strength,
                trend.correlation_with_performance,
                trend.seasonality_detected,
                json.dumps([cp.isoformat() for cp in trend.change_points])
            ))
            conn.commit()

    def get_aggregate_importance(self,
                               feature_names: Optional[List[str]] = None,
                               time_window: Tuple[datetime, datetime] = None,
                               aggregation_method: str = 'mean') -> Dict[str, float]:
        """
        Get aggregate importance scores over a time period.

        Args:
            feature_names: Features to aggregate
            time_window: Time window for aggregation
            aggregation_method: Method for aggregation ('mean', 'median', 'max', 'min')

        Returns:
            Dictionary mapping feature names to aggregate scores
        """
        if time_window is None:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)
            time_window = (start_time, end_time)

        history_df = self.get_importance_history(
            feature_names=feature_names,
            start_time=time_window[0],
            end_time=time_window[1]
        )

        if history_df.empty:
            return {}

        aggregate_scores = {}
        for feature in history_df['feature_name'].unique():
            feature_scores = history_df[history_df['feature_name'] == feature]['importance_score']

            if aggregation_method == 'mean':
                aggregate_scores[feature] = float(feature_scores.mean())
            elif aggregation_method == 'median':
                aggregate_scores[feature] = float(feature_scores.median())
            elif aggregation_method == 'max':
                aggregate_scores[feature] = float(feature_scores.max())
            elif aggregation_method == 'min':
                aggregate_scores[feature] = float(feature_scores.min())
            else:
                aggregate_scores[feature] = float(feature_scores.mean())

        return aggregate_scores

    def get_feature_importance_summary(self,
                                      feature_names: Optional[List[str]] = None,
                                      lookback_days: int = 30) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive importance summary for features.

        Args:
            feature_names: Features to summarize
            lookback_days: Number of days to look back

        Returns:
            Dictionary with feature importance summaries
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)

        history_df = self.get_importance_history(
            feature_names=feature_names,
            start_time=start_time,
            end_time=end_time
        )

        if history_df.empty:
            return {}

        summaries = {}
        for feature in history_df['feature_name'].unique():
            feature_data = history_df[history_df['feature_name'] == feature]
            scores = feature_data['importance_score']

            summary = {
                'feature_name': feature,
                'current_importance': float(scores.iloc[-1]) if len(scores) > 0 else 0.0,
                'average_importance': float(scores.mean()),
                'importance_std': float(scores.std()),
                'importance_trend': self._calculate_simple_trend(scores),
                'importance_volatility': float(scores.std() / scores.mean()) if scores.mean() > 0 else 0.0,
                'data_points': len(scores),
                'first_observation': feature_data['timestamp'].min().isoformat(),
                'last_observation': feature_data['timestamp'].max().isoformat(),
                'methods_used': feature_data['method'].unique().tolist()
            }

            summaries[feature] = summary

        return summaries

    def _calculate_simple_trend(self, scores: pd.Series) -> str:
        """Calculate simple trend direction."""
        if len(scores) < 3:
            return "insufficient_data"

        recent_half = scores.iloc[len(scores)//2:]
        early_half = scores.iloc[:len(scores)//2]

        recent_mean = recent_half.mean()
        early_mean = early_half.mean()

        if abs(recent_mean - early_mean) < 0.01 * scores.std():
            return "stable"
        elif recent_mean > early_mean:
            return "increasing"
        else:
            return "decreasing"

    def cleanup_old_data(self):
        """Clean up old data to maintain database size."""
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)

        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Delete old snapshots
            cursor.execute('''
                DELETE FROM importance_snapshots
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))

            # Delete old alerts (keep last 90 days)
            alert_cutoff = datetime.now() - timedelta(days=90)
            cursor.execute('''
                DELETE FROM drift_alerts
                WHERE timestamp < ?
            ''', (alert_cutoff.isoformat(),))

            # Delete old trends (keep last 30 days)
            trend_cutoff = datetime.now() - timedelta(days=30)
            cursor.execute('''
                DELETE FROM importance_trends
                WHERE timestamp < ?
            ''', (trend_cutoff.isoformat(),))

            conn.commit()

        logger.info(f"Cleaned up data older than {cutoff_date}")

    def export_report(self,
                     export_path: str,
                     feature_names: Optional[List[str]] = None,
                     lookback_days: int = 30) -> Dict[str, Any]:
        """
        Export comprehensive importance report.

        Args:
            export_path: Path to save report
            feature_names: Features to include in report
            lookback_days: Number of days to include

        Returns:
            Report dictionary
        """
        report = {
            'export_timestamp': datetime.now().isoformat(),
            'lookback_days': lookback_days,
            'feature_summaries': self.get_feature_importance_summary(feature_names, lookback_days),
            'recent_drift_alerts': [],
            'current_trends': [],
            'aggregate_importance': self.get_aggregate_importance(feature_names)
        }

        # Get recent drift alerts
        recent_alerts_cutoff = datetime.now() - timedelta(days=7)
        with sqlite3.connect(self.storage_path) as conn:
            alerts_df = pd.read_sql_query('''
                SELECT * FROM drift_alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', conn, params=(recent_alerts_cutoff.isoformat(),))

        if not alerts_df.empty:
            for _, row in alerts_df.iterrows():
                report['recent_drift_alerts'].append({
                    'feature_name': row['feature_name'],
                    'timestamp': row['timestamp'],
                    'severity': row['severity'],
                    'drift_score': row['drift_score'],
                    'method': row['method'],
                    'description': row['description']
                })

        # Get current trends
        trends = self.analyze_trends(feature_names, lookback_days)
        for trend in trends.values():
            report['current_trends'].append({
                'feature_name': trend.feature_name,
                'trend_direction': trend.trend_direction,
                'trend_strength': trend.trend_strength,
                'correlation_with_performance': trend.correlation_with_performance,
                'seasonality_detected': trend.seasonality_detected,
                'change_points_count': len(trend.change_points)
            })

        # Save to file
        with open(export_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report exported to {export_path}")
        return report

    def get_importance_statistics(self) -> Dict[str, Any]:
        """Get overall statistics for the importance tracking system."""
        with sqlite3.connect(self.storage_path) as conn:
            cursor = conn.cursor()

            # Get basic counts
            cursor.execute('SELECT COUNT(*) FROM importance_snapshots')
            total_snapshots = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM drift_alerts')
            total_alerts = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM importance_trends')
            total_trends = cursor.fetchone()[0]

            # Get feature counts
            cursor.execute('SELECT COUNT(DISTINCT feature_name) FROM importance_snapshots')
            total_features = cursor.fetchone()[0]

            # Get date ranges
            cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM importance_snapshots')
            date_range = cursor.fetchone()

            # Get method usage
            cursor.execute('''
                SELECT method, COUNT(*) as count
                FROM importance_snapshots
                GROUP BY method
                ORDER BY count DESC
            ''')
            method_usage = cursor.fetchall()

            # Get recent drift alerts by severity
            recent_cutoff = datetime.now() - timedelta(days=7)
            cursor.execute('''
                SELECT severity, COUNT(*) as count
                FROM drift_alerts
                WHERE timestamp >= ?
                GROUP BY severity
            ''', (recent_cutoff.isoformat(),))
            recent_severity_counts = cursor.fetchall()

        return {
            'total_snapshots': total_snapshots,
            'total_alerts': total_alerts,
            'total_trends': total_trends,
            'total_features': total_features,
            'data_range': {
                'start': date_range[0],
                'end': date_range[1]
            },
            'method_usage': dict(method_usage),
            'recent_drift_severity_counts': dict(recent_severity_counts),
            'database_size_mb': self.storage_path.stat().st_size / (1024 * 1024)
        }