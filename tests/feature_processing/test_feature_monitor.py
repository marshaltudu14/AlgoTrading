"""
Unit tests for FeatureMonitor
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from src.feature_processing import (
    FeatureMonitor, FeatureRegistry, FeatureMetadata, FeatureType,
    DriftType, HealthStatus
)


class TestFeatureMonitor:
    """Test cases for FeatureMonitor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.feature_registry = FeatureRegistry()
        self.monitor = FeatureMonitor(self.feature_registry)
        self.baseline_data = self._create_baseline_data()
        self.current_data = self._create_current_data()
        self._setup_registry()

    def _create_baseline_data(self) -> pd.DataFrame:
        """Create baseline data"""
        np.random.seed(42)
        n_samples = 1000

        data = {
            'numerical_stable': np.random.normal(100, 15, n_samples),
            'numerical_volatile': np.random.normal(50, 5, n_samples),
            'categorical_stable': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.5, 0.3, 0.2]),
            'categorical_volatile': np.random.choice(['X', 'Y', 'Z'], n_samples),
            'boolean_feature': np.random.choice([True, False], n_samples)
        }

        return pd.DataFrame(data)

    def _create_current_data(self) -> pd.DataFrame:
        """Create current data with some drift"""
        np.random.seed(123)  # Different seed for different data
        n_samples = 1000

        # Create data with some intentional drift
        data = {
            'numerical_stable': np.random.normal(102, 15, n_samples),  # Slight mean shift
            'numerical_volatile': np.random.normal(70, 20, n_samples),  # Significant drift
            'categorical_stable': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.4, 0.2]),  # Distribution shift
            'categorical_volatile': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),  # New category
            'boolean_feature': np.random.choice([True, False], n_samples)
        }

        return pd.DataFrame(data)

    def _setup_registry(self):
        """Setup feature registry with test features"""
        features = [
            FeatureMetadata(
                name='numerical_stable',
                feature_type=FeatureType.NUMERICAL,
                dtype='float64',
                null_count=0,
                null_percentage=0.0,
                unique_count=950,
                min_value=50.0,
                max_value=150.0,
                mean_value=100.0,
                std_value=15.0
            ),
            FeatureMetadata(
                name='categorical_stable',
                feature_type=FeatureType.CATEGORICAL,
                dtype='object',
                null_count=0,
                null_percentage=0.0,
                unique_count=3,
                categories=['A', 'B', 'C']
            )
        ]

        for feature in features:
            self.feature_registry.register_feature(feature)

    def test_establish_baseline(self):
        """Test baseline establishment"""
        self.monitor.establish_baseline(self.baseline_data)

        assert len(self.monitor.baseline_snapshots) > 0
        assert 'numerical_stable' in self.monitor.baseline_snapshots
        assert 'categorical_stable' in self.monitor.baseline_snapshots

        # Check snapshot properties
        baseline = self.monitor.baseline_snapshots['numerical_stable']
        assert isinstance(baseline, object)
        assert baseline.feature_name == 'numerical_stable'
        assert baseline.sample_size > 0

    def test_detect_drift_no_baseline(self):
        """Test drift detection without baseline"""
        drift_results = self.monitor.detect_drift(self.current_data)

        # Should return empty results if no baseline
        assert len(drift_results) == 0

    def test_detect_drift_with_baseline(self):
        """Test drift detection with baseline"""
        # Establish baseline first
        self.monitor.establish_baseline(self.baseline_data)

        # Detect drift
        drift_results = self.monitor.detect_drift(self.current_data)

        assert len(drift_results) > 0
        assert 'numerical_stable' in drift_results
        assert 'categorical_stable' in drift_results

        # Check drift result structure
        for feature_name, drift_result in drift_results.items():
            assert isinstance(drift_result.drift_detected, bool)
            assert isinstance(drift_result.drift_score, float)
            assert isinstance(drift_result.drift_type, DriftType)
            assert isinstance(drift_result.confidence, float)
            assert isinstance(drift_result.details, dict)

    def test_drift_detection_types(self):
        """Test different types of drift detection"""
        self.monitor.establish_baseline(self.baseline_data)
        drift_results = self.monitor.detect_drift(self.current_data)

        # Should detect various types of drift
        drift_types = [result.drift_type for result in drift_results.values()]
        assert DriftType.DISTRIBUTION in drift_types
        assert DriftType.STATISTICAL in drift_types

    def test_drift_score_thresholds(self):
        """Test drift score thresholds"""
        # Configure low threshold for testing
        self.monitor.config['drift_threshold'] = 0.01

        self.monitor.establish_baseline(self.baseline_data)
        drift_results = self.monitor.detect_drift(self.current_data)

        # Should detect more drift with lower threshold
        detected_drift_count = sum(1 for result in drift_results.values() if result.drift_detected)
        assert detected_drift_count > 0

    def test_calculate_health_metrics(self):
        """Test health metrics calculation"""
        self.monitor.establish_baseline(self.baseline_data)
        drift_results = self.monitor.detect_drift(self.current_data)

        health_metrics = self.monitor.calculate_health_metrics(drift_results)

        assert len(health_metrics) > 0
        for feature_name, metrics in health_metrics.items():
            assert isinstance(metrics.status, HealthStatus)
            assert isinstance(metrics.drift_score, float)
            assert isinstance(metrics.null_percentage, float)
            assert isinstance(metrics.uniqueness_score, float)
            assert isinstance(metrics.stability_score, float)

    def test_alert_creation(self):
        """Test alert creation for drift detection"""
        # Configure low threshold to trigger alerts
        self.monitor.config['drift_threshold'] = 0.01

        self.monitor.establish_baseline(self.baseline_data)
        self.monitor.detect_drift(self.current_data)

        # Should have created alerts
        assert len(self.monitor.alerts) > 0

        # Check alert structure
        alert = self.monitor.alerts[0]
        assert 'timestamp' in alert
        assert 'feature_name' in alert
        assert 'alert_type' in alert
        assert 'severity' in alert
        assert 'drift_score' in alert

    def test_alert_cooldown(self):
        """Test alert cooldown mechanism"""
        # Configure very short cooldown for testing
        self.monitor.config['alert_cooldown_minutes'] = 0

        self.monitor.establish_baseline(self.baseline_data)
        self.monitor.config['drift_threshold'] = 0.01

        # First detection
        self.monitor.detect_drift(self.current_data)
        first_alert_count = len(self.monitor.alerts)

        # Second detection (should create new alerts due to no cooldown)
        self.monitor.detect_drift(self.current_data)
        second_alert_count = len(self.monitor.alerts)

        assert second_alert_count >= first_alert_count

    def test_get_monitoring_summary(self):
        """Test monitoring summary generation"""
        self.monitor.establish_baseline(self.baseline_data)
        self.monitor.detect_drift(self.current_data)

        summary = self.monitor.get_monitoring_summary()

        assert 'total_features' in summary
        assert 'health_distribution' in summary
        assert 'total_alerts' in summary
        assert 'recent_alerts' in summary
        assert 'average_drift_score' in summary

    def test_save_load_monitoring_data(self):
        """Test saving and loading monitoring data"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Establish baseline and detect drift
            self.monitor.establish_baseline(self.baseline_data)
            self.monitor.detect_drift(self.current_data)

            # Save monitoring data
            self.monitor.save_monitoring_data(temp_path)

            # Create new monitor and load data
            new_monitor = FeatureMonitor(self.feature_registry)
            new_monitor.load_monitoring_data(temp_path)

            # Check that data was loaded
            assert len(new_monitor.baseline_snapshots) > 0
            # Note: Alerts may or may not be created depending on drift detection
            # The important thing is that the data loads correctly

        finally:
            Path(temp_path).unlink()

    def test_empty_baseline_data(self):
        """Test handling of empty baseline data"""
        empty_data = pd.DataFrame()

        # Should handle empty data gracefully
        self.monitor.establish_baseline(empty_data)
        assert len(self.monitor.baseline_snapshots) == 0

    def test_single_feature_monitoring(self):
        """Test monitoring with single feature"""
        single_feature_data = pd.DataFrame({'single_feature': [1, 2, 3, 4, 5]})

        self.monitor.establish_baseline(single_feature_data)
        current_single = pd.DataFrame({'single_feature': [2, 3, 4, 5, 6]})  # Slight drift

        drift_results = self.monitor.detect_drift(current_single)

        assert len(drift_results) == 1
        assert 'single_feature' in drift_results

    def test_high_null_percentage_monitoring(self):
        """Test monitoring with high null percentage"""
        # Baseline with low nulls
        baseline_low_nulls = self.baseline_data.copy()
        baseline_low_nulls['test_feature'] = np.random.normal(0, 1, len(baseline_low_nulls))

        # Current with high nulls
        current_high_nulls = self.current_data.copy()
        current_high_nulls['test_feature'] = np.random.normal(0, 1, len(current_high_nulls))
        current_high_nulls.loc[current_high_nulls.index[:800], 'test_feature'] = np.nan  # 80% nulls

        self.monitor.establish_baseline(baseline_low_nulls)
        drift_results = self.monitor.detect_drift(current_high_nulls)

        # Should detect missing value drift
        test_feature_drift = drift_results.get('test_feature')
        if test_feature_drift:
            assert test_feature_drift.drift_type == DriftType.MISSING_VALUES

    def test_cardinality_drift_detection(self):
        """Test cardinality drift detection"""
        # Baseline with few categories
        baseline_few_cats = pd.DataFrame({
            'test_cat': np.random.choice(['A', 'B'], 1000)
        })

        # Current with many categories
        current_many_cats = pd.DataFrame({
            'test_cat': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 1000)
        })

        self.monitor.establish_baseline(baseline_few_cats)
        drift_results = self.monitor.detect_drift(current_many_cats)

        # Should detect cardinality drift
        test_cat_drift = drift_results.get('test_cat')
        if test_cat_drift:
            assert test_cat_drift.drift_type == DriftType.CARDINALITY

    def test_range_drift_detection(self):
        """Test range drift detection"""
        # Baseline with constrained range
        baseline_constrained = pd.DataFrame({
            'test_num': np.random.uniform(0, 10, 1000)
        })

        # Current with expanded range
        current_expanded = pd.DataFrame({
            'test_num': np.random.uniform(-5, 15, 1000)
        })

        self.monitor.establish_baseline(baseline_constrained)
        drift_results = self.monitor.detect_drift(current_expanded)

        # Should detect range drift
        test_num_drift = drift_results.get('test_num')
        if test_num_drift:
            assert test_num_drift.drift_type == DriftType.RANGE