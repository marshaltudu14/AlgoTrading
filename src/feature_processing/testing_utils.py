"""
Feature Processing Testing Utilities

This module provides comprehensive testing utilities for feature detection,
validation, and monitoring components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import tempfile
import json
from hypothesis import given, strategies as st, assume
from hypothesis.extra.pandas import data_frames, column
import pytest

from .feature_detector import FeatureDetector, FeatureMetadata, FeatureType
from .feature_registry import FeatureRegistry
from .dynamic_handler import DynamicFeatureHandler, ValidationResult
from .feature_monitor import FeatureMonitor, DriftResult

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a feature processing test"""
    test_name: str
    passed: bool
    duration_seconds: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Result of a complete test suite"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    test_results: List[TestResult]
    duration_seconds: float


class FeatureTestingUtils:
    """
    Comprehensive testing utilities for feature processing
    """

    def __init__(self, feature_registry: Optional[FeatureRegistry] = None):
        """
        Initialize testing utilities

        Args:
            feature_registry: Feature registry instance
        """
        self.feature_registry = feature_registry or FeatureRegistry()
        self.test_results: List[TestResult] = []

    def generate_test_data(self, n_samples: int = 1000,
                          feature_types: Optional[List[FeatureType]] = None,
                          null_percentage: float = 0.05,
                          include_edge_cases: bool = True) -> pd.DataFrame:
        """
        Generate synthetic test data with various feature types

        Args:
            n_samples: Number of samples to generate
            feature_types: Types of features to include
            null_percentage: Percentage of null values to include
            include_edge_cases: Whether to include edge cases

        Returns:
            Generated test DataFrame
        """
        if feature_types is None:
            feature_types = [FeatureType.NUMERICAL, FeatureType.CATEGORICAL, FeatureType.BOOLEAN]

        np.random.seed(42)  # For reproducible tests
        data = {}

        # Generate numerical features
        if FeatureType.NUMERICAL in feature_types:
            data['numerical_normal'] = np.random.normal(100, 15, n_samples)
            data['numerical_uniform'] = np.random.uniform(0, 1, n_samples)
            data['numerical_skewed'] = np.random.exponential(2, n_samples)
            data['numerical_integers'] = np.random.randint(1, 100, n_samples)

            # Add outliers if edge cases enabled
            if include_edge_cases:
                outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.01), replace=False)
                data['numerical_normal'][outlier_indices] = np.random.normal(1000, 100, len(outlier_indices))

        # Generate categorical features
        if FeatureType.CATEGORICAL in feature_types:
            categories = ['A', 'B', 'C', 'D', 'E']
            data['categorical_few'] = np.random.choice(categories, n_samples)

            # High cardinality categorical
            many_categories = [f"cat_{i}" for i in range(100)]
            data['categorical_many'] = np.random.choice(many_categories, n_samples)

            # Imbalanced categorical
            imbalanced_probs = [0.7, 0.2, 0.05, 0.03, 0.02]
            data['categorical_imbalanced'] = np.random.choice(
                ['majority', 'minority1', 'minority2', 'rare1', 'rare2'],
                n_samples, p=imbalanced_probs
            )

        # Generate boolean features
        if FeatureType.BOOLEAN in feature_types:
            data['boolean_balanced'] = np.random.choice([True, False], n_samples)
            data['boolean_imbalanced'] = np.random.choice([True, False], n_samples, p=[0.9, 0.1])

        # Add identifier and target features
        data['id_feature'] = [f"id_{i}" for i in range(n_samples)]
        data['target_feature'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add null values
        if null_percentage > 0:
            for column in df.columns:
                if column not in ['id_feature']:  # Keep identifiers clean
                    null_indices = np.random.choice(
                        n_samples, size=int(n_samples * null_percentage), replace=False
                    )
                    df.loc[null_indices, column] = np.nan

        return df

    def test_feature_detection(self, test_data: pd.DataFrame) -> TestResult:
        """
        Test feature detection functionality

        Args:
            test_data: Test data to use

        Returns:
            Test result
        """
        start_time = datetime.now()
        test_name = "feature_detection"

        try:
            detector = FeatureDetector()
            detected_features = detector.detect_features(test_data, "test_data")

            # Test expectations
            expected_min_features = 5  # Minimum we expect to detect
            actual_features = len(detected_features)

            passed = actual_features >= expected_min_features

            details = {
                'expected_min_features': expected_min_features,
                'actual_features': actual_features,
                'detected_feature_types': {},
                'detection_duration': (datetime.now() - start_time).total_seconds()
            }

            # Count by feature type
            for name, metadata in detected_features.items():
                feature_type = metadata.feature_type.value
                details['detected_feature_types'][feature_type] = \
                    details['detected_feature_types'].get(feature_type, 0) + 1

            result = TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details=details
            )

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details={},
                error_message=str(e)
            )

        self.test_results.append(result)
        return result

    def test_feature_validation(self, test_data: pd.DataFrame) -> TestResult:
        """
        Test feature validation functionality

        Args:
            test_data: Test data to use

        Returns:
            Test result
        """
        start_time = datetime.now()
        test_name = "feature_validation"

        try:
            # First detect features to populate registry
            detector = FeatureDetector()
            detected_features = detector.detect_features(test_data, "validation_test")

            # Register features
            for metadata in detected_features.values():
                self.feature_registry.register_feature(metadata)

            # Create handler and validate
            handler = DynamicFeatureHandler(self.feature_registry)
            validation_result = handler.validate_features(test_data)

            # Test expectations
            passed = validation_result.is_valid or validation_result.status.value == "partially_compatible"

            details = {
                'validation_status': validation_result.status.value,
                'is_valid': validation_result.is_valid,
                'issues_count': len(validation_result.issues),
                'warnings_count': len(validation_result.warnings),
                'missing_features': len(validation_result.missing_features),
                'new_features': len(validation_result.new_features),
                'validation_duration': (datetime.now() - start_time).total_seconds()
            }

            result = TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details=details
            )

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details={},
                error_message=str(e)
            )

        self.test_results.append(result)
        return result

    def test_drift_detection(self, test_data: pd.DataFrame) -> TestResult:
        """
        Test drift detection functionality

        Args:
            test_data: Test data to use

        Returns:
            Test result
        """
        start_time = datetime.now()
        test_name = "drift_detection"

        try:
            # Create monitor
            monitor = FeatureMonitor(self.feature_registry)

            # Establish baseline with first half of data
            baseline_data = test_data.iloc[:len(test_data)//2]
            monitor.establish_baseline(baseline_data)

            # Detect drift with second half (with some modifications)
            current_data = test_data.iloc[len(test_data)//2:].copy()

            # Introduce some drift in numerical columns
            numerical_cols = current_data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols[:2]:  # Modify first 2 numerical columns
                if col in current_data.columns:
                    # Add drift by shifting distribution
                    current_data[col] = current_data[col] * 1.2 + np.random.normal(0, 5, len(current_data))

            # Detect drift
            drift_results = monitor.detect_drift(current_data)

            # Test expectations
            passed = len(drift_results) > 0  # Should detect some drift

            details = {
                'features_tested': len(drift_results),
                'drift_detected_count': sum(1 for r in drift_results.values() if r.drift_detected),
                'average_drift_score': np.mean([r.drift_score for r in drift_results.values()]) if drift_results else 0,
                'drift_types_detected': {},
                'drift_detection_duration': (datetime.now() - start_time).total_seconds()
            }

            # Count drift types
            for result in drift_results.values():
                if result.drift_detected:
                    drift_type = result.drift_type.value
                    details['drift_types_detected'][drift_type] = \
                        details['drift_types_detected'].get(drift_type, 0) + 1

            result = TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details=details
            )

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details={},
                error_message=str(e)
            )

        self.test_results.append(result)
        return result

    def test_edge_cases(self) -> TestResult:
        """
        Test edge cases and boundary conditions

        Returns:
            Test result
        """
        start_time = datetime.now()
        test_name = "edge_cases"

        try:
            test_results = []

            # Test 1: Empty DataFrame
            try:
                empty_df = pd.DataFrame()
                detector = FeatureDetector()
                empty_result = detector.detect_features(empty_df, "empty_test")
                test_results.append(("empty_dataframe", len(empty_result) == 0))
            except Exception as e:
                test_results.append(("empty_dataframe", False, str(e)))

            # Test 2: DataFrame with single column
            try:
                single_df = pd.DataFrame({'single_col': [1, 2, 3, 4, 5]})
                detector = FeatureDetector()
                single_result = detector.detect_features(single_df, "single_test")
                test_results.append(("single_column", len(single_result) == 1))
            except Exception as e:
                test_results.append(("single_column", False, str(e)))

            # Test 3: DataFrame with all nulls
            try:
                null_df = pd.DataFrame({'null_col': [np.nan, np.nan, np.nan]})
                detector = FeatureDetector()
                null_result = detector.detect_features(null_df, "null_test")
                test_results.append(("all_nulls", len(null_result) == 0))
            except Exception as e:
                test_results.append(("all_nulls", False, str(e)))

            # Test 4: DataFrame with mixed types
            try:
                mixed_df = pd.DataFrame({
                    'mixed_col': [1, 'a', 3.14, True, None]
                })
                detector = FeatureDetector()
                mixed_result = detector.detect_features(mixed_df, "mixed_test")
                test_results.append(("mixed_types", len(mixed_result) > 0))
            except Exception as e:
                test_results.append(("mixed_types", False, str(e)))

            # Calculate overall pass rate
            passed_tests = sum(1 for result in test_results if isinstance(result[1], bool) and result[1])
            total_tests = len(test_results)
            passed = passed_tests == total_tests

            details = {
                'total_edge_cases': total_tests,
                'passed_edge_cases': passed_tests,
                'individual_results': test_results,
                'edge_case_duration': (datetime.now() - start_time).total_seconds()
            }

            result = TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details=details
            )

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details={},
                error_message=str(e)
            )

        self.test_results.append(result)
        return result

    def test_performance_benchmarks(self, test_data: pd.DataFrame) -> TestResult:
        """
        Test performance benchmarks

        Args:
            test_data: Test data to use

        Returns:
            Test result
        """
        start_time = datetime.now()
        test_name = "performance_benchmarks"

        try:
            benchmarks = {}

            # Benchmark 1: Feature detection speed
            start_bench = datetime.now()
            detector = FeatureDetector()
            for _ in range(10):  # Run multiple times for average
                detector.detect_features(test_data, "benchmark")
            detection_time = (datetime.now() - start_bench).total_seconds() / 10
            benchmarks['feature_detection_avg_time'] = detection_time

            # Benchmark 2: Validation speed
            handler = DynamicFeatureHandler(self.feature_registry)
            start_bench = datetime.now()
            for _ in range(10):
                handler.validate_features(test_data)
            validation_time = (datetime.now() - start_bench).total_seconds() / 10
            benchmarks['validation_avg_time'] = validation_time

            # Benchmark 3: Drift detection speed
            monitor = FeatureMonitor(self.feature_registry)
            monitor.establish_baseline(test_data)
            start_bench = datetime.now()
            for _ in range(5):
                monitor.detect_drift(test_data)
            drift_time = (datetime.now() - start_bench).total_seconds() / 5
            benchmarks['drift_detection_avg_time'] = drift_time

            # Performance thresholds (in seconds)
            thresholds = {
                'feature_detection_max_time': 1.0,
                'validation_max_time': 0.5,
                'drift_detection_max_time': 2.0
            }

            # Check if benchmarks meet thresholds
            passed = (
                benchmarks['feature_detection_avg_time'] <= thresholds['feature_detection_max_time'] and
                benchmarks['validation_avg_time'] <= thresholds['validation_max_time'] and
                benchmarks['drift_detection_avg_time'] <= thresholds['drift_detection_max_time']
            )

            details = {
                'benchmarks': benchmarks,
                'thresholds': thresholds,
                'benchmark_duration': (datetime.now() - start_time).total_seconds()
            }

            result = TestResult(
                test_name=test_name,
                passed=passed,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details=details
            )

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                passed=False,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                details={},
                error_message=str(e)
            )

        self.test_results.append(result)
        return result

    def run_comprehensive_test_suite(self, test_data: Optional[pd.DataFrame] = None) -> TestSuiteResult:
        """
        Run comprehensive test suite

        Args:
            test_data: Test data to use (if None, generates synthetic data)

        Returns:
            Test suite result
        """
        suite_start_time = datetime.now()

        if test_data is None:
            test_data = self.generate_test_data(n_samples=1000)

        test_functions = [
            self.test_feature_detection,
            self.test_feature_validation,
            self.test_drift_detection,
            self.test_edge_cases,
            self.test_performance_benchmarks
        ]

        test_results = []
        for test_func in test_functions:
            try:
                if test_func.__name__ == 'test_edge_cases':
                    result = test_func()  # Edge cases don't need test data
                else:
                    result = test_func(test_data)
                test_results.append(result)
            except Exception as e:
                logger.error(f"Test {test_func.__name__} failed: {e}")
                test_results.append(TestResult(
                    test_name=test_func.__name__,
                    passed=False,
                    duration_seconds=0,
                    details={},
                    error_message=str(e)
                ))

        passed_tests = sum(1 for result in test_results if result.passed)
        failed_tests = len(test_results) - passed_tests

        suite_result = TestSuiteResult(
            suite_name="comprehensive_feature_processing",
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=test_results,
            duration_seconds=(datetime.now() - suite_start_time).total_seconds()
        )

        return suite_result

    def generate_test_report(self, test_suite_result: TestSuiteResult,
                           output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate comprehensive test report

        Args:
            test_suite_result: Test suite result
            output_path: Path to save report (if None, returns as string)

        Returns:
            Test report as string
        """
        report_lines = [
            f"# Feature Processing Test Report",
            f"Generated: {datetime.now().isoformat()}",
            f"",
            f"## Summary",
            f"- Suite: {test_suite_result.suite_name}",
            f"- Total Tests: {test_suite_result.total_tests}",
            f"- Passed: {test_suite_result.passed_tests}",
            f"- Failed: {test_suite_result.failed_tests}",
            f"- Success Rate: {(test_suite_result.passed_tests/test_suite_result.total_tests)*100:.1f}%",
            f"- Duration: {test_suite_result.duration_seconds:.2f} seconds",
            f"",
            f"## Test Results",
            f""
        ]

        for result in test_suite_result.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report_lines.extend([
                f"### {result.test_name}",
                f"- Status: {status}",
                f"- Duration: {result.duration_seconds:.3f}s",
                f""
            ])

            if result.error_message:
                report_lines.append(f"- Error: {result.error_message}")
                report_lines.append("")

            if result.details:
                report_lines.append("#### Details:")
                for key, value in result.details.items():
                    report_lines.append(f"- {key}: {value}")
                report_lines.append("")

        report = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Test report saved to {output_path}")

        return report

    def save_test_results(self, file_path: Union[str, Path]):
        """Save test results to file"""
        results_data = {
            'test_results': [
                {
                    'test_name': result.test_name,
                    'passed': result.passed,
                    'duration_seconds': result.duration_seconds,
                    'details': result.details,
                    'error_message': result.error_message
                } for result in self.test_results
            ],
            'saved_at': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)

    def load_test_results(self, file_path: Union[str, Path]):
        """Load test results from file"""
        with open(file_path, 'r') as f:
            results_data = json.load(f)

        self.test_results = []
        for result_data in results_data['test_results']:
            self.test_results.append(TestResult(
                test_name=result_data['test_name'],
                passed=result_data['passed'],
                duration_seconds=result_data['duration_seconds'],
                details=result_data['details'],
                error_message=result_data['error_message']
            ))