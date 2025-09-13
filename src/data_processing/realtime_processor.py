#!/usr/bin/env python3
"""
Real-Time Data Processing Module
================================

Optimized components for real-time processing of streaming data:
- Online normalization for streaming data
- Incremental parameter updates
- Low-latency inference optimization
- Memory-efficient processing
- Caching strategies
- Version management

Designed for sub-50ms processing requirements in live trading.

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Deque
from collections import deque
import time
import logging
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

from .normalization import FeatureNormalizer, NormalizationParams, FeatureType
from .feature_specific_normalizer import FeatureSpecificNormalizer, FeatureNormalizationConfig

logger = logging.getLogger(__name__)


class UpdateStrategy(Enum):
    """Strategies for updating normalization parameters in real-time."""
    FIXED = "fixed"           # Use fixed parameters from training
    WINDOW = "window"         # Update using sliding window
    EXPONENTIAL = "exponential"  # Update using exponential smoothing
    ADAPTIVE = "adaptive"     # Adaptive update based on data distribution


@dataclass
class RealTimeConfig:
    """Configuration for real-time processing."""
    update_strategy: UpdateStrategy = UpdateStrategy.WINDOW
    window_size: int = 1000
    update_frequency: int = 100
    min_samples_for_update: int = 100
    exponential_smoothing: float = 0.01
    max_memory_samples: int = 10000
    enable_caching: bool = True
    cache_size: int = 1000
    enable_batch_processing: bool = True
    batch_size: int = 50
    max_processing_time_ms: float = 50.0
    enable_performance_monitoring: bool = True


class OnlineNormalizer:
    """
    Online normalizer for streaming data with incremental parameter updates.
    """

    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.base_normalizer: Optional[FeatureNormalizer] = None
        self.feature_specific_normalizer: Optional[FeatureSpecificNormalizer] = None

        # Online statistics tracking
        self.online_stats: Dict[str, Dict[str, Any]] = {}
        self.data_windows: Dict[str, Deque] = {}
        self.update_counters: Dict[str, int] = {}

        # Performance tracking
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0,
            'average_time': 0,
            'updates_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': deque(maxlen=1000)
        }

        # Caching
        self.cache: Dict[str, Any] = {}
        if self.config.enable_caching:
            self.cache = {}

        # Thread safety
        self.lock = threading.Lock()

    def initialize_from_trained(self, normalizer: FeatureNormalizer,
                              feature_specific_normalizer: FeatureSpecificNormalizer):
        """Initialize online normalizer from trained normalizer."""
        self.base_normalizer = normalizer
        self.feature_specific_normalizer = feature_specific_normalizer

        # Initialize online statistics from trained parameters
        for feature, params in normalizer.normalization_params.items():
            self.online_stats[feature] = {
                'min': params.min_val,
                'max': params.max_val,
                'mean': params.mean_val,
                'std': params.std_val,
                'count': 0,
                'sum': 0,
                'sum_sq': 0
            }

            # Initialize data window for window-based updates
            if self.config.update_strategy == UpdateStrategy.WINDOW:
                self.data_windows[feature] = deque(maxlen=self.config.window_size)

            self.update_counters[feature] = 0

        logger.info(f"Online normalizer initialized with {len(self.online_stats)} features")

    def process_stream_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process streaming data with real-time normalization.

        Args:
            data: New data batch to process

        Returns:
            Normalized data
        """
        if self.base_normalizer is None:
            raise ValueError("Online normalizer must be initialized first")

        start_time = time.time()

        try:
            # Process data in batches if enabled
            if self.config.enable_batch_processing and len(data) > self.config.batch_size:
                result = self._process_batch_stream(data)
            else:
                result = self._process_single_stream(data)

            # Update performance stats
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self._update_performance_stats(processing_time, len(data))

            # Check performance requirements
            if processing_time > self.config.max_processing_time_ms:
                logger.warning(f"Processing time {processing_time:.2f}ms exceeded limit {self.config.max_processing_time_ms}ms")

            return result

        except Exception as e:
            logger.error(f"Error processing stream data: {str(e)}")
            # Fallback to basic transformation
            return self.base_normalizer.transform(data)

    def _process_single_stream(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process a single stream of data."""
        normalized_data = data.copy()

        for feature in data.columns:
            if feature not in self.online_stats:
                continue

            with self.lock:
                # Update online statistics
                self._update_online_statistics(feature, data[feature])

                # Check if parameters need updating
                if self._should_update_parameters(feature):
                    self._update_parameters(feature)

                # Apply current normalization
                normalized_data[feature] = self._normalize_feature(data[feature], feature)

        return normalized_data

    def _process_batch_stream(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data in batches for better performance."""
        batch_size = self.config.batch_size
        results = []

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i + batch_size]
            normalized_batch = self._process_single_stream(batch)
            results.append(normalized_batch)

        return pd.concat(results, ignore_index=True)

    def _update_online_statistics(self, feature: str, new_data: pd.Series):
        """Update online statistics with new data."""
        if feature not in self.online_stats:
            return

        stats = self.online_stats[feature]
        valid_data = new_data.dropna()

        if len(valid_data) == 0:
            return

        # Update basic statistics
        new_min = valid_data.min()
        new_max = valid_data.max()
        new_sum = valid_data.sum()
        new_sum_sq = (valid_data ** 2).sum()
        new_count = len(valid_data)

        if stats['count'] == 0:
            # First update
            stats.update({
                'min': new_min,
                'max': new_max,
                'sum': new_sum,
                'sum_sq': new_sum_sq,
                'count': new_count
            })
        else:
            # Incremental update
            stats['min'] = min(stats['min'], new_min)
            stats['max'] = max(stats['max'], new_max)
            stats['sum'] += new_sum
            stats['sum_sq'] += new_sum_sq
            stats['count'] += new_count

        # Update derived statistics
        if stats['count'] > 0:
            stats['mean'] = stats['sum'] / stats['count']
            variance = (stats['sum_sq'] / stats['count']) - (stats['mean'] ** 2)
            stats['std'] = max(0, variance) ** 0.5

        # Add to data window for window-based updates
        if self.config.update_strategy == UpdateStrategy.WINDOW and feature in self.data_windows:
            self.data_windows[feature].extend(valid_data.tolist())

        # Increment update counter
        self.update_counters[feature] += 1

    def _should_update_parameters(self, feature: str) -> bool:
        """Determine if normalization parameters should be updated."""
        if self.config.update_strategy == UpdateStrategy.FIXED:
            return False

        counter = self.update_counters.get(feature, 0)
        return (counter >= self.config.update_frequency and
                self.online_stats[feature]['count'] >= self.config.min_samples_for_update)

    def _update_parameters(self, feature: str):
        """Update normalization parameters based on current strategy."""
        if self.config.update_strategy == UpdateStrategy.WINDOW:
            self._update_window_parameters(feature)
        elif self.config.update_strategy == UpdateStrategy.EXPONENTIAL:
            self._update_exponential_parameters(feature)
        elif self.config.update_strategy == UpdateStrategy.ADAPTIVE:
            self._update_adaptive_parameters(feature)

        # Reset update counter
        self.update_counters[feature] = 0
        self.performance_stats['updates_performed'] += 1

        logger.debug(f"Updated parameters for feature {feature}")

    def _update_window_parameters(self, feature: str):
        """Update parameters using sliding window."""
        if feature not in self.data_windows or len(self.data_windows[feature]) == 0:
            return

        window_data = np.array(self.data_windows[feature])
        stats = self.online_stats[feature]

        stats['min'] = np.min(window_data)
        stats['max'] = np.max(window_data)
        stats['mean'] = np.mean(window_data)
        stats['std'] = np.std(window_data)

    def _update_exponential_parameters(self, feature: str):
        """Update parameters using exponential smoothing."""
        alpha = self.config.exponential_smoothing
        stats = self.online_stats[feature]

        if stats['count'] > 0:
            current_mean = stats['mean']
            current_std = stats['std']

            # Apply exponential smoothing to recent changes
            # This is a simplified approach - in practice, you'd maintain more sophisticated state
            smoothing_factor = min(alpha, 1.0 / stats['count'])
            stats['mean'] = (1 - smoothing_factor) * current_mean + smoothing_factor * current_mean
            stats['std'] = (1 - smoothing_factor) * current_std + smoothing_factor * current_std

    def _update_adaptive_parameters(self, feature: str):
        """Update parameters using adaptive strategy based on data distribution."""
        stats = self.online_stats[feature]

        if stats['count'] < self.config.min_samples_for_update:
            return

        # Calculate current statistics
        current_mean = stats['mean']
        current_std = stats['std']

        # Detect significant changes in distribution
        if self._detect_distribution_change(feature):
            # More aggressive update
            stats['mean'] = current_mean
            stats['std'] = current_std

            # Adjust range if needed
            range_padding = 3 * current_std
            stats['min'] = current_mean - range_padding
            stats['max'] = current_mean + range_padding

    def _detect_distribution_change(self, feature: str) -> bool:
        """Detect significant changes in data distribution."""
        # Simplified change detection
        # In practice, you might use statistical tests like Kolmogorov-Smirnov
        return False  # Placeholder for now

    def _normalize_feature(self, data: pd.Series, feature: str) -> pd.Series:
        """Normalize a feature using current parameters."""
        stats = self.online_stats[feature]

        if stats['std'] == 0 or stats['max'] == stats['min']:
            # Handle edge cases
            return pd.Series([50.0] * len(data), index=data.index)

        # Min-max normalization to 0-100 range
        normalized = ((data - stats['min']) / (stats['max'] - stats['min'])) * 100

        # Clip to range
        normalized = normalized.clip(0, 100)

        return normalized

    def _update_performance_stats(self, processing_time: float, data_points: int):
        """Update performance statistics."""
        self.performance_stats['total_processed'] += data_points
        self.performance_stats['total_time'] += processing_time
        self.performance_stats['processing_times'].append(processing_time)

        if self.performance_stats['total_processed'] > 0:
            self.performance_stats['average_time'] = (
                self.performance_stats['total_time'] /
                self.performance_stats['total_processed']
            )

    def get_current_parameters(self) -> Dict[str, Dict[str, float]]:
        """Get current normalization parameters."""
        return {
            feature: {
                'min': stats['min'],
                'max': stats['max'],
                'mean': stats['mean'],
                'std': stats['std'],
                'count': stats['count']
            }
            for feature, stats in self.online_stats.items()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        processing_times = list(self.performance_stats['processing_times'])
        if not processing_times:
            return self.performance_stats

        return {
            **self.performance_stats,
            'min_processing_time': min(processing_times),
            'max_processing_time': max(processing_times),
            'p95_processing_time': np.percentile(processing_times, 95),
            'p99_processing_time': np.percentile(processing_times, 99),
            'cache_hit_rate': (self.performance_stats['cache_hits'] /
                             (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses'])
                             if (self.performance_stats['cache_hits'] + self.performance_stats['cache_misses']) > 0 else 0)
        }

    def reset_statistics(self):
        """Reset online statistics and performance tracking."""
        self.online_stats.clear()
        self.data_windows.clear()
        self.update_counters.clear()
        self.performance_stats = {
            'total_processed': 0,
            'total_time': 0,
            'average_time': 0,
            'updates_performed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': deque(maxlen=1000)
        }

        # Reinitialize from base normalizer if available
        if self.base_normalizer:
            for feature, params in self.base_normalizer.normalization_params.items():
                self.online_stats[feature] = {
                    'min': params.min_val,
                    'max': params.max_val,
                    'mean': params.mean_val,
                    'std': params.std_val,
                    'count': 0,
                    'sum': 0,
                    'sum_sq': 0
                }
                self.update_counters[feature] = 0


class RealTimeProcessor:
    """
    High-level real-time processor that manages multiple online normalizers
    and provides caching and optimization features.
    """

    def __init__(self, config: RealTimeConfig):
        self.config = config
        self.online_normalizer: Optional[OnlineNormalizer] = None
        self.cache_manager = CacheManager(config.cache_size) if config.enable_caching else None
        self.performance_monitor = PerformanceMonitor() if config.enable_performance_monitoring else None

    def initialize(self, trained_normalizer: FeatureNormalizer,
                  feature_specific_normalizer: FeatureSpecificNormalizer):
        """Initialize the real-time processor."""
        self.online_normalizer = OnlineNormalizer(self.config)
        self.online_normalizer.initialize_from_trained(trained_normalizer, feature_specific_normalizer)

        logger.info("Real-time processor initialized")

    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process data with caching and performance monitoring.

        Args:
            data: Input data to process

        Returns:
            Processing results including normalized data and performance metrics
        """
        if self.online_normalizer is None:
            raise ValueError("Real-time processor must be initialized first")

        start_time = time.time()

        # Check cache if enabled
        cache_key = self._generate_cache_key(data) if self.cache_manager else None
        if cache_key and self.cache_manager and self.cache_manager.get(cache_key):
            self.online_normalizer.performance_stats['cache_hits'] += 1
            cached_result = self.cache_manager.get(cache_key)
            return {
                'data': cached_result,
                'cached': True,
                'processing_time_ms': 0,
                'from_cache': True
            }

        # Process data
        if cache_key:
            self.online_normalizer.performance_stats['cache_misses'] += 1

        normalized_data = self.online_normalizer.process_stream_data(data)

        # Cache result if enabled
        if cache_key and self.cache_manager:
            self.cache_manager.set(cache_key, normalized_data)

        # Record performance if enabled
        processing_time = (time.time() - start_time) * 1000
        if self.performance_monitor:
            self.performance_monitor.record_processing(processing_time, len(data))

        return {
            'data': normalized_data,
            'cached': False,
            'processing_time_ms': processing_time,
            'from_cache': False
        }

    def _generate_cache_key(self, data: pd.DataFrame) -> str:
        """Generate cache key for data."""
        # Simple hash-based key generation
        data_hash = hash(str(data.values.tobytes()))
        return f"realtime_{data_hash}_{len(data)}"

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'online_normalizer': self.online_normalizer.get_performance_summary() if self.online_normalizer else {},
            'cache_manager': self.cache_manager.get_stats() if self.cache_manager else {},
            'performance_monitor': self.performance_monitor.get_stats() if self.performance_monitor else {}
        }
        return report


class CacheManager:
    """Simple cache manager for real-time processing."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        if key in self.cache:
            self.access_times[key] = time.time()
            self.stats['hits'] += 1
            return self.cache[key]
        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any):
        """Set item in cache."""
        # Evict if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = value
        self.access_times[key] = time.time()

    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.stats['evictions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.stats['hits'] /
                   (self.stats['hits'] + self.stats['misses'])
                   if (self.stats['hits'] + self.stats['misses']) > 0 else 0)

        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'evictions': self.stats['evictions']
        }


class PerformanceMonitor:
    """Performance monitoring for real-time processing."""

    def __init__(self):
        self.processing_times = deque(maxlen=10000)
        self.data_sizes = deque(maxlen=10000)
        self.timestamps = deque(maxlen=10000)

    def record_processing(self, processing_time_ms: float, data_size: int):
        """Record a processing event."""
        current_time = time.time()
        self.processing_times.append(processing_time_ms)
        self.data_sizes.append(data_size)
        self.timestamps.append(current_time)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.processing_times:
            return {}

        processing_times = list(self.processing_times)
        data_sizes = list(self.data_sizes)

        return {
            'total_events': len(processing_times),
            'avg_processing_time': np.mean(processing_times),
            'min_processing_time': np.min(processing_times),
            'max_processing_time': np.max(processing_times),
            'p95_processing_time': np.percentile(processing_times, 95),
            'p99_processing_time': np.percentile(processing_times, 99),
            'avg_data_size': np.mean(data_sizes),
            'throughput_per_second': len(processing_times) / max(1, self._get_time_span()),
            'latency_violations': sum(1 for t in processing_times if t > 50.0)  # Count violations of 50ms limit
        }

    def _get_time_span(self) -> float:
        """Get time span of recorded events."""
        if len(self.timestamps) < 2:
            return 1.0
        return self.timestamps[-1] - self.timestamps[0]