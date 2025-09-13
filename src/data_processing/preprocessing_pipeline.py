#!/usr/bin/env python3
"""
Integrated Preprocessing Pipeline
================================

Complete preprocessing pipeline that combines:
- Feature generation
- Feature-specific normalization
- Missing value handling
- Outlier detection and processing
- Batch processing for large datasets
- Real-time processing capabilities

This pipeline provides end-to-end data preprocessing for the
transformer-based trading system.

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import time
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
from enum import Enum

from .normalization import FeatureNormalizer, OutlierHandlingStrategy, ImputationStrategy, FeatureType
from .feature_specific_normalizer import FeatureSpecificNormalizer, FeatureNormalizationConfig, TechnicalIndicatorType
from .feature_generator import DynamicFileProcessor

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for different use cases."""
    BATCH = "batch"          # Process large historical datasets
    STREAMING = "streaming"  # Process real-time streaming data
    INCREMENTAL = "incremental"  # Process new data incrementally


@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    processing_mode: ProcessingMode = ProcessingMode.BATCH
    enable_gpu: bool = True
    batch_size: int = 10000
    max_workers: Optional[int] = None
    enable_caching: bool = True
    cache_dir: str = "cache/preprocessing"
    save_intermediate: bool = False
    intermediate_dir: str = "intermediate/preprocessing"
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 1000


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for trading data.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.feature_normalizer = FeatureNormalizer()
        self.feature_specific_normalizer = FeatureSpecificNormalizer()
        self.feature_processor = DynamicFileProcessor()

        # Performance tracking
        self.performance_stats = {
            'total_processing_time': 0,
            'feature_generation_time': 0,
            'normalization_time': 0,
            'outlier_handling_time': 0,
            'missing_value_time': 0,
            'data_points_processed': 0,
            'features_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Caching
        self.cache: Dict[str, Any] = {}
        if self.config.enable_caching:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup directories
        if self.config.save_intermediate:
            self.intermediate_dir = Path(self.config.intermediate_dir)
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Preprocessing pipeline initialized")

    def process_historical_data(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process historical data through the complete pipeline.

        Args:
            input_dir: Directory containing raw data files
            output_dir: Directory for processed output

        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        logger.info(f"Starting historical data processing from {input_dir}")

        # Step 1: Generate features
        logger.info("Step 1: Generating features...")
        feature_start = time.time()
        feature_results = self._generate_features(input_dir, output_dir)
        self.performance_stats['feature_generation_time'] = time.time() - feature_start

        if not feature_results['success']:
            return {
                'success': False,
                'error': f"Feature generation failed: {feature_results['error']}",
                'stage': 'feature_generation'
            }

        # Step 2: Load and combine all feature files
        logger.info("Step 2: Loading feature data...")
        combined_data = self._load_feature_data(output_dir)
        if combined_data is None:
            return {
                'success': False,
                'error': "Failed to load feature data",
                'stage': 'data_loading'
            }

        # Step 3: Normalize features
        logger.info("Step 3: Normalizing features...")
        norm_start = time.time()
        normalized_data = self._normalize_features(combined_data)
        self.performance_stats['normalization_time'] = time.time() - norm_start

        # Step 4: Handle missing values and outliers
        logger.info("Step 4: Handling missing values and outliers...")
        cleaning_start = time.time()
        final_data = self._clean_data(normalized_data)
        self.performance_stats['missing_value_time'] = time.time() - cleaning_start

        # Step 5: Save processed data
        logger.info("Step 5: Saving processed data...")
        self._save_processed_data(final_data, output_dir)

        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats.update({
            'total_processing_time': total_time,
            'data_points_processed': len(final_data),
            'features_generated': len([col for col in final_data.columns if col not in ['open', 'high', 'low', 'close']])
        })

        return {
            'success': True,
            'total_time': total_time,
            'feature_generation': feature_results,
            'data_shape': final_data.shape,
            'performance_stats': self.performance_stats,
            'output_files': list(Path(output_dir).glob("processed_*.parquet"))
        }

    def process_real_time_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process new real-time data using pre-fitted normalization parameters.

        Args:
            new_data: New data to process

        Returns:
            Processed data
        """
        if not self.feature_normalizer.normalization_params:
            raise ValueError("Normalizer must be fitted before processing real-time data")

        start_time = time.time()

        # Step 1: Generate features for new data
        feature_data = self.feature_processor.process_dataframe(new_data)

        # Step 2: Apply normalization using existing parameters
        normalized_data = self.feature_normalizer.transform(feature_data)

        # Step 3: Final cleaning
        final_data = self._clean_data(normalized_data)

        processing_time = time.time() - start_time
        logger.info(f"Real-time processing completed in {processing_time:.3f} seconds")

        return final_data

    def _generate_features(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """Generate features from raw data."""
        try:
            self.feature_processor.data_folder = Path(input_dir)
            self.feature_processor.processed_folder = Path(output_dir)
            results = self.feature_processor.process_all_files(
                parallel=True,
                max_workers=self.config.max_workers or mp.cpu_count()
            )

            if results:
                return {
                    'success': True,
                    'files_processed': len(results),
                    'results': results
                }
            else:
                return {
                    'success': False,
                    'error': "No files were processed",
                    'files_processed': 0
                }

        except Exception as e:
            logger.error(f"Feature generation failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'files_processed': 0
            }

    def _load_feature_data(self, output_dir: str) -> Optional[pd.DataFrame]:
        """Load and combine all feature files."""
        try:
            feature_files = list(Path(output_dir).glob("features_*.parquet"))
            if not feature_files:
                logger.error("No feature files found")
                return None

            dataframes = []
            for file_path in feature_files:
                try:
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if not dataframes:
                logger.error("No valid feature data loaded")
                return None

            combined_data = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(combined_data)} rows with {len(combined_data.columns)} features")
            return combined_data

        except Exception as e:
            logger.error(f"Failed to load feature data: {str(e)}")
            return None

    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using appropriate strategies."""
        try:
            # Select feature columns (exclude OHLCV data)
            feature_columns = [col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'datetime']]
            if not feature_columns:
                logger.warning("No feature columns found for normalization")
                return data

            feature_data = data[feature_columns]

            # Auto-configure feature-specific normalization
            self.feature_specific_normalizer.auto_configure_features(feature_columns)

            # Fit the main normalizer
            self.feature_normalizer.fit(feature_data)

            # Apply normalization
            normalized_features = self.feature_normalizer.transform(feature_data)

            # Combine with original data
            result_data = data.copy()
            result_data[feature_columns] = normalized_features

            logger.info(f"Normalized {len(feature_columns)} features")
            return result_data

        except Exception as e:
            logger.error(f"Normalization failed: {str(e)}")
            return data

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling remaining missing values and outliers."""
        try:
            initial_rows = len(data)

            # Remove rows with excessive missing values
            missing_threshold = 0.3  # 30% missing values
            missing_ratios = data.isna().mean(axis=1)
            data = data[missing_ratios <= missing_threshold]

            # Forward fill remaining missing values for time series
            data = data.fillna(method='ffill').fillna(method='bfill')

            # Remove any remaining rows with NaN values
            data = data.dropna()

            removed_rows = initial_rows - len(data)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} rows with missing data. Final: {len(data)} rows")

            return data

        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            return data

    def _save_processed_data(self, data: pd.DataFrame, output_dir: str):
        """Save processed data to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

            # Save in chunks if data is too large
            if len(data) > 100000:
                chunk_size = 100000
                for i in range(0, len(data), chunk_size):
                    chunk = data.iloc[i:i + chunk_size]
                    chunk_path = output_path / f"processed_{i // chunk_size}.parquet"
                    chunk.to_parquet(chunk_path, index=False)
                    logger.info(f"Saved chunk {i // chunk_size} to {chunk_path}")
            else:
                output_file = output_path / "processed_data.parquet"
                data.to_parquet(output_file, index=False)
                logger.info(f"Saved processed data to {output_file}")

            # Save normalization parameters
            params_file = output_path / "normalization_params.json"
            self.feature_normalizer.save_parameters(str(params_file))

            # Save feature-specific configurations
            config_file = output_path / "feature_configs.json"
            feature_configs = {}
            for feature, config in self.feature_specific_normalizer.feature_configs.items():
                feature_configs[feature] = {
                    'feature_type': config.feature_type.value,
                    'indicator_type': config.indicator_type.value if config.indicator_type else None,
                    'custom_range': config.custom_range,
                    'outlier_multiplier': config.outlier_multiplier,
                    'imputation_strategy': config.imputation_strategy.value,
                    'outlier_handling': config.outlier_handling.value
                }

            with open(config_file, 'w') as f:
                json.dump(feature_configs, f, indent=2)

            logger.info("Saved normalization parameters and feature configurations")

        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")

    def load_preprocessing_state(self, state_dir: str):
        """Load preprocessing state (normalization parameters, configs) from directory."""
        try:
            state_path = Path(state_dir)

            # Load normalization parameters
            params_file = state_path / "normalization_params.json"
            if params_file.exists():
                self.feature_normalizer.load_parameters(str(params_file))

            # Load feature configurations
            config_file = state_path / "feature_configs.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    feature_configs = json.load(f)

                self.feature_specific_normalizer.feature_configs = {}
                for feature, config_data in feature_configs.items():
                    config = FeatureNormalizationConfig(
                        feature_name=feature,
                        feature_type=FeatureType(config_data['feature_type']),
                        indicator_type=TechnicalIndicatorType(config_data['indicator_type']) if config_data['indicator_type'] else None,
                        custom_range=tuple(config_data['custom_range']) if config_data['custom_range'] else None,
                        outlier_multiplier=config_data['outlier_multiplier'],
                        imputation_strategy=ImputationStrategy(config_data['imputation_strategy']),
                        outlier_handling=OutlierHandlingStrategy(config_data['outlier_handling'])
                    )
                    self.feature_specific_normalizer.feature_configs[feature] = config

            logger.info("Loaded preprocessing state successfully")

        except Exception as e:
            logger.error(f"Failed to load preprocessing state: {str(e)}")

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of preprocessing configuration and results."""
        return {
            'configuration': {
                'processing_mode': self.config.processing_mode.value,
                'enable_gpu': self.config.enable_gpu,
                'batch_size': self.config.batch_size,
                'max_workers': self.config.max_workers,
                'enable_caching': self.config.enable_caching,
                'cache_dir': str(self.cache_dir) if self.config.enable_caching else None
            },
            'performance_stats': self.performance_stats,
            'feature_normalization': {
                'num_features_configured': len(self.feature_normalizer.normalization_params),
                'feature_types': {feature: params.feature_type.value for feature, params in self.feature_normalizer.normalization_params.items()}
            },
            'feature_specific_configs': len(self.feature_specific_normalizer.feature_configs)
        }

    def validate_preprocessing_pipeline(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the preprocessing pipeline with test data.

        Args:
            test_data: Test data for validation

        Returns:
            Validation results
        """
        try:
            # Process test data
            processed_data = self.process_real_time_data(test_data)

            # Validate results
            validation_results = {
                'success': True,
                'input_shape': test_data.shape,
                'output_shape': processed_data.shape,
                'missing_values': processed_data.isna().sum().sum(),
                'value_ranges': {},
                'processing_time': self.performance_stats.get('total_processing_time', 0)
            }

            # Check value ranges for normalized features
            feature_columns = [col for col in processed_data.columns if col not in ['open', 'high', 'low', 'close']]
            for col in feature_columns:
                validation_results['value_ranges'][col] = {
                    'min': processed_data[col].min(),
                    'max': processed_data[col].max(),
                    'mean': processed_data[col].mean(),
                    'std': processed_data[col].std()
                }

            return validation_results

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'input_shape': test_data.shape if 'test_data' in locals() else None
            }