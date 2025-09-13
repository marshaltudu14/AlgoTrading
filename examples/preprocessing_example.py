#!/usr/bin/env python3
"""
Preprocessing Pipeline Example
=============================

This example demonstrates how to use the complete preprocessing pipeline
for trading data, including:

1. Feature generation from raw OHLCV data
2. Feature-specific normalization
3. Missing value and outlier handling
4. Real-time processing capabilities
5. Performance monitoring

Usage:
    python examples/preprocessing_example.py

Author: AlgoTrading System
Version: 1.0
"""

import sys
import numpy as np
import pandas as pd
import time
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing.preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig, ProcessingMode
from src.data_processing.realtime_processor import RealTimeProcessor, RealTimeConfig, UpdateStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample OHLCV data for demonstration."""
    np.random.seed(42)

    # Generate realistic price data
    dates = pd.date_range('2023-01-01', periods=5000, freq='1H')
    base_price = 100.0

    # Generate price movements with some trends and volatility
    price_changes = np.random.normal(0, 0.002, len(dates))  # 0.2% std dev
    prices = base_price * (1 + price_changes).cumprod()

    # Create OHLCV data
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0.002, 0.001, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0.002, 0.001, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, len(dates)).astype(int)
    })

    # Ensure realistic OHLC relationships
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def save_sample_data(data: pd.DataFrame, output_dir: Path):
    """Save sample data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split into multiple files to demonstrate batch processing
    chunk_size = 1000
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i + chunk_size]
        file_path = output_dir / f"sample_data_{i // chunk_size}.csv"
        chunk.to_csv(file_path, index=False)
        logger.info(f"Saved sample data to {file_path}")


def demonstrate_historical_processing():
    """Demonstrate historical data processing."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING HISTORICAL DATA PROCESSING")
    logger.info("=" * 60)

    # Create sample data
    logger.info("Creating sample data...")
    sample_data = create_sample_data()

    # Setup directories
    base_dir = Path("temp/preprocessing_example")
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"

    # Save sample data
    logger.info("Saving sample data to CSV files...")
    save_sample_data(sample_data, input_dir)

    # Create preprocessing pipeline
    config = PreprocessingConfig(
        processing_mode=ProcessingMode.BATCH,
        enable_performance_monitoring=True,
        batch_size=500,
        max_workers=4
    )

    logger.info("Initializing preprocessing pipeline...")
    pipeline = PreprocessingPipeline(config)

    # Process historical data
    logger.info("Processing historical data...")
    start_time = time.time()
    result = pipeline.process_historical_data(str(input_dir), str(output_dir))
    processing_time = time.time() - start_time

    # Display results
    logger.info("\n" + "=" * 40)
    logger.info("HISTORICAL PROCESSING RESULTS")
    logger.info("=" * 40)
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Success: {result['success']}")
    logger.info(f"Data shape: {result['data_shape']}")
    logger.info(f"Files processed: {result['feature_generation']['files_processed']}")

    if result['success']:
        # Display performance stats
        stats = result['performance_stats']
        logger.info(f"Data points processed: {stats['data_points_processed']:,}")
        logger.info(f"Features generated: {stats['features_generated']}")
        logger.info(f"Feature generation time: {stats['feature_generation_time']:.2f}s")
        logger.info(f"Normalization time: {stats['normalization_time']:.2f}s")

        # Display normalization summary
        summary = pipeline.get_preprocessing_summary()
        logger.info(f"\nNormalization configurations: {summary['feature_normalization']['num_features_configured']}")

    return pipeline, result


def demonstrate_realtime_processing(pipeline: PreprocessingPipeline):
    """Demonstrate real-time data processing."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING REAL-TIME DATA PROCESSING")
    logger.info("=" * 60)

    # Create real-time processor
    realtime_config = RealTimeConfig(
        update_strategy=UpdateStrategy.WINDOW,
        window_size=500,
        update_frequency=100,
        max_processing_time_ms=50.0,
        enable_performance_monitoring=True
    )

    logger.info("Initializing real-time processor...")
    realtime_processor = RealTimeProcessor(realtime_config)

    # Initialize with trained normalizer
    realtime_processor.initialize(
        pipeline.feature_normalizer,
        pipeline.feature_specific_normalizer
    )

    # Simulate real-time data stream
    logger.info("Simulating real-time data stream...")
    base_price = 110.0
    stream_data = []

    for i in range(100):
        # Generate new data point
        price_change = np.random.normal(0, 0.002)
        new_price = base_price * (1 + price_change)
        base_price = new_price

        new_data = pd.DataFrame({
            'datetime': [pd.Timestamp.now() + pd.Timedelta(minutes=i)],
            'open': [new_price * (1 + np.random.normal(0, 0.001))],
            'high': [new_price * (1 + np.abs(np.random.normal(0.002, 0.001)))],
            'low': [new_price * (1 - np.abs(np.random.normal(0.002, 0.001)))],
            'close': [new_price],
            'volume': [int(np.random.lognormal(10, 1))]
        })

        # Process real-time data
        start_time = time.time()
        result = realtime_processor.process_data(new_data)
        processing_time = (time.time() - start_time) * 1000

        stream_data.append({
            'timestamp': i,
            'processing_time_ms': processing_time,
            'from_cache': result['from_cache'],
            'features_count': len(result['data'].columns)
        })

        if i % 20 == 0:
            logger.info(f"Processed {i+1} data points, latest: {processing_time:.2f}ms")

    # Analyze real-time performance
    processing_times = [d['processing_time_ms'] for d in stream_data]
    logger.info(f"\nReal-time processing performance:")
    logger.info(f"Average processing time: {np.mean(processing_times):.2f}ms")
    logger.info(f"Min processing time: {np.min(processing_times):.2f}ms")
    logger.info(f"Max processing time: {np.max(processing_times):.2f}ms")
    logger.info(f"P95 processing time: {np.percentile(processing_times, 95):.2f}ms")
    logger.info(f"P99 processing time: {np.percentile(processing_times, 99):.2f}ms")

    # Check performance requirements
    violations = sum(1 for t in processing_times if t > 50.0)
    logger.info(f"Performance violations (>50ms): {violations}/{len(processing_times)} ({violations/len(processing_times)*100:.1f}%)")

    # Display performance report
    performance_report = realtime_processor.get_performance_report()
    logger.info(f"\nCache performance:")
    if 'cache_manager' in performance_report:
        cache_stats = performance_report['cache_manager']
        logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        logger.info(f"Cache size: {cache_stats['size']}/{cache_stats['max_size']}")

    return realtime_processor


def demonstrate_validation(pipeline: PreprocessingPipeline):
    """Demonstrate pipeline validation."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATING PIPELINE VALIDATION")
    logger.info("=" * 60)

    # Create test data for validation
    test_data = pd.DataFrame({
        'datetime': pd.date_range('2023-12-01', periods=200, freq='1H'),
        'open': np.random.uniform(105, 115, 200),
        'high': np.random.uniform(115, 120, 200),
        'low': np.random.uniform(100, 105, 200),
        'close': np.random.uniform(108, 118, 200)
    })

    # Validate pipeline
    logger.info("Validating pipeline with test data...")
    validation_results = pipeline.validate_preprocessing_pipeline(test_data)

    # Display validation results
    logger.info("\nValidation Results:")
    logger.info(f"Success: {validation_results['success']}")
    logger.info(f"Input shape: {validation_results['input_shape']}")
    logger.info(f"Output shape: {validation_results['output_shape']}")
    logger.info(f"Missing values: {validation_results['missing_values']}")
    logger.info(f"Processing time: {validation_results['processing_time']:.3f}s")

    if validation_results['success']:
        # Display feature ranges
        value_ranges = validation_results['value_ranges']
        logger.info(f"\nFeature ranges (normalized 0-100):")
        for feature, ranges in list(value_ranges.items())[:5]:  # Show first 5 features
            logger.info(f"  {feature}: min={ranges['min']:.2f}, max={ranges['max']:.2f}, mean={ranges['mean']:.2f}")


def main():
    """Main function to run the preprocessing example."""
    logger.info("ALGOTRADING PREPROCESSING PIPELINE EXAMPLE")
    logger.info("=" * 60)

    try:
        # Demonstrate historical processing
        pipeline, historical_result = demonstrate_historical_processing()

        if historical_result['success']:
            # Demonstrate real-time processing
            realtime_processor = demonstrate_realtime_processing(pipeline)

            # Demonstrate validation
            demonstrate_validation(pipeline)

            logger.info("\n" + "=" * 60)
            logger.info("EXAMPLE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("Key features demonstrated:")
            logger.info("✓ Feature generation from OHLCV data")
            logger.info("✓ Min-max normalization to 0-100 range")
            logger.info("✓ Feature-specific normalization strategies")
            logger.info("✓ Missing value and outlier handling")
            logger.info("✓ Batch processing for large datasets")
            logger.info("✓ Real-time streaming data processing")
            logger.info("✓ Performance monitoring and optimization")
            logger.info("✓ Caching strategies")
            logger.info("✓ Pipeline validation")
            logger.info("\nExample data saved to: temp/preprocessing_example/")
        else:
            logger.error("Historical processing failed. Check error messages above.")

    except Exception as e:
        logger.error(f"Example failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()