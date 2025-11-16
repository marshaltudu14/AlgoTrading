#!/usr/bin/env python3
"""
Integrated Data Processing Pipeline
==================================

Complete pipeline that processes raw historical data through:
1. Feature Generation (technical indicators, signals)
2. Reasoning Generation (human-like trading reasoning)

This pipeline automates the entire process from raw OHLCV data to
training-ready data with comprehensive reasoning annotations.

Usage:
    python backend/src/data_processing/pipeline.py
    python backend/src/data_processing/pipeline.py --input-dir backend/data/raw --output-dir backend/data/final
    python backend/src/data_processing/pipeline.py --features-only
    python backend/src/data_processing/pipeline.py --reasoning-only

Author: AlgoTrading System
Version: 1.0
"""

import os
import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # Add project root to sys.path

from src.config.settings import get_settings
from src.data_processing.feature_generator import DynamicFileProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataProcessingPipeline:
    """
    Integrated pipeline for processing raw historical data into training-ready
    data with features and reasoning annotations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data processing pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or get_settings()
        self.setup_directories()
        self.feature_generator = None
        logger.info("DataProcessingPipeline initialized")

    def setup_directories(self):
        """Setup required directories for the pipeline."""
        paths_config = self.config.get('paths', {})
        directories = [
            paths_config.get('raw_data_dir', 'backend/data/raw'),
            paths_config.get('final_data_dir', 'backend/data/final'),
            Path(paths_config.get('reports_dir', 'reports')) / 'pipeline',
            Path(paths_config.get('reports_dir', 'reports')) / 'quality',
            paths_config.get('logs_dir', 'logs'),
            paths_config.get('temp_dir', 'temp')
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def run_complete_pipeline(self, input_dir: str = None,
                            output_dir: str = None,
                            parallel: bool = True,
                            max_workers: int = None) -> Dict[str, Any]:
        paths_config = self.config.get('paths', {})
        input_dir = input_dir or paths_config.get('raw_data_dir', 'backend/data/raw')
        output_dir = output_dir or paths_config.get('final_data_dir', 'backend/data/final')
        
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            logger.info("STEP 1: FEATURE GENERATION")
            logger.info("-" * 40)
            feature_results = self.run_feature_generation(input_dir, output_dir, parallel, max_workers)
            
            if not feature_results['success']:
                return {
                    'success': False,
                    'error': f"Feature generation failed: {feature_results['error']}",
                    'step_failed': 'feature_generation'
                }
            
            logger.info(f"Feature generation completed: {feature_results['files_processed']} files")
            
            total_time = time.time() - start_time
            
            summary = {
                'success': True,
                'total_time_seconds': total_time,
                'total_time_formatted': self.format_time(total_time),
                'feature_generation': feature_results,
                'total_files_processed': feature_results.get('files_processed', 0),
                'total_rows_processed': feature_results.get('total_rows', 0),
                'output_directory': output_dir
            }
            
            self.generate_pipeline_report(summary)
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return {'success': False, 'error': str(e), 'step_failed': 'unknown'}
    
    def run_feature_generation(self, input_dir: str, output_dir: str, parallel: bool = True, max_workers: int = None) -> Dict[str, Any]:
        try:
            logger.info("Running feature generator...")
            
            processor = DynamicFileProcessor(data_folder=input_dir)
            processor.processed_folder = Path(output_dir) # Override the processed folder
            results = processor.process_all_files(parallel=parallel, max_workers=max_workers)

            if results:
                logger.info("Feature generator completed successfully")
                output_data_path = Path(output_dir)
                feature_files = list(output_data_path.glob("features_*.parquet"))
                total_rows = 0
                for feature_file in feature_files:
                    try:
                        df = pd.read_parquet(feature_file)
                        total_rows += len(df)
                    except Exception as e:
                        logger.warning(f"Could not count rows in {feature_file}: {e}")

                return {
                    'success': True,
                    'files_processed': len(feature_files),
                    'total_files': len(results),
                    'processed_files': [f.name for f in feature_files],
                    'total_rows': total_rows,
                    'output_directory': output_dir
                }
            else:
                logger.error("Feature generator failed: No files processed")
                return {
                    'success': False,
                    'error': "Feature generator failed: No files processed",
                    'files_processed': 0
                }

        except Exception as e:
            logger.error(f"Feature generation step failed: {str(e)}")
            return {'success': False, 'error': str(e), 'files_processed': 0}
    
    def generate_pipeline_report(self, summary: Dict[str, Any]):
        report_path = Path('reports/pipeline') / f"pipeline_report_{int(time.time())}.txt"
        with open(report_path, 'w') as f:
            f.write("DATA PROCESSING PIPELINE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Pipeline Status: {'SUCCESS' if summary['success'] else 'FAILED'}\n")
            f.write(f"Total Processing Time: {summary.get('total_time_formatted', 'Unknown')}\n")
            f.write(f"Files Processed: {summary.get('total_files_processed', 0)}\n")
            f.write(f"Rows Processed: {summary.get('total_rows_processed', 0):,}\n")
            f.write(f"Output Directory: {summary.get('output_directory', 'Unknown')}\n\n")
            feature_results = summary.get('feature_generation', {})
            f.write("FEATURE GENERATION:\n")
            f.write(f"  Status: {'SUCCESS' if feature_results.get('success') else 'FAILED'}\n")
            f.write(f"  Files Processed: {feature_results.get('files_processed', 0)}\n")
            f.write(f"  Total Rows: {feature_results.get('total_rows', 0):,}\n\n")
            f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logger.info(f"Pipeline report saved: {report_path}")
    
    def format_time(self, seconds: float) -> str:
        if seconds < 60: return f"{seconds:.1f} seconds"
        elif seconds < 3600: return f"{seconds / 60:.1f} minutes"
        else: return f"{seconds / 3600:.1f} hours"


def main():
    pipeline = DataProcessingPipeline()
    paths_config = pipeline.config.get('paths', {})
    parser = argparse.ArgumentParser(description='Integrated Data Processing Pipeline')
    parser.add_argument('--input-dir', default=paths_config.get('raw_data_dir', 'backend/data/raw'),
                       help='Directory containing raw historical data')
    parser.add_argument('--output-dir', default=paths_config.get('final_data_dir', 'backend/data/final'),
                       help='Directory for final processed data')
    parser.add_argument('--features-only', action='store_true', help='Run only feature generation step')
    parser.add_argument('--config-file', help='Custom configuration file path')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of worker processes (default: CPU count)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTEGRATED DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    try:
        if args.features_only:
            print("Running FEATURES ONLY mode...")
            result = pipeline.run_feature_generation(
                args.input_dir, 
                args.output_dir,
                parallel=not args.no_parallel,
                max_workers=args.max_workers
            )
        else:
            print("Running COMPLETE PIPELINE...")
            result = pipeline.run_complete_pipeline(
                args.input_dir, 
                args.output_dir,
                parallel=not args.no_parallel,
                max_workers=args.max_workers
            )

        if result['success']:
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            if not args.features_only:
                print(f"Total processing time: {result.get('total_time_formatted', 'Unknown')}")
                print(f"Files processed: {result.get('total_files_processed', 0)}")
                print(f"Rows processed: {result.get('total_rows_processed', 0):,}")
                print(f"Final data location: {result.get('output_directory', 'Unknown')}")
            else:
                print(f"Files processed: {result.get('files_processed', 0)}")
                if 'total_rows' in result:
                    print(f"Rows processed: {result.get('total_rows', 0):,}")
            print("\nYour data is now ready for model training!")
        else:
            print("\n" + "=" * 80)
            print("PIPELINE FAILED")
            print("=" * 80)
            print(f"Error: {result.get('error', 'Unknown error')}")
            if 'step_failed' in result:
                print(f"Failed at step: {result['step_failed']}")
            print("\nCheck the log files for detailed error information.")
        print("=" * 80)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        print(f"\nFatal Error: {str(e)}")
        print("Check the log files for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
