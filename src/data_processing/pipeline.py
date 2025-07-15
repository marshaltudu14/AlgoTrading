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
    python src/data_processing/pipeline.py
    python src/data_processing/pipeline.py --input-dir data/raw --output-dir data/final
    python src/data_processing/pipeline.py --features-only
    python src/data_processing/pipeline.py --reasoning-only

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
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_processing_pipeline.log')
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
        self.config = config or {}
        
        # Setup directories
        self.setup_directories()
        
        # Initialize components
        self.feature_generator = None
        self.reasoning_orchestrator = None
        
        logger.info("DataProcessingPipeline initialized")
    
    def setup_directories(self):
        """Setup required directories for the pipeline."""
        directories = [
            'data/raw',
            'data/processed',
            'data/processed/reasoning',
            'data/final',
            'reports/pipeline',
            'reports/quality',
            'logs',
            'temp'
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def run_complete_pipeline(self, input_dir: str = "data/raw",
                            output_dir: str = "data/final") -> Dict[str, Any]:
        """
        Run the complete pipeline: Features -> Reasoning -> Final Data
        
        Args:
            input_dir: Directory containing raw historical CSV files
            output_dir: Directory to save final processed data
            
        Returns:
            Pipeline execution summary
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Feature Generation
            logger.info("STEP 1: FEATURE GENERATION")
            logger.info("-" * 40)

            feature_results = self.run_feature_generation(input_dir, "data/processed")
            
            if not feature_results['success']:
                return {
                    'success': False,
                    'error': f"Feature generation failed: {feature_results['error']}",
                    'step_failed': 'feature_generation'
                }
            
            logger.info(f"Feature generation completed: {feature_results['files_processed']} files")
            
            # Step 2: Reasoning Generation
            logger.info("\nSTEP 2: REASONING GENERATION")
            logger.info("-" * 40)

            reasoning_results = self.run_reasoning_generation("data/processed", "data/processed/reasoning")
            
            if not reasoning_results['success']:
                return {
                    'success': False,
                    'error': f"Reasoning generation failed: {reasoning_results['error']}",
                    'step_failed': 'reasoning_generation'
                }
            
            logger.info(f"Reasoning generation completed: {reasoning_results['files_processed']} files")
            
            # Step 3: Final Data Organization
            logger.info("\nSTEP 3: FINAL DATA ORGANIZATION")
            logger.info("-" * 40)
            
            final_results = self.organize_final_data("data/processed/reasoning", output_dir)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Generate summary
            summary = {
                'success': True,
                'total_time_seconds': total_time,
                'total_time_formatted': self.format_time(total_time),
                'feature_generation': feature_results,
                'reasoning_generation': reasoning_results,
                'final_organization': final_results,
                'total_files_processed': feature_results.get('files_processed', 0),
                'total_rows_processed': reasoning_results.get('total_rows', 0),
                'average_quality_score': reasoning_results.get('average_quality', 0),
                'output_directory': output_dir
            }
            
            self.generate_pipeline_report(summary)
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'step_failed': 'unknown'
            }
    
    def run_feature_generation(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Run feature generation step.

        Args:
            input_dir: Directory with raw historical data
            output_dir: Directory to save feature data

        Returns:
            Feature generation results
        """
        try:
            logger.info("Running feature generator...")

            # Import feature generator directly
            import importlib.util
            spec = importlib.util.spec_from_file_location("feature_generator",
                                                        "src/data_processing/feature_generator.py")
            feature_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(feature_module)

            # Get the processor class and run it
            processor = feature_module.DynamicFileProcessor()
            results = processor.process_all_files()

            if results:
                logger.info("Feature generator completed successfully")

                # Count processed files and rows
                processed_data_path = Path("data/processed")
                feature_files = list(processed_data_path.glob("features_*.csv"))

                total_rows = 0
                for feature_file in feature_files:
                    try:
                        df = pd.read_csv(feature_file)
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
            return {
                'success': False,
                'error': str(e),
                'files_processed': 0
            }
    
    def run_reasoning_generation(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Run reasoning generation step.

        Args:
            input_dir: Directory with feature data
            output_dir: Directory to save reasoning data

        Returns:
            Reasoning generation results
        """
        try:
            logger.info("Running reasoning generator...")

            # Import reasoning system
            from src.reasoning_system.core.enhanced_orchestrator import EnhancedReasoningOrchestrator
            from config.config import get_config

            config = get_config()

            # Use enhanced reasoning orchestrator (only option now)
            self.reasoning_orchestrator = EnhancedReasoningOrchestrator(config)
            logger.info("Using Enhanced Reasoning System")

            # Find feature files to process
            input_path = Path(input_dir)
            feature_files = list(input_path.glob("features_*.csv"))

            if not feature_files:
                return {
                    'success': False,
                    'error': f"No feature files found in {input_dir}",
                    'files_processed': 0
                }

            results = []
            total_rows_processed = 0
            total_quality_score = 0

            for feature_file in feature_files:
                result = self.reasoning_orchestrator.process_file(str(feature_file), str(Path(output_dir) / feature_file.name.replace("features_", "reasoning_")))
                results.append(result)
                if result['status'] == 'success':
                    total_rows_processed += result.get('input_rows', 0)
                    total_quality_score += result.get('quality_score', 0) * result.get('input_rows', 0) # Weighted average

            avg_quality = (total_quality_score / total_rows_processed) if total_rows_processed > 0 else 0

            summary = {
                'results': results,
                'total_files': len(feature_files),
                'total_rows': total_rows_processed,
                'average_quality': avg_quality
            }

            # Check if summary has results
            if 'results' not in summary:
                return {
                    'success': False,
                    'error': f"No results from reasoning orchestrator: {summary}",
                    'files_processed': 0
                }

            # Extract results
            successful = [r for r in summary['results'] if r['status'] == 'success']
            failed = [r for r in summary['results'] if r['status'] == 'error']

            total_rows = sum(r.get('input_rows', 0) for r in successful)
            avg_quality = sum(r.get('quality_score', 0) for r in successful) / len(successful) if successful else 0

            logger.info("Reasoning generator completed successfully")

            return {
                'success': len(successful) > 0,
                'files_processed': len(successful),
                'total_files': summary.get('total_files', 0),
                'failed_files': len(failed),
                'total_rows': total_rows,
                'average_quality': avg_quality,
                'output_directory': output_dir,
                'detailed_results': summary['results']
            }

        except Exception as e:
            logger.error(f"Reasoning generation step failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'files_processed': 0
            }
    
    def organize_final_data(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Organize final data with proper naming and structure.
        
        Args:
            input_dir: Directory with reasoning data
            output_dir: Final output directory
            
        Returns:
            Organization results
        """
        try:
            input_path = Path(input_dir)
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Find reasoning files
            reasoning_files = list(input_path.glob("reasoning_*.csv"))
            
            if not reasoning_files:
                return {
                    'success': False,
                    'error': f"No reasoning files found in {input_dir}",
                    'files_organized': 0
                }
            
            organized_files = []
            
            for reasoning_file in reasoning_files:
                # Generate final filename (remove reasoning_ prefix)
                final_name = reasoning_file.name.replace('reasoning_', 'final_')
                final_path = output_path / final_name

                # Read and clean the CSV file to remove any unwanted index columns
                df = pd.read_csv(reasoning_file)

                # Remove any unnamed index columns
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

                # Save cleaned file to final location
                df.to_csv(final_path, index=False)

                organized_files.append(final_name)
                logger.info(f"Organized: {final_name}")
            
            # Create metadata file
            metadata = {
                'pipeline_version': '1.0',
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'files_processed': len(organized_files),
                'file_list': organized_files,
                'columns_included': [
                    'datetime', 'open', 'high', 'low', 'close',
                    'technical_indicators', 'signals', 'reasoning_columns'
                ]
            }
            
            import json
            with open(output_path / 'processing_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'files_organized': len(organized_files),
                'organized_files': organized_files,
                'output_directory': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Final data organization failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'files_organized': 0
            }
    
    def generate_pipeline_report(self, summary: Dict[str, Any]):
        """Generate comprehensive pipeline report."""
        report_path = Path('reports/pipeline') / f"pipeline_report_{int(time.time())}.txt"
        
        with open(report_path, 'w') as f:
            f.write("DATA PROCESSING PIPELINE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Pipeline Status: {'SUCCESS' if summary['success'] else 'FAILED'}\n")
            f.write(f"Total Processing Time: {summary.get('total_time_formatted', 'Unknown')}\n")
            f.write(f"Files Processed: {summary.get('total_files_processed', 0)}\n")
            f.write(f"Rows Processed: {summary.get('total_rows_processed', 0):,}\n")
            f.write(f"Average Quality Score: {summary.get('average_quality_score', 0):.1f}\n")
            f.write(f"Output Directory: {summary.get('output_directory', 'Unknown')}\n\n")
            
            # Feature generation details
            feature_results = summary.get('feature_generation', {})
            f.write("FEATURE GENERATION:\n")
            f.write(f"  Status: {'SUCCESS' if feature_results.get('success') else 'FAILED'}\n")
            f.write(f"  Files Processed: {feature_results.get('files_processed', 0)}\n")
            f.write(f"  Total Rows: {feature_results.get('total_rows', 0):,}\n\n")
            
            # Reasoning generation details
            reasoning_results = summary.get('reasoning_generation', {})
            f.write("REASONING GENERATION:\n")
            f.write(f"  Status: {'SUCCESS' if reasoning_results.get('success') else 'FAILED'}\n")
            f.write(f"  Files Processed: {reasoning_results.get('files_processed', 0)}\n")
            f.write(f"  Average Quality: {reasoning_results.get('average_quality', 0):.1f}\n\n")
            
            f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Pipeline report saved: {report_path}")
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"


def main():
    """Main function for the integrated pipeline."""
    parser = argparse.ArgumentParser(description='Integrated Data Processing Pipeline')
    parser.add_argument('--input-dir', default='data/raw',
                       help='Directory containing raw historical data (default: data/raw)')
    parser.add_argument('--output-dir', default='data/final',
                       help='Directory for final processed data (default: data/final)')
    parser.add_argument('--features-only', action='store_true',
                       help='Run only feature generation step')
    parser.add_argument('--reasoning-only', action='store_true',
                       help='Run only reasoning generation step')
    parser.add_argument('--config-file',
                       help='Custom configuration file path')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("INTEGRATED DATA PROCESSING PIPELINE")
    print("=" * 80)
    print("Processing: Raw Historical Data -> Features -> Reasoning -> Training Data")
    print()
    
    try:
        # Initialize pipeline
        pipeline = DataProcessingPipeline()

        print("ENHANCED REASONING SYSTEM:")
        print("  - Decision column with signal-based logic (no signal references in text)")
        print("  - Historical pattern analysis (20-50 and 100-200 candle timeframes)")
        print("  - Feature relationship analysis across 65+ indicators")
        print("  - Market condition detection with confidence scoring")
        print("  - Advanced natural language generation (>60% unique content)")
        print("  - Fast processing (<1 second per row target)")
        print()

        if args.features_only:
            print("Running FEATURES ONLY mode...")
            result = pipeline.run_feature_generation(args.input_dir, "data/processed")

        elif args.reasoning_only:
            print("Running REASONING ONLY mode...")
            result = pipeline.run_reasoning_generation("data/processed", "data/processed/reasoning")
            
        else:
            print("Running COMPLETE PIPELINE...")
            result = pipeline.run_complete_pipeline(args.input_dir, args.output_dir)
        
        # Display results
        if result['success']:
            print("\n" + "=" * 80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            if not args.features_only and not args.reasoning_only:
                print(f"Total processing time: {result.get('total_time_formatted', 'Unknown')}")
                print(f"Files processed: {result.get('total_files_processed', 0)}")
                print(f"Rows processed: {result.get('total_rows_processed', 0):,}")
                print(f"Average quality score: {result.get('average_quality_score', 0):.1f}")
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
