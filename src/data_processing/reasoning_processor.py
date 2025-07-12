#!/usr/bin/env python3
"""
Reasoning Processor
==================

Main script for processing feature data and adding comprehensive reasoning columns.
Integrates all reasoning system components to generate human-like trading reasoning.

Usage:
    python src/data_processing/reasoning_processor.py

The script will:
1. Scan data/processed folder for feature files
2. Generate reasoning for each file
3. Save enhanced files with reasoning columns
4. Generate quality reports

Author: AlgoTrading System
Version: 1.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from reasoning_system import ReasoningOrchestrator
from config.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('reasoning_processor.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Setup required directories for reasoning processing."""
    directories = [
        'data/processed',
        'data/processed/reasoning',
        'reports/quality'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def get_reasoning_config() -> Dict[str, Any]:
    """
    Get configuration for reasoning generation.
    
    Returns:
        Configuration dictionary for reasoning system
    """
    base_config = get_config()
    
    # Add reasoning-specific configuration
    reasoning_config = {
        'context_window_size': 100,
        'quality_validation': {
            'min_quality_score': 70,
            'check_price_references': True,
            'check_logical_consistency': True,
            'check_professional_language': True
        },
        'text_generation': {
            'min_reasoning_length': 50,
            'max_reasoning_length': 300,
            'use_professional_enhancement': True
        },
        'processing': {
            'batch_size': 1000,
            'progress_reporting_interval': 1000,
            'save_quality_reports': True
        }
    }
    
    # Merge with base config
    base_config.update(reasoning_config)
    return base_config


def process_single_file(orchestrator: ReasoningOrchestrator, input_file: Path, 
                       output_dir: Path) -> Dict[str, Any]:
    """
    Process a single feature file and add reasoning columns.
    
    Args:
        orchestrator: ReasoningOrchestrator instance
        input_file: Path to input feature file
        output_dir: Directory to save output file
        
    Returns:
        Processing result dictionary
    """
    logger.info(f"Processing file: {input_file.name}")
    
    # Generate output filename
    output_filename = input_file.name.replace('features_', 'reasoning_')
    output_file = output_dir / output_filename
    
    # Process the file
    result = orchestrator.process_file(str(input_file), str(output_file))
    
    if result['status'] == 'success':
        logger.info(f"Successfully processed {input_file.name}")
        logger.info(f"  - Input rows: {result['input_rows']}")
        logger.info(f"  - Output rows: {result['output_rows']}")
        logger.info(f"  - Reasoning columns added: {result['reasoning_columns_added']}")
        logger.info(f"  - Quality score: {result['quality_score']:.1f}")
        logger.info(f"  - Output file: {output_filename}")
    else:
        logger.error(f"Failed to process {input_file.name}: {result['error']}")
    
    return result


def generate_summary_report(results: list, output_dir: Path):
    """
    Generate a summary report of all processing results.
    
    Args:
        results: List of processing results
        output_dir: Directory to save report
    """
    report_file = output_dir / 'processing_summary.txt'
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    total_input_rows = sum(r.get('input_rows', 0) for r in successful)
    total_output_rows = sum(r.get('output_rows', 0) for r in successful)
    avg_quality_score = sum(r.get('quality_score', 0) for r in successful) / len(successful) if successful else 0
    
    with open(report_file, 'w') as f:
        f.write("REASONING PROCESSING SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total files processed: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n\n")
        
        if successful:
            f.write("SUCCESS STATISTICS:\n")
            f.write(f"Total input rows: {total_input_rows:,}\n")
            f.write(f"Total output rows: {total_output_rows:,}\n")
            f.write(f"Average quality score: {avg_quality_score:.1f}\n\n")
            
            f.write("SUCCESSFUL FILES:\n")
            for result in successful:
                f.write(f"  - {result.get('input_file', 'Unknown')}: ")
                f.write(f"{result.get('input_rows', 0)} rows, ")
                f.write(f"quality {result.get('quality_score', 0):.1f}\n")
        
        if failed:
            f.write("\nFAILED FILES:\n")
            for result in failed:
                f.write(f"  - {result.get('input_file', 'Unknown')}: {result.get('error', 'Unknown error')}\n")
        
        f.write(f"\nReport generated: {Path.cwd()}\n")
        f.write(f"Reasoning data saved to: {output_dir}\n")
    
    logger.info(f"Summary report saved to: {report_file}")


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process feature data and add reasoning columns')
    parser.add_argument('--input-dir', default='data/processed',
                       help='Directory containing feature files (default: data/processed)')
    parser.add_argument('--output-dir', default='data/processed/reasoning',
                       help='Directory to save reasoning files (default: data/processed/reasoning)')
    parser.add_argument('--file-pattern', default='features_*.csv',
                       help='Pattern to match input files (default: features_*.csv)')
    parser.add_argument('--quality-reports', action='store_true',
                       help='Generate detailed quality reports')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AUTOMATED TRADING REASONING GENERATION SYSTEM")
    print("=" * 80)
    print("Generating human-like trading reasoning for feature data...")
    print("Components: Pattern Recognition, Context Analysis, Psychology Assessment")
    print("           Execution Decisions, Risk Assessment, Alternative Scenarios")
    print()
    
    try:
        # Setup directories
        setup_directories()
        
        # Get configuration
        config = get_reasoning_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize reasoning orchestrator
        logger.info("Initializing reasoning orchestrator...")
        orchestrator = ReasoningOrchestrator(config)
        
        # Setup paths
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Find input files
        input_files = list(input_dir.glob(args.file_pattern))
        
        if not input_files:
            logger.warning(f"No files found matching pattern '{args.file_pattern}' in {input_dir}")
            print(f"\nNo files found in {input_dir} matching pattern '{args.file_pattern}'")
            print("Please ensure feature files are present in the input directory.")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        print(f"Found {len(input_files)} files to process:")
        for file in input_files:
            print(f"  - {file.name}")
        print()
        
        # Process each file
        results = []
        
        for i, input_file in enumerate(input_files, 1):
            print(f"Processing file {i}/{len(input_files)}: {input_file.name}")
            
            result = process_single_file(orchestrator, input_file, output_dir)
            results.append(result)
            
            print()
        
        # Generate summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        print("=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        
        if successful:
            total_rows = sum(r['input_rows'] for r in successful)
            avg_quality = sum(r['quality_score'] for r in successful) / len(successful)
            
            print(f"Successfully processed: {len(successful)} files")
            print(f"Total rows processed: {total_rows:,}")
            print(f"Average quality score: {avg_quality:.1f}")
            print(f"Output directory: {output_dir}")
            
            print("\nFiles processed:")
            for result in successful:
                print(f"  - {result['output_file']}: {result['input_rows']} rows, "
                      f"quality {result['quality_score']:.1f}")
        
        if failed:
            print(f"\nFailed to process: {len(failed)} files")
            for result in failed:
                print(f"  - {result.get('input_file', 'Unknown')}: {result['error']}")
        
        # Generate detailed reports
        if args.quality_reports and successful:
            print(f"\nGenerating quality reports...")
            quality_dir = Path('reports/quality')
            quality_dir.mkdir(parents=True, exist_ok=True)
            generate_summary_report(results, quality_dir)
        
        print(f"\nReasoning generation complete!")
        print(f"Enhanced files with reasoning columns saved in: {output_dir}")
        
        if successful:
            print("\nNext steps:")
            print("1. Review the generated reasoning columns")
            print("2. Validate quality scores and reasoning coherence")
            print("3. Use the enhanced data for model training")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error in main processing: {str(e)}")
        print(f"\nError: {str(e)}")
        print("Check the log file for detailed error information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
