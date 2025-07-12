#!/usr/bin/env python3
"""
AlgoTrading Pipeline Launcher
============================

Simple launcher script for the AlgoTrading data processing pipeline.

Usage:
    python run_pipeline.py                    # Run complete pipeline
    python run_pipeline.py --features-only    # Run only feature generation
    python run_pipeline.py --reasoning-only   # Run only reasoning generation
    python run_pipeline.py --test             # Run tests
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Main launcher function."""
    
    # Check if this is a test run
    if "--test" in sys.argv:
        print("Running pipeline tests...")
        result = subprocess.run([sys.executable, "scripts/test_pipeline.py"])
        sys.exit(result.returncode)
    
    # Otherwise, run the main pipeline
    print("Launching AlgoTrading Data Processing Pipeline...")
    print("=" * 60)
    
    # Pass all arguments to the pipeline
    pipeline_args = [sys.executable, "src/data_processing/pipeline.py"] + sys.argv[1:]
    
    try:
        result = subprocess.run(pipeline_args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
