#!/usr/bin/env python3
"""
Test script to verify parallel processing fix
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.feature_generator import DynamicFileProcessor

def test_parallel_processing():
    """Test that parallel processing works correctly"""
    print("Testing parallel processing for feature generation...")
    
    # Initialize processor
    processor = DynamicFileProcessor()
    
    # Test parallel processing
    print("Testing parallel processing...")
    parallel_results = processor.process_all_files(parallel=True)
    print(f"Parallel processing results: {len(parallel_results)} files processed")
    
    print("Test completed!")

if __name__ == "__main__":
    test_parallel_processing()