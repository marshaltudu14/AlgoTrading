"""
Feature caching system for AlgoTrading.
Tracks feature definitions and only reprocesses data when they change.
"""
import os
import json
import hashlib
import inspect
from typing import Dict, Any, Optional
import logging

from core.logging_setup import get_logger

logger = get_logger(__name__)

def get_function_hash(func) -> str:
    """
    Generate a hash of a function's source code to track changes.
    
    Args:
        func: Function to hash
        
    Returns:
        Hash string of the function's source code
    """
    source = inspect.getsource(func)
    return hashlib.md5(source.encode()).hexdigest()

def get_feature_hash(processor_module) -> str:
    """
    Generate a hash of the feature processing functions to track changes.
    
    Args:
        processor_module: Module containing the processing functions
        
    Returns:
        Hash string of the processing functions
    """
    # Get the source code of the process_df function
    process_df_source = inspect.getsource(processor_module.process_df)
    
    # Create a hash of the source code
    return hashlib.md5(process_df_source.encode()).hexdigest()

def save_feature_hash(processor_module, output_dir: str = 'processed_data') -> None:
    """
    Save the current feature hash to a file.
    
    Args:
        processor_module: Module containing the processing functions
        output_dir: Directory to save the hash file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    feature_hash = get_feature_hash(processor_module)
    
    hash_file = os.path.join(output_dir, 'feature_hash.json')
    
    hash_data = {
        'feature_hash': feature_hash,
        'processor_name': processor_module.__name__
    }
    
    with open(hash_file, 'w') as f:
        json.dump(hash_data, f, indent=2)
    
    logger.info(f"Saved feature hash: {feature_hash} for {processor_module.__name__}")

def check_feature_hash(processor_module, output_dir: str = 'processed_data') -> bool:
    """
    Check if the feature hash has changed.
    
    Args:
        processor_module: Module containing the processing functions
        output_dir: Directory containing the hash file
        
    Returns:
        True if the hash has changed or doesn't exist, False otherwise
    """
    hash_file = os.path.join(output_dir, 'feature_hash.json')
    
    # If the hash file doesn't exist, we need to process
    if not os.path.exists(hash_file):
        logger.info("Feature hash file not found, processing required")
        return True
    
    # Load the saved hash
    with open(hash_file, 'r') as f:
        hash_data = json.load(f)
    
    saved_hash = hash_data.get('feature_hash')
    saved_processor = hash_data.get('processor_name')
    
    # Calculate the current hash
    current_hash = get_feature_hash(processor_module)
    current_processor = processor_module.__name__
    
    # If the processor has changed, we need to process
    if saved_processor != current_processor:
        logger.info(f"Processor changed from {saved_processor} to {current_processor}, processing required")
        return True
    
    # If the hash has changed, we need to process
    if saved_hash != current_hash:
        logger.info(f"Feature hash changed from {saved_hash} to {current_hash}, processing required")
        return True
    
    logger.info(f"Feature hash unchanged ({current_hash}), no processing required")
    return False

def should_process_data(processor_module, output_dir: str = 'processed_data', force: bool = False) -> bool:
    """
    Determine if data processing is needed based on feature changes.
    
    Args:
        processor_module: Module containing the processing functions
        output_dir: Directory containing processed data and hash file
        force: Force processing regardless of hash
        
    Returns:
        True if processing is needed, False otherwise
    """
    if force:
        logger.info("Forced processing requested")
        return True
    
    # Check if the output directory exists
    if not os.path.exists(output_dir):
        logger.info(f"Output directory {output_dir} does not exist, processing required")
        return True
    
    # Check if there are any processed files
    csv_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.csv')]
    if not csv_files:
        logger.info("No processed files found, processing required")
        return True
    
    # Check if the feature hash has changed
    return check_feature_hash(processor_module, output_dir)
