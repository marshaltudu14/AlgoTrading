"""
Data processing utility for AlgoTrading.
Processes data with feature caching to avoid unnecessary reprocessing.
"""
import os
import argparse
import logging
from typing import Optional

from core.logging_setup import get_logger
from data.processor import process_all as process_basic
from data_processing.enhanced_processor import process_all as process_enhanced
from config import HIST_DIR, PROCESSED_DIR, RR_RATIO

logger = get_logger(__name__)

def process_data(
    input_dir: str = HIST_DIR,
    output_dir: str = PROCESSED_DIR,
    rr_ratio: float = RR_RATIO,
    enhanced: bool = True,  # Changed default to True
    normalize: bool = True,
    force: bool = False
) -> bool:
    """
    Process data with feature caching.

    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed CSV files
        rr_ratio: Risk-reward ratio for signal generation
        enhanced: Whether to use enhanced features
        normalize: Whether to normalize features (only for enhanced)
        force: Force processing regardless of feature hash

    Returns:
        bool: True if processing was performed, False if skipped
    """
    if enhanced:
        logger.info("Using enhanced processor")
        return process_enhanced(
            input_dir=input_dir,
            output_dir=output_dir,
            rr_ratio=rr_ratio,
            normalize=normalize,
            force=force
        )
    else:
        logger.info("Using basic processor")
        return process_basic(
            input_dir=input_dir,
            output_dir=output_dir,
            rr_ratio=rr_ratio,
            force=force
        )

def main():
    """Parse command line arguments and process data."""
    parser = argparse.ArgumentParser(description="Process data for AlgoTrading")

    parser.add_argument(
        "--input_dir",
        type=str,
        default=HIST_DIR,
        help="Directory containing raw CSV files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=PROCESSED_DIR,
        help="Directory to save processed CSV files"
    )

    parser.add_argument(
        "--rr_ratio",
        type=float,
        default=RR_RATIO,
        help="Risk-reward ratio for signal generation"
    )

    parser.add_argument(
        "--basic",
        action="store_true",
        help="Use basic features instead of enhanced features"
    )

    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="Normalize features (only for enhanced)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing regardless of feature hash"
    )

    args = parser.parse_args()

    # Process data
    processed = process_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rr_ratio=args.rr_ratio,
        enhanced=not args.basic,  # Use enhanced by default unless --basic is specified
        normalize=args.normalize,
        force=args.force
    )

    if processed:
        logger.info("Data processing completed")
    else:
        logger.info("Data processing skipped (no changes detected)")

if __name__ == "__main__":
    main()
