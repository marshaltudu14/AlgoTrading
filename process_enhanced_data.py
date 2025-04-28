"""
Process historical data with enhanced features.
This script uses the enhanced processor to generate feature-rich datasets.
"""
import os
import argparse
import pandas as pd
from tqdm import tqdm
from data_processing.enhanced_processor import process_df

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process data with enhanced features")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="historical_data",
        help="Directory containing raw CSV files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_data",
        help="Directory to save processed CSV files"
    )
    
    parser.add_argument(
        "--rr_ratio",
        type=float,
        default=2.0,
        help="Risk-reward ratio for signal generation"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize features"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default="enhanced_",
        help="Prefix for output files"
    )
    
    return parser.parse_args()


def process_all_files(input_dir, output_dir, rr_ratio=2.0, normalize=True, prefix="enhanced_"):
    """
    Process all CSV files in the input directory and save to output directory.
    
    Args:
        input_dir: Directory containing raw CSV files
        output_dir: Directory to save processed CSV files
        rr_ratio: Risk-reward ratio for signal generation
        normalize: Whether to normalize features
        prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.csv')]
    
    # Process each file
    for fname in tqdm(csv_files, desc="Processing files"):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"{prefix}{fname}")
        
        # Load data
        df = pd.read_csv(in_path)
        
        # Process data
        try:
            processed = process_df(df, rr_ratio=rr_ratio, normalize=normalize)
            
            # Save processed data
            processed.to_csv(out_path, index=False)
            
            # Save feature columns
            exclude = {'signal', 'datetime', 'Unnamed: 0'}
            feature_cols = [c for c in processed.columns if c not in exclude]
            
            import json
            with open(out_path.replace('.csv', '.features.json'), 'w') as f:
                json.dump(feature_cols, f, indent=2)
                
            print(f"Processed {fname} -> {prefix}{fname} with {len(feature_cols)} features")
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")


def main():
    """Main function."""
    args = parse_args()
    
    print(f"Processing data with enhanced features:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Risk-reward ratio: {args.rr_ratio}")
    print(f"  Normalize features: {args.normalize}")
    print(f"  Output prefix: {args.prefix}")
    
    process_all_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rr_ratio=args.rr_ratio,
        normalize=args.normalize,
        prefix=args.prefix
    )
    
    print("Processing completed successfully")


if __name__ == "__main__":
    main()
