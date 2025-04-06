import os
import pandas as pd
import time
import gc
from pathlib import Path

# Assuming src is in the same parent directory or PYTHONPATH is set
try:
    from src.data_handler import FullFeaturePipeline
    from src import config # To access potential config paths if needed
except ImportError:
    print("Error: Ensure 'src' directory is accessible or in PYTHONPATH.")
    # Attempt fallback if run from the root directory
    try:
        from src.data_handler import FullFeaturePipeline
        from src import config
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        exit(1)

def process_raw_data_files(raw_dir: str | Path, processed_dir: str | Path):
    """
    Processes all raw CSV files in the raw_dir using FullFeaturePipeline
    and saves them as Parquet files in the processed_dir.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    # Create the processed directory if it doesn't exist
    processed_path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured processed data directory exists: {processed_path}")

    raw_files = list(raw_path.glob('*.csv'))
    if not raw_files:
        print(f"No raw CSV files found in {raw_path}. Exiting.")
        return

    print(f"Found {len(raw_files)} raw CSV files to process.")

    total_start_time = time.time()

    for i, csv_file in enumerate(raw_files):
        file_start_time = time.time()
        print(f"\nProcessing file {i+1}/{len(raw_files)}: {csv_file.name}...")

        try:
            # 1. Load raw data
            raw_df = pd.read_csv(csv_file)
            print(f"  Loaded raw data: {raw_df.shape}")

            if raw_df.empty:
                print(f"  Skipping empty file: {csv_file.name}")
                continue

            # Check for required columns before pipeline
            required_cols = ['datetime', 'open', 'high', 'low', 'close'] # Volume is optional/dropped
            if not all(col in raw_df.columns for col in required_cols):
                 print(f"  Skipping file {csv_file.name}: Missing one or more required columns {required_cols}.")
                 continue

            # 2. Instantiate and run the pipeline
            pipeline = FullFeaturePipeline(raw_df)
            pipeline.run_pipeline() # Executes preprocess, clean, indicators, time features
            processed_df = pipeline.get_processed_df() # Gets final df, drops NaNs

            if processed_df.empty:
                print(f"  Skipping file {csv_file.name}: Processed DataFrame is empty (likely all NaNs).")
                continue

            # 3. Construct output path and save as Parquet
            parquet_filename = csv_file.stem + ".parquet"
            output_path = processed_path / parquet_filename

            processed_df.to_parquet(output_path, index=True) # Keep the datetime index
            file_end_time = time.time()
            print(f"  Successfully processed and saved to: {output_path}")
            print(f"  Processed shape: {processed_df.shape}")
            print(f"  Time taken for this file: {file_end_time - file_start_time:.2f} seconds.")

            # Explicitly delete dataframes and collect garbage to manage memory
            del raw_df
            del processed_df
            del pipeline
            gc.collect()

        except Exception as e:
            print(f"  ERROR processing file {csv_file.name}: {e}")
            # Optional: Add more robust error handling, like moving failed files
            continue # Continue to the next file

    total_end_time = time.time()
    print(f"\nFinished processing all files.")
    print(f"Total time taken: {total_end_time - total_start_time:.2f} seconds.")


if __name__ == "__main__":
    # Define directories relative to the script location or project root
    # Assuming script is run from the project root directory
    RAW_DATA_DIRECTORY = Path("data/historical_raw")
    PROCESSED_DATA_DIRECTORY = Path("data/historical_processed")

    print("Starting data processing script...")
    process_raw_data_files(RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY)
    print("Data processing script finished.")
