import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
# Import config variables
from src.config import RL_PROCESSED_DATA_DIR, RL_STATS_FILE

# Configuration
# PROCESSED_DATA_DIR = 'data/historical_processed' # Replaced by config
# OUTPUT_STATS_FILE = 'normalization_stats.json' # Replaced by config
# Features for which to calculate normalization statistics
# We calculate stats for the base features used in the state representation
# Note: These are the base features needed by the environment state
FEATURES_TO_NORMALIZE = [
    'close', # Needed directly and for diffs
    'atr_14',
    'rsi_14',
    'MACDh_12_26_9',
    'adx_14',
    'ema_50', # Needed for diff
    'ema_200' # Needed for diff
]
# Small epsilon to prevent division by zero if std dev is zero
EPSILON = 1e-8

def calculate_stats(processed_data_dir=RL_PROCESSED_DATA_DIR, output_stats_file=RL_STATS_FILE):
    """
    Calculates mean and standard deviation for specified features
    across all processed CSV files and saves them to a JSON file.
    """
    all_stats = {}

    if not os.path.exists(processed_data_dir):
        print(f"Error: Processed data directory not found at {processed_data_dir}")
        return

    print(f"Calculating normalization statistics from files in {processed_data_dir}...")

    try:
        file_list = [f for f in os.listdir(processed_data_dir) if f.endswith('_processed.csv')]
    except FileNotFoundError:
        print(f"Error: Could not list files in {processed_data_dir}. Does the directory exist?")
        return

    if not file_list:
        print(f"Error: No '*_processed.csv' files found in {processed_data_dir}.")
        return

    for filename in tqdm(file_list, desc="Processing files"):
        file_path = os.path.join(processed_data_dir, filename)
        try:
            df = pd.read_csv(file_path)
            file_stats = {}

            # Ensure datetime column is parsed if needed later, though not normalized here
            if 'datetime' in df.columns:
                 try:
                     df['datetime'] = pd.to_datetime(df['datetime'])
                 except Exception:
                     print(f"Warning: Could not parse datetime in {filename}. Skipping datetime conversion.")


            missing_features = [f for f in FEATURES_TO_NORMALIZE if f not in df.columns]
            if missing_features:
                print(f"Warning: File {filename} is missing features: {', '.join(missing_features)}. Skipping these features for this file.")
                
            
            available_features = [f for f in FEATURES_TO_NORMALIZE if f in df.columns]

            for feature in available_features:
                # Convert column to numeric, coercing errors to NaN
                numeric_col = pd.to_numeric(df[feature], errors='coerce')
                # Drop NaN/inf values for calculation
                valid_data = numeric_col.replace([np.inf, -np.inf], np.nan).dropna()

                if not valid_data.empty:
                    mean = valid_data.mean()
                    std = valid_data.std()
                    # Handle cases where std dev is zero or very close to it
                    if pd.isna(std) or std < EPSILON:
                        std = EPSILON
                        print(f"Warning: Standard deviation for '{feature}' in {filename} is zero or NaN. Using epsilon ({EPSILON}).")

                    file_stats[feature] = {'mean': float(mean), 'std': float(std)}
                else:
                     print(f"Warning: No valid numeric data found for feature '{feature}' in {filename}. Skipping stats calculation for this feature.")
                     file_stats[feature] = {'mean': 0.0, 'std': EPSILON} # Provide default values

            all_stats[filename] = file_stats

        except pd.errors.EmptyDataError:
            print(f"Warning: File {filename} is empty. Skipping.")
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Save the calculated stats
    try:
        with open(output_stats_file, 'w') as f:
            json.dump(all_stats, f, indent=4)
        print(f"Normalization statistics saved to {output_stats_file}")
    except Exception as e:
        print(f"Error saving statistics file to {output_stats_file}: {e}")

if __name__ == "__main__":
    # Uses defaults from src.config if no args provided
    calculate_stats()
