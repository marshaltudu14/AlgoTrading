import os
import pandas as pd
import gc
import json
from fyers_apiv3 import fyersModel

# Import modules from the 'src' package
from src import config
from src import fyers_auth
from src import data_handler

def run_setup():
    """
    Main function to run the data fetching and processing setup.
    """
    print("Starting data setup process...")

    # --- 1. Authentication ---
    access_token = None
    try:
        print("Attempting Fyers authentication...")
        access_token = fyers_auth.get_fyers_access_token()
        if not access_token:
            print("Authentication failed. Exiting.")
            return
        print("Authentication successful.")
        fyers = fyersModel.FyersModel(client_id=config.APP_ID, token=access_token)
        # Optional: Verify connection
        profile = fyers.get_profile()
        print(f"Successfully connected. Profile FY_ID: {profile.get('data', {}).get('fy_id')}")

    except Exception as e:
        print(f"Error during authentication: {e}")
        return # Stop execution if auth fails

    # --- 2. Fetch and Save Raw Historical Data ---
    print("\nFetching and saving raw historical data...")
    raw_folder_path = config.HISTORY_CONFIG["raw_data_folder"]
    os.makedirs(raw_folder_path, exist_ok=True) # Ensure directory exists

    for instrument_name, info in config.INDEX_MAPPING.items():
        for tf in config.HISTORY_CONFIG["timeframes"]:
            file_name = f"{instrument_name}_{tf}.csv"
            file_path = os.path.join(raw_folder_path, file_name)

            if os.path.exists(file_path):
                print(f"Raw file already exists: {file_path}. Skipping fetch.")
                continue

            print(f"Fetching raw data for {instrument_name} ({tf} min)...")
            df_raw = data_handler.fetch_train_candle_data(
                fyers_instance=fyers,
                days_count=config.HISTORY_CONFIG["fetch_days_limit"],
                index_symbol=info["symbol"],
                interval_minutes=tf
            )

            if df_raw is not None and not df_raw.empty:
                df_raw.to_csv(file_path, index=False)
                print(f"Saved raw data to {file_path}")
            else:
                print(f"No raw data returned or error for {instrument_name} ({tf} min).")

            # Cleanup memory
            del df_raw
            gc.collect()
            time.sleep(0.5) # Pause between fetches

    print("Raw data fetching complete.")

    # --- 3. Process Raw Data and Save Processed Files ---
    print("\nProcessing raw data and saving processed files...")
    processed_folder_path = config.HISTORY_CONFIG["processed_data_folder"]
    os.makedirs(processed_folder_path, exist_ok=True) # Ensure directory exists

    # --- Determine expected columns from the current pipeline ---
    expected_columns = []
    try:
        print("Determining expected columns from current pipeline...")
        # Create a minimal dummy DataFrame matching raw data structure
        dummy_data = {
            'datetime': [pd.Timestamp.now().timestamp() - 60, pd.Timestamp.now().timestamp()],
            'open': [100, 101], 'high': [102, 102], 'low': [99, 100], 'close': [101, 101.5], 'volume': [1000, 1100]
        }
        dummy_df_raw = pd.DataFrame(dummy_data)
        dummy_pipeline = data_handler.FullFeaturePipeline(dummy_df_raw)
        dummy_pipeline.run_pipeline()
        dummy_processed_df = dummy_pipeline.get_processed_df()
        # Include the index name ('datetime') in expected columns
        expected_columns = ['datetime'] + dummy_processed_df.columns.tolist()
        print(f"Expected columns: {expected_columns}")
        del dummy_df_raw, dummy_pipeline, dummy_processed_df # Cleanup dummy data
        gc.collect()
    except Exception as e:
        print(f"Error determining expected columns: {e}. Cannot perform column check.")
        # If we can't determine expected columns, we might fall back to always processing or always skipping.
        # For safety, let's fall back to the original behavior (skip if exists).
        expected_columns = [] # Reset to disable check

    temp_data_list = [] # To store info for dynamic config
    raw_files = [f for f in os.listdir(raw_folder_path) if f.endswith(".csv")]

    for file in raw_files:
        file_path = os.path.join(raw_folder_path, file)
        try:
            # Extract instrument name and timeframe from filename
            parts = file.replace(".csv", "").rsplit("_", 1)
            if len(parts) != 2:
                print(f"Skipping invalid filename format: {file}")
                continue
            instrument_name, timeframe_str = parts
            timeframe = int(timeframe_str)

            # Check if instrument and timeframe are valid according to config
            if instrument_name not in config.INDEX_MAPPING:
                print(f"Skipping {file}: Instrument '{instrument_name}' not in INDEX_MAPPING.")
                continue
            if timeframe not in config.HISTORY_CONFIG["timeframes"]:
                 print(f"Skipping {file}: Timeframe '{timeframe}' not in configured timeframes.")
                 continue

            processed_file_name = f"{instrument_name}_{timeframe}_processed.csv"
            processed_file_path = os.path.join(processed_folder_path, processed_file_name)

            # --- Check if file exists and if columns match ---
            process_this_file = True # Default to processing
            if os.path.exists(processed_file_path):
                if expected_columns: # Only check if we could determine expected columns
                    try:
                        # Read only header to get columns
                        existing_columns = pd.read_csv(processed_file_path, nrows=0).columns.tolist()
                        if set(existing_columns) == set(expected_columns):
                            print(f"Processed file exists and columns match: {processed_file_path}. Skipping processing.")
                            process_this_file = False
                        else:
                            print(f"Processed file exists but columns mismatch: {processed_file_path}. Reprocessing...")
                            # Optionally log the difference:
                            # print(f"  Expected: {sorted(expected_columns)}")
                            # print(f"  Existing: {sorted(existing_columns)}")
                    except Exception as read_err:
                        print(f"Error reading existing processed file header {processed_file_path}: {read_err}. Reprocessing...")
                else:
                    # Fallback if expected columns couldn't be determined: skip if exists
                    print(f"Processed file already exists (column check disabled): {processed_file_path}. Skipping processing.")
                    process_this_file = False

            if process_this_file:
                print(f"Processing raw file: {file_path}...")
                try:
                    raw_df = pd.read_csv(file_path)
                except Exception as read_err:
                     print(f"Error reading raw file {file_path}: {read_err}. Skipping.")
                     continue # Skip to next file if raw file can't be read

                # Ensure no duplicates based on datetime before processing
                raw_df.drop_duplicates(subset='datetime', keep='first', inplace=True)

                if raw_df.empty:
                     print(f"Raw file {file_path} is empty or became empty after deduplication. Skipping.")
                     continue

                pipeline = data_handler.FullFeaturePipeline(raw_df)
                pipeline.run_pipeline() # Executes the full processing
                processed_df = pipeline.get_processed_df()

                if processed_df is not None and not processed_df.empty:
                    # Save with datetime index
                    processed_df.to_csv(processed_file_path, index=True, index_label='datetime')
                    print(f"Saved processed data to {processed_file_path}")
                else:
                    print(f"Processing resulted in empty DataFrame for {file}. Not saving.")

                # Cleanup memory
                del raw_df
                del processed_df
                gc.collect()

            # Add info for dynamic config generation (even if skipped processing)
            temp_data_list.append((instrument_name, timeframe, processed_file_path))

        except ValueError:
             print(f"Skipping {file}: Invalid timeframe '{timeframe_str}'.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            gc.collect() # Attempt cleanup on error

    print("Data processing complete.")

    # --- 4. Generate and Save Dynamic Instrument Configuration ---
    print("\nGenerating dynamic instrument configuration...")
    # Sort by timeframe descending, then by instrument name alphabetically
    temp_data_list.sort(key=lambda x: (-x[1], x[0]))

    dynamic_instruments = []
    for instrument_name, timeframe, csv_path in temp_data_list:
         # Check if the processed file actually exists before adding to config
         if os.path.exists(csv_path):
             lot_size = config.INDEX_MAPPING[instrument_name]["quantity"]
             dynamic_name = f"{instrument_name.upper()}_{timeframe}M" # Consistent naming
             dynamic_instruments.append({
                 "name": dynamic_name,
                 "file_path": csv_path, # Store the path to the processed file
                 "lot_size": lot_size,
                 "transaction_cost": 20.0  # Default brokerage, can be customized
             })
         else:
              print(f"Warning: Processed file {csv_path} not found. Excluding from dynamic config.")


    final_dynamic_config = {"instruments": dynamic_instruments}

    # Save the dynamic config to a JSON file in the base directory
    dynamic_config_path = os.path.join(config.BASE_DIR, 'dynamic_config.json')
    config.save_dynamic_config(final_dynamic_config, dynamic_config_path)
    print(f"Dynamic instrument configuration saved to: {dynamic_config_path}")

    print("\n--- Data Setup Summary ---")
    print(f"Raw data directory: {config.HISTORY_CONFIG['raw_data_folder']}")
    print(f"Processed data directory: {config.HISTORY_CONFIG['processed_data_folder']}")
    print(f"Generated dynamic config for {len(final_dynamic_config['instruments'])} instruments.")
    print("--------------------------")

    print("\nData setup process finished.")


if __name__ == "__main__":
    run_setup()
