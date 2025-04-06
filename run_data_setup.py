import os
import os
import pandas as pd
import gc
# import json # Removed as dynamic config generation is removed
import time # Added for sleep
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
                continue # Indent this under the if

            # Dedent the following block to align with the 'if' statement
            print(f"Fetching raw data for {instrument_name} ({tf} min)...")
            # Corrected function call and arguments to match data_handler.py
            df_raw = data_handler.fetch_candle_data(
                fyers_instance=fyers,
                symbol=info["symbol"],
                resolution=str(tf), # Ensure resolution is string
                days_to_fetch=config.HISTORY_CONFIG["fetch_days_limit"]
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

    # --- Steps 3 & 4 (Processing and Dynamic Config) Removed ---
    # The script now only fetches and saves raw data.

    print("\n--- Data Setup Summary ---")
    print(f"Raw data directory: {config.HISTORY_CONFIG['raw_data_folder']}")
    # print(f"Processed data directory: {config.HISTORY_CONFIG['processed_data_folder']}") # Removed
    # print(f"Generated dynamic config for {len(final_dynamic_config['instruments'])} instruments.") # Removed
    print("--------------------------")

    print("\nData setup process finished (Raw data fetch only).")


if __name__ == "__main__":
    run_setup()
