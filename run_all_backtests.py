import pandas as pd
from pathlib import Path
import re
import warnings
import traceback
from datetime import datetime

# Assuming src is in the same parent directory or PYTHONPATH is set
try:
    from src import config
    from src.custom_backtester import run_custom_backtest, calculate_performance_metrics
    from src.signals import get_inside_candle_signals
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Error: Ensure 'src' directory is accessible or in PYTHONPATH.")
    exit(1)

# --- Configuration ---
PROCESSED_DATA_DIR = Path("data/historical_processed")
STRATEGY_NAME = "Inside Candle Breakout (Custom)"
STRATEGY_SIGNAL_FUNC = get_inside_candle_signals
# RESULTS_FILE = Path("memory-bank/strategies.md") # No longer used directly here

# Backtest Parameters (Match defaults or specify)
INITIAL_CASH = 100_000
COMMISSION_PER_TRADE = 20.0 # Fixed amount per leg
STOP_LOSS_ATR_MULTIPLIER = 1.0
RISK_REWARD_RATIO = 2.0
ATR_PERIOD = 14

def format_metrics(metrics: dict) -> dict:
    """Formats metrics dictionary for printing/logging."""
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, (float, np.float64)):
            formatted[key] = f"{value:.2f}"
        elif isinstance(value, pd.Timestamp):
            formatted[key] = value.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(value, pd.Timedelta):
            formatted[key] = str(value)
        else:
            formatted[key] = value
    return formatted

def format_indian(num):
    """Formats a number into Indian Lakh/Crore system."""
    if not isinstance(num, (int, float, np.number)):
        return num # Return as is if not a number
    num = float(num)
    if num < 1_00_000:
        return f"{num:,.2f}" # Below 1 Lakh
    elif num < 1_00_00_000:
        return f"{num / 1_00_000:,.2f} L" # Lakhs
    else:
        return f"{num / 1_00_00_000:,.2f} Cr" # Crores

def run_batch_backtests():
    """Runs backtests for all processed files and logs results."""
    all_results = []
    processed_files = sorted(list(PROCESSED_DATA_DIR.glob("*.parquet")))

    if not processed_files:
        print(f"Error: No processed Parquet files found in {PROCESSED_DATA_DIR}")
        return

    print(f"Found {len(processed_files)} processed files. Starting batch backtest...")
    print("-" * 60)
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Parameters: SL ATR Mult=1.0, RR=2.0, ATR Period={ATR_PERIOD}, Commission=40.00 (Round Trip)")
    print("-" * 60)

    for data_file_path in processed_files:
        print(f"Processing: {data_file_path.name}...")
        instrument_match = re.match(r"([a-zA-Z_]+)_(\d+)", data_file_path.stem)
        if not instrument_match:
            print(f"  Skipping: Could not determine instrument/timeframe from filename.")
            continue

        instrument_name = instrument_match.group(1)
        timeframe = instrument_match.group(2)

        if instrument_name not in config.INDEX_MAPPING:
            print(f"  Skipping: Instrument '{instrument_name}' not found in config.INDEX_MAPPING.")
            continue

        lot_size = config.INDEX_MAPPING[instrument_name].get("quantity")
        if lot_size is None:
            print(f"  Skipping: 'quantity' (lot size) not found for '{instrument_name}'.")
            continue

        try:
            # 1. Load Data
            df = pd.read_parquet(data_file_path)
            if df.empty:
                print("  Skipping: Loaded DataFrame is empty.")
                continue

            # 2. Prepare Data
            if not isinstance(df.index, pd.DatetimeIndex):
                print("  Skipping: DataFrame index is not a DatetimeIndex.")
                continue
            rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
            cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
            df_bt = df.rename(columns=cols_to_rename)
            required_cols = ['Open', 'High', 'Low', 'Close', f'atr_{ATR_PERIOD}']
            if not all(col in df_bt.columns for col in required_cols):
                missing = set(required_cols) - set(df_bt.columns)
                print(f"  Skipping: Missing required columns: {missing}")
                continue

            # 3. Generate Signals
            signals = STRATEGY_SIGNAL_FUNC(df_bt)

            # 4. Run Backtest
            trades_log, performance_metrics = run_custom_backtest(
                df=df_bt,
                signals=signals,
                initial_cash=INITIAL_CASH,
                lot_size=lot_size,
                commission_per_trade=COMMISSION_PER_TRADE,
                stop_loss_atr_multiplier=STOP_LOSS_ATR_MULTIPLIER,
                risk_reward_ratio=RISK_REWARD_RATIO,
                atr_period=ATR_PERIOD
            )

            # Add context to metrics
            performance_metrics['Instrument'] = instrument_name
            performance_metrics['Timeframe (min)'] = timeframe
            # Ensure Lot Size is included (it should be from the modified function)
            if 'Lot Size' not in performance_metrics:
                 performance_metrics['Lot Size'] = lot_size # Add if missing

            all_results.append(performance_metrics)
            print(f"  Finished: Trades={performance_metrics.get('Number of Trades', 'N/A')}, Return={performance_metrics.get('Total Return [%]', 'N/A'):.2f}%")

        except Exception as e:
            print(f"  ERROR processing {data_file_path.name}: {e}")
            # traceback.print_exc() # Uncomment for detailed error stack

    # --- Log Results ---
    if not all_results:
        print("\nNo backtests completed successfully.")
        return

    print("\n--- Batch Backtest Summary ---")
    results_df = pd.DataFrame(all_results)

    # Convert Timeframe to numeric for sorting
    results_df['Timeframe (min)'] = pd.to_numeric(results_df['Timeframe (min)'], errors='coerce')

    # Sort by Instrument (alphabetical) and then Timeframe (descending)
    results_df = results_df.sort_values(by=['Instrument', 'Timeframe (min)'], ascending=[True, False])

    # Select and reorder columns for the final summary - including ALL metrics
    # Get all columns from the first result (assuming all results have the same keys)
    if all_results:
        all_metric_keys = list(all_results[0].keys())
        # Define preferred order, put others at the end
        preferred_order = [
            'Instrument', 'Timeframe (min)', 'Lot Size', 'Start Date', 'End Date', 'Duration',
            'Initial Cash', 'Final Equity', 'Total Return [%]', 'Total PnL', 'Total Points',
            'Number of Trades', 'Number of Wins', 'Number of Losses', 'Win Rate [%]',
            'Average Win PnL', 'Average Loss PnL', 'Average Win Points', 'Average Loss Points',
            'Max Points Captured', 'Max Points Lost',
            'Profit Factor', 'Expectancy PnL', 'Expectancy Points',
            'Max Drawdown [%]', 'Total Commission'
        ]
        # Combine preferred order with any remaining keys (ensure no duplicates)
        remaining_keys = [key for key in all_metric_keys if key not in preferred_order]
        summary_cols = preferred_order + remaining_keys
    else:
        summary_cols = [] # Should not happen if check above works, but prevents error

    # Filter out potential missing columns during error (though less likely now)
    summary_cols = [col for col in summary_cols if col in results_df.columns]
    results_summary = results_df[summary_cols].copy()

    # Format numeric columns (excluding cash for now)
    numeric_cols_to_format = [
        'Total Return [%]', 'Total PnL', 'Total Points', 'Win Rate [%]',
        'Average Win PnL', 'Average Loss PnL', 'Average Win Points', 'Average Loss Points',
        'Max Points Captured', 'Max Points Lost',
        'Profit Factor', 'Expectancy PnL', 'Expectancy Points',
        'Max Drawdown [%]', 'Total Commission'
    ]
    for col in numeric_cols_to_format:
        if col in results_summary.columns:
            # Ensure the column exists before trying to format
             results_summary[col] = pd.to_numeric(results_summary[col], errors='coerce').round(2)

    # Format Cash columns using Indian system AFTER converting other numerics
    if 'Initial Cash' in results_summary.columns:
        results_summary['Initial Cash'] = results_summary['Initial Cash'].apply(format_indian)
    if 'Final Equity' in results_summary.columns:
        results_summary['Final Equity'] = results_summary['Final Equity'].apply(format_indian)


    print(results_summary.to_string())

    # Append to strategy log file
    # Generate strategy-specific filename
    safe_strategy_name = re.sub(r'[^\w\-]+', '_', STRATEGY_NAME) # Replace non-alphanumeric with underscore
    results_file_path = Path("memory-bank") / f"results_{safe_strategy_name}.md"
    print(f"\nAppending results to {results_file_path}...")
    try:
        # Ensure memory-bank directory exists (optional, Path.mkdir can handle it)
        results_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file_path, "a") as f:
            # Add a header if the file is new/empty
            if f.tell() == 0:
                 f.write(f"# Backtest Results: {STRATEGY_NAME}\n\n")
                 f.write("This file logs the results of backtest runs for this specific strategy.\n")

            f.write("\n\n" + "="*80 + "\n")
            f.write(f"Batch Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Strategy: {STRATEGY_NAME}\n")
            f.write(f"Parameters: SL ATR Mult=1.0, RR=2.0, ATR Period={ATR_PERIOD}, Commission=40.00 (Round Trip)\n")
            f.write("-" * 80 + "\n\n")
            f.write(results_summary.to_markdown(index=False))
            f.write("\n\n" + "="*80 + "\n")
        print("Results appended successfully.")
    except Exception as e:
        print(f"Error appending results to {results_file_path}: {e}")


if __name__ == "__main__":
    # Import numpy here if not imported globally in the try block
    # This is needed for the format_metrics function if run directly
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
         print("Error: numpy or pandas not found. Please install requirements.")
         exit(1)

    run_batch_backtests()
    print("\nBatch backtesting script finished.")
