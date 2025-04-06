import pandas as pd
from pathlib import Path
import re
import warnings
import pprint # For pretty printing results
import numpy as np # Add numpy import

# Assuming src is in the same parent directory or PYTHONPATH is set
try:
    from src import config # Import config to get lot size
    from src.custom_backtester import run_custom_backtest # Import the custom backtester
    from src.signals import get_inside_candle_signals # Import the signal function
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Error: Ensure 'src' directory is accessible or in PYTHONPATH.")
    exit(1)

# Suppress specific warnings if needed
# warnings.filterwarnings("ignore", category=FutureWarning)

def run_inside_candle_custom_backtest(data_file_path: str | Path,
                                      cash: int = 100_000,
                                      commission_per_trade: float = 5.0, # Example fixed commission
                                      stop_loss_atr_multiplier: float = 1.5,
                                      risk_reward_ratio: float = 2.0,
                                      atr_period: int = 14):
    """
    Runs the custom backtest for the Inside Candle strategy.

    Args:
        data_file_path: Path to the processed Parquet data file (e.g., "Nifty_5.parquet").
        cash: Initial portfolio cash.
        commission_per_trade: Fixed commission cost per trade (applied on entry and exit).
        stop_loss_atr_multiplier: ATR multiplier for SL.
        risk_reward_ratio: RR ratio for TP.
        atr_period: Period used for ATR calculation.
    """
    # --- Determine Instrument and Lot Size ---
    data_path = Path(data_file_path)
    instrument_match = re.match(r"([a-zA-Z_]+)_\d+", data_path.stem)
    if not instrument_match:
        print(f"Error: Could not determine instrument name from filename: {data_path.name}")
        return None

    instrument_name = instrument_match.group(1)
    if instrument_name not in config.INDEX_MAPPING:
        print(f"Error: Instrument '{instrument_name}' not found in config.INDEX_MAPPING.")
        return None

    lot_size = config.INDEX_MAPPING[instrument_name].get("quantity")
    if lot_size is None:
         print(f"Error: 'quantity' (lot size) not found for '{instrument_name}' in config.INDEX_MAPPING.")
         return None

    print(f"Determined Instrument: {instrument_name}, Lot Size: {lot_size}")
    # --- End Instrument/Lot Size Determination ---

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure you have run 'run_data_processing.py' first.")
        return None

    print(f"\n--- Running Custom Backtest ---")
    print(f"Strategy: Inside Candle Breakout (Custom)")
    print(f"Data File: {data_path.name}")
    print(f"Initial Cash: {cash:,.2f}")
    print(f"Commission per Trade (Entry+Exit): {commission_per_trade * 2:.2f}") # Show total round-trip
    print(f"SL ATR Multiplier: {stop_loss_atr_multiplier}")
    print(f"Risk/Reward Ratio: {risk_reward_ratio}")
    print(f"ATR Period: {atr_period}")

    try:
        # 1. Load Processed Data
        df = pd.read_parquet(data_path)
        print(f"Loaded data shape: {df.shape}")

        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return None

        # 2. Prepare Data for Custom Backtester
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Error: DataFrame index is not a DatetimeIndex.")
            return None

        # Rename columns to match backtester expectations (TitleCase)
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
        }
        cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df_bt = df.rename(columns=cols_to_rename)

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', f'atr_{atr_period}']
        if not all(col in df_bt.columns for col in required_cols):
            missing = set(required_cols) - set(df_bt.columns)
            print(f"Error: Missing required columns for backtesting: {missing}")
            return None

        print(f"Data prepared for backtesting. Head:\n{df_bt.head()}")

        # 3. Generate Signals
        print("Generating signals...")
        signals = get_inside_candle_signals(df_bt)
        print(f"Generated {len(signals[signals != 0])} non-zero signals.")

        # 4. Run Custom Backtest
        print("Running custom backtest simulation...")
        trades_log, performance_metrics = run_custom_backtest(
            df=df_bt,
            signals=signals, # Pass generated signals
            initial_cash=cash,
            lot_size=lot_size,
            commission_per_trade=commission_per_trade,
            stop_loss_atr_multiplier=stop_loss_atr_multiplier,
            risk_reward_ratio=risk_reward_ratio,
            atr_period=atr_period
        )
        print("Custom backtest finished.")

        # 5. Print Results
        print("\n--- Custom Backtest Results ---")
        # Format metrics for printing
        formatted_metrics = {}
        for key, value in performance_metrics.items():
            if isinstance(value, (float, np.float64)):
                formatted_metrics[key] = f"{value:.2f}" # Format floats to 2 decimal places
            elif isinstance(value, pd.Timestamp):
                 formatted_metrics[key] = value.strftime('%Y-%m-%d %H:%M:%S') # Format timestamps
            elif isinstance(value, pd.Timedelta):
                 formatted_metrics[key] = str(value) # Keep timedelta as string
            else:
                formatted_metrics[key] = value # Keep others as is

        # Print formatted metrics nicely
        max_key_len = max(len(key) for key in formatted_metrics.keys())
        for key, value in formatted_metrics.items():
            print(f"{key:<{max_key_len}} : {value}")


        if not trades_log.empty:
            print("\n--- Trades Log (Last 10) ---")
            print(trades_log.tail(10).to_string())
        else:
            print("\n--- Trades Log ---")
            print("No trades were executed.")

        # Optional: Save trades log to CSV
        # trades_log_filename = f"custom_backtest_trades_{data_path.stem}_SL{stop_loss_atr_multiplier}_RR{risk_reward_ratio}.csv"
        # trades_log.to_csv(trades_log_filename)
        # print(f"\nTrades log saved to: {trades_log_filename}")

        return performance_metrics

    except Exception as e:
        print(f"An error occurred during custom backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # --- Configuration ---
    PROCESSED_DATA_DIR = Path("data/historical_processed")
    # Select a specific file to backtest
    DATA_FILE = PROCESSED_DATA_DIR / "Nifty_5.parquet"

    # --- Run ---
    run_inside_candle_custom_backtest(
        DATA_FILE,
        cash=100_000,
        commission_per_trade=5.0, # Example: 5 units of currency per entry/exit leg
        stop_loss_atr_multiplier=1.5,
        risk_reward_ratio=2.0,
        atr_period=14
    )

    # --- Example: Run another test ---
    # DATA_FILE_2 = PROCESSED_DATA_DIR / "Bank_Nifty_15.parquet"
    # if DATA_FILE_2.exists():
    #     run_inside_candle_custom_backtest(
    #         DATA_FILE_2,
    #         cash=50000,
    #         commission_per_trade=10.0,
    #         stop_loss_atr_multiplier=1.8,
    #         risk_reward_ratio=1.5
    #     )
    # else:
    #     print(f"\nSkipping second test: {DATA_FILE_2} not found.")

    print("\nCustom backtesting script finished.")
