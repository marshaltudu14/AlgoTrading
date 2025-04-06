import pandas as pd
import pandas_ta as ta # Import pandas_ta
from backtesting import Backtest
from pathlib import Path
import warnings
import re # Import regex for extracting instrument name

# Assuming src is in the same parent directory or PYTHONPATH is set
try:
    from src.strategy import InsideCandleStrategy # Import the Inside Candle strategy
    from src import config # Import config to get lot size
    # from src.strategy import EMACrossoverStrategy # Keep for reference
except ImportError:
    print("Error: Ensure 'src' directory is accessible or in PYTHONPATH.")
    # Attempt fallback if run from the root directory
    try:
        from src.strategy import InsideCandleStrategy
        from src import config
    except ImportError as e:
        print(f"Failed to import required modules: {e}")
        exit(1)

# Suppress specific warnings from backtesting.py if they become noisy
# warnings.filterwarnings("ignore", category=FutureWarning, module="backtesting")
# warnings.filterwarnings("ignore", message="Some prices are larger than initial cash value") # Example

def run_strategy_backtest(data_file_path: str | Path, strategy_class, cash: int = 100_000, commission: float = 0.0002):
    """
    Runs a backtest for a given strategy on a specific processed data file,
    setting lot size based on the instrument name derived from the filename.

    Args:
        data_file_path: Path to the processed Parquet data file (e.g., "Nifty_5.parquet").
        strategy_class: The strategy class to test (e.g., InsideCandleStrategy).
        cash: Initial portfolio cash.
        commission: Broker commission per trade (e.g., 0.0002 for 0.02%).
    """
    # --- Determine Instrument and Lot Size ---
    data_path = Path(data_file_path)
    instrument_match = re.match(r"([a-zA-Z_]+)_\d+", data_path.stem) # Extract base name like "Nifty" or "Bank_Nifty"
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

    print(f"\n--- Running Backtest ---")
    print(f"Strategy: {strategy_class.__name__}")
    print(f"Data File: {data_path.name}")
    print(f"Initial Cash: {cash:,.2f}")
    print(f"Commission: {commission:.4f}")

    try:
        # 1. Load Processed Data
        df = pd.read_parquet(data_path)
        print(f"Loaded data shape: {df.shape}")

        # --- Removed EMA Pre-calculation ---


        if df.empty:
            print("Error: Loaded DataFrame is empty.")
            return None

        # 2. Prepare Data for backtesting.py
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Error: DataFrame index is not a DatetimeIndex.")
            return None

        # Rename columns to match backtesting.py expectations (TitleCase)
        # The pipeline saves lowercase columns.
        rename_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close'
            # Volume is not present/needed by default after processing
        }
        # Check which columns actually exist before renaming
        cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        if len(cols_to_rename) != 4:
             missing = set(rename_map.keys()) - set(df.columns)
             print(f"Warning: Missing expected columns for renaming: {missing}. Proceeding with available columns.")
             # Check if essential OHLC are present even if not renamed
             if not all(col in df.columns or col in cols_to_rename for col in ['open', 'high', 'low', 'close']):
                 print("Error: Essential OHLC columns are missing from the data.")
                 return None

        df_bt = df.rename(columns=cols_to_rename)

        # Add Volume column if strategy requires it and it exists (it shouldn't after pipeline)
        # We need to ensure 'atr_14' column is present for the strategy
        if 'atr_14' not in df.columns:
             print("Error: Required column 'atr_14' not found in the processed data.")
             return None
        # Ensure ATR column is carried over (needed if strategy uses it, even simplified one might)
        df_bt['atr_14'] = df['atr_14']


        print(f"Data prepared for backtesting (columns renamed, ATR added). Head:\n{df_bt.head()}")

        # 3. Set Strategy Parameters Dynamically
        # Modify the class attributes before initializing Backtest
        # Set parameters relevant to the InsideCandleStrategy
        strategy_class.lot_size = lot_size
        # Set default SL/TP parameters if the strategy has them
        if hasattr(strategy_class, 'stop_loss_atr_multiplier'):
            strategy_class.stop_loss_atr_multiplier = 1.5 # Example default
        if hasattr(strategy_class, 'risk_reward_ratio'):
            strategy_class.risk_reward_ratio = 2.0 # Example default

        print(f"Set Strategy Parameters: lot_size={strategy_class.lot_size}, "
              f"SL_ATR_Mult={getattr(strategy_class, 'stop_loss_atr_multiplier', 'N/A')}, "
              f"RR={getattr(strategy_class, 'risk_reward_ratio', 'N/A')}")


        # 4. Initialize Backtest
        bt = Backtest(
            df_bt,               # Data prepared for the library
             strategy_class,      # Your strategy class (now with updated params)
             cash=cash,
             commission=commission,
             # trade_on_close=True, # Removed: Use default entry at next bar's open
             exclusive_orders=True # Prevent overlapping entry/exit signals in same bar
         )

        # 5. Run Backtest
        print("Running backtest simulation...")
        stats = bt.run()
        print("Backtest finished.")

        # 6. Print Statistics
        print("\n--- Backtest Results ---")
        print(stats)

        # Optional: Print specific stats
        # print(f"\nReturn: {stats['Return [%]']:.2f}%")
        # print(f"Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        # print(f"Win Rate: {stats['Win Rate [%]']:.2f}%")
        # print(f"Number of Trades: {stats['# Trades']}")

        # 7. Plot Results (optional)
        print("\nGenerating plot...")
        # Adjust filename to include SL/TP parameters
        sl_mult = getattr(strategy_class, 'stop_loss_atr_multiplier', 'NA')
        rr = getattr(strategy_class, 'risk_reward_ratio', 'NA')
        plot_filename = f"backtest_{strategy_class.__name__}_{data_path.stem}_SL{sl_mult}_RR{rr}.html"
        try:
            bt.plot(filename=plot_filename, open_browser=False)
            print(f"Plot saved to: {plot_filename}")
        except Exception as plot_err:
            print(f"Warning: Could not generate plot. Error: {plot_err}")


        return stats # Return stats for potential further analysis

    except Exception as e:
        print(f"An error occurred during backtesting: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # --- Configuration ---
    PROCESSED_DATA_DIR = Path("data/historical_processed")
    # Select a specific file to backtest
    # Make sure this file exists after running run_data_processing.py
    # Example: Use Nifty 5-minute data
    DATA_FILE = PROCESSED_DATA_DIR / "Nifty_5.parquet"
    STRATEGY = InsideCandleStrategy # Use the Inside Candle strategy

    # --- Run ---
    run_strategy_backtest(DATA_FILE, STRATEGY)

    # --- Example: Run another test ---
    # DATA_FILE_2 = PROCESSED_DATA_DIR / "Bank_Nifty_15.parquet"
    # if DATA_FILE_2.exists():
    #     run_strategy_backtest(DATA_FILE_2, STRATEGY, cash=50000, commission=0.0003)
    # else:
    #     print(f"\nSkipping second test: {DATA_FILE_2} not found.")

    print("\nBacktesting script finished.")
