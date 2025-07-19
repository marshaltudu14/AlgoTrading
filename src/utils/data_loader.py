import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, final_data_dir: str = "data/final", raw_data_dir: str = "data/raw"):
        self.final_data_path = Path(final_data_dir)
        self.raw_data_path = Path(raw_data_dir)

    def load_all_processed_data(self) -> pd.DataFrame:
        all_data = []
        if not self.final_data_path.exists() or not self.final_data_path.is_dir():
            logging.warning(f"Data directory not found or is not a directory: {self.final_data_path}")
            return pd.DataFrame()

        csv_files = list(self.final_data_path.glob("*.csv"))
        if not csv_files:
            logging.warning(f"No CSV files found in {self.final_data_path}. Returning empty DataFrame.")
            return pd.DataFrame()

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                all_data.append(df)
            except pd.errors.ParserError as e:
                logging.error(f"ParserError reading {file_path}: {e}. Skipping file.")
                continue
            except Exception as e:
                logging.error(f"Unexpected error reading {file_path}: {e}. Skipping file.")
                continue

            # Basic validation: check if DataFrame is empty or has unexpected columns
            if df.empty:
                logging.warning(f"File {file_path} resulted in an empty DataFrame. Skipping.")
                continue
            # Assuming all processed data files should have at least 2 columns (e.g., datetime and some value)
            # This can be made more specific if we know the exact expected columns.
            if len(df.columns) < 2:
                logging.warning(f"File {file_path} has fewer than 2 columns. Skipping as potentially malformed.")
                continue

        if not all_data:
            logging.warning("No data successfully loaded from any CSV files. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            concatenated_df = pd.concat(all_data, ignore_index=True)
            # Basic data validation/cleaning (e.g., drop duplicates, handle NaNs if necessary)
            # For now, just dropping duplicates as a basic step. More specific cleaning can be added later.
            concatenated_df.drop_duplicates(inplace=True)
            return concatenated_df
        except Exception as e:
            logging.error(f"Error concatenating DataFrames: {e}")
            return pd.DataFrame()

    def load_raw_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        file_path = self.raw_data_path / f"{symbol}.csv"
        if not file_path.exists():
            logging.error(f"Raw data file not found for symbol {symbol}: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)

            # Check for required columns
            required_columns = ['datetime', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_columns):
                logging.error(f"Missing required columns in {file_path}. Expected: {required_columns}")
                return pd.DataFrame()

            # Convert datetime column to datetime objects
            df['datetime'] = pd.to_datetime(df['datetime'])

            # OHLCV validation
            invalid_ohlc = df[~((df['high'] >= df['low']) &
                                (df['high'] >= df['open']) &
                                (df['high'] >= df['close']) &
                                (df['low'] <= df['open']) &
                                (df['low'] <= df['close']))]

            if not invalid_ohlc.empty:
                logging.warning(f"OHLCV data validation failed for {len(invalid_ohlc)} rows in {file_path}. These rows will be dropped.")
                df = df.drop(invalid_ohlc.index)

            if df.empty:
                logging.warning(f"After validation, DataFrame for {symbol} is empty. Returning empty DataFrame.")
                return pd.DataFrame()

            return df

        except pd.errors.ParserError as e:
            logging.error(f"ParserError reading {file_path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Unexpected error reading {file_path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()