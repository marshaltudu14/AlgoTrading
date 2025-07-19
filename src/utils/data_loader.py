import pandas as pd
import os
import logging
import random
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, final_data_dir: str = "data/final", raw_data_dir: str = "data/raw"):
        self.final_data_dir = final_data_dir
        self.raw_data_dir = raw_data_dir

    def load_all_processed_data(self) -> pd.DataFrame:
        all_data = []
        for filename in os.listdir(self.final_data_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.final_data_dir, filename)
                try:
                    df = pd.read_csv(filepath)
                    all_data.append(df)
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
        if not all_data:
            logging.warning(f"No CSV files found in {self.final_data_dir}")
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def load_raw_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        filepath = os.path.join(self.raw_data_dir, f"{symbol}.csv")
        try:
            df = pd.read_csv(filepath)
            # Basic OHLC validation
            if not ((df['high'] >= df['low']).all() and
                    (df['high'] >= df['open']).all() and
                    (df['high'] >= df['close']).all() and
                    (df['low'] <= df['open']).all() and
                    (df['low'] <= df['close']).all()):
                logging.warning(f"OHLC validation failed for {symbol}.csv")
            
            return df
        except FileNotFoundError:
            logging.error(f"File not found for symbol: {symbol} at {filepath}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading raw data for {symbol}: {e}")
            return pd.DataFrame()

    def get_available_tasks(self) -> List[Tuple[str, str]]:
        tasks = set()
        for filename in os.listdir(self.final_data_dir):
            if filename.endswith(".csv"):
                # Assuming filename format like 'INSTRUMENT_TIMEFRAME.csv'
                parts = filename.replace(".csv", "").split('_')
                if len(parts) >= 2:
                    instrument_type = "_".join(parts[:-1])
                    timeframe = parts[-1]
                    tasks.add((instrument_type, timeframe))
        return sorted(list(tasks))

    def sample_tasks(self, num_tasks: int) -> List[Tuple[str, str]]:
        available_tasks = self.get_available_tasks()
        if len(available_tasks) < num_tasks:
            logging.warning(f"Requested {num_tasks} tasks, but only {len(available_tasks)} are available. Returning all available tasks.")
            return available_tasks
        return random.sample(available_tasks, num_tasks)

    def get_task_data(self, instrument_type: str, timeframe: str) -> pd.DataFrame:
        filename = f"{instrument_type}_{timeframe}.csv"
        filepath = os.path.join(self.final_data_dir, filename)
        try:
            df = pd.read_csv(filepath)
            return df
        except FileNotFoundError:
            logging.error(f"Task data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading task data for {filename}: {e}")
            return pd.DataFrame()
