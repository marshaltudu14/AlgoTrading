import pandas as pd
import os
import logging
import random
from typing import List, Tuple, Iterator, Optional, Union, Dict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

from src.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, final_data_dir: str = None, raw_data_dir: str = None,
                 chunk_size: int = 10000, use_parquet: bool = True):
        settings = get_settings()
        paths_config = settings.get('paths', {})
        
        self.final_data_dir = final_data_dir or paths_config.get('final_data_dir', 'data/final')
        self.raw_data_dir = raw_data_dir or paths_config.get('raw_data_dir', 'data/raw')
        self.chunk_size = chunk_size
        self.use_parquet = use_parquet
        self.testing_mode = False  # Initialize testing mode
        self.in_memory_data = {}   # Initialize in-memory data storage

        # Create parquet directories if they don't exist
        self.parquet_final_dir = os.path.join(self.final_data_dir, "parquet")
        self.parquet_raw_dir = os.path.join(self.raw_data_dir, "parquet")
        os.makedirs(self.parquet_final_dir, exist_ok=True)
        os.makedirs(self.parquet_raw_dir, exist_ok=True)



    def load_all_processed_data(self) -> pd.DataFrame:
        all_data = []
        for filename in os.listdir(self.final_data_dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(self.final_data_dir, filename)
                try:
                    # CRITICAL: Set datetime_readable as index when loading processed features
                    df = pd.read_csv(filepath, index_col=0)
                    all_data.append(df)
                except Exception as e:
                    logging.warning(f"Could not read {filename}: {e}")
        if not all_data:
            logging.warning(f"No CSV files found in {self.final_data_dir}")
            return pd.DataFrame()
        return pd.concat(all_data, ignore_index=True)

    def load_data_chunks(self, symbol: Optional[str] = None,
                        data_type: str = "raw") -> Iterator[pd.DataFrame]:
        """
        Load data in chunks for memory-efficient processing.

        Args:
            symbol: Specific symbol to load (if None, loads all data)
            data_type: "raw" or "final" data directory

        Yields:
            pd.DataFrame: Chunks of data
        """
        data_dir = self.raw_data_dir if data_type == "raw" else self.final_data_dir
        parquet_dir = self.parquet_raw_dir if data_type == "raw" else self.parquet_final_dir

        if symbol:
            # Load specific symbol
            yield from self._load_symbol_chunks(symbol, data_dir, parquet_dir)
        else:
            # Load all files
            file_pattern = "*.parquet" if self.use_parquet else "*.csv"
            search_dir = parquet_dir if self.use_parquet else data_dir

            for filepath in Path(search_dir).glob(file_pattern):
                yield from self._load_file_chunks(filepath)

    def _load_symbol_chunks(self, symbol: str, data_dir: str,
                           parquet_dir: str) -> Iterator[pd.DataFrame]:
        """Load chunks for a specific symbol."""
        if self.use_parquet:
            parquet_path = os.path.join(parquet_dir, f"{symbol}.parquet")
            if os.path.exists(parquet_path):
                yield from self._load_parquet_chunks(parquet_path)
            else:
                logging.error(f"No data found for symbol: {symbol}")
        else:
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            yield from self._load_csv_chunks(csv_path)

    def _load_file_chunks(self, filepath: Path) -> Iterator[pd.DataFrame]:
        """Load chunks from a single file."""
        if filepath.suffix == '.parquet':
            yield from self._load_parquet_chunks(str(filepath))
        elif filepath.suffix == '.csv':
            yield from self._load_csv_chunks(str(filepath))

    def _load_csv_chunks(self, filepath: str) -> Iterator[pd.DataFrame]:
        """Load CSV file in chunks."""
        try:
            # Determine if this is a features file (no datetime required)
            is_features_file = 'features_' in os.path.basename(filepath)
            require_datetime = not is_features_file

            # CRITICAL: Set datetime_readable as index when loading processed features
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size, index_col=0):
                # Apply basic validation to each chunk
                if self._validate_chunk(chunk, require_datetime=require_datetime):
                    yield chunk
                else:
                    logging.warning(f"Skipping invalid chunk in {filepath}")
        except Exception as e:
            logging.error(f"Error loading CSV chunks from {filepath}: {e}")

    def _load_parquet_chunks(self, filepath: str) -> Iterator[pd.DataFrame]:
        """Load Parquet file in chunks."""
        try:
            # Determine if this is a features file (no datetime required)
            is_features_file = 'features_' in os.path.basename(filepath)
            require_datetime = not is_features_file

            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                df = batch.to_pandas()
                if self._validate_chunk(df, require_datetime=require_datetime):
                    yield df
                else:
                    logging.warning(f"Skipping invalid chunk in {filepath}")
        except Exception as e:
            logging.error(f"Error loading Parquet chunks from {filepath}: {e}")

    def _validate_chunk(self, chunk: pd.DataFrame, require_datetime: bool = True) -> bool:
        """Validate a data chunk."""
        if chunk.empty:
            return False

        # Check for required columns (basic validation)
        required_cols = ['open', 'high', 'low', 'close']
        if require_datetime:
            required_cols = ['datetime'] + required_cols

        if not all(col in chunk.columns for col in required_cols):
            # For features files, datetime might not be present
            if not require_datetime and all(col in chunk.columns for col in ['open', 'high', 'low', 'close']):
                pass  # Valid features file
            else:
                return False

        # Basic OHLC validation
        try:
            if not ((chunk['high'] >= chunk['low']).all() and
                    (chunk['high'] >= chunk['open']).all() and
                    (chunk['high'] >= chunk['close']).all() and
                    (chunk['low'] <= chunk['open']).all() and
                    (chunk['low'] <= chunk['close']).all()):
                return False
        except Exception:
            return False

        return True

    

    def get_storage_info(self) -> Dict:
        """Get information about data storage formats and sizes."""
        info = {
            "use_parquet": self.use_parquet,
            "chunk_size": self.chunk_size,
            "directories": {
                "raw_csv": self.raw_data_dir,
                "final_csv": self.final_data_dir,
                "raw_parquet": self.parquet_raw_dir,
                "final_parquet": self.parquet_final_dir
            },
            "file_counts": {},
            "storage_sizes": {}
        }

        # Count files and calculate sizes
        for dir_name, dir_path in info["directories"].items():
            if os.path.exists(dir_path):
                if "parquet" in dir_name:
                    files = list(Path(dir_path).glob("*.parquet"))
                else:
                    files = list(Path(dir_path).glob("*.csv"))

                info["file_counts"][dir_name] = len(files)
                total_size = sum(f.stat().st_size for f in files if f.exists())
                info["storage_sizes"][dir_name] = f"{total_size / (1024*1024):.2f} MB"
            else:
                info["file_counts"][dir_name] = 0
                info["storage_sizes"][dir_name] = "0 MB"

        return info

    def benchmark_loading_performance(self, symbol: str, iterations: int = 3) -> Dict:
        """Benchmark loading performance between CSV and Parquet formats."""
        import time

        results = {
            "symbol": symbol,
            "iterations": iterations,
            "csv_times": [],
            "parquet_times": [],
            "csv_avg": 0,
            "parquet_avg": 0,
            "speedup": 0
        }

        # Test CSV loading
        csv_path = os.path.join(self.raw_data_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            for i in range(iterations):
                start_time = time.time()
                df = pd.read_csv(csv_path)
                end_time = time.time()
                results["csv_times"].append(end_time - start_time)
            results["csv_avg"] = sum(results["csv_times"]) / iterations

        # Test Parquet loading
        parquet_path = os.path.join(self.parquet_raw_dir, f"{symbol}.parquet")
        if not os.path.exists(parquet_path) and os.path.exists(csv_path):
            self._convert_csv_to_parquet(csv_path, parquet_path)

        if os.path.exists(parquet_path):
            for i in range(iterations):
                start_time = time.time()
                df = pd.read_parquet(parquet_path)
                end_time = time.time()
                results["parquet_times"].append(end_time - start_time)
            results["parquet_avg"] = sum(results["parquet_times"]) / iterations

        # Calculate speedup
        if results["parquet_avg"] > 0 and results["csv_avg"] > 0:
            results["speedup"] = results["csv_avg"] / results["parquet_avg"]

        return results

    def load_final_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """Load processed final data for a specific symbol."""
        logging.info(f"Looking for data for symbol: {symbol} in directory: {self.final_data_dir}")

        # Only look for the correct pattern: features_{symbol}.parquet
        filename = f"features_{symbol}.parquet"
        filepath = os.path.join(self.final_data_dir, filename)
        
        logging.info(f"Checking file: {filepath}")
        if os.path.exists(filepath):
            logging.info(f"File exists: {filepath}")
            try:
                # Load from Parquet file
                df = pd.read_parquet(filepath)
                
                # Ensure all numeric columns are float32 to prevent dtype mismatches
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = df[numeric_columns].astype(np.float32)
                
                logging.info(f"Loaded final data for {symbol}: {len(df)} rows from {filename}")
                return df
            except Exception as e:
                logging.error(f"Error loading {filepath}: {e}")
        else:
            logging.info(f"File does not exist: {filepath}")
            # List available files for debugging
            if os.path.exists(self.final_data_dir):
                available_files = [f for f in os.listdir(self.final_data_dir) if f.endswith(".csv")]
                logging.info(f"Available CSV files in directory: {available_files}")

        logging.error(f"No final data found for symbol: {symbol}")
        return pd.DataFrame()

    def get_data_length(self, symbol: str, data_type: str = "final") -> int:
        """Get the total number of rows for a symbol from its Parquet file."""
        if data_type != "final":
            logging.warning("get_data_length is optimized for final Parquet data and should not be used for raw data.")
            return 0

        # The file path is constructed based on the symbol.
        # e.g., symbol 'Bank_Nifty_5' -> 'features_Bank_Nifty_5.parquet'
        parquet_path = os.path.join(self.final_data_dir, f"features_{symbol}.parquet")

        if os.path.exists(parquet_path):
            try:
                parquet_file = pq.ParquetFile(parquet_path)
                return parquet_file.metadata.num_rows
            except Exception as e:
                logging.error(f"Error reading Parquet metadata for {parquet_path}: {e}")
                return 0
        else:
            logging.error(f"No Parquet data file found for symbol: {symbol} at {parquet_path}")
            return 0

    def get_base_symbol(self, symbol: str) -> str:
        """
        Extract base symbol by removing timeframe suffix.

        Examples:
        - Bank_Nifty_5 -> Bank_Nifty
        - Nifty_2 -> Nifty
        - RELIANCE_1 -> RELIANCE
        """
        import re
        # Remove timeframe suffix pattern (underscore followed by digits)
        base_symbol = re.sub(r'_\d+$', '', symbol)
        return base_symbol

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
        try:
            for filename in os.listdir(self.final_data_dir):
                if filename.endswith(".csv"):
                    # Handle filename format like 'features_INSTRUMENT_TIMEFRAME.csv' or 'INSTRUMENT_TIMEFRAME.csv'
                    base_name = filename.replace(".csv", "")

                    # Remove "features_" prefix if present
                    if base_name.startswith("features_"):
                        base_name = base_name[9:]  # Remove "features_" prefix

                    parts = base_name.split('_')
                    if len(parts) >= 2:
                        instrument_type = "_".join(parts[:-1])
                        timeframe = parts[-1]
                        tasks.add((instrument_type, timeframe))
                        logging.info(f"Found task: ({instrument_type}, {timeframe}) from file {filename}")
        except Exception as e:
            logging.error(f"Error scanning for available tasks: {e}")

        logging.info(f"Total available tasks: {len(tasks)}")
        return sorted(list(tasks))

    def sample_tasks(self, num_tasks: int) -> List[Tuple[str, str]]:
        available_tasks = self.get_available_tasks()

        # Handle empty tasks case
        if not available_tasks:
            logging.warning(f"No tasks available in {self.final_data_dir}. Creating fallback tasks.")
            # Create fallback tasks based on available data files
            fallback_tasks = self._create_fallback_tasks()
            if fallback_tasks:
                return fallback_tasks[:num_tasks] if num_tasks > 0 else fallback_tasks
            else:
                logging.error("No data files found for fallback tasks.")
                return []

        # Handle invalid num_tasks
        if num_tasks <= 0:
            logging.warning(f"Invalid num_tasks: {num_tasks}. Returning first available task.")
            return [available_tasks[0]] if available_tasks else []

        # If we have fewer tasks than requested, return all available with repetition if needed
        if len(available_tasks) < num_tasks:
            logging.warning(f"Requested {num_tasks} tasks, but only {len(available_tasks)} are available.")
            # Repeat tasks to meet the requirement
            repeated_tasks = []
            for i in range(num_tasks):
                repeated_tasks.append(available_tasks[i % len(available_tasks)])
            return repeated_tasks

        # Use safe sampling to avoid "Cannot choose from an empty sequence" error
        try:
            return random.sample(available_tasks, num_tasks)
        except ValueError as e:
            logging.error(f"Error sampling tasks: {e}. Available tasks: {len(available_tasks)}, Requested: {num_tasks}")
            # Return all available tasks as fallback
            return available_tasks[:num_tasks] if len(available_tasks) >= num_tasks else available_tasks

    def _create_fallback_tasks(self) -> List[Tuple[str, str]]:
        """Create fallback tasks from available data files."""
        fallback_tasks = []

        try:
            # Check for any CSV files in the final data directory
            if os.path.exists(self.final_data_dir):
                for filename in os.listdir(self.final_data_dir):
                    if filename.endswith(".csv"):
                        # Extract instrument and timeframe from filename
                        base_name = filename.replace(".csv", "")
                        if base_name.startswith("features_"):
                            base_name = base_name[9:]  # Remove "features_" prefix

                        # Try to split into instrument and timeframe
                        parts = base_name.split('_')
                        if len(parts) >= 2:
                            instrument = "_".join(parts[:-1])
                            timeframe = parts[-1]
                            fallback_tasks.append((instrument, timeframe))
                        else:
                            # Use the whole name as instrument with default timeframe
                            fallback_tasks.append((base_name, "5"))

            # Also check the raw data directory for additional tasks
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.endswith(".csv"):
                        base_name = filename.replace(".csv", "")
                        # Create task from raw data filename
                        fallback_tasks.append((base_name, "5"))

            if not fallback_tasks:
                # Create multiple minimal fallback tasks for better MAML training
                logging.warning("No CSV files found, creating minimal fallback tasks")
                fallback_tasks = [
                    ("Nifty", "5"),
                    ("BankNifty", "5"),
                    ("Nifty", "15"),
                    ("BankNifty", "15")
                ]

        except Exception as e:
            logging.error(f"Error creating fallback tasks: {e}")
            # Create multiple fallback tasks to avoid empty sequence
            fallback_tasks = [
                ("Nifty", "5"),
                ("BankNifty", "5"),
                ("Nifty", "15"),
                ("BankNifty", "15")
            ]

        logging.info(f"Created {len(fallback_tasks)} fallback tasks: {fallback_tasks}")
        return fallback_tasks

    def get_task_data(self, instrument_type: str, timeframe: str) -> pd.DataFrame:
        filename = f"{instrument_type}_{timeframe}.csv"
        filepath = os.path.join(self.final_data_dir, filename)
        try:
            # CRITICAL: Set datetime_readable as index when loading processed features
            df = pd.read_csv(filepath, index_col=0)
            return df
        except FileNotFoundError:
            logging.error(f"Task data file not found: {filename}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading task data for {filename}: {e}")
            return pd.DataFrame()
