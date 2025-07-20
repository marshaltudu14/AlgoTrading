import pandas as pd
import os
import logging
import random
from typing import List, Tuple, Iterator, Optional, Union, Dict
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, final_data_dir: str = "data/final", raw_data_dir: str = "data/raw",
                 chunk_size: int = 10000, use_parquet: bool = True):
        self.final_data_dir = final_data_dir
        self.raw_data_dir = raw_data_dir
        self.chunk_size = chunk_size
        self.use_parquet = use_parquet

        # Create parquet directories if they don't exist
        self.parquet_final_dir = os.path.join(final_data_dir, "parquet")
        self.parquet_raw_dir = os.path.join(raw_data_dir, "parquet")
        os.makedirs(self.parquet_final_dir, exist_ok=True)
        os.makedirs(self.parquet_raw_dir, exist_ok=True)

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
                # Convert CSV to Parquet first
                csv_path = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(csv_path):
                    self._convert_csv_to_parquet(csv_path, parquet_path)
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
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
                # Apply basic validation to each chunk
                if self._validate_chunk(chunk):
                    yield chunk
                else:
                    logging.warning(f"Skipping invalid chunk in {filepath}")
        except Exception as e:
            logging.error(f"Error loading CSV chunks from {filepath}: {e}")

    def _load_parquet_chunks(self, filepath: str) -> Iterator[pd.DataFrame]:
        """Load Parquet file in chunks."""
        try:
            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                df = batch.to_pandas()
                if self._validate_chunk(df):
                    yield df
                else:
                    logging.warning(f"Skipping invalid chunk in {filepath}")
        except Exception as e:
            logging.error(f"Error loading Parquet chunks from {filepath}: {e}")

    def _validate_chunk(self, chunk: pd.DataFrame) -> bool:
        """Validate a data chunk."""
        if chunk.empty:
            return False

        # Check for required columns (basic validation)
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        if not all(col in chunk.columns for col in required_cols):
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

    def _convert_csv_to_parquet(self, csv_path: str, parquet_path: str) -> None:
        """Convert CSV file to Parquet format for faster loading."""
        try:
            logging.info(f"Converting {csv_path} to Parquet format...")
            df = pd.read_csv(csv_path)

            # Basic data cleaning and validation
            if self._validate_chunk(df):
                # Convert datetime column to proper datetime type
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])

                # Save as Parquet
                df.to_parquet(parquet_path, index=False, engine='pyarrow')
                logging.info(f"Successfully converted to {parquet_path}")
            else:
                logging.error(f"Invalid data in {csv_path}, skipping conversion")
        except Exception as e:
            logging.error(f"Error converting {csv_path} to Parquet: {e}")

    def load_data_segment(self, symbol: str, start_idx: int, end_idx: int,
                         data_type: str = "raw") -> pd.DataFrame:
        """
        Load a specific segment of data for on-demand loading.

        Args:
            symbol: Symbol to load
            start_idx: Starting row index
            end_idx: Ending row index
            data_type: "raw" or "final" data directory

        Returns:
            pd.DataFrame: Data segment
        """
        data_dir = self.raw_data_dir if data_type == "raw" else self.final_data_dir
        parquet_dir = self.parquet_raw_dir if data_type == "raw" else self.parquet_final_dir

        if self.use_parquet:
            parquet_path = os.path.join(parquet_dir, f"{symbol}.parquet")
            if not os.path.exists(parquet_path):
                # Convert CSV to Parquet first
                csv_path = os.path.join(data_dir, f"{symbol}.csv")
                if os.path.exists(csv_path):
                    self._convert_csv_to_parquet(csv_path, parquet_path)
                else:
                    logging.error(f"No data found for symbol: {symbol}")
                    return pd.DataFrame()

            return self._load_parquet_segment(parquet_path, start_idx, end_idx)
        else:
            csv_path = os.path.join(data_dir, f"{symbol}.csv")
            return self._load_csv_segment(csv_path, start_idx, end_idx)

    def _load_csv_segment(self, filepath: str, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a specific segment from CSV file."""
        try:
            # Use skiprows and nrows for efficient segment loading
            nrows = end_idx - start_idx
            df = pd.read_csv(filepath, skiprows=range(1, start_idx + 1), nrows=nrows)

            if self._validate_chunk(df):
                return df
            else:
                logging.warning(f"Invalid data segment in {filepath}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading CSV segment from {filepath}: {e}")
            return pd.DataFrame()

    def _load_parquet_segment(self, filepath: str, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a specific segment from Parquet file."""
        try:
            # Parquet allows efficient row-level access
            df = pd.read_parquet(filepath, engine='pyarrow')
            segment = df.iloc[start_idx:end_idx]

            if self._validate_chunk(segment):
                return segment
            else:
                logging.warning(f"Invalid data segment in {filepath}")
                return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error loading Parquet segment from {filepath}: {e}")
            return pd.DataFrame()

    def get_data_length(self, symbol: str, data_type: str = "raw") -> int:
        """Get the total number of rows for a symbol."""
        data_dir = self.raw_data_dir if data_type == "raw" else self.final_data_dir
        parquet_dir = self.parquet_raw_dir if data_type == "raw" else self.parquet_final_dir

        if self.use_parquet:
            parquet_path = os.path.join(parquet_dir, f"{symbol}.parquet")
            if os.path.exists(parquet_path):
                try:
                    parquet_file = pq.ParquetFile(parquet_path)
                    return parquet_file.metadata.num_rows
                except Exception as e:
                    logging.error(f"Error reading Parquet metadata: {e}")
                    return 0

        # Fallback to CSV
        csv_path = os.path.join(data_dir, f"{symbol}.csv")
        try:
            # Count lines efficiently
            with open(csv_path, 'r') as f:
                return sum(1 for _ in f) - 1  # Subtract header
        except Exception as e:
            logging.error(f"Error counting CSV rows: {e}")
            return 0

    def convert_all_csv_to_parquet(self, data_type: str = "raw") -> None:
        """Convert all CSV files to Parquet format for better performance."""
        data_dir = self.raw_data_dir if data_type == "raw" else self.final_data_dir
        parquet_dir = self.parquet_raw_dir if data_type == "raw" else self.parquet_final_dir

        csv_files = list(Path(data_dir).glob("*.csv"))
        if not csv_files:
            logging.info(f"No CSV files found in {data_dir}")
            return

        logging.info(f"Converting {len(csv_files)} CSV files to Parquet format...")

        for csv_file in csv_files:
            parquet_file = os.path.join(parquet_dir, f"{csv_file.stem}.parquet")
            if not os.path.exists(parquet_file):
                self._convert_csv_to_parquet(str(csv_file), parquet_file)
            else:
                logging.info(f"Parquet file already exists: {parquet_file}")

        logging.info("Parquet conversion completed")

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
        # Try different possible filename patterns
        possible_files = [
            f"{symbol}.csv",
            f"features_{symbol}.csv",
            f"processed_{symbol}.csv"
        ]

        for filename in possible_files:
            filepath = os.path.join(self.final_data_dir, filename)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    logging.info(f"Loaded final data for {symbol}: {len(df)} rows")
                    return df
                except Exception as e:
                    logging.error(f"Error loading {filepath}: {e}")
                    continue

        logging.error(f"No final data found for symbol: {symbol}")
        return pd.DataFrame()

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
