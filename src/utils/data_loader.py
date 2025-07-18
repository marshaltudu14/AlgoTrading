import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Loads processed data from the specified directory.
    """
    def __init__(self, data_dir: str = "data/final"):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_all_data(self) -> pd.DataFrame:
        """
        Loads all CSV files from the data directory and concatenates them into a single DataFrame.
        
        Returns:
            pd.DataFrame: A concatenated DataFrame of all processed data.
        """
        all_files = list(self.data_dir.glob("*.csv"))
        if not all_files:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return pd.DataFrame()

        list_df = []
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path)
                list_df.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                continue
        
        if not list_df:
            logger.error("No dataframes were successfully loaded.")
            return pd.DataFrame()

        combined_df = pd.concat(list_df, ignore_index=True)
        logger.info(f"Successfully combined {len(list_df)} files into a single DataFrame with {len(combined_df)} rows.")
        return combined_df

    def load_data_for_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Loads data for a specific symbol (e.g., 'Nifty_2').
        
        Args:
            symbol (str): The base name of the CSV file (e.g., 'Nifty_2').
            
        Returns:
            pd.DataFrame: DataFrame for the specified symbol.
        """
        file_path = self.data_dir / f"final_{symbol}.csv"
        if not file_path.exists():
            logger.warning(f"Data file for symbol '{symbol}' not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows for symbol '{symbol}' from {file_path.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading data for symbol '{symbol}' from {file_path.name}: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example Usage:
    data_loader = DataLoader()
    
    # Load all data
    all_data_df = data_loader.load_all_data()
    if not all_data_df.empty:
        print(f"Shape of all combined data: {all_data_df.shape}")
        print(all_data_df.head())
    
    print("\n" + "="*50 + "\n")
    
    # Load data for a specific symbol
    nifty_data_df = data_loader.load_data_for_symbol("Nifty_2")
    if not nifty_data_df.empty:
        print(f"Shape of Nifty_2 data: {nifty_data_df.shape}")
        print(nifty_data_df.head())
