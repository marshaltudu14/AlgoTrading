"""
Dataset classes for training the TradingTransformer model.
Includes datasets for behavioral cloning, reward modeling, and RL fine-tuning.
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import os
import random
from config import WINDOW_SIZE, INSTRUMENT_IDS, TIMEFRAME_IDS, PROCESSED_DIR


class TradingDataset(Dataset):
    """
    Base dataset for trading data.
    Loads data from processed CSV files and creates windows for training.
    """
    def __init__(
        self,
        instruments: List[str],
        timeframes: List[int],
        window_size: int = WINDOW_SIZE,
        feature_cols: List[str] = None,
        include_signal: bool = False,
        validation: bool = False,
        validation_split: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            instruments: List of instruments to include
            timeframes: List of timeframes to include
            window_size: Size of the observation window
            feature_cols: List of feature columns to use (default: None, uses basic features)
            include_signal: Whether to include the signal column in the data
            validation: Whether this is a validation dataset
            validation_split: Fraction of data to use for validation
            seed: Random seed for reproducibility
        """
        self.instruments = instruments
        self.timeframes = timeframes
        self.window_size = window_size
        self.include_signal = include_signal
        self.validation = validation
        self.validation_split = validation_split
        self.seed = seed
        
        if feature_cols is None:
            self.feature_cols = ['open', 'high', 'low', 'close', 'ATR']
        else:
            self.feature_cols = feature_cols
            
        # Load all data
        self.data_frames = {}
        self.windows = []
        self._load_data()
        
    def _load_data(self):
        """Load data from processed CSV files and create windows."""
        random.seed(self.seed)
        
        for instrument in self.instruments:
            for timeframe in self.timeframes:
                # Load data from CSV
                file_path = os.path.join(
                    PROCESSED_DIR, 
                    f"{instrument.replace(' ', '_')}_{timeframe}.csv"
                )
                
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} not found, skipping.")
                    continue
                
                df = pd.read_csv(file_path)
                
                # Ensure all required columns are present
                missing_cols = [col for col in self.feature_cols if col not in df.columns]
                if missing_cols:
                    print(f"Warning: Missing columns {missing_cols} in {file_path}, skipping.")
                    continue
                
                if self.include_signal and 'signal' not in df.columns:
                    print(f"Warning: Signal column not found in {file_path}, skipping.")
                    continue
                
                # Store the dataframe
                key = (instrument, timeframe)
                self.data_frames[key] = df
                
                # Create windows
                n_samples = len(df) - self.window_size
                
                # Split into train/validation
                indices = list(range(n_samples))
                random.shuffle(indices)
                
                split_idx = int(n_samples * self.validation_split)
                
                if self.validation:
                    # Use the first split_idx indices for validation
                    indices = indices[:split_idx]
                else:
                    # Use the rest for training
                    indices = indices[split_idx:]
                
                # Create window metadata
                for idx in indices:
                    self.windows.append({
                        'instrument': instrument,
                        'timeframe': timeframe,
                        'start_idx': idx,
                        'end_idx': idx + self.window_size
                    })
    
    def __len__(self):
        """Return the number of windows in the dataset."""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a window of data.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary containing the window data
        """
        window = self.windows[idx]
        instrument = window['instrument']
        timeframe = window['timeframe']
        start_idx = window['start_idx']
        end_idx = window['end_idx']
        
        # Get the dataframe
        df = self.data_frames[(instrument, timeframe)]
        
        # Extract the window
        window_df = df.iloc[start_idx:end_idx]
        
        # Extract features
        features = window_df[self.feature_cols].values.astype(np.float32)
        
        # Create the sample
        sample = {
            'features': torch.tensor(features, dtype=torch.float32),
            'instrument_id': torch.tensor(INSTRUMENT_IDS[instrument], dtype=torch.long),
            'timeframe_id': torch.tensor(TIMEFRAME_IDS[timeframe], dtype=torch.long)
        }
        
        # Add signal if requested
        if self.include_signal:
            signal = window_df['signal'].values.astype(np.int64)
            sample['signal'] = torch.tensor(signal, dtype=torch.long)
        
        return sample


class BehavioralCloningDataset(TradingDataset):
    """
    Dataset for behavioral cloning.
    Converts signals to actions for supervised learning.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the dataset with signal column included."""
        kwargs['include_signal'] = True
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        """
        Get a window of data with actions derived from signals.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary containing the window data and actions
        """
        sample = super().__getitem__(idx)
        
        # Convert signals to actions
        # 0=hold, 1=buy target hit, 2=buy SL hit, 3=sell target hit, 4=sell SL hit
        signals = sample['signal']
        
        # Create actions: 0=hold, 1=buy, 2=sell
        actions = torch.zeros_like(signals)
        
        # Buy action for buy target hit (signal=1)
        actions[signals == 1] = 1
        
        # Sell action for sell target hit (signal=3)
        actions[signals == 3] = 2
        
        # Add actions to sample
        sample['actions'] = actions
        
        # For the last step, we use the action for the current state
        # This is what we want to predict
        sample['target_action'] = actions[-1]
        
        return sample


class RewardModelDataset(TradingDataset):
    """
    Dataset for training the reward model.
    Creates preference pairs based on signals.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the dataset with signal column included."""
        kwargs['include_signal'] = True
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        """
        Get a window of data with preference pairs.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary containing the window data and preference pairs
        """
        sample = super().__getitem__(idx)
        
        # Get the last state in the window
        last_state = sample['features'][-1]
        
        # Get the signal for the last state
        signal = sample['signal'][-1].item()
        
        # Create preference pairs
        # We'll create pairs of (state, action) with preferences
        
        # Preferred actions based on signal
        if signal == 1:  # Buy target hit
            preferred_action = 1  # Buy
            non_preferred_action = 0  # Hold
        elif signal == 3:  # Sell target hit
            preferred_action = 2  # Sell
            non_preferred_action = 0  # Hold
        else:  # Hold or SL hit
            preferred_action = 0  # Hold
            non_preferred_action = random.choice([1, 2])  # Buy or Sell
        
        sample['state'] = last_state
        sample['preferred_action'] = torch.tensor(preferred_action, dtype=torch.long)
        sample['non_preferred_action'] = torch.tensor(non_preferred_action, dtype=torch.long)
        
        return sample


class RLDataset(TradingDataset):
    """
    Dataset for RL fine-tuning.
    Provides windows of data for RL training.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the dataset without signal column for RL training."""
        kwargs['include_signal'] = False
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        """
        Get a window of data for RL training.
        
        Args:
            idx: Index of the window
            
        Returns:
            Dictionary containing the window data
        """
        return super().__getitem__(idx)


# Utility functions to create data loaders
def create_bc_dataloaders(
    instruments: List[str],
    timeframes: List[int],
    batch_size: int = 64,
    validation_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create data loaders for behavioral cloning.
    
    Args:
        instruments: List of instruments to include
        timeframes: List of timeframes to include
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = BehavioralCloningDataset(
        instruments=instruments,
        timeframes=timeframes,
        validation=False,
        validation_split=validation_split,
        seed=seed
    )
    
    val_dataset = BehavioralCloningDataset(
        instruments=instruments,
        timeframes=timeframes,
        validation=True,
        validation_split=validation_split,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_rm_dataloaders(
    instruments: List[str],
    timeframes: List[int],
    batch_size: int = 64,
    validation_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42
):
    """
    Create data loaders for reward modeling.
    
    Args:
        instruments: List of instruments to include
        timeframes: List of timeframes to include
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = RewardModelDataset(
        instruments=instruments,
        timeframes=timeframes,
        validation=False,
        validation_split=validation_split,
        seed=seed
    )
    
    val_dataset = RewardModelDataset(
        instruments=instruments,
        timeframes=timeframes,
        validation=True,
        validation_split=validation_split,
        seed=seed
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_rl_dataset(
    instruments: List[str],
    timeframes: List[int],
    validation_split: float = 0.0,
    seed: int = 42
):
    """
    Create a dataset for RL fine-tuning.
    
    Args:
        instruments: List of instruments to include
        timeframes: List of timeframes to include
        validation_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        RLDataset instance
    """
    return RLDataset(
        instruments=instruments,
        timeframes=timeframes,
        validation=False,
        validation_split=validation_split,
        seed=seed
    )
