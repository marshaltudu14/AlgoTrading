"""
Universal Data Normalizer for HRM Trading Model
Handles percentage-based P&L and universal feature normalization
Eliminates instrument-specific bias while preserving trading signal quality
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import logging
import joblib
import os

logger = logging.getLogger(__name__)


class UniversalDataNormalizer:
    """
    Universal data normalizer that eliminates instrument bias
    Key principles:
    1. Convert all P&L to percentage terms
    2. Normalize all features to 0-100 scale using MinMax
    3. No instrument-specific hardcoded rules
    4. Preserve relative relationships in data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.scalers = {}
        self.normalization_stats = {}
        self.is_fitted = False
        
        # Normalization configuration
        self.feature_range = (0, 100)  # 0-100 scale for all features
        self.reward_range = (-100, 100)  # -100 to +100 for rewards
        
        
        # Columns to exclude from normalization (non-numerical or not useful)
        self.exclude_columns = ['datetime_readable']
        
    def fit(self, data: pd.DataFrame, price_column: str = 'close') -> 'UniversalDataNormalizer':
        """
        Fit the normalizer on training data
        
        Args:
            data: Training dataset
            price_column: Column to use as reference price for percentage calculations
        """
        logger.info("Fitting universal data normalizer...")
        
        # Make a copy to avoid modifying original
        data_copy = data.copy()
        
        # Fit scalers for all numeric columns
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        columns_to_normalize = [col for col in numeric_columns if col not in self.exclude_columns]
        
        for column in columns_to_normalize:
            if column in data_copy.columns:
                scaler = MinMaxScaler(feature_range=self.feature_range)
                
                # Handle NaN values
                values = data_copy[column].values.reshape(-1, 1)
                mask = ~np.isnan(values.flatten())
                
                if mask.sum() > 0:  # Only fit if we have valid values
                    scaler.fit(values[mask])
                    self.scalers[column] = scaler
                    
                    # Store normalization statistics
                    self.normalization_stats[column] = {
                        'min': float(np.nanmin(data_copy[column])),
                        'max': float(np.nanmax(data_copy[column])),
                        'mean': float(np.nanmean(data_copy[column])),
                        'std': float(np.nanstd(data_copy[column]))
                    }
        
        self.is_fitted = True
        logger.info(f"Normalizer fitted on {len(columns_to_normalize)} columns")
        return self
    
    def transform(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Transform data using fitted normalizers
        
        Args:
            data: Data to transform
            price_column: Reference price column for percentage calculations
            
        Returns:
            Normalized data
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transforming")
            
        # Make a copy to avoid modifying original
        data_copy = data.copy()
        
        # Apply normalization
        for column, scaler in self.scalers.items():
            if column in data_copy.columns:
                values = data_copy[column].values.reshape(-1, 1)
                
                # Handle NaN values
                mask = ~np.isnan(values.flatten())
                if mask.sum() > 0:
                    values[mask] = scaler.transform(values[mask])
                    data_copy[column] = values.flatten()
        
        return data_copy
    
    def fit_transform(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data, price_column).transform(data, price_column)
    
    
    def normalize_reward(self, reward: float, capital_pct_change: float = None) -> float:
        """
        Normalize reward to -100 to +100 range based on percentage P&L
        
        Args:
            reward: Raw reward value
            capital_pct_change: Percentage change in capital (for context)
            
        Returns:
            Normalized reward in -100 to +100 range
        """
        # If we have percentage capital change, use it for more meaningful scaling
        if capital_pct_change is not None:
            # Scale based on percentage change in capital
            # 10% gain = +100 reward, 10% loss = -100 reward
            normalized_reward = np.clip(capital_pct_change * 10, -100, 100)
        else:
            # Fallback: simple clipping to range
            normalized_reward = np.clip(reward, -100, 100)
        
        return float(normalized_reward)
    
    def calculate_percentage_pnl(self, current_capital: float, previous_capital: float) -> float:
        """
        Calculate percentage-based P&L
        
        Args:
            current_capital: Current capital amount
            previous_capital: Previous capital amount
            
        Returns:
            Percentage change in capital
        """
        if previous_capital == 0:
            return 0.0
            
        pct_change = ((current_capital - previous_capital) / previous_capital) * 100
        return float(pct_change)
    
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform normalized data back to original scale
        Useful for debugging and analysis
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before inverse transforming")
            
        data_copy = data.copy()
        
        for column, scaler in self.scalers.items():
            if column in data_copy.columns:
                values = data_copy[column].values.reshape(-1, 1)
                
                # Handle NaN values
                mask = ~np.isnan(values.flatten())
                if mask.sum() > 0:
                    values[mask] = scaler.inverse_transform(values[mask])
                    data_copy[column] = values.flatten()
        
        return data_copy
    
    def save(self, filepath: str):
        """Save the fitted normalizer"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted normalizer")
            
        save_data = {
            'scalers': self.scalers,
            'normalization_stats': self.normalization_stats,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(save_data, filepath)
        logger.info(f"Normalizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a fitted normalizer"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Normalizer file not found: {filepath}")
            
        save_data = joblib.load(filepath)
        
        self.scalers = save_data['scalers']
        self.normalization_stats = save_data['normalization_stats']
        self.config = save_data.get('config', {})
        self.is_fitted = save_data.get('is_fitted', True)
        
        logger.info(f"Normalizer loaded from {filepath}")
    
    def get_feature_stats(self) -> Dict:
        """Get normalization statistics for all features"""
        return self.normalization_stats.copy()
    
    def print_normalization_summary(self):
        """Print a summary of normalization statistics"""
        print("\n" + "="*60)
        print("📊 DATA NORMALIZATION SUMMARY")
        print("="*60)
        print(f"Normalized Features: {len(self.scalers)}")
        print(f"Feature Range: {self.feature_range}")
        print(f"Reward Range: {self.reward_range}")
        print("\nTop 10 Feature Statistics:")
        print("-" * 60)
        
        for i, (feature, stats) in enumerate(list(self.normalization_stats.items())[:10]):
            print(f"{feature:20s} | Min: {stats['min']:8.3f} | Max: {stats['max']:8.3f} | "
                  f"Mean: {stats['mean']:8.3f} | Std: {stats['std']:8.3f}")
        
        if len(self.normalization_stats) > 10:
            print(f"... and {len(self.normalization_stats) - 10} more features")
        
        print("="*60 + "\n")


def create_universal_normalizer(config: Dict = None) -> UniversalDataNormalizer:
    """Factory function to create a universal data normalizer"""
    return UniversalDataNormalizer(config)