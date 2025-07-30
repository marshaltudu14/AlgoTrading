"""
Test suite for capital-aware quantity adjustment in TradingEnv.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader

class TestCapitalAwareTradingEnv:
    """Test suite for capital-aware quantity adjustment in TradingEnv."""
    
    def create_test_data(self, symbol="Bank_Nifty", num_points=100):
        """Create test data for trading environment."""
        np.random.seed(42)
        base_price = 45000 if symbol == "Bank_Nifty" else 2500
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        returns = np.random.normal(0.0001, 0.015, num_points)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 50000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_capital_aware_quantity_adjustment_options_insufficient_capital(self):
        """Test that TradingEnv adjusts quantity when capital is insufficient for options."""
        # Create test data
        df = self.create_test_data("Bank_Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment with very limited capital
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=15000,  # Very limited capital
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Reset environment
            obs = env.reset()
            
            # Try to execute a large trade that should be adjusted
            action = (0, 5)  # BUY_LONG with 5 lots (should be too expensive)
            
            # Get initial capital
            initial_capital = env.engine.get_account_state()['capital']
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Check that the trade was executed (possibly with adjusted quantity)
            account_state = env.engine.get_account_state()
            
            # If a position was opened, it should be with adjusted quantity
            if account_state['is_position_open']:
                # Position should be less than 5 lots due to capital constraints
                assert account_state['current_position_quantity'] < 5
                assert account_state['current_position_quantity'] >= 0
            
            # Capital should have decreased (trade was executed)
            assert account_state['capital'] <= initial_capital
    
    def test_capital_aware_quantity_adjustment_options_sufficient_capital(self):
        """Test that TradingEnv doesn't adjust quantity when capital is sufficient for options."""
        # Create test data
        df = self.create_test_data("Bank_Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment with sufficient capital
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=100000,  # Sufficient capital
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Reset environment
            obs = env.reset()
            
            # Try to execute a small trade that should not be adjusted
            action = (0, 2)  # BUY_LONG with 2 lots (should be affordable)
            
            # Get initial capital
            initial_capital = env.engine.get_account_state()['capital']
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Check that the trade was executed with the requested quantity
            account_state = env.engine.get_account_state()
            
            # Position should be exactly 2 lots (no adjustment needed)
            if account_state['is_position_open']:
                assert account_state['current_position_quantity'] == 2
            
            # Capital should have decreased appropriately
            assert account_state['capital'] < initial_capital
    
    def test_capital_aware_quantity_adjustment_stocks(self):
        """Test that TradingEnv adjusts quantity for stocks when capital is insufficient."""
        # Create test data
        df = self.create_test_data("RELIANCE")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "RELIANCE.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_RELIANCE.csv"), index=False)
            
            # Create environment with limited capital
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="RELIANCE",
                initial_capital=10000,  # Limited capital for stocks
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Reset environment
            obs = env.reset()
            
            # Try to execute a large trade that should be adjusted
            action = (0, 5)  # BUY_LONG with 5 shares (might be too expensive)
            
            # Get initial capital
            initial_capital = env.engine.get_account_state()['capital']
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Check that the trade was executed (possibly with adjusted quantity)
            account_state = env.engine.get_account_state()
            
            # If a position was opened, it should be with appropriate quantity
            if account_state['is_position_open']:
                assert account_state['current_position_quantity'] >= 1
                assert account_state['current_position_quantity'] <= 5
            
            # Capital should have decreased (trade was executed)
            assert account_state['capital'] <= initial_capital
    
    def test_hold_action_not_affected(self):
        """Test that HOLD actions are not affected by capital-aware adjustment."""
        # Create test data
        df = self.create_test_data("Bank_Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=50000,
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Reset environment
            obs = env.reset()
            
            # Execute HOLD action
            action = (4, 0)  # HOLD
            
            # Get initial capital
            initial_capital = env.engine.get_account_state()['capital']
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Check that capital and position remain unchanged
            account_state = env.engine.get_account_state()
            
            assert account_state['capital'] == initial_capital  # No change in capital
            assert not account_state['is_position_open']  # No position opened
    
    def test_zero_quantity_after_adjustment(self):
        """Test behavior when quantity is adjusted to zero due to insufficient capital."""
        # Create test data
        df = self.create_test_data("Bank_Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment with very limited capital
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=1000,  # Very limited capital (insufficient for any trade)
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Reset environment
            obs = env.reset()
            
            # Try to execute a trade that should be adjusted to zero
            action = (0, 3)  # BUY_LONG with 3 lots (should be too expensive)
            
            # Get initial capital
            initial_capital = env.engine.get_account_state()['capital']
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Check that no trade was executed (quantity adjusted to 0)
            account_state = env.engine.get_account_state()
            
            # No position should be opened
            assert not account_state['is_position_open']
            
            # Capital should remain unchanged (no trade executed)
            assert account_state['capital'] == initial_capital
