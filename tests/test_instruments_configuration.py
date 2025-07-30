"""
Comprehensive tests for instruments.yaml configuration.
Tests both OPTIONS and STOCKS data types across different timeframes.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import yaml
from unittest.mock import Mock, patch

from src.utils.instrument_loader import load_instruments
from src.config.instrument import Instrument
from src.backtesting.environment import TradingEnv
from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader

class TestInstrumentsConfiguration:
    """Test suite for instruments.yaml configuration."""
    
    def test_load_instruments_yaml(self):
        """Test loading instruments from YAML configuration."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)
        
        # Should have both OPTIONS and STOCKS
        assert len(instruments) >= 5  # At least 5 instruments
        
        # Check OPTIONS instruments
        assert "Bank_Nifty" in instruments
        assert "Nifty" in instruments
        
        bank_nifty = instruments["Bank_Nifty"]
        assert bank_nifty.type == "OPTION"
        assert bank_nifty.lot_size == 25
        assert bank_nifty.tick_size == 0.05
        
        nifty = instruments["Nifty"]
        assert nifty.type == "OPTION"
        assert nifty.lot_size == 50
        assert nifty.tick_size == 0.05
        
        # Check STOCK instruments
        assert "RELIANCE" in instruments
        assert "TCS" in instruments
        assert "HDFC" in instruments
        
        reliance = instruments["RELIANCE"]
        assert reliance.type == "STOCK"
        assert reliance.lot_size == 1
        assert reliance.tick_size == 0.05
    
    def test_instrument_types_validation(self):
        """Test that instrument types are correctly validated."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)

        for instrument in instruments.values():
            # Type should be either OPTION or STOCK
            assert instrument.type in ["OPTION", "STOCK"]
            
            # Lot size should be positive
            assert instrument.lot_size > 0
            
            # Tick size should be positive
            assert instrument.tick_size > 0
            
            # OPTIONS should have larger lot sizes
            if instrument.type == "OPTION":
                assert instrument.lot_size >= 25
            
            # STOCKS should have lot size of 1
            if instrument.type == "STOCK":
                assert instrument.lot_size == 1

class TestOptionsDataProcessing:
    """Test suite for options data processing and premium conversion."""
    
    def create_test_options_data(self, symbol="Bank_Nifty", num_points=100):
        """Create realistic options test data."""
        np.random.seed(42)
        
        # Generate realistic Bank Nifty price data
        base_price = 45000 if symbol == "Bank_Nifty" else 19000  # Typical levels
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        
        # Generate price series with realistic volatility
        returns = np.random.normal(0.0001, 0.015, num_points)  # Higher volatility for options
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 50000)  # Higher volume for options
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_options_premium_conversion_bank_nifty(self):
        """Test options premium conversion for Bank Nifty."""
        # Create test data
        df = self.create_test_options_data("Bank_Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)

            # Create processed data with features
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)

            # Create data loader and environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=100000,
                lookback_window=10,
                episode_length=50,
                use_streaming=False
            )
            
            # Test environment initialization
            obs = env.reset()
            assert obs is not None
            
            # Test premium calculation
            current_price = df['close'].iloc[-1]
            atr = abs(df['high'] - df['low']).mean()  # Simple ATR approximation
            
            premium = env._calculate_proxy_premium(current_price, atr)
            
            # Premium should be reasonable for Bank Nifty
            assert premium >= 50.0  # Minimum premium for Bank Nifty
            assert premium <= current_price * 0.05  # Maximum 5% of underlying
            
            # Premium should be proportional to volatility
            assert isinstance(premium, float)
            assert premium > 0
    
    def test_options_premium_conversion_nifty(self):
        """Test options premium conversion for Nifty."""
        # Create test data
        df = self.create_test_options_data("Nifty")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Save test data
            df.to_csv(os.path.join(raw_dir, "Nifty.csv"), index=False)

            # Create processed data with features
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Nifty.csv"), index=False)

            # Create data loader and environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Nifty",
                initial_capital=100000,
                lookback_window=10,
                episode_length=50,
                use_streaming=False
            )
            
            # Test environment initialization
            obs = env.reset()
            assert obs is not None
            
            # Test premium calculation
            current_price = df['close'].iloc[-1]
            atr = abs(df['high'] - df['low']).mean()
            
            premium = env._calculate_proxy_premium(current_price, atr)
            
            # Premium should be reasonable for Nifty
            assert premium >= 25.0  # Minimum premium for Nifty
            assert premium <= current_price * 0.05  # Maximum 5% of underlying
    
    def test_options_lot_size_calculation(self):
        """Test that options lot sizes are correctly applied."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)

        # Test Bank Nifty lot size
        bank_nifty = instruments["Bank_Nifty"]
        engine = BacktestingEngine(initial_capital=100000, instrument=bank_nifty)

        # Test trade execution with lot size
        current_price = 45000
        proxy_premium = 200  # Typical premium

        # Execute 2 lots trade
        quantity = 2
        _, _ = engine.execute_trade("BUY_LONG", quantity, current_price, proxy_premium)

        # Check that position quantity reflects lot size
        state = engine.get_account_state()
        expected_position = quantity * bank_nifty.lot_size  # 2 * 25 = 50
        assert state["current_position_quantity"] == expected_position

        # Test Nifty lot size
        nifty = instruments["Nifty"]
        engine_nifty = BacktestingEngine(initial_capital=100000, instrument=nifty)

        # Execute 1 lot trade
        quantity = 1
        _, _ = engine_nifty.execute_trade("BUY_LONG", quantity, 19000, 150)

        # Check that position quantity reflects lot size
        state = engine_nifty.get_account_state()
        expected_position = quantity * nifty.lot_size  # 1 * 50 = 50
        assert state["current_position_quantity"] == expected_position

class TestStockDataProcessing:
    """Test suite for stock data processing."""
    
    def create_test_stock_data(self, symbol="RELIANCE", num_points=100):
        """Create realistic stock test data."""
        np.random.seed(42)
        
        # Generate realistic stock price data
        base_price = 2500 if symbol == "RELIANCE" else 3500  # Typical stock prices
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        
        # Generate price series with stock-like volatility
        returns = np.random.normal(0.0001, 0.01, num_points)  # Lower volatility for stocks
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLC from price series
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.003)))
            low = price * (1 - abs(np.random.normal(0, 0.003)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(100, 5000)  # Lower volume for stocks
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_stock_trading_environment(self):
        """Test stock trading environment."""
        # Create test data
        df = self.create_test_stock_data("RELIANCE")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)

            # Save test data
            df.to_csv(os.path.join(raw_dir, "RELIANCE.csv"), index=False)

            # Create processed data with features
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_RELIANCE.csv"), index=False)

            # Create data loader and environment
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="RELIANCE",
                initial_capital=100000,
                lookback_window=10,
                episode_length=50,
                use_streaming=False
            )
            
            # Test environment initialization
            obs = env.reset()
            assert obs is not None
            
            # Test stock trading (no premium calculation)
            action = (0, 10)  # BUY 10 shares
            _, reward, done, info = env.step(action)

            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)
    
    def test_stock_lot_size_calculation(self):
        """Test that stock lot sizes are correctly applied."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)

        # Test stock lot size
        reliance = instruments["RELIANCE"]
        engine = BacktestingEngine(initial_capital=100000, instrument=reliance)

        # Test trade execution with stock lot size
        current_price = 2500

        # Execute 100 shares trade
        quantity = 100
        _, _ = engine.execute_trade("BUY_LONG", quantity, current_price)

        # Check that position quantity reflects lot size (should be exactly quantity for stocks)
        state = engine.get_account_state()
        expected_position = quantity * reliance.lot_size  # 100 * 1 = 100
        assert state["current_position_quantity"] == expected_position

class TestMultiTimeframeSupport:
    """Test suite for multi-timeframe support."""
    
    def test_different_timeframes(self):
        """Test that different timeframes are supported."""
        # Create test data for different timeframes
        timeframes = ["1", "5", "15", "60"]  # 1min, 5min, 15min, 1hour
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Create data for each timeframe
            for tf in timeframes:
                # Create test data
                np.random.seed(42)
                dates = pd.date_range('2023-01-01', periods=100, freq=f'{tf}min')
                
                data = {
                    'datetime': dates,
                    'open': np.random.uniform(45000, 46000, 100),
                    'high': np.random.uniform(46000, 47000, 100),
                    'low': np.random.uniform(44000, 45000, 100),
                    'close': np.random.uniform(45000, 46000, 100),
                    'volume': np.random.randint(1000, 10000, 100)
                }
                
                df = pd.DataFrame(data)
                
                # Save raw data
                df.to_csv(os.path.join(raw_dir, f"Bank_Nifty_{tf}.csv"), index=False)
                
                # Save processed data
                df_processed = df.copy()
                df_processed['returns'] = df_processed['close'].pct_change()
                df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
                df_processed.to_csv(os.path.join(final_dir, f"features_Bank_Nifty_{tf}.csv"), index=False)
            
            # Test data loader with different timeframes
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            
            # Get available tasks
            tasks = loader.get_available_tasks()
            
            # Should have tasks for different timeframes
            assert len(tasks) >= len(timeframes)
            
            # Check that timeframes are correctly parsed
            timeframe_symbols = [task[0] for task in tasks]
            for tf in timeframes:
                assert any(tf in symbol for symbol in timeframe_symbols)

class TestInstrumentIntegration:
    """Test integration between instruments and trading components."""
    
    def test_options_vs_stocks_trading_differences(self):
        """Test that options and stocks are handled differently in trading."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)

        # Test options instrument
        bank_nifty = instruments["Bank_Nifty"]
        options_engine = BacktestingEngine(initial_capital=100000, instrument=bank_nifty)

        # Test stock instrument
        reliance = instruments["RELIANCE"]
        stock_engine = BacktestingEngine(initial_capital=100000, instrument=reliance)

        # Execute similar trades
        quantity = 2
        price = 45000
        proxy_premium = 200

        # Options trade (uses premium)
        _, _ = options_engine.execute_trade("BUY_LONG", quantity, price, proxy_premium)
        options_state = options_engine.get_account_state()

        # Stock trade (no premium)
        _, _ = stock_engine.execute_trade("BUY_LONG", quantity, 2500)
        stock_state = stock_engine.get_account_state()

        # Options should have different capital calculation due to premium
        # Stock should use direct price * quantity calculation
        assert options_state["current_position_quantity"] == quantity * bank_nifty.lot_size
        assert stock_state["current_position_quantity"] == quantity * reliance.lot_size

        # Capital should be different due to different cost calculations
        assert options_state["capital"] != stock_state["capital"]
    
    def test_instrument_configuration_validation(self):
        """Test that instrument configuration is properly validated."""
        # Test loading with valid configuration
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)

        # All instruments should be valid
        for symbol, instrument in instruments.items():
            assert isinstance(instrument, Instrument)
            assert instrument.symbol == symbol
            assert instrument.type in ["OPTION", "STOCK"]
            assert instrument.lot_size > 0
            assert instrument.tick_size > 0
