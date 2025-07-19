import unittest
from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader
from src.backtesting.environment import TradingEnv
from pathlib import Path
import shutil
import numpy as np
import logging

# Suppress logging during tests for cleaner output
logging.disable(logging.CRITICAL)


class TestTradingEnv(unittest.TestCase):

    def setUp(self):
        self.initial_capital = 100000.0
        self.lookback_window = 2
        self.test_data_raw_dir = Path("test_data_raw_env")
        self.test_data_raw_dir.mkdir(exist_ok=True)

        # Create a dummy CSV file for testing
        self._create_raw_csv("TEST_SYMBOL.csv",
            "datetime,open,high,low,close,volume\n"
            "2023-01-01,100,105,95,102,1000\n"
            "2023-01-02,102,108,98,105,1200\n"
            "2023-01-03,105,110,100,108,1500\n"
            "2023-01-04,108,112,102,110,1300\n"
            "2023-01-05,110,115,105,112,1400"
        )
        self.data_loader = DataLoader(raw_data_dir=str(self.test_data_raw_dir))

    def tearDown(self):
        if self.test_data_raw_dir.exists():
            shutil.rmtree(self.test_data_raw_dir)

    def _create_raw_csv(self, filename, content):
        file_path = self.test_data_raw_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def test_env_initialization(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        self.assertIsInstance(env, TradingEnv)
        self.assertEqual(env.initial_capital, self.initial_capital)
        self.assertEqual(env.lookback_window, self.lookback_window)
        self.assertEqual(env.current_step, self.lookback_window)
        self.assertFalse(env.df.empty)
        self.assertIsInstance(env.engine, BacktestingEngine)

    def test_env_reset(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertEqual(env.current_step, self.lookback_window)
        self.assertEqual(info["capital"], self.initial_capital)

    def test_env_step_hold(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        obs, info = env.reset()
        
        # Step 1: HOLD
        action = 4 # HOLD
        obs, reward, done, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertFalse(done)
        self.assertEqual(env.current_step, self.lookback_window + 1)
        self.assertEqual(reward, 0.0) # No change in P&L for HOLD

    def test_env_step_buy_long(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        obs, info = env.reset()
        
        # Step 1: BUY_LONG
        action = 0 # BUY_LONG
        obs, reward, done, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertFalse(done)
        self.assertEqual(info["current_position_quantity"], 1) # Default quantity is 1
        self.assertLess(info["capital"], self.initial_capital) # Capital should decrease due to trade and brokerage

    def test_env_step_close_long(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        obs, info = env.reset()
        
        # Step 1: BUY_LONG
        env.step(0) # Buy 1 share
        
        # Step 2: CLOSE_LONG
        action = 2 # CLOSE_LONG
        obs, reward, done, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertFalse(done)
        self.assertEqual(info["current_position_quantity"], 0) # Position should be closed
        self.assertNotEqual(info["realized_pnl"], 0.0) # Should have some P&L

    def test_env_episode_done(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=self.initial_capital, lookback_window=self.lookback_window)
        obs, info = env.reset()
        
        # Fast forward to the end of the data
        # Perform a buy and sell to ensure some P&L
        env.step(0) # BUY_LONG
        for _ in range(len(env.df) - env.current_step -1):
            obs, reward, done, info = env.step(4) # HOLD action until done
            if done:
                break
        env.step(2) # CLOSE_LONG
        obs, reward, done, info = env.step(4) # Final step to trigger done
        
        self.assertTrue(done)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertNotEqual(reward, 0.0) # Final reward should be total P&L

    def test_env_invalid_action_penalty(self):
        env = TradingEnv(symbol="TEST_SYMBOL", data_loader=self.data_loader, initial_capital=10.0, lookback_window=self.lookback_window) # Low capital
        obs, info = env.reset()
        
        # Attempt BUY_LONG with insufficient capital
        action = 0 # BUY_LONG
        obs, reward, done, info = env.step(action)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(info, dict)
        self.assertFalse(done)
        self.assertEqual(reward, -1.0) # Should receive penalty for invalid action
        self.assertEqual(info["current_position_quantity"], 0) # No position opened
        self.assertEqual(info["capital"], 10.0) # Capital should not change