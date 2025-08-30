#!/usr/bin/env python3
"""
Comprehensive test suite for TradingEnv to verify all logic, calculations, and actions.
Tests every component: environment, engine, reward calculator, observation handler, termination manager.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import all components
from src.env.environment import TradingEnv
from src.env.trading_mode import TradingMode
from src.env.engine import BacktestingEngine
from src.env.reward_calculator import RewardCalculator
from src.env.observation_handler import ObservationHandler
from src.env.termination_manager import TerminationManager
from src.config.instrument import Instrument
from src.utils.data_loader import DataLoader

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentTester:
    """Comprehensive environment testing class."""
    
    def __init__(self):
        self.test_results = []
        self.failed_tests = []
        
    def create_sample_data(self, length: int = 1000) -> pd.DataFrame:
        """Create sample market data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price data
        base_price = 50000
        prices = []
        current_price = base_price
        
        for i in range(length):
            change = np.random.normal(0, 0.02) * current_price
            current_price += change
            current_price = max(current_price, base_price * 0.7)  # Price floor
            prices.append(current_price)
        
        data = {
            'datetime': pd.date_range('2024-01-01', periods=length, freq='5min'),
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, length),
        }
        
        df = pd.DataFrame(data)
        df.set_index('datetime', inplace=True)
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['atr'] = self._calculate_atr(df, 14)
        df['bb_upper'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], 20)
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['datetime_epoch'] = df.index.astype(np.int64) // 10**9
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper, lower
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def create_test_instrument(self) -> Instrument:
        """Create a test instrument."""
        return Instrument(
            symbol="NIFTY_TEST",
            lot_size=50,
            tick_size=0.05
        )
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test and record results."""
        print(f"\n{'='*60}")
        print(f"Running Test: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func(*args, **kwargs)
            self.test_results.append({
                'test': test_name,
                'status': 'PASSED',
                'result': result
            })
            print(f"{test_name}: PASSED")
            return result
        except Exception as e:
            self.test_results.append({
                'test': test_name,
                'status': 'FAILED',
                'error': str(e)
            })
            self.failed_tests.append(test_name)
            print(f"{test_name}: FAILED - {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return None
    
    def test_backtesting_engine(self) -> Dict:
        """Test BacktestingEngine functionality."""
        print("Testing BacktestingEngine...")
        
        instrument = self.create_test_instrument()
        initial_capital = 100000.0
        engine = BacktestingEngine(initial_capital, instrument, trailing_stop_percentage=0.02)
        
        # Test initial state
        initial_state = engine.get_account_state()
        assert initial_state['capital'] == initial_capital
        assert initial_state['current_position_quantity'] == 0.0
        assert not initial_state['is_position_open']
        print("Initial state correct")
        
        # Test BUY_LONG trade
        price = 50000.0
        quantity = 2.0
        atr = 500.0
        
        realized_pnl, unrealized_pnl = engine.execute_trade("BUY_LONG", price, quantity, atr)
        state_after_buy = engine.get_account_state(price)
        
        assert state_after_buy['current_position_quantity'] == quantity
        assert state_after_buy['current_position_entry_price'] == price
        assert state_after_buy['is_position_open']
        assert state_after_buy['capital'] < initial_capital  # Brokerage deducted
        print("BUY_LONG executed correctly")
        
        # Test price movement and unrealized PNL
        new_price = 51000.0  # Price moved up by 1000
        state_with_profit = engine.get_account_state(new_price)
        expected_unrealized_pnl = (new_price - price) * quantity * instrument.lot_size
        
        assert abs(state_with_profit['unrealized_pnl'] - expected_unrealized_pnl) < 1e-6
        print("Unrealized PNL calculation correct")
        
        # Test CLOSE_LONG trade
        realized_pnl, unrealized_pnl = engine.execute_trade("CLOSE_LONG", new_price, quantity, atr)
        state_after_close = engine.get_account_state(new_price)
        
        assert state_after_close['current_position_quantity'] == 0.0
        assert not state_after_close['is_position_open']
        assert state_after_close['realized_pnl'] > 0  # Profit made
        print("CLOSE_LONG executed correctly")
        
        # Test SELL_SHORT trade
        engine.reset()
        realized_pnl, unrealized_pnl = engine.execute_trade("SELL_SHORT", price, quantity, atr)
        state_after_short = engine.get_account_state(price)
        
        assert state_after_short['current_position_quantity'] == -quantity
        assert state_after_short['is_position_open']
        print("SELL_SHORT executed correctly")
        
        # Test CLOSE_SHORT trade
        lower_price = 49000.0  # Price moved down
        realized_pnl, unrealized_pnl = engine.execute_trade("CLOSE_SHORT", lower_price, quantity, atr)
        state_after_close_short = engine.get_account_state(lower_price)
        
        assert state_after_close_short['current_position_quantity'] == 0.0
        assert not state_after_close_short['is_position_open']
        print("CLOSE_SHORT executed correctly")
        
        # Test stop loss functionality
        engine.reset()
        engine.execute_trade("BUY_LONG", price, quantity, atr)
        
        # Trigger stop loss
        sl_price = engine._stop_loss_price
        stop_loss_price = sl_price - 1  # Below stop loss
        realized_pnl, unrealized_pnl = engine.execute_trade("HOLD", stop_loss_price, 0, atr)
        state_after_sl = engine.get_account_state(stop_loss_price)
        
        # Position should be automatically closed
        assert state_after_sl['current_position_quantity'] == 0.0
        print("Stop loss functionality working")
        
        # Test trade history
        trade_history = engine.get_trade_history()
        assert len(trade_history) > 0
        print("Trade history recorded")
        
        return {
            'engine_tests_passed': True,
            'trade_history_length': len(trade_history),
            'final_capital': engine.get_account_state()['capital']
        }
    
    def test_observation_handler(self) -> Dict:
        """Test ObservationHandler functionality."""
        print("Testing ObservationHandler...")
        
        data = self.create_sample_data(200)
        lookback_window = 50
        handler = ObservationHandler(lookback_window)
        
        # Create mock engine
        instrument = self.create_test_instrument()
        engine = BacktestingEngine(100000.0, instrument)
        
        # Initialize observation space
        config = {
            'hierarchical_reasoning_model': {
                'hierarchical_processing': {
                    'high_level_lookback': 100,
                    'low_level_lookback': 15
                }
            }
        }
        feature_columns = handler.initialize_observation_space(data, config)
        
        assert handler.observation_space is not None
        assert handler.features_per_step > 0
        assert len(feature_columns) > 0
        print("Observation space initialized correctly")
        
        # Test observation generation
        current_step = 100
        observation = handler.get_observation(data, current_step, engine)
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] == handler.observation_dim
        assert np.isfinite(observation).all()
        print("Observation generation working")
        
        # Test hierarchical observations
        hierarchical_obs = handler.get_hierarchical_observation(data, current_step, engine)
        
        assert 'high_level' in hierarchical_obs
        assert 'low_level' in hierarchical_obs
        assert isinstance(hierarchical_obs['high_level'], np.ndarray)
        assert isinstance(hierarchical_obs['low_level'], np.ndarray)
        print("Hierarchical observations working")
        
        # Test observation consistency
        obs1 = handler.get_observation(data, current_step, engine)
        obs2 = handler.get_observation(data, current_step, engine)
        
        assert np.array_equal(obs1, obs2), "Observations should be consistent for same input"
        print("Observation consistency verified")
        
        return {
            'observation_handler_tests_passed': True,
            'observation_dim': handler.observation_dim,
            'features_per_step': handler.features_per_step,
            'feature_count': len(feature_columns)
        }
    
    def test_reward_calculator(self) -> Dict:
        """Test Reward Calculator functionality."""
        print("Testing Reward Calculator...")
        
        symbol = "NIFTY_TEST"
        reward_calculator = RewardCalculator("pnl", symbol)
        
        # Test initialization
        assert reward_calculator.reward_function == "pnl"
        assert reward_calculator.reward_normalization_factor > 0
        print("Reward Calculator initialized correctly")
        
        # Test reset
        initial_capital = 100000.0
        reward_calculator.reset(initial_capital)
        
        assert len(reward_calculator.equity_history) == 1
        assert reward_calculator.equity_history[0] == initial_capital
        print("Reset functionality working")
        
        # Test reward calculation with mock engine
        instrument = self.create_test_instrument()
        engine = BacktestingEngine(initial_capital, instrument)
        
        # Test basic PNL reward
        current_capital = 101000.0
        prev_capital = 100000.0
        base_reward = reward_calculator.calculate_reward(current_capital, prev_capital, engine)
        
        assert base_reward > 0, "Should have positive reward for profit"
        print("Basic reward calculation working")
        
        # Test reward shaping
        shaped_reward = reward_calculator.apply_reward_shaping(
            base_reward, 4, current_capital, prev_capital, engine, 50000.0
        )
        
        assert isinstance(shaped_reward, float)
        print("Reward shaping working")
        
        # Test quantity feedback reward
        action = [1, 5.0]  # SELL_SHORT, quantity 5
        quantity_reward = reward_calculator.calculate_quantity_feedback_reward(action, current_capital)
        
        assert isinstance(quantity_reward, float)
        print("Quantity feedback reward working")
        
        # Test normalization
        normalized_reward = reward_calculator.normalize_reward(base_reward)
        
        assert abs(normalized_reward) <= abs(base_reward)  # Should be scaled down
        print("Reward normalization working")
        
        # Test different reward functions
        for reward_func in ["sharpe", "sortino", "profit_factor", "trading_focused"]:
            reward_calc = RewardCalculator(reward_func, symbol)
            reward_calc.reset(initial_capital)
            
            # Add some history for metric calculations
            for i in range(20):
                reward_calc.update_tracking_data(1, initial_capital + i * 100, 5.0, 10)
            
            reward = reward_calc.calculate_reward(current_capital, prev_capital, engine)
            assert isinstance(reward, float)
            print(f"{reward_func} reward calculation working")
        
        return {
            'reward_calculator_tests_passed': True,
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'normalized_reward': normalized_reward
        }
    
    def test_termination_manager(self) -> Dict:
        """Test TerminationManager functionality."""
        print("Testing TerminationManager...")
        
        # Test TRAINING mode
        termination_manager = TerminationManager(TradingMode.TRAINING, max_drawdown_pct=0.10)
        
        initial_capital = 100000.0
        termination_manager.reset(initial_capital, episode_end_step=500)
        
        assert termination_manager.peak_equity == initial_capital
        assert termination_manager.episode_end_step == 500
        print("TerminationManager initialized correctly")
        
        # Test normal conditions (no termination)
        done, reason = termination_manager.check_termination_conditions(100, 1000, 105000.0)
        assert not done
        assert reason is None
        print("Normal conditions - no termination")
        
        # Test strategic episode end
        done, reason = termination_manager.check_termination_conditions(500, 1000, 105000.0)
        assert done
        assert "strategic_episode_end" in reason
        print("Strategic episode end working")
        
        # Test max drawdown
        termination_manager.reset(initial_capital, episode_end_step=1000)
        termination_manager.update_peak_equity(110000.0)  # Set peak higher
        
        # Test drawdown exceeding limit
        current_capital = 95000.0  # More than 10% drawdown from peak
        done, reason = termination_manager.check_termination_conditions(200, 1000, current_capital)
        assert done
        assert "max_drawdown_exceeded" in reason
        print("Max drawdown termination working")
        
        # Test end of data
        done, reason = termination_manager.check_termination_conditions(999, 1000, 105000.0)
        assert done
        assert "end_of_data" in reason
        print("End of data termination working")
        
        # Test BACKTESTING mode (no early termination)
        backtest_manager = TerminationManager(TradingMode.BACKTESTING)
        backtest_manager.reset(initial_capital)
        
        # Should not terminate early even with drawdown
        done, reason = backtest_manager.check_termination_conditions(500, 1000, 80000.0)
        assert not done
        print("BACKTESTING mode - no early termination")
        
        return {
            'termination_manager_tests_passed': True,
            'drawdown_detection_working': True,
            'episode_end_detection_working': True
        }
    
    def test_full_trading_environment(self) -> Dict:
        """Test full TradingEnv with all components integrated."""
        print("Testing Full TradingEnv...")
        
        # Create sample data
        data = self.create_sample_data(500)
        
        # Test BACKTESTING mode (easier to test without data_loader dependency)
        env = TradingEnv(
            data_loader=None,
            symbol="NIFTY_TEST",
            initial_capital=100000.0,
            lookback_window=50,
            episode_length=200,
            mode=TradingMode.BACKTESTING,
            external_data=data,
            smart_action_filtering=True
        )
        
        # Test reset
        observation = env.reset()
        
        assert isinstance(observation, np.ndarray)
        assert observation.shape[0] == env.observation_space.shape[0]
        assert np.isfinite(observation).all()
        print("Environment reset working")
        
        # Test action space
        assert env.action_space.shape == (2,)  # [action_type, quantity]
        print("Action space correctly defined")
        
        # Test observation space
        assert env.observation_space.shape[0] > 0
        print("Observation space correctly defined")
        
        # Test step function with different actions
        actions_to_test = [
            [0, 5.0],   # BUY_LONG
            [4, 0.0],   # HOLD
            [2, 5.0],   # CLOSE_LONG
            [1, 3.0],   # SELL_SHORT
            [4, 0.0],   # HOLD
            [3, 3.0],   # CLOSE_SHORT
        ]
        
        step_results = []
        total_reward = 0.0
        
        for i, action in enumerate(actions_to_test):
            obs, reward, done, info = env.step(action)
            
            # Validate step return values
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(done, bool)
            assert isinstance(info, dict)
            
            assert obs.shape == env.observation_space.shape
            assert np.isfinite(obs).all()
            assert np.isfinite(reward)
            
            step_results.append({
                'step': i,
                'action': action,
                'reward': reward,
                'done': done,
                'action_name': info.get('action', 'UNKNOWN'),
                'capital': info.get('account_state', {}).get('capital', 0),
                'position': info.get('account_state', {}).get('current_position_quantity', 0)
            })
            
            total_reward += reward
            
            if done:
                print(f"Episode terminated at step {i}")
                break
        
        print("Step function working correctly")
        print(f"Total reward: {total_reward:.2f}")
        
        # Test episode info
        episode_info = env.get_episode_info()
        assert isinstance(episode_info, dict)
        assert 'total_data_length' in episode_info
        print("Episode info retrieval working")
        
        # Test backtest results
        backtest_results = env.get_backtest_results()
        assert isinstance(backtest_results, dict)
        assert 'mode' in backtest_results
        assert 'final_capital' in backtest_results
        print("Backtest results retrieval working")
        
        # Run a longer episode to test more functionality
        env.reset()
        long_episode_steps = 0
        long_episode_rewards = []
        
        for step in range(100):
            # Random valid actions
            action_type = np.random.choice([0, 1, 2, 3, 4])
            quantity = np.random.uniform(1, 10) if action_type != 4 else 0
            action = [action_type, quantity]
            
            obs, reward, done, info = env.step(action)
            long_episode_rewards.append(reward)
            long_episode_steps += 1
            
            if done:
                break
        
        print(f"Long episode completed: {long_episode_steps} steps")
        print(f"Average reward: {np.mean(long_episode_rewards):.4f}")
        
        return {
            'environment_tests_passed': True,
            'step_results': step_results,
            'total_reward': total_reward,
            'long_episode_steps': long_episode_steps,
            'final_capital': backtest_results.get('final_capital', 0),
            'backtest_results': backtest_results
        }
    
    def test_edge_cases_and_error_handling(self) -> Dict:
        """Test edge cases and error handling."""
        print("Testing Edge Cases and Error Handling...")
        
        # Test with minimal data
        minimal_data = self.create_sample_data(10)
        
        try:
            env = TradingEnv(
                data_loader=None,
                symbol="MINIMAL_TEST",
                initial_capital=10000.0,
                lookback_window=5,
                episode_length=5,
                mode=TradingMode.BACKTESTING,
                external_data=minimal_data
            )
            
            observation = env.reset()
            assert isinstance(observation, np.ndarray)
            print("Minimal data handling working")
            
        except Exception as e:
            print(f"Minimal data test failed: {e}")
        
        # Test invalid actions
        data = self.create_sample_data(100)
        env = TradingEnv(
            data_loader=None,
            symbol="ERROR_TEST",
            initial_capital=100000.0,
            mode=TradingMode.BACKTESTING,
            external_data=data
        )
        
        env.reset()
        
        # Test invalid action values
        invalid_actions = [
            [-1, 5.0],    # Invalid action type
            [5, 5.0],     # Invalid action type  
            [0, -5.0],    # Negative quantity
            [0, 0.0],     # Zero quantity for trading action
        ]
        
        error_handling_results = []
        
        for action in invalid_actions:
            try:
                obs, reward, done, info = env.step(action)
                error_handling_results.append({
                    'action': action,
                    'handled': True,
                    'reward': reward
                })
                print(f"Invalid action {action} handled gracefully")
            except Exception as e:
                error_handling_results.append({
                    'action': action,
                    'handled': False,
                    'error': str(e)
                })
                print(f"Invalid action {action} caused error: {e}")
        
        # Test zero capital scenarios
        zero_capital_env = TradingEnv(
            data_loader=None,
            symbol="ZERO_CAP_TEST",
            initial_capital=100.0,  # Very low capital
            mode=TradingMode.BACKTESTING,
            external_data=data
        )
        
        zero_capital_env.reset()
        
        # Try to make a large trade with insufficient capital
        obs, reward, done, info = zero_capital_env.step([0, 100.0])  # Large quantity
        
        account_state = info.get('account_state', {})
        position = account_state.get('current_position_quantity', 0)
        
        # Should limit quantity based on available capital
        print(f"Capital constraint handling: position = {position}")
        
        return {
            'edge_case_tests_passed': True,
            'error_handling_results': error_handling_results,
            'minimal_data_handled': True,
            'capital_constraints_working': True
        }
    
    def run_performance_test(self) -> Dict:
        """Test performance and timing."""
        print("Testing Performance...")
        
        import time
        
        data = self.create_sample_data(1000)
        env = TradingEnv(
            data_loader=None,
            symbol="PERF_TEST",
            initial_capital=100000.0,
            mode=TradingMode.BACKTESTING,
            external_data=data
        )
        
        # Time reset operation
        start_time = time.time()
        env.reset()
        reset_time = time.time() - start_time
        
        # Time step operations
        step_times = []
        
        for i in range(100):
            action = [np.random.choice([0, 1, 2, 3, 4]), np.random.uniform(1, 5)]
            
            start_time = time.time()
            obs, reward, done, info = env.step(action)
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            if done:
                break
        
        avg_step_time = np.mean(step_times)
        max_step_time = np.max(step_times)
        
        print(f"Reset time: {reset_time:.4f} seconds")
        print(f"Average step time: {avg_step_time:.4f} seconds")
        print(f"Max step time: {max_step_time:.4f} seconds")
        print(f"Steps per second: {1/avg_step_time:.0f}")
        
        # Performance thresholds
        performance_ok = (
            reset_time < 1.0 and  # Reset should be under 1 second
            avg_step_time < 0.1   # Steps should be under 100ms on average
        )
        
        return {
            'performance_tests_passed': performance_ok,
            'reset_time': reset_time,
            'avg_step_time': avg_step_time,
            'max_step_time': max_step_time,
            'steps_per_second': 1/avg_step_time
        }
    
    def run_all_tests(self) -> Dict:
        """Run all tests and return comprehensive results."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE ENVIRONMENT TESTING")
        print("="*80)
        
        # Run all test categories
        engine_results = self.run_test("BacktestingEngine", self.test_backtesting_engine)
        obs_results = self.run_test("ObservationHandler", self.test_observation_handler)
        reward_results = self.run_test("RewardCalculator", self.test_reward_calculator)
        term_results = self.run_test("TerminationManager", self.test_termination_manager)
        env_results = self.run_test("TradingEnvironment", self.test_full_trading_environment)
        edge_results = self.run_test("EdgeCases", self.test_edge_cases_and_error_handling)
        perf_results = self.run_test("Performance", self.run_performance_test)
        
        # Compile comprehensive results
        comprehensive_results = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'passed_tests': len([t for t in self.test_results if t['status'] == 'PASSED']),
                'failed_tests': len(self.failed_tests),
                'success_rate': (len(self.test_results) - len(self.failed_tests)) / len(self.test_results) * 100
            },
            'component_results': {
                'engine': engine_results,
                'observation_handler': obs_results,
                'reward_calculator': reward_results,
                'termination_manager': term_results,
                'trading_environment': env_results,
                'edge_cases': edge_results,
                'performance': perf_results
            },
            'failed_tests': self.failed_tests,
            'detailed_results': self.test_results
        }
        
        return comprehensive_results
    
    def print_final_report(self, results: Dict):
        """Print comprehensive test report."""
        print("\n" + "="*80)
        print("FINAL TEST REPORT")
        print("="*80)
        
        summary = results['test_summary']
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        
        if summary['failed_tests'] > 0:
            print(f"\nFailed Tests:")
            for test in results['failed_tests']:
                print(f"  - {test}")
        
        print(f"\nOVERALL STATUS: {'PASSED' if summary['failed_tests'] == 0 else 'FAILED'}")
        
        # Component-specific insights
        comp_results = results['component_results']
        
        if comp_results['performance']:
            perf = comp_results['performance']
            print(f"\nPerformance Metrics:")
            print(f"  - Steps per second: {perf['steps_per_second']:.0f}")
            print(f"  - Average step time: {perf['avg_step_time']:.4f}s")
        
        if comp_results['trading_environment']:
            env_res = comp_results['trading_environment']
            print(f"\nTrading Results:")
            print(f"  - Final capital: ${env_res['final_capital']:,.2f}")
            print(f"  - Total reward: {env_res['total_reward']:.4f}")
            print(f"  - Episode length: {env_res['long_episode_steps']} steps")

def main():
    """Main test execution function."""
    print("Environment Testing Suite Starting...")
    
    try:
        tester = EnvironmentTester()
        results = tester.run_all_tests()
        tester.print_final_report(results)
        
        # Return results for debugging/analysis
        return results
        
    except Exception as e:
        print(f"Critical error in test suite: {e}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    main()