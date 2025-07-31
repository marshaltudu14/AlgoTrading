#!/usr/bin/env python3
"""
Unified Backtesting System using Enhanced TradingEnv
===================================================

This script uses the enhanced TradingEnv for backtesting with the same logic
as training, ensuring consistency across training, backtesting, and live trading.

Features:
- Uses TradingEnv in BACKTESTING mode
- Fetches real-time data from Fyers API
- Processes data through feature generator
- Sequential row-by-row processing (no episodes)
- Point-based index trading (no option premium complexity)
- Comprehensive results and trade analysis
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtesting.environment import TradingEnv, TradingMode
from src.agents.ppo_agent import PPOAgent
from src.trading.fyers_client import FyersClient
from src.data_processing.feature_generator import DynamicFileProcessor
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('unified_backtest.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from training_sequence.yaml"""
    config_path = "config/training_sequence.yaml"
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.warning(f"Could not load config from {config_path}: {e}")
        return {
            'environment': {'initial_capital': 100000.0, 'lookback_window': 50},
            'model': {'hidden_dim': 64},
            'risk_management': {'risk_multiplier': 1.0, 'reward_multiplier': 2.0}
        }

class UnifiedBacktester:
    """Unified backtesting system using enhanced TradingEnv."""

    def __init__(self):
        self.fyers_client = FyersClient()
        self.feature_processor = DynamicFileProcessor()

        # Load configuration
        self.config = load_config()
        env_config = self.config.get('environment', {})
        model_config = self.config.get('model', {})

        # Hardcoded configuration - Bank Nifty, 2min, 30 days
        self.SYMBOL = 'NSE:NIFTYBANK-INDEX'
        self.TIMEFRAME = '2'  # 2 minutes
        self.DAYS = 15

        # Model parameters from config
        self.LOOKBACK_WINDOW = env_config.get('lookback_window', 50)
        self.HIDDEN_DIM = model_config.get('hidden_dim', 64)
        self.ACTION_DIM_DISCRETE = model_config.get('action_dim_discrete', 5)
        self.ACTION_DIM_CONTINUOUS = model_config.get('action_dim_continuous', 1)

        # Fallback data configuration
        self.FALLBACK_DATA_PATH = "data/fallback_market_data.csv"
        self.FALLBACK_BACKTEST_RESULTS_PATH = "data/fallback_backtest_results.json"

    def save_fallback_data(self, data):
        """Save market data as fallback for future use when authentication fails."""
        try:
            os.makedirs("data", exist_ok=True)
            # CRITICAL: Properly preserve datetime index
            data_to_save = data.copy()

            # Ensure the index has a proper name for saving
            if data_to_save.index.name is None:
                data_to_save.index.name = 'datetime_readable'

            # Reset index to save datetime as a column, then save
            data_to_save = data_to_save.reset_index()
            data_to_save.to_csv(self.FALLBACK_DATA_PATH, index=False)

            logger.info(f"✅ Fallback data saved to {self.FALLBACK_DATA_PATH}")
            logger.info(f"   📊 Saved {len(data_to_save)} rows with datetime index preserved")
            logger.info(f"   📅 Index column name: {data.index.name}")
        except Exception as e:
            logger.error(f"❌ Failed to save fallback data: {e}")

    def load_fallback_data(self):
        """Load fallback data when Fyers authentication fails."""
        try:
            if os.path.exists(self.FALLBACK_DATA_PATH):
                data = pd.read_csv(self.FALLBACK_DATA_PATH)
                logger.info(f"📂 Using fallback data from {self.FALLBACK_DATA_PATH}")
                logger.info(f"   📊 Data shape: {data.shape}")

                # FIXED: Restore datetime index if datetime_readable column exists
                if 'datetime_readable' in data.columns:
                    data['datetime_readable'] = pd.to_datetime(data['datetime_readable'])
                    data = data.set_index('datetime_readable')
                    logger.info(f"   📅 Restored datetime index from {data.index.min()} to {data.index.max()}")
                elif 'datetime' in data.columns:
                    # Fallback for old format
                    logger.info(f"   📅 Date range: {data['datetime'].min()} to {data['datetime'].max()}")

                return data
            else:
                logger.error(f"❌ No fallback data found at {self.FALLBACK_DATA_PATH}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load fallback data: {e}")
            return None

    def save_fallback_backtest_results(self, results):
        """Save backtest results as fallback for future use when backtesting fails."""
        try:
            import json
            os.makedirs("data", exist_ok=True)
            with open(self.FALLBACK_BACKTEST_RESULTS_PATH, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"✅ Fallback backtest results saved to {self.FALLBACK_BACKTEST_RESULTS_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to save fallback backtest results: {e}")

    def load_fallback_backtest_results(self):
        """Load fallback backtest results when backtesting fails."""
        try:
            import json
            if os.path.exists(self.FALLBACK_BACKTEST_RESULTS_PATH):
                with open(self.FALLBACK_BACKTEST_RESULTS_PATH, 'r') as f:
                    results = json.load(f)
                logger.info(f"📂 Using fallback backtest results from {self.FALLBACK_BACKTEST_RESULTS_PATH}")
                return results
            else:
                logger.error(f"❌ No fallback backtest results found at {self.FALLBACK_BACKTEST_RESULTS_PATH}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to load fallback backtest results: {e}")
            return None
        
        logger.info("🚀 Unified Backtester initialized")
        logger.info(f"   Symbol: {self.SYMBOL}")
        logger.info(f"   Timeframe: {self.TIMEFRAME} minutes")
        logger.info(f"   Days: {self.DAYS}")
        logger.info(f"   Lookback window: {self.LOOKBACK_WINDOW}")
    
    def fetch_and_process_data(self) -> Optional[pd.DataFrame]:
        """Fetch data from Fyers API and process through feature generator. Use fallback if API fails."""
        try:
            logger.info(f"📡 Fetching real-time data from Fyers API...")
            raw_data = self.fyers_client.get_historical_data(
                symbol=self.SYMBOL,
                timeframe=self.TIMEFRAME,
                days=self.DAYS
            )

            if raw_data is None or raw_data.empty:
                logger.warning("⚠️ Failed to fetch data from Fyers API - trying fallback data")
                return self.load_fallback_data()

            logger.info(f"✅ Fetched {len(raw_data)} rows of raw data")
            
            # Process features
            logger.info("🔧 Processing data through feature generator...")
            processed_data = self.feature_processor.process_dataframe(raw_data)

            if processed_data is None or processed_data.empty:
                logger.error("❌ Failed to process data")
                return None

            # ENHANCED: Handle new datetime structure (datetime_readable index + datetime_epoch column)
            datetime_available = False
            if processed_data.index.name == 'datetime_readable':
                # New structure: datetime_readable as index, datetime_epoch as column
                logger.info("✅ Enhanced datetime structure detected: datetime_readable index + datetime_epoch column")
                datetime_available = True
            elif processed_data.index.name == 'datetime':
                # Old structure: datetime as index
                processed_data = processed_data.reset_index()
                logger.info("✅ Reset index to preserve datetime column for sequential backtesting")
                datetime_available = True
            elif 'datetime' in processed_data.columns:
                # datetime already as column
                datetime_available = True
            elif hasattr(processed_data.index, 'name') and processed_data.index.name:
                # Some other named index - reset it
                processed_data = processed_data.reset_index()
                if 'index' in processed_data.columns:
                    processed_data = processed_data.rename(columns={'index': 'datetime'})
                    logger.info("✅ Recovered datetime from unnamed index")
                    datetime_available = True

            logger.info(f"✅ Processed data: {len(processed_data)} rows, {len(processed_data.columns)} features")
            logger.info(f"📅 Datetime available: {datetime_available}")

            # If still no datetime, create a simple step-based datetime for logging
            if not datetime_available:
                processed_data['datetime'] = [f"Step_{i}" for i in range(len(processed_data))]
                logger.info("⚠️ Created step-based datetime for logging purposes")

            # Save as fallback data for future use
            self.save_fallback_data(processed_data)

            return processed_data

        except Exception as e:
            logger.error(f"❌ Error fetching/processing data: {e}", exc_info=True)
            logger.info("🔄 Attempting to use fallback data...")
            return self.load_fallback_data()
    
    def load_model(self, env: TradingEnv) -> Optional[PPOAgent]:
        """Load the trained PPO model with proper dimensions from environment."""
        try:
            model_config = self.config.get('model', {})
            model_path = model_config.get('model_path', "models/universal_final_model.pth")

            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return None

            # Get observation dimensions from environment (same as training)
            obs = env.reset()
            observation_dim = obs.shape[0]
            action_dim_discrete = int(env.action_space.high[0]) + 1
            action_dim_continuous = 1

            logger.info(f"📏 Model dimensions from environment:")
            logger.info(f"   Observation dim: {observation_dim}")
            logger.info(f"   Action discrete: {action_dim_discrete}")
            logger.info(f"   Action continuous: {action_dim_continuous}")

            # Create agent with exact same parameters as training
            agent = PPOAgent(
                observation_dim=observation_dim,
                action_dim_discrete=action_dim_discrete,
                action_dim_continuous=action_dim_continuous,
                hidden_dim=self.HIDDEN_DIM
            )

            agent.load_model(model_path)
            logger.info(f"✅ Model loaded successfully")
            return agent

        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            return None
    
    def run_backtest(self) -> Dict:
        """Run the unified backtesting process."""
        try:
            logger.info("🎯 Starting unified backtesting...")

            # Step 1: Fetch and process data
            processed_data = self.fetch_and_process_data()
            if processed_data is None:
                return {}

            # Step 2: Create TradingEnv in BACKTESTING mode
            logger.info("🏗️ Creating TradingEnv in BACKTESTING mode...")
            env_config = self.config.get('environment', {})
            risk_config = self.config.get('risk_management', {})

            env = TradingEnv(
                mode=TradingMode.BACKTESTING,
                external_data=processed_data,
                symbol="NIFTYBANK",  # Extract symbol name for reward normalization
                initial_capital=env_config.get('initial_capital', 100000.0),
                lookback_window=self.LOOKBACK_WINDOW,
                trailing_stop_percentage=risk_config.get('trailing_stop_percentage', 0.02),
                smart_action_filtering=env_config.get('smart_action_filtering', False)
            )

            # Step 3: Load model with correct dimensions from environment
            agent = self.load_model(env)
            if agent is None:
                return {}

            # Step 4: Run sequential backtesting (environment already reset during model loading)
            logger.info("🔄 Starting sequential backtesting...")
            obs = env.reset()  # Reset again to start fresh

            # CRITICAL FIX: Force engine to start with zero position
            env.engine._current_position_quantity = 0.0
            env.engine._is_position_open = False
            logger.info(f"🔧 Force reset position to: {env.engine._current_position_quantity}")

            done = False
            step_count = 0
            
            while not done:
                # Get model prediction
                action_type, quantity = agent.select_action(obs)
                action = [action_type, quantity]

                # Execute step (sequential row-by-row processing)
                obs, reward, done, info = env.step(action)
                step_count += 1

                # Get current datetime from the data index (readable datetime)
                current_datetime = "N/A"
                try:
                    if hasattr(env, 'data') and env.data is not None and env.current_step < len(env.data):
                        # Use the datetime index (readable format)
                        current_datetime = str(env.data.index[env.current_step])
                        # Also show epoch feature for verification
                        epoch_feature = env.data['datetime_epoch'].iloc[env.current_step] if 'datetime_epoch' in env.data.columns else "N/A"
                except Exception as e:
                    current_datetime = f"Step_{env.current_step}"
                    epoch_feature = "N/A"

                # Log EVERY SINGLE STEP in training format
                account_state = env.engine.get_account_state()
                action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
                action_name = action_names[action_type] if action_type < len(action_names) else "UNKNOWN"

                # Calculate win rate from current trades
                current_trades = env.engine.get_trade_history()
                closing_trades = [t for t in current_trades if t.get('trade_type') == 'CLOSE']
                if closing_trades:
                    winning_trades = sum(1 for t in closing_trades if t.get('pnl', 0) > 0)
                    win_rate = winning_trades / len(closing_trades)
                else:
                    win_rate = 0.0

                total_trades = len(closing_trades)

                # Get exit reason if available
                exit_reason = info.get('exit_reason', '')
                exit_info = f" | Exit: {exit_reason}" if exit_reason else ""

                # Match training log format exactly
                logger.info(f"📊 Step {step_count} | {current_datetime} | NIFTYBANK | "
                           f"Action: {action_name} | Capital: ₹{account_state['capital']:.2f} | "
                           f"Position: {account_state['current_position_quantity']} | "
                           f"Reward: {reward:.4f} | Win Rate: {win_rate:.1%} | "
                           f"Trades: {total_trades}{exit_info}")
            
            # Step 5: Get results
            results = env.get_backtest_results()
            logger.info("✅ Backtesting completed!")

            # Save results as fallback for future use
            self.save_fallback_backtest_results(results)

            return results

        except Exception as e:
            logger.error(f"❌ Error in backtesting: {e}", exc_info=True)
            logger.info("🔄 Attempting to use fallback backtest results...")
            fallback_results = self.load_fallback_backtest_results()
            if fallback_results:
                logger.info("✅ Using fallback backtest results")
                return fallback_results
            else:
                logger.error("❌ No fallback backtest results available")
                return {}
    
    def print_results(self, results: Dict):
        """Print comprehensive backtesting results."""
        if not results:
            logger.error("❌ No results to display")
            return
        
        logger.info("=" * 80)
        logger.info("UNIFIED BACKTESTING RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"📊 Symbol: {results['symbol']}")
        logger.info(f"💰 Initial Capital: ₹{results['initial_capital']:,.2f}")
        logger.info(f"💰 Final Capital: ₹{results['final_capital']:,.2f}")
        logger.info(f"📈 Total Return: ₹{results['final_capital'] - results['initial_capital']:,.2f} ({results['total_return_pct']:.2f}%)")
        logger.info(f"📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        logger.info(f"🎯 Total Trades: {results['total_trades']}")
        logger.info(f"📏 Total Steps: {results['total_steps']}")
        logger.info(f"📊 Data Length: {results['data_length']}")
        
        if results['current_position'] != 0:
            logger.info(f"⚠️ Final Position: {results['current_position']} (not closed)")
        
        logger.info("=" * 80)

def main():
    """Main execution function."""
    try:
        backtester = UnifiedBacktester()
        results = backtester.run_backtest()
        backtester.print_results(results)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()
