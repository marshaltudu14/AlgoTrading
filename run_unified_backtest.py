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
from src.config.config import INITIAL_CAPITAL, RISK_REWARD_CONFIG, MODEL_CONFIG

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

class UnifiedBacktester:
    """Unified backtesting system using enhanced TradingEnv."""
    
    def __init__(self):
        self.fyers_client = FyersClient()
        self.feature_processor = DynamicFileProcessor()
        
        # Hardcoded configuration - Bank Nifty, 2min, 30 days
        self.SYMBOL = 'NSE:NIFTYBANK-INDEX'
        self.TIMEFRAME = '2'  # 2 minutes
        self.DAYS = 30

        # Model parameters from shared config
        self.LOOKBACK_WINDOW = MODEL_CONFIG['lookback_window']
        self.HIDDEN_DIM = MODEL_CONFIG['hidden_dim']
        self.ACTION_DIM_DISCRETE = MODEL_CONFIG['action_dim_discrete']
        self.ACTION_DIM_CONTINUOUS = MODEL_CONFIG['action_dim_continuous']
        
        logger.info("🚀 Unified Backtester initialized")
        logger.info(f"   Symbol: {self.SYMBOL}")
        logger.info(f"   Timeframe: {self.TIMEFRAME} minutes")
        logger.info(f"   Days: {self.DAYS}")
        logger.info(f"   Lookback window: {self.LOOKBACK_WINDOW}")
    
    def fetch_and_process_data(self) -> Optional[pd.DataFrame]:
        """Fetch data from Fyers API and process through feature generator."""
        try:
            logger.info(f"📡 Fetching real-time data from Fyers API...")
            raw_data = self.fyers_client.get_historical_data(
                symbol=self.SYMBOL,
                timeframe=self.TIMEFRAME,
                days=self.DAYS
            )
            
            if raw_data is None or raw_data.empty:
                logger.error("❌ Failed to fetch data from Fyers API")
                return None
                
            logger.info(f"✅ Fetched {len(raw_data)} rows of raw data")
            
            # Process features
            logger.info("🔧 Processing data through feature generator...")
            processed_data = self.feature_processor.process_dataframe(raw_data)
            
            if processed_data is None or processed_data.empty:
                logger.error("❌ Failed to process data")
                return None
                
            logger.info(f"✅ Processed data: {len(processed_data)} rows, {len(processed_data.columns)} features")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error fetching/processing data: {e}", exc_info=True)
            return None
    
    def load_model(self, env: TradingEnv) -> Optional[PPOAgent]:
        """Load the trained PPO model with proper dimensions from environment."""
        try:
            model_path = "models/universal_final_model.pth"
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
            env = TradingEnv(
                mode=TradingMode.BACKTESTING,
                external_data=processed_data,
                initial_capital=INITIAL_CAPITAL,
                lookback_window=self.LOOKBACK_WINDOW,
                trailing_stop_percentage=RISK_REWARD_CONFIG['trailing_stop_percentage']
            )

            # Step 3: Load model with correct dimensions from environment
            agent = self.load_model(env)
            if agent is None:
                return {}

            # Step 4: Run sequential backtesting (environment already reset during model loading)
            logger.info("🔄 Starting sequential backtesting...")
            obs = env.reset()  # Reset again to start fresh
            done = False
            step_count = 0
            
            while not done:
                # Get model prediction
                action_type, quantity = agent.select_action(obs)
                action = [action_type, quantity]
                
                # Execute step
                obs, reward, done, info = env.step(action)
                step_count += 1
                
                # Log progress every 500 steps
                if step_count % 500 == 0:
                    account_state = env.engine.get_account_state()
                    logger.info(f"📊 Step {step_count}: Capital: ₹{account_state['capital']:.2f}, Position: {account_state['current_position_quantity']}")
            
            # Step 5: Get results
            results = env.get_backtest_results()
            logger.info("✅ Backtesting completed!")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}", exc_info=True)
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
