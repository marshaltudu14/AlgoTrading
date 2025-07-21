#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Trained RL Models
Processes raw data and backtests using saved models from training sequence.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.moe_agent import MoEAgent
from src.backtesting.environment import TradingEnv
from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader
from src.data_processing.feature_generator import DynamicFileProcessor
from src.utils.instrument_loader import load_instruments
from src.utils.metrics import calculate_comprehensive_metrics
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    """Comprehensive backtesting system for trained RL models."""
    
    def __init__(self, model_path: str, raw_data_dir: str = "data/raw", 
                 processed_data_dir: str = "data/processed"):
        self.model_path = model_path
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.data_processor = DynamicFileProcessor()
        self.instruments = load_instruments('config/instruments.yaml')
        
        # Ensure processed data directory exists
        os.makedirs(processed_data_dir, exist_ok=True)
        
    def discover_raw_data_files(self) -> List[str]:
        """Discover all raw data files in the raw data directory."""
        patterns = ['*.csv', '*.parquet', '*.feather']
        files = []
        
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.raw_data_dir, pattern)))
            
        logger.info(f"Discovered {len(files)} raw data files")
        return files
    
    def process_raw_data(self, file_path: str) -> Optional[str]:
        """Process a single raw data file and save to processed directory."""
        try:
            # Extract symbol from filename
            filename = os.path.basename(file_path)
            symbol = filename.split('.')[0]  # Remove extension
            
            logger.info(f"Processing raw data for {symbol}")
            
            # Load raw data
            if file_path.endswith('.csv'):
                raw_data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                raw_data = pd.read_parquet(file_path)
            elif file_path.endswith('.feather'):
                raw_data = pd.read_feather(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None
                
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in raw_data.columns for col in required_columns):
                logger.error(f"Missing required columns in {file_path}. Required: {required_columns}")
                return None
                
            # Process the data using the feature generator
            processed_data = self.data_processor.process_dataframe(raw_data)
            
            if processed_data.empty:
                logger.error(f"Data processing failed for {symbol}")
                return None
                
            # Save processed data
            processed_file_path = os.path.join(self.processed_data_dir, f"{symbol}.csv")
            processed_data.to_csv(processed_file_path, index=False)
            
            logger.info(f"Processed data saved to {processed_file_path} ({len(processed_data)} rows)")
            return processed_file_path
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def load_trained_model(self, symbol: str):
        """Load the trained model for backtesting."""
        try:
            # First priority: Check for autonomous champion agent
            autonomous_path = f"models/autonomous_agents/{symbol}_autonomous_final.pth"
            if os.path.exists(autonomous_path):
                logger.info(f"Loading autonomous champion agent from {autonomous_path}")
                from src.training.autonomous_trainer import load_champion_agent
                return load_champion_agent(autonomous_path)

            # Fallback: Try to load other models
            model_paths = [
                f"models/{symbol}_final_model.pth",
                f"models/{symbol}_maml_stage3_final.pth",
                f"models/{symbol}_moe_stage2.pth",
                f"models/{symbol}_ppo_stage1.pth"
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    logger.info(f"Loading model from {path}")
                    
                    # Create agent with same parameters as training
                    agent = MoEAgent(
                        observation_dim=1246,  # Match training configuration
                        action_dim=2,
                        num_experts=3,
                        expert_hidden_dim=64,
                        gating_hidden_dim=32
                    )
                    
                    agent.load_model(path)
                    logger.info(f"Successfully loaded model for {symbol}")
                    return agent
                    
            logger.error(f"No trained model found for {symbol}. Checked paths: {model_paths}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}")
            return None
    
    def run_backtest(self, symbol: str, processed_data_path: str, 
                    initial_capital: float = 100000.0) -> Optional[Dict]:
        """Run backtest for a single symbol."""
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            # Load trained model
            agent = self.load_trained_model(symbol)
            if agent is None:
                return None
                
            # Load processed data
            data_loader = DataLoader(self.processed_data_dir)
            
            # Create trading environment
            env = TradingEnv(
                data_loader=data_loader,
                symbol=symbol,
                initial_capital=initial_capital,
                lookback_window=20,
                episode_length=None,  # Use full dataset
                use_streaming=False
            )
            
            # Run backtest
            observation = env.reset()
            done = False
            step_count = 0
            
            logger.info(f"Running backtest on {len(env.data)} data points")
            
            while not done:
                # Get action from trained model
                action = agent.select_action(observation, training=False)  # Inference mode
                
                # Execute action in environment
                observation, reward, done, info = env.step(action)
                step_count += 1
                
                if step_count % 1000 == 0:
                    logger.info(f"Processed {step_count} steps...")
            
            # Get final results
            final_account = env.engine.get_account_state()
            trade_history = env.engine.get_trade_history()
            
            # Calculate comprehensive metrics
            metrics = calculate_comprehensive_metrics(
                trade_history=trade_history,
                capital_history=env.equity_history,
                initial_capital=initial_capital,
                total_episodes=1,
                total_reward=final_account['capital'] - initial_capital
            )
            
            # Add backtest-specific information
            backtest_results = {
                'symbol': symbol,
                'initial_capital': initial_capital,
                'final_capital': final_account['capital'],
                'total_pnl': final_account['capital'] - initial_capital,
                'data_points_processed': step_count,
                'trades_executed': len(trade_history),
                'metrics': metrics,
                'trade_history': trade_history,
                'final_account': final_account
            }
            
            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"Final Capital: ₹{final_account['capital']:,.2f}")
            logger.info(f"Total P&L: ₹{backtest_results['total_pnl']:,.2f}")
            logger.info(f"Total Trades: {len(trade_history)}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}")
            return None
    
    def generate_backtest_report(self, results: List[Dict], output_file: str = None):
        """Generate comprehensive backtest report."""
        if not results:
            logger.warning("No backtest results to report")
            return
            
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("🎯 COMPREHENSIVE BACKTEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Symbols Tested: {len(results)}")
        report_lines.append("")
        
        total_initial = sum(r['initial_capital'] for r in results)
        total_final = sum(r['final_capital'] for r in results)
        total_pnl = total_final - total_initial
        
        report_lines.append("📊 PORTFOLIO SUMMARY:")
        report_lines.append(f"   Total Initial Capital: ₹{total_initial:,.2f}")
        report_lines.append(f"   Total Final Capital: ₹{total_final:,.2f}")
        report_lines.append(f"   Total P&L: ₹{total_pnl:,.2f}")
        report_lines.append(f"   Total Return: {(total_pnl/total_initial)*100:.2f}%")
        report_lines.append("")
        
        # Individual symbol results
        for result in results:
            metrics = result['metrics']
            report_lines.append(f"📈 {result['symbol'].upper()}:")
            report_lines.append(f"   Final Capital: ₹{result['final_capital']:,.2f}")
            report_lines.append(f"   Total P&L: ₹{result['total_pnl']:,.2f}")
            report_lines.append(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            report_lines.append(f"   Total Trades: {metrics.get('total_trades', 0)}")
            report_lines.append(f"   Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report_lines.append(f"   Max Drawdown: {metrics.get('max_drawdown_percentage', 0):.2f}%")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Print to console
        print(report_content)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Backtest report saved to {output_file}")


def main():
    """Main function for running backtests."""
    parser = argparse.ArgumentParser(description="Backtest trained RL trading models")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backtest")
    parser.add_argument("--raw-data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital for backtesting")
    parser.add_argument("--process-data", action="store_true", help="Process raw data before backtesting")
    parser.add_argument("--report-file", help="Save backtest report to file")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained models")

    args = parser.parse_args()

    # Initialize backtest runner
    runner = BacktestRunner(
        model_path=args.model_dir,
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir
    )

    # Process raw data if requested
    processed_files = []
    if args.process_data:
        logger.info("Processing raw data files...")
        raw_files = runner.discover_raw_data_files()

        for raw_file in raw_files:
            processed_file = runner.process_raw_data(raw_file)
            if processed_file:
                processed_files.append(processed_file)

        logger.info(f"Successfully processed {len(processed_files)} files")

    # Determine symbols to backtest
    if args.symbols:
        symbols = args.symbols
    else:
        # Auto-discover symbols from processed data or models
        if processed_files:
            symbols = [os.path.basename(f).split('.')[0] for f in processed_files]
        else:
            # Look for existing processed files
            processed_pattern = os.path.join(args.processed_data_dir, "*.csv")
            processed_files = glob.glob(processed_pattern)
            symbols = [os.path.basename(f).split('.')[0] for f in processed_files]

    if not symbols:
        logger.error("No symbols found for backtesting. Use --process-data or specify --symbols")
        return

    logger.info(f"Running backtests for {len(symbols)} symbols: {symbols}")

    # Run backtests
    results = []
    for symbol in symbols:
        processed_file = os.path.join(args.processed_data_dir, f"{symbol}.csv")

        if not os.path.exists(processed_file):
            logger.warning(f"Processed data not found for {symbol}: {processed_file}")
            continue

        result = runner.run_backtest(symbol, processed_file, args.initial_capital)
        if result:
            results.append(result)

    # Generate report
    if results:
        runner.generate_backtest_report(results, args.report_file)
        logger.info(f"Backtest completed successfully for {len(results)} symbols")
    else:
        logger.error("No successful backtests completed")


if __name__ == "__main__":
    main()
