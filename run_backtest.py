#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Trained PPO Models.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import torch
from datetime import datetime
import glob
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.ppo_agent import PPOAgent
from src.backtesting.environment import TradingEnv
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
        logging.FileHandler('backtest_log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    """Backtesting system for trained PPO models."""

    def __init__(self, model_dir: str, processed_data_dir: str = "data/processed", raw_data_dir: str = "data/raw"):
        self.model_dir = model_dir
        self.processed_data_dir = processed_data_dir
        self.raw_data_dir = raw_data_dir
        self.data_processor = DynamicFileProcessor()
        self.instruments = load_instruments('config/instruments.yaml')
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def discover_raw_data_files(self) -> List[str]:
        """Discover all raw data files in the raw data directory."""
        patterns = ['*.csv', '*.parquet', '*.feather']
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.raw_data_dir, pattern)))
        logger.info(f"Discovered {len(files)} raw data files.")
        return files

    def process_raw_data(self, file_path: str) -> Optional[str]:
        """Process a single raw data file and save to processed directory."""
        try:
            filename = os.path.basename(file_path)
            symbol = os.path.splitext(filename)[0]
            logger.info(f"Processing raw data for {symbol}")

            if file_path.endswith('.csv'):
                raw_data = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                raw_data = pd.read_parquet(file_path)
            elif file_path.endswith('.feather'):
                raw_data = pd.read_feather(file_path)
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in raw_data.columns for col in required_columns):
                logger.error(f"Missing required columns in {file_path}. Required: {required_columns}")
                return None

            processed_data = self.data_processor.process_dataframe(raw_data)
            if processed_data.empty:
                logger.error(f"Data processing failed for {symbol}")
                return None

            processed_file_path = os.path.join(self.processed_data_dir, f"{symbol}.csv")
            processed_data.to_csv(processed_file_path, index=False)
            logger.info(f"Processed data saved to {processed_file_path} ({len(processed_data)} rows)")
            return processed_file_path
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None

    def load_trained_model(self, symbol: str, env: TradingEnv) -> Optional[PPOAgent]:
        """Load the trained PPO model for backtesting."""
        model_path = os.path.join(self.model_dir, f"{symbol}_ppo_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"Model not found for {symbol} at {model_path}")
            return None

        try:
            logger.info(f"Loading model from {model_path}")
            obs = env.reset()
            observation_dim = obs.shape[0]
            action_dim_discrete = int(env.action_space.high[0]) + 1
            action_dim_continuous = 1

            agent = PPOAgent(
                observation_dim=observation_dim,
                action_dim_discrete=action_dim_discrete,
                action_dim_continuous=action_dim_continuous,
                hidden_dim=64
            )
            # The load_model method in PPOAgent needs to handle loading the state dict
            if hasattr(agent, 'load_model'):
                 agent.load_model(model_path)
            else:
                 agent.load_state_dict(torch.load(model_path))

            logger.info(f"Successfully loaded model for {symbol}")
            return agent
        except Exception as e:
            logger.error(f"Error loading model for {symbol}: {e}", exc_info=True)
            return None

    def run_backtest(self, symbol: str, initial_capital: float = 100000.0) -> Optional[Dict]:
        """Run backtest for a single symbol."""
        try:
            logger.info(f"Starting backtest for {symbol}")
            
            data_loader = DataLoader(self.processed_data_dir)
            
            env = TradingEnv(
                data_loader=data_loader,
                symbol=symbol,
                initial_capital=initial_capital,
                lookback_window=20,
                episode_length=None,  # Use full dataset
                use_streaming=False
            )
            
            agent = self.load_trained_model(symbol, env)
            if agent is None:
                return None

            observation = env.reset()
            done = False
            step_count = 0
            
            logger.info(f"Running backtest on {len(env.data)} data points")
            
            while not done:
                action = agent.select_action(observation)
                observation, reward, done, info = env.step(action)
                step_count += 1
                
                if step_count % 1000 == 0:
                    logger.info(f"Processed {step_count} steps...")
            
            final_account = env.engine.get_account_state()
            trade_history = env.engine.get_trade_history()
            
            metrics = calculate_comprehensive_metrics(
                trade_history=trade_history,
                capital_history=env.equity_history,
                initial_capital=initial_capital,
                total_episodes=1,
                total_reward=final_account['capital'] - initial_capital
            )
            
            backtest_results = {
                'symbol': symbol,
                'initial_capital': initial_capital,
                'final_capital': final_account['capital'],
                'total_pnl': final_account['capital'] - initial_capital,
                'data_points_processed': step_count,
                'trades_executed': len(trade_history),
                'metrics': metrics
            }
            
            logger.info(f"Backtest completed for {symbol}")
            logger.info(f"Final Capital: ₹{final_account['capital']:,.2f}")
            logger.info(f"Total P&L: ₹{backtest_results['total_pnl']:,.2f}")
            logger.info(f"Total Trades: {len(trade_history)}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running backtest for {symbol}: {e}", exc_info=True)
            return None

    def generate_report(self, results: List[Dict], output_file: str = None):
        """Generate a comprehensive backtest report."""
        if not results:
            logger.warning("No backtest results to report.")
            return

        report_lines = [
            "=" * 80,
            "PPO Backtest Report",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Symbols Tested: {len(results)}",
            ""
        ]

        for result in results:
            metrics = result['metrics']
            report_lines.extend([
                f"--- Symbol: {result['symbol']} ---",
                f"  Initial Capital: ₹{result['initial_capital']:,.2f}",
                f"  Final Capital:   ₹{result['final_capital']:,.2f}",
                f"  Total P&L:       ₹{result['total_pnl']:,.2f}",
                f"  Total Return:    {metrics.get('total_return_percentage', 0):.2f}%",
                f"  Total Trades:    {metrics.get('total_trades', 0)}",
                f"  Win Rate:        {metrics.get('win_rate', 0) * 100:.2f}%",
                f"  Profit Factor:   {metrics.get('profit_factor', 0):.2f}",
                f"  Max Drawdown:    {metrics.get('max_drawdown_percentage', 0):.2f}%",
                f"  Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):.2f}",
                ""
            ])

        report_content = "\n".join(report_lines)
        print(report_content)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Backtest report saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Backtest trained PPO trading models.")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backtest.")
    parser.add_argument("--raw-data-dir", default="data/raw", help="Raw data directory.")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Processed data directory.")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained models.")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital for backtesting.")
    parser.add_argument("--process-data", action="store_true", help="Process raw data before backtesting.")
    parser.add_argument("--report-file", help="Save backtest report to a file.")
    
    args = parser.parse_args()

    runner = BacktestRunner(
        model_dir=args.model_dir,
        processed_data_dir=args.processed_data_dir,
        raw_data_dir=args.raw_data_dir
    )

    if args.process_data:
        logger.info("Processing raw data files...")
        raw_files = runner.discover_raw_data_files()
        for raw_file in raw_files:
            runner.process_raw_data(raw_file)

    symbols_to_backtest = args.symbols
    if not symbols_to_backtest:
        # Discover from models if not specified
        model_files = glob.glob(os.path.join(args.model_dir, "*_ppo_model.pth"))
        symbols_to_backtest = [os.path.basename(f).replace("_ppo_model.pth", "") for f in model_files]

    if not symbols_to_backtest:
        logger.error("No symbols found to backtest. Please train a model first or specify symbols.")
        return

    logger.info(f"Running backtests for: {', '.join(symbols_to_backtest)}")
    
    results = []
    for symbol in symbols_to_backtest:
        processed_file = os.path.join(args.processed_data_dir, f"{symbol}.csv")
        if not os.path.exists(processed_file):
            logger.warning(f"Processed data not found for {symbol}, skipping backtest.")
            continue
        
        result = runner.run_backtest(symbol, args.initial_capital)
        if result:
            results.append(result)

    if results:
        runner.generate_report(results, args.report_file)
        logger.info("All backtests completed.")
    else:
        logger.error("No backtests were successfully completed.")

if __name__ == "__main__":
    main()