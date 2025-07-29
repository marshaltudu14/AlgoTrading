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
import torch
from datetime import datetime, timedelta
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
from src.trading.fyers_client import FyersClient
import src.config as config

# Configure logging - single overwritable file for easier tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_errors_warnings.txt', mode='w'),  # Overwrite mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BacktestRunner:
    """Comprehensive backtesting system for trained RL models with Fyers API integration."""

    # Hardcoded parameters for Nifty backtesting (can be changed as needed)
    NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"
    TIMEFRAME = "5"  # 5-minute intervals
    LOOKBACK_DAYS = 30  # Past 30 days

    def __init__(self, model_path: str, raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed", use_fyers_api: bool = True):
        self.model_path = model_path
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.use_fyers_api = use_fyers_api
        self.data_processor = DynamicFileProcessor()
        self.instruments = load_instruments('config/instruments.yaml')

        # Initialize Fyers client if using API
        if self.use_fyers_api:
            try:
                logger.info("Initializing Fyers API client...")
                self.fyers_client = FyersClient()
                logger.info("Fyers API client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Fyers client: {e}")
                self.fyers_client = None

    def fetch_nifty_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch Nifty index data from Fyers API for the past 30 days in 5-minute intervals.

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self.fyers_client:
            logger.error("Fyers client not initialized")
            return None

        try:
            logger.info(f"Fetching {self.NIFTY_SYMBOL} data for past {self.LOOKBACK_DAYS} days...")
            logger.info(f"Timeframe: {self.TIMEFRAME} minutes")

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.LOOKBACK_DAYS)

            logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

            # Fetch data using Fyers client
            data = self.fyers_client.get_historical_data(
                symbol=self.NIFTY_SYMBOL,
                timeframe=self.TIMEFRAME,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )

            if data is not None and not data.empty:
                logger.info(f"Successfully fetched {len(data)} data points")
                logger.info(f"Data columns: {list(data.columns)}")
                logger.info(f"Date range in data: {data.index[0]} to {data.index[-1]}")
                return data
            else:
                logger.error("No data received from Fyers API")
                return None

        except Exception as e:
            logger.error(f"Error fetching data from Fyers API: {e}")
            return None

    def run_backtest_with_fyers_data(self, initial_capital: float = 100000.0) -> Optional[Dict]:
        """
        Run backtest using Nifty data fetched from Fyers API.

        Args:
            initial_capital: Starting capital for backtest

        Returns:
            Dictionary with backtest results or None if failed
        """
        try:
            logger.info("Starting Nifty backtest with Fyers API data...")

            # Fetch Nifty data from Fyers API
            raw_data = self.fetch_nifty_data()
            if raw_data is None:
                logger.error("Failed to fetch data from Fyers API")
                return None

            # Process the raw data to generate features
            logger.info("Processing raw data to generate features...")

            # Prepare data for processing - add datetime column if not present
            if 'datetime' not in raw_data.columns:
                raw_data.reset_index(inplace=True)
                if raw_data.index.name == 'datetime':
                    raw_data['datetime'] = raw_data.index
                else:
                    raw_data['datetime'] = raw_data.index

            # Process the data using DynamicFileProcessor
            processed_data = self.data_processor.process_dataframe(raw_data)

            if processed_data is None or processed_data.empty:
                logger.error("Failed to process raw data")
                return None

            logger.info(f"Data processed successfully!")
            logger.info(f"Processed data shape: {processed_data.shape}")

            # Load the universal trained model
            universal_model_path = os.path.join(self.model_path, "universal_final_model.pth")

            if not os.path.exists(universal_model_path):
                logger.error(f"Universal model not found at: {universal_model_path}")
                return None

            try:
                logger.info(f"Loading universal model from: {universal_model_path}")

                # Create a new MoEAgent and load the state dict
                agent = MoEAgent(
                    input_dim=63,  # Based on processed data columns
                    action_dim=3,  # Typical action space (buy, sell, hold)
                    num_experts=4,
                    hidden_dim=256
                )

                # Load the saved state dict
                checkpoint = torch.load(universal_model_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent.load_state_dict(checkpoint)

                agent.eval()  # Set to evaluation mode
                logger.info(f"Successfully loaded universal model!")

            except Exception as e:
                logger.error(f"Failed to load universal model: {e}")
                return None

            logger.info(f"Using processed data with {len(processed_data)} rows and {len(processed_data.columns)} columns")

            # Create data loader with the processed data directly
            # For backtesting, we'll pass the DataFrame directly to the environment
            data_loader = None  # We'll use processed_data directly

            # Create trading environment
            env = TradingEnv(
                data_loader=data_loader,
                symbol="NIFTY",  # Use generic symbol for environment
                initial_capital=int(initial_capital),  # Ensure integer for production
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
                'symbol': self.NIFTY_SYMBOL,
                'timeframe': self.TIMEFRAME,
                'lookback_days': self.LOOKBACK_DAYS,
                'initial_capital': int(initial_capital),
                'final_capital': int(final_account['capital']),
                'total_pnl': int(final_account['capital'] - initial_capital),
                'data_points_processed': step_count,
                'trades_executed': len(trade_history),
                'metrics': metrics,
                'trade_history': trade_history,
                'final_account': final_account
            }

            logger.info(f"Backtest completed for {self.NIFTY_SYMBOL}")
            logger.info(f"Final Capital: ₹{final_account['capital']:,.2f}")
            logger.info(f"Total P&L: ₹{backtest_results['total_pnl']:,.2f}")
            logger.info(f"Total Trades: {len(trade_history)}")

            return backtest_results

        except Exception as e:
            logger.error(f"Error running backtest with Fyers data: {e}")
            return None

    def generate_comprehensive_fyers_report(self, result: Dict):
        """Generate comprehensive report for Fyers API backtest results."""
        if not result:
            logger.warning("No backtest results to report")
            return

        metrics = result['metrics']

        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("🎯 COMPREHENSIVE NIFTY BACKTEST REPORT - FYERS API DATA")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Symbol: {result['symbol']}")
        report_lines.append(f"Timeframe: {result['timeframe']} minutes")
        report_lines.append(f"Lookback Period: {result['lookback_days']} days")
        report_lines.append(f"Data Points Processed: {result['data_points_processed']:,}")
        report_lines.append("")

        # Capital and P&L Summary
        report_lines.append("💰 CAPITAL & P&L SUMMARY:")
        report_lines.append(f"   Initial Capital: ₹{result['initial_capital']:,}")
        report_lines.append(f"   Final Capital: ₹{result['final_capital']:,}")
        report_lines.append(f"   Total P&L: ₹{result['total_pnl']:,}")
        report_lines.append(f"   Total Return: {metrics.get('total_return_percentage', 0):.2f}%")
        report_lines.append("")

        # Trading Statistics
        report_lines.append("📊 TRADING STATISTICS:")
        report_lines.append(f"   Total Trades: {metrics.get('total_trades', 0)}")
        report_lines.append(f"   Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
        report_lines.append(f"   Average P&L per Trade: ₹{metrics.get('avg_pnl_per_trade', 0):,.2f}")
        report_lines.append(f"   Average Winning Trade: ₹{metrics.get('avg_winning_trade', 0):,.2f}")
        report_lines.append(f"   Average Losing Trade: ₹{metrics.get('avg_losing_trade', 0):,.2f}")
        report_lines.append(f"   Largest Winning Trade: ₹{metrics.get('largest_winning_trade', 0):,.2f}")
        report_lines.append(f"   Largest Losing Trade: ₹{metrics.get('largest_losing_trade', 0):,.2f}")
        report_lines.append(f"   Max Consecutive Wins: {metrics.get('max_consecutive_wins', 0)}")
        report_lines.append(f"   Max Consecutive Losses: {metrics.get('max_consecutive_losses', 0)}")
        report_lines.append("")

        # Risk & Performance Metrics
        report_lines.append("📈 RISK & PERFORMANCE METRICS:")
        report_lines.append(f"   Profit Factor: {metrics.get('profit_factor', 0):.3f}")
        report_lines.append(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        report_lines.append(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}")
        report_lines.append(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}")
        report_lines.append(f"   Recovery Factor: {metrics.get('recovery_factor', 0):.3f}")
        report_lines.append(f"   Expectancy: ₹{metrics.get('expectancy', 0):,.2f}")
        report_lines.append(f"   Maximum Drawdown: {metrics.get('max_drawdown_percentage', 0):.2f}%")
        report_lines.append("")

        # Model Performance
        report_lines.append("🤖 MODEL PERFORMANCE:")
        report_lines.append(f"   Total Reward: {metrics.get('total_reward', 0):,.2f}")
        report_lines.append(f"   Average Reward per Episode: {metrics.get('avg_reward_per_episode', 0):,.2f}")
        report_lines.append("")

        # Performance Evaluation
        report_lines.append("🎯 PERFORMANCE EVALUATION:")
        win_rate = metrics.get('win_rate', 0) * 100
        profit_factor = metrics.get('profit_factor', 0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_dd = abs(metrics.get('max_drawdown_percentage', 0))

        if win_rate >= 60 and profit_factor >= 1.5 and sharpe_ratio >= 1.0 and max_dd <= 10:
            performance_rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif win_rate >= 50 and profit_factor >= 1.2 and sharpe_ratio >= 0.5 and max_dd <= 15:
            performance_rating = "GOOD ⭐⭐⭐⭐"
        elif win_rate >= 40 and profit_factor >= 1.0 and max_dd <= 20:
            performance_rating = "AVERAGE ⭐⭐⭐"
        elif profit_factor >= 1.0:
            performance_rating = "BELOW AVERAGE ⭐⭐"
        else:
            performance_rating = "POOR ⭐"

        report_lines.append(f"   Overall Rating: {performance_rating}")
        report_lines.append("")

        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS:")
        if profit_factor < 1.0:
            report_lines.append("   ⚠️  Profit factor below 1.0 - Strategy is losing money")
        if win_rate < 0.4:
            report_lines.append("   ⚠️  Low win rate - Consider improving entry signals")
        if max_dd > 20:
            report_lines.append("   ⚠️  High drawdown - Implement better risk management")
        if sharpe_ratio < 0.5:
            report_lines.append("   ⚠️  Low Sharpe ratio - Risk-adjusted returns need improvement")
        if metrics.get('total_trades', 0) < 10:
            report_lines.append("   ⚠️  Low trade frequency - Consider longer backtest period")

        if profit_factor >= 1.5 and win_rate >= 0.5 and max_dd <= 15:
            report_lines.append("   ✅ Strategy shows promising results for live trading")

        report_lines.append("")
        report_lines.append("=" * 100)

        report_content = "\n".join(report_lines)

        # Print to console
        print(report_content)

        # Save to file
        report_file = f"nifty_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)

        logger.info(f"Comprehensive report saved to: {report_file}")
        return report_file
        
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
    """Main function for running backtests with Fyers API integration."""
    parser = argparse.ArgumentParser(description="Backtest trained RL trading models with Fyers API")
    parser.add_argument("--use-fyers", action="store_true", default=True,
                       help="Use Fyers API for Nifty data (default: True)")
    parser.add_argument("--use-local-data", action="store_true",
                       help="Use local data files instead of Fyers API")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to backtest (for local data mode)")
    parser.add_argument("--raw-data-dir", default="data/raw", help="Raw data directory")
    parser.add_argument("--processed-data-dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Initial capital for backtesting")
    parser.add_argument("--process-data", action="store_true", help="Process raw data before backtesting")
    parser.add_argument("--report-file", help="Save backtest report to file")
    parser.add_argument("--model-dir", default="models", help="Directory containing trained models")

    args = parser.parse_args()

    # Determine mode: Fyers API (default) or local data
    use_fyers_api = not args.use_local_data

    # Initialize backtest runner
    runner = BacktestRunner(
        model_path=args.model_dir,
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        use_fyers_api=use_fyers_api
    )

    if use_fyers_api:
        # Fyers API mode - fetch Nifty data and run backtest
        logger.info("🚀 Starting Nifty backtest with Fyers API data...")
        logger.info(f"Symbol: {runner.NIFTY_SYMBOL}")
        logger.info(f"Timeframe: {runner.TIMEFRAME} minutes")
        logger.info(f"Lookback: {runner.LOOKBACK_DAYS} days")
        logger.info(f"Initial Capital: ₹{args.initial_capital:,.2f}")

        # Run Fyers API backtest
        result = runner.run_backtest_with_fyers_data(args.initial_capital)

        if result:
            # Generate comprehensive report
            runner.generate_comprehensive_fyers_report(result)
            logger.info("✅ Nifty backtest completed successfully!")
        else:
            logger.error("❌ Nifty backtest failed")

    else:
        # Local data mode - original functionality
        logger.info("Using local data files for backtesting...")

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
