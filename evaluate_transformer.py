"""
Evaluation script for the TradingTransformer model.
Evaluates the model on historical data and reports performance metrics.
"""
import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from models.trading_transformer import TradingTransformer, create_trading_transformer
from models.inference import TradingTransformerInference
from envs.trading_env import TradingEnv
from envs.data_fetcher import fetch_candle_data
from data_processing.processor import process_df
from config import (
    INSTRUMENTS, 
    TIMEFRAMES, 
    TRANSFORMER_CONFIG, 
    DEVICE,
    MODEL_PATH,
    INSTRUMENT_IDS,
    TIMEFRAME_IDS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the TradingTransformer model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the trained model"
    )
    
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=list(INSTRUMENTS.keys()),
        help="List of instruments to evaluate on"
    )
    
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=TIMEFRAMES,
        help="List of timeframes to evaluate on"
    )
    
    parser.add_argument(
        "--eval_days",
        type=int,
        default=90,
        help="Number of days to evaluate on"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic predictions"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate performance plots"
    )
    
    return parser.parse_args()


def evaluate_model(
    model_path: str,
    instruments: List[str],
    timeframes: List[int],
    eval_days: int = 90,
    deterministic: bool = True,
    output_dir: str = "evaluation_results",
    plot: bool = False
) -> pd.DataFrame:
    """
    Evaluate the model on historical data.
    
    Args:
        model_path: Path to the trained model
        instruments: List of instruments to evaluate on
        timeframes: List of timeframes to evaluate on
        eval_days: Number of days to evaluate on
        deterministic: Whether to use deterministic predictions
        output_dir: Directory to save evaluation results
        plot: Whether to generate performance plots
        
    Returns:
        DataFrame with evaluation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model inference wrapper
    inference = TradingTransformerInference(model_path=model_path)
    
    # Initialize results
    records = []
    
    # Evaluate on each instrument and timeframe
    for instrument in instruments:
        logger.info(f"Evaluating instrument: {instrument}")
        
        for timeframe in timeframes:
            logger.info(f"  Timeframe: {timeframe}...")
            
            # Create environment
            env = TradingEnv(instrument, timeframe)
            
            # Reset environment
            obs, _ = env.reset()
            
            # Initialize metrics
            done = False
            steps = 0
            actions = []
            capitals = [env.capital]
            positions = [env.position]
            
            # Run episode
            while not done:
                # Get action from model
                features = obs
                instrument_id = INSTRUMENT_IDS[instrument]
                timeframe_id = TIMEFRAME_IDS[timeframe]
                
                # Convert to tensors
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                instrument_id_tensor = torch.tensor([instrument_id], dtype=torch.long).to(DEVICE)
                timeframe_id_tensor = torch.tensor([timeframe_id], dtype=torch.long).to(DEVICE)
                
                # Get action
                action, _ = inference.predict(features, instrument, timeframe, deterministic)
                
                # Take step in environment
                obs, reward, done, _, info = env.step(action)
                
                # Record metrics
                steps += 1
                actions.append(action)
                capitals.append(env.capital)
                positions.append(env.position)
            
            # Calculate metrics
            final_capital = info['capital']
            max_drawdown = info['max_drawdown']
            trade_count = info['trade_count']
            win_rate = info['win_rate']
            
            # Calculate returns
            initial_capital = capitals[0]
            returns = (final_capital - initial_capital) / initial_capital * 100
            
            # Calculate action distribution
            action_counts = np.bincount(actions, minlength=3)
            hold_pct = action_counts[0] / len(actions) * 100
            buy_pct = action_counts[1] / len(actions) * 100
            sell_pct = action_counts[2] / len(actions) * 100
            
            # Log results
            logger.info(
                f"    Done in {steps} steps. "
                f"Final capital: {final_capital:.2f}, "
                f"Return: {returns:.2f}%, "
                f"Max DD: {max_drawdown:.2f}, "
                f"Trades: {trade_count}, "
                f"Win rate: {win_rate:.2f}, "
                f"Actions: Hold={hold_pct:.1f}%, Buy={buy_pct:.1f}%, Sell={sell_pct:.1f}%"
            )
            
            # Add to records
            records.append({
                'instrument': instrument,
                'timeframe': timeframe,
                'final_capital': final_capital,
                'returns': returns,
                'max_drawdown': max_drawdown,
                'trade_count': trade_count,
                'win_rate': win_rate,
                'hold_pct': hold_pct,
                'buy_pct': buy_pct,
                'sell_pct': sell_pct,
                'steps': steps
            })
            
            # Generate plot
            if plot:
                plt.figure(figsize=(12, 8))
                
                # Plot capital
                plt.subplot(2, 1, 1)
                plt.plot(capitals)
                plt.title(f"{instrument} @ {timeframe}min - Capital")
                plt.xlabel("Steps")
                plt.ylabel("Capital")
                plt.grid(True)
                
                # Plot positions
                plt.subplot(2, 1, 2)
                plt.plot(positions)
                plt.title(f"{instrument} @ {timeframe}min - Position")
                plt.xlabel("Steps")
                plt.ylabel("Position")
                plt.yticks([0, 1], ["None", "Long"])
                plt.grid(True)
                
                # Save plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{instrument}_{timeframe}.png"))
                plt.close()
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save results
    df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    
    # Print summary
    logger.info("\nEvaluation summary:")
    logger.info(f"Average return: {df['returns'].mean():.2f}%")
    logger.info(f"Average max drawdown: {df['max_drawdown'].mean():.2f}")
    logger.info(f"Average trade count: {df['trade_count'].mean():.2f}")
    logger.info(f"Average win rate: {df['win_rate'].mean():.2f}")
    logger.info(f"Average hold percentage: {df['hold_pct'].mean():.2f}%")
    logger.info(f"Average buy percentage: {df['buy_pct'].mean():.2f}%")
    logger.info(f"Average sell percentage: {df['sell_pct'].mean():.2f}%")
    
    return df


def main():
    """Main function."""
    args = parse_args()
    
    # Evaluate model
    df = evaluate_model(
        model_path=args.model_path,
        instruments=args.instruments,
        timeframes=args.timeframes,
        eval_days=args.eval_days,
        deterministic=args.deterministic,
        output_dir=args.output_dir,
        plot=args.plot
    )
    
    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()
