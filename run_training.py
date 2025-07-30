#!/usr/bin/env python3
"""
Script for running PPO training.
"""

import os
import sys
import argparse
import logging
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.ppo_agent import PPOAgent
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.config.config import INITIAL_CAPITAL
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_available_symbols(data_dir: str = "data/final") -> List[str]:
    """Get list of available trading symbols from data directory."""
    symbols = []

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist")
        return symbols

    # Scan for all CSV files with features_ prefix
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and filename.startswith('features_'):
            symbol = filename.replace('features_', '').replace('.csv', '')
            symbols.append(symbol)

    # Also check parquet directory if it exists
    parquet_dir = os.path.join(data_dir, "parquet")
    if os.path.exists(parquet_dir):
        for filename in os.listdir(parquet_dir):
            if filename.endswith('.parquet'):
                symbol = filename.replace('.parquet', '')
                if symbol not in symbols:
                    symbols.append(symbol)

    symbols = sorted(list(set(symbols)))
    logger.info(f"🔍 Found {len(symbols)} symbols in {data_dir}: {symbols}")
    return symbols

def run_ppo_training(
    symbol: str,
    num_episodes: int,
    data_dir: str,
    testing_mode: bool = False
):
    """
    Run single-threaded PPO training for a symbol.
    """
    logger.info(f"Starting PPO training for symbol: {symbol}")

    data_loader = DataLoader(final_data_dir=data_dir, use_parquet=True, testing_mode=testing_mode)

    env = TradingEnv(
        data_loader=data_loader,
        symbol=symbol,
        initial_capital=INITIAL_CAPITAL,
        lookback_window=20,
        episode_length=500,
        reward_function="trading_focused",
        use_streaming=False
    )

    obs = env.reset()
    observation_dim = obs.shape[0]
    action_dim_discrete = int(env.action_space.high[0]) + 1
    action_dim_continuous = 1

    logger.info(f"Environment dimensions: obs={observation_dim}, action_discrete={action_dim_discrete}, action_continuous={action_dim_continuous}")

    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim_discrete=action_dim_discrete,
        action_dim_continuous=action_dim_continuous,
        hidden_dim=64,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        epsilon_clip=0.2,
        k_epochs=3
    )

    trainer = Trainer(agent, num_episodes=num_episodes, log_interval=10)
    logger.info(f"Training PPO agent for {num_episodes} episodes...")
    trainer.train(data_loader, symbol, INITIAL_CAPITAL)

    model_path = f"models/{symbol}_ppo_model.pth"
    os.makedirs("models", exist_ok=True)
    # Assuming agent has a save_model method
    if hasattr(agent, 'save_model'):
        agent.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
    else:
        logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")


    logger.info(f"PPO training completed for {symbol}")


def main():
    """Main entry point for RL training."""
    parser = argparse.ArgumentParser(description="Run PPO training for trading")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols to train on")
    parser.add_argument("--data-dir", default="data/final", help="Data directory")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes for training")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    args = parser.parse_args()

    if args.testing:
        logger.info("🧪 TESTING MODE ENABLED - Using in-memory test data")
        # Test both stock and option instruments to verify environment handling
        symbols = ["RELIANCE_1", "Bank_Nifty_5"]
        args.data_dir = 'data/test'  # This will be overridden by in-memory data
    else:
        if args.symbols:
            symbols = args.symbols
        else:
            symbols = get_available_symbols(args.data_dir)
            if not symbols:
                logger.error("No symbols found. Please specify symbols or ensure data directory contains CSV files.")
                return

    for symbol in symbols:
        try:
            run_ppo_training(symbol, num_episodes=args.episodes, data_dir=args.data_dir, testing_mode=args.testing)
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}", exc_info=True)
            continue

if __name__ == "__main__":
    main()