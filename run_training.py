#!/usr/bin/env python3
"""
Script for running PPO training.
"""

import os
import sys
import argparse
import logging
import yaml
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agents.ppo_agent import PPOAgent
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.training.trainer import Trainer
from src.training.universal_trainer import UniversalTrainer
from src.training.curriculum_trainer import CurriculumTrainer
from src.utils.test_data_generator import create_test_data_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_config(config_path: str = "config/training_sequence.yaml") -> dict:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file {config_path} not found. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

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
    testing_mode: bool = False,
    config: dict = None
):
    """
    Run single-threaded PPO training for a symbol.
    """
    logger.info(f"Starting PPO training for symbol: {symbol}")

    # Load configuration if not provided
    if config is None:
        config = load_training_config()

    # Get configuration sections
    env_config = config.get('environment', {})
    model_config = config.get('model', {})

    # Use appropriate data directory for testing vs production
    final_data_dir = "data/test/final" if testing_mode else data_dir
    data_loader = DataLoader(final_data_dir=final_data_dir, use_parquet=True)

    env = TradingEnv(
        data_loader=data_loader,
        symbol=symbol,
        initial_capital=env_config.get('initial_capital', 100000.0),
        lookback_window=env_config.get('lookback_window', 50),
        episode_length=env_config.get('episode_length', 500),
        reward_function=env_config.get('reward_function', "trading_focused"),
        use_streaming=env_config.get('use_streaming', False),
        trailing_stop_percentage=env_config.get('trailing_stop_percentage', 0.02)
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
        hidden_dim=model_config.get('hidden_dim', 64),
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        epsilon_clip=0.2,
        k_epochs=3
    )

    trainer = Trainer(agent, num_episodes=num_episodes, log_interval=10)
    logger.info(f"Training PPO agent for {num_episodes} episodes...")
    trainer.train(data_loader, symbol, env_config.get('initial_capital', 100000.0))

    # Only save model in production mode (not testing)
    if not testing_mode:
        # Use universal model path instead of symbol-specific
        model_path = "models/universal_final_model.pth"
        os.makedirs("models", exist_ok=True)

        if hasattr(agent, 'save_model'):
            agent.save_model(model_path)
            logger.info(f"✅ Universal model saved to {model_path}")
        else:
            logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")
    else:
        logger.info("🧪 Testing mode - Model not saved")

    logger.info(f"PPO training completed for {symbol}")


def run_universal_ppo_training(
    symbols: List[str],
    num_episodes: int,
    data_dir: str,
    testing_mode: bool = False,
    config: dict = None
):
    """
    Run universal PPO training that rotates through symbols per episode.
    This creates a single universal model trained on diverse market data.
    """
    logger.info(f"Starting Universal PPO training with {len(symbols)} symbols: {symbols}")
    logger.info(f"🔄 Symbol rotation: Each episode will use a different symbol for diverse market exposure")

    # Load configuration if not provided
    if config is None:
        config = load_training_config()

    # Get configuration sections
    env_config = config.get('environment', {})
    model_config = config.get('model', {})

    # Use appropriate data directory for testing vs production
    final_data_dir = "data/test/final" if testing_mode else data_dir
    data_loader = DataLoader(final_data_dir=final_data_dir, use_parquet=True)

    # Create environment with first symbol to get dimensions
    env = TradingEnv(
        data_loader=data_loader,
        symbol=symbols[0],
        initial_capital=env_config.get('initial_capital', 100000.0),
        lookback_window=env_config.get('lookback_window', 50),
        episode_length=env_config.get('episode_length', 500),
        reward_function=env_config.get('reward_function', "trading_focused"),
        use_streaming=env_config.get('use_streaming', False),
        trailing_stop_percentage=env_config.get('trailing_stop_percentage', 0.02)
    )

    # Get observation and action dimensions from environment
    obs = env.reset()
    observation_dim = obs.shape[0]
    action_dim_discrete = int(env.action_space.high[0]) + 1
    action_dim_continuous = 1

    logger.info(f"Environment dimensions: obs={observation_dim}, action_discrete={action_dim_discrete}, action_continuous={action_dim_continuous}")

    # Create PPO agent
    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim_discrete=action_dim_discrete,
        action_dim_continuous=action_dim_continuous,
        hidden_dim=model_config.get('hidden_dim', 64),
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        epsilon_clip=0.2,
        k_epochs=3
    )

    # Create universal trainer that handles symbol rotation
    trainer = UniversalTrainer(agent, symbols, data_loader, num_episodes=num_episodes, log_interval=10, config=config)
    logger.info(f"Training Universal PPO agent for {num_episodes} episodes with symbol rotation...")
    trainer.train()

    # Only save model in production mode (not testing)
    if not testing_mode:
        # Use universal model path instead of symbol-specific
        model_path = "models/universal_final_model.pth"
        os.makedirs("models", exist_ok=True)

        if hasattr(agent, 'save_model'):
            agent.save_model(model_path)
            logger.info(f"✅ Universal model saved to {model_path}")
        else:
            logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")
    else:
        logger.info("🧪 Testing mode - Model not saved")

    logger.info(f"Universal PPO training completed for all symbols")


def run_curriculum_ppo_training(num_episodes: int = 10, testing_mode: bool = False):
    """
    Run curriculum PPO training with timeframe progression.

    Args:
        num_episodes: Number of episodes to train
        testing_mode: Whether to run in testing mode (no model saving)
    """
    logger.info("🎓 Starting Curriculum PPO Training")
    logger.info("=" * 60)

    # Load configuration
    config = load_training_config()

    # Initialize data loader
    data_loader = DataLoader()

    # Create curriculum trainer first to discover data
    temp_trainer = CurriculumTrainer(None, data_loader, num_episodes=1, config=config)

    # Determine observation dimension from actual data
    if temp_trainer.curriculum_batches:
        from src.backtesting.environment import TradingMode
        sample_env = TradingEnv(
            data_loader=data_loader,
            symbol=temp_trainer.curriculum_batches[0]['symbols'][0],
            initial_capital=config.get('initial_capital', 100000),
            lookback_window=config.get('lookback_window', 50),
            episode_length=50,  # Small episode for dimension detection
            mode=TradingMode.TRAINING
        )
        sample_obs = sample_env.reset()
        observation_dim = len(sample_obs)
        logger.info(f"🔧 Detected observation dimension: {observation_dim}")
    else:
        observation_dim = 1186  # Fallback
        logger.warning("🔧 Using fallback observation dimension: 1186")

    # Create PPO agent with correct dimensions
    agent = PPOAgent(
        observation_dim=observation_dim,
        action_dim_discrete=5,
        action_dim_continuous=1,
        hidden_dim=256,
        lr_actor=0.0003,
        lr_critic=0.001,
        gamma=0.99,
        epsilon_clip=0.2,
        k_epochs=3
    )

    # Create curriculum trainer
    trainer = CurriculumTrainer(agent, data_loader, num_episodes=num_episodes, log_interval=5, config=config)
    logger.info(f"Training PPO agent with curriculum learning for {num_episodes} episodes...")

    # Execute curriculum training
    training_results = trainer.train()

    # Only save model in production mode (not testing)
    if not testing_mode:
        # Use universal model path
        model_path = "models/universal_curriculum_model.pth"
        os.makedirs("models", exist_ok=True)

        if hasattr(agent, 'save_model'):
            agent.save_model(model_path)
            logger.info(f"✅ Curriculum model saved to {model_path}")
        else:
            logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")
    else:
        logger.info("🧪 Testing mode - Model not saved")

    logger.info(f"🎓 Curriculum PPO training completed")
    return training_results


def main():
    """Main entry point for RL training."""
    parser = argparse.ArgumentParser(description="Run PPO training for trading")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols to train on")
    parser.add_argument("--data-dir", default="data/final", help="Data directory")
    parser.add_argument("--episodes", type=int, help="Number of episodes for training (overrides config)")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum training (use basic symbol rotation instead)")
    args = parser.parse_args()

    # Load configuration
    config = load_training_config()

    # Determine episodes based on testing mode and configuration
    if args.testing:
        logger.info("🧪 TESTING MODE ENABLED - Using testing configuration")
        # Use testing overrides from config
        if 'testing_overrides' in config and 'training_sequence' in config['testing_overrides']:
            episodes = config['testing_overrides']['training_sequence']['stage_1_ppo']['episodes']
            logger.info(f"📊 Using testing episodes from config: {episodes}")
        else:
            episodes = 5  # Fallback for testing
            logger.info(f"📊 Using fallback testing episodes: {episodes}")

        # Create test data files for both stock and option instruments
        symbols = ["RELIANCE_1", "Bank_Nifty_5"]
        args.data_dir = 'data/test'

        # Use test data configuration if available
        if 'testing_overrides' in config and 'test_data' in config['testing_overrides']:
            test_config = config['testing_overrides']['test_data']
            num_rows = test_config.get('num_rows', 500)
            symbols = test_config.get('symbols', symbols)
        else:
            num_rows = 500

        create_test_data_files(
            data_dir=args.data_dir,
            create_multiple_instruments=True,
            num_rows=num_rows
        )
    else:
        # Production mode - use production configuration
        if 'training_sequence' in config and 'stage_1_ppo' in config['training_sequence']:
            episodes = config['training_sequence']['stage_1_ppo']['episodes']
            logger.info(f"📊 Using production episodes from config: {episodes}")
        else:
            episodes = 100  # Full episodes as per config (not reduced)
            logger.info(f"📊 Using fallback production episodes: {episodes}")

        if args.symbols:
            symbols = args.symbols
        else:
            symbols = get_available_symbols(args.data_dir)
            if not symbols:
                logger.error("No symbols found. Please specify symbols or ensure data directory contains CSV files.")
                return

    # Override episodes if explicitly provided via command line
    if args.episodes is not None:
        episodes = args.episodes
        logger.info(f"📊 Overriding episodes with command line value: {episodes}")

    # Choose training method based on arguments (curriculum is default)
    try:
        if args.no_curriculum:
            logger.info("🔄 Using universal training (symbol rotation)")
            run_universal_ppo_training(symbols, num_episodes=episodes, data_dir=args.data_dir, testing_mode=args.testing, config=config)
        else:
            logger.info("🎓 Using curriculum training (timeframe progression) - DEFAULT")
            run_curriculum_ppo_training(num_episodes=episodes, testing_mode=args.testing)
    except Exception as e:
        logger.error(f"Failed to run training: {e}", exc_info=True)

if __name__ == "__main__":
    main()