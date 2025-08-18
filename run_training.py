#!/usr/bin/env python3
"""
Script for running HRM (Hierarchical Reasoning Model) training.
"""

import os
import sys
import argparse
import logging
import yaml
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.training.trainer import Trainer
from src.training.universal_trainer import UniversalTrainer
from src.utils.test_data_generator import create_test_data_files
from src.utils.iteration_manager import IterationManager
from src.utils.research_logger import ResearchLogger

# Configure clean, minimal logging for training
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Reduce verbosity for specific modules during training
logging.getLogger('src.backtesting.environment').setLevel(logging.WARNING)
logging.getLogger('src.utils.data_loader').setLevel(logging.WARNING)
logging.getLogger('src.data_processing.feature_generator').setLevel(logging.WARNING)
logging.getLogger('src.utils.data_feeding_strategy').setLevel(logging.WARNING)
logging.getLogger('src.utils.config_loader').setLevel(logging.WARNING)
logging.getLogger('src.training.curriculum_trainer').setLevel(logging.WARNING)
logging.getLogger('src.training.trainer').setLevel(logging.WARNING)
logging.getLogger('src.training.universal_trainer').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def load_training_config(config_path: str = "config/settings.yaml") -> dict:
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

def run_hrm_training(
    symbol: str,
    num_episodes: int,
    data_dir: str,
    testing_mode: bool = False,
    config: dict = None
):
    """
    Run single-threaded HRM training for a symbol.
    """
    logger.info(f"🚀 Starting HRM training: {symbol}")

    # Load configuration if not provided
    if config is None:
        config = load_training_config()

    # Get configuration sections
    env_config = config.get('environment', {})
    model_config = config.get('model', {})

    # Use appropriate data directory for testing vs production
    data_processing_config = config.get('data_processing', {})
    final_data_dir = os.path.join(data_processing_config.get('test_folder', 'data/test'), 'final') if testing_mode else data_dir
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

    # Dynamically get dimensions from the environment
    observation_dim = env.observation_space.shape[0]
    action_dim_discrete = int(env.action_space.high[0]) + 1
    logger.info(f"🔧 Environment configured with Observation Dim: {observation_dim}")

    # Update config with the true, environment-derived dimension before creating the model
    config_copy = config.copy()
    config_copy['model'] = config_copy.get('model', {})
    config_copy['model']['observation_dim'] = observation_dim

    # Ensure hierarchical_reasoning_model.input_embedding.input_dim is also updated
    hrm_config = config_copy.get('hierarchical_reasoning_model', {})
    input_embedding_config = hrm_config.get('input_embedding', {})
    input_embedding_config['input_dim'] = observation_dim
    hrm_config['input_embedding'] = input_embedding_config
    config_copy['hierarchical_reasoning_model'] = hrm_config

    agent = HierarchicalReasoningModel(config_copy)

    trainer = Trainer(agent, num_episodes=num_episodes, log_interval=10)
    logger.info(f"🎯 Training {num_episodes} episodes")
    trainer.train(data_loader, symbol, env_config.get('initial_capital', 100000.0))

    # Only save model in production mode (not testing)
    if not testing_mode:
        # Use universal model path instead of symbol-specific
        model_path = model_config.get('model_path', 'models/universal_final_model.pth')
        os.makedirs("models", exist_ok=True)

        if hasattr(agent, 'save_model'):
            agent.save_model(model_path)
            logger.info(f"✅ Universal model saved to {model_path}")
        else:
            logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")
    else:
        logger.info("🧪 Testing mode - Model not saved")

    logger.info(f"✅ Training complete: {symbol}")


def run_universal_hrm_training(
    symbols: List[str],
    num_episodes: int,
    data_dir: str,
    testing_mode: bool = False,
    config: dict = None,
    iteration_manager = None,
    research_logger = None
):
    """
    Run universal HRM training that rotates through symbols per episode.
    This creates a single universal model trained on diverse market data.
    """
    logger.info(f"🚀 Universal HRM training: {len(symbols)} symbols")

    # Load configuration if not provided
    if config is None:
        config = load_training_config()

    # Get configuration sections
    env_config = config.get('environment', {})
    model_config = config.get('model', {})

    # Use appropriate data directory for testing vs production
    data_processing_config = config.get('data_processing', {})
    final_data_dir = os.path.join(data_processing_config.get('test_folder', 'data/test'), 'final') if testing_mode else data_dir
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

    # Call reset to initialize observation_space
    env.reset()

    # Dynamically get dimensions from the environment
    observation_dim = env.observation_space.shape[0]
    logger.info(f"🔧 Environment configured with Observation Dim: {observation_dim}")

    # Update config with the true, environment-derived dimension before creating the model
    config_copy = config.copy()
    config_copy['model'] = config_copy.get('model', {})
    config_copy['model']['observation_dim'] = observation_dim

    # Ensure hierarchical_reasoning_model.input_embedding.input_dim is also updated
    hrm_config = config_copy.get('hierarchical_reasoning_model', {})
    input_embedding_config = hrm_config.get('input_embedding', {})
    input_embedding_config['input_dim'] = observation_dim
    hrm_config['input_embedding'] = input_embedding_config
    config_copy['hierarchical_reasoning_model'] = hrm_config

    # Create HRM agent
    agent = HierarchicalReasoningModel(config_copy)

    # Create universal trainer that handles symbol rotation
    trainer = UniversalTrainer(
        agent, symbols, data_loader, 
        num_episodes=num_episodes, 
        log_interval=10, 
        config=config,
        research_logger=research_logger
    )
    logger.info(f"🎯 Training {num_episodes} episodes with symbol rotation")
    trainer.train()

    # Only save model in production mode (not testing)
    if not testing_mode:
        # Use universal model path instead of symbol-specific
        model_path = model_config.get('model_path', 'models/universal_final_model.pth')
        # Ensure models directory exists
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        if hasattr(agent, 'save_model'):
            agent.save_model(model_path)
            logger.info(f"✅ Universal model saved to {model_path}")
            
            # Copy model to iteration directory
            if iteration_manager:
                iteration_manager.save_model_artifacts(model_path)
        else:
            logger.warning(f"Agent does not have save_model method. Cannot save model to {model_path}")
    else:
        logger.info("🧪 Testing mode - Model not saved")

    # Save final training metrics
    if iteration_manager and hasattr(trainer, 'get_training_summary'):
        training_summary = trainer.get_training_summary()
        iteration_manager.save_training_metrics(training_summary)

    logger.info("✅ Universal training complete")




def main():
    """Main entry point for RL training."""
    # Enable detailed logging for direct training runs
    os.environ['DETAILED_BACKTEST_LOGGING'] = 'true'
    logger.info("Detailed trade logging enabled for direct training run")

    parser = argparse.ArgumentParser(description="Run HRM training for trading")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols to train on")
    parser.add_argument("--data-dir", default="data/final", help="Data directory")
    parser.add_argument("--episodes", type=int, help="Number of episodes for training (overrides config)")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode")
    parser.add_argument("--noLog", action="store_true", help="Use progress bar instead of console logs")
    args = parser.parse_args()

    # Load configuration
    config = load_training_config()

    # Determine data directory for DataLoader based on testing mode
    data_processing_config = config.get('data_processing', {})
    if args.testing:
        # In testing mode, data is generated in data/test
        data_dir_for_loader = os.path.join(data_processing_config.get('test_folder', 'data/test'), 'final')
    else:
        # In production mode, data is in args.data_dir (default: data/final)
        data_dir_for_loader = args.data_dir

    # Initialize data_loader here, it's needed for both training paths
    data_loader = DataLoader(final_data_dir=data_dir_for_loader)

    # Determine episodes based on testing mode and configuration
    if args.testing:
        logger.info("🧪 TESTING MODE ENABLED - Using testing configuration")
        # Use testing overrides from config
        if 'testing_overrides' in config and 'training_sequence' in config['testing_overrides']:
            episodes = config['testing_overrides']['training_sequence']['stage_1_hrm']['episodes']
            logger.info(f"🧪 Testing mode: {episodes} episodes")
        else:
            episodes = 5  # Fallback for testing
            logger.info(f"🧪 Testing mode: {episodes} episodes (fallback)")

        # Create test data files for both stock and option instruments
        symbols = ["RELIANCE_1", "Bank_Nifty_5"]
        # args.data_dir is used by create_test_data_files, ensure it points to the test folder
        args.data_dir = data_processing_config.get('test_folder', 'data/test')

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
        if 'training_sequence' in config and 'stage_1_hrm' in config['training_sequence']:
            episodes = config['training_sequence']['stage_1_hrm']['episodes']
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

    # Set up iteration management and research logging
    try:
        # Initialize iteration manager
        iteration_manager = IterationManager(config)
        
        # Temporarily initialize data loader for iteration setup
        data_processing_config = config.get('data_processing', {})
        if args.testing:
            data_dir_for_iteration = os.path.join(data_processing_config.get('test_folder', 'data/test'), 'final')
        else:
            data_dir_for_iteration = args.data_dir
        temp_data_loader = DataLoader(final_data_dir=data_dir_for_iteration)
        
        # Setup iteration directory
        iteration_dir = iteration_manager.setup_iteration(temp_data_loader, symbols)
        
        # Initialize research logger
        research_logger = ResearchLogger(config, iteration_dir, use_progress_bar=args.noLog)
        
        logger.info(f"🔬 Research iteration: {iteration_manager.current_iteration}")
        logger.info(f"📁 Iteration directory: {iteration_dir}")
        logger.info("🎯 Using Universal Training with Timeframe-Aware Symbol Rotation")
        
        # Run training with enhanced logging
        run_universal_hrm_training(
            symbols, 
            num_episodes=episodes, 
            data_dir=args.data_dir, 
            testing_mode=args.testing, 
            config=config,
            iteration_manager=iteration_manager,
            research_logger=research_logger
        )
        
    except Exception as e:
        logger.error(f"Failed to run training: {e}", exc_info=True)

if __name__ == "__main__":
    main()