#!/usr/bin/env python3
"""
Example script for running parallel training with Ray RLlib.
"""

import os
import sys
import argparse
import logging
from typing import List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.training.parallel_trainer import ParallelTrainer
from src.training.parallel_config import ParallelTrainingConfig, get_recommended_config
from src.training.trainer import Trainer
from src.training.sequence_manager import TrainingSequenceManager
from src.agents.ppo_agent import PPOAgent
from src.agents.moe_agent import MoEAgent
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.config.config import INITIAL_CAPITAL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enable detailed logging for trading components
logging.getLogger('src.backtesting.engine').setLevel(logging.INFO)
logging.getLogger('src.backtesting.environment').setLevel(logging.INFO)
logging.getLogger('src.training.trainer').setLevel(logging.INFO)

def get_available_symbols(data_dir: str = "data/final") -> List[str]:
    """Get list of available trading symbols from data directory."""
    symbols = []
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} does not exist")
        return symbols

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv') and filename.startswith('features_'):
            symbol = filename.replace('features_', '').replace('.csv', '')
            symbols.append(symbol)
    
    logger.info(f"Found {len(symbols)} symbols: {symbols}")
    return symbols

def run_single_symbol_training(
    symbol: str,
    config: dict,
    num_iterations: int = None,
    algorithm: str = "PPO"
) -> dict:
    """
    Run parallel training for a single symbol.
    
    Args:
        symbol: Trading symbol to train on
        config: Training configuration
        num_iterations: Number of training iterations
        algorithm: RL algorithm to use
        
    Returns:
        Training results
    """
    logger.info(f"Starting parallel training for symbol: {symbol}")

    # Use YAML config iterations if not specified
    if num_iterations is None:
        from src.training.sequence_manager import TrainingSequenceManager
        manager = TrainingSequenceManager()
        if algorithm.upper() == "PPO":
            num_iterations = manager.config['training_sequence']['stage_1_ppo'].get('episodes', 500)
        elif algorithm.upper() == "MOE":
            num_iterations = manager.config['training_sequence']['stage_2_moe'].get('episodes', 800)
        elif algorithm.upper() == "MAML":
            num_iterations = manager.config['training_sequence']['stage_3_maml'].get('meta_iterations', 150)
        elif algorithm.upper() == "AUTONOMOUS":
            num_iterations = manager.config['training_sequence']['stage_4_autonomous'].get('generations', 50)
        else:
            num_iterations = 500  # fallback to PPO default
        logger.info(f"Using {num_iterations} iterations from YAML configuration")

    # Update config for this symbol
    config["env_config"]["symbol"] = symbol
    config["training_config"]["algorithm"] = algorithm

    # Create and setup trainer
    trainer = ParallelTrainer(config)
    trainer.setup_algorithm(algorithm)
    
    try:
        # Run training
        results = trainer.train(
            num_iterations=num_iterations,
            checkpoint_freq=config["checkpoint_config"]["checkpoint_freq"],
            checkpoint_dir=f"checkpoints/{symbol}_{algorithm.lower()}"
        )
        
        # Run evaluation
        eval_results = trainer.evaluate(
            num_episodes=config["checkpoint_config"]["evaluation_episodes"]
        )
        
        # Combine results
        final_results = {
            "symbol": symbol,
            "algorithm": algorithm,
            "training_results": results,
            "evaluation_results": eval_results
        }
        
        logger.info(f"Training completed for {symbol}: Best reward = {results['best_reward']:.2f}")
        return final_results
        
    except Exception as e:
        logger.error(f"Training failed for {symbol}: {e}")
        raise
    finally:
        trainer.cleanup()

def run_multi_symbol_training(
    symbols: List[str],
    config: dict,
    num_iterations: int = None,
    algorithm: str = "PPO",
    mode: str = "sequential"
) -> List[dict]:
    """
    Run parallel training for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        config: Base training configuration
        num_iterations: Number of training iterations per symbol
        algorithm: RL algorithm to use
        mode: Training mode ("sequential" or "parallel")
        
    Returns:
        List of training results for each symbol
    """
    logger.info(f"Starting multi-symbol training for {len(symbols)} symbols in {mode} mode")

    # Use YAML config iterations if not specified
    if num_iterations is None:
        from src.training.sequence_manager import TrainingSequenceManager
        manager = TrainingSequenceManager()
        if algorithm.upper() == "PPO":
            num_iterations = manager.config['training_sequence']['stage_1_ppo'].get('episodes', 500)
        elif algorithm.upper() == "MOE":
            num_iterations = manager.config['training_sequence']['stage_2_moe'].get('episodes', 800)
        elif algorithm.upper() == "MAML":
            num_iterations = manager.config['training_sequence']['stage_3_maml'].get('meta_iterations', 150)
        elif algorithm.upper() == "AUTONOMOUS":
            num_iterations = manager.config['training_sequence']['stage_4_autonomous'].get('generations', 50)
        else:
            num_iterations = 500  # fallback to PPO default
        logger.info(f"Using {num_iterations} iterations from YAML configuration")

    all_results = []

    if mode == "sequential":
        # Train symbols one by one
        for symbol in symbols:
            try:
                results = run_single_symbol_training(symbol, config, num_iterations, algorithm)
                all_results.append(results)
            except Exception as e:
                logger.error(f"Failed to train {symbol}: {e}")
                continue
    
    elif mode == "parallel":
        # TODO: Implement true parallel training across symbols
        # This would require more complex coordination
        logger.warning("Parallel multi-symbol training not yet implemented, falling back to sequential")
        return run_multi_symbol_training(symbols, config, num_iterations, algorithm, "sequential")
    
    return all_results

def run_simple_training(
    symbol: str,
    num_episodes: int = None,
    agent_type: str = "PPO",
    testing_mode: bool = False
) -> dict:
    """
    Run simple single-threaded training for a symbol.

    Args:
        symbol: Trading symbol to train on
        num_episodes: Number of training episodes
        agent_type: Type of agent ("PPO" or "MoE")

    Returns:
        Training results
    """
    logger.info(f"Starting simple training for symbol: {symbol}")

    # Use YAML config episodes if not specified
    if num_episodes is None:
        from src.training.sequence_manager import TrainingSequenceManager
        manager = TrainingSequenceManager()
        if agent_type.upper() == "PPO":
            num_episodes = manager.config['training_sequence']['stage_1_ppo'].get('episodes', 500)
        elif agent_type.upper() == "MOE":
            num_episodes = manager.config['training_sequence']['stage_2_moe'].get('episodes', 800)
        elif agent_type.upper() == "MAML":
            num_episodes = manager.config['training_sequence']['stage_3_maml'].get('meta_iterations', 150)
        elif agent_type.upper() == "AUTONOMOUS":
            num_episodes = manager.config['training_sequence']['stage_4_autonomous'].get('generations', 50)
        else:
            num_episodes = 500  # fallback to PPO default
        logger.info(f"Using {num_episodes} episodes from YAML configuration")

    # Initialize data loader (use test data files if in testing mode)
    data_dir = "data/test" if testing_mode else "data/final"
    if testing_mode:
        logger.info(f"🧪 Using test data directory: {data_dir}")
        # Create test data files using actual pipeline (creates both STOCK and OPTION data)
        from src.utils.test_data_generator import create_test_data_files
        create_test_data_files(
            data_dir="data/test",
            symbol=symbol,
            num_rows=150,
            create_both=True,
            create_multiple_instruments=True  # Creates RELIANCE (STOCK) and Bank_Nifty_5 (OPTION)
        )

    data_loader = DataLoader(final_data_dir=data_dir, use_parquet=True)

    # Create environment (disable streaming for consistent dimensions)
    env = TradingEnv(
        data_loader=data_loader,
        symbol=symbol,
        initial_capital=INITIAL_CAPITAL,
        lookback_window=20,
        episode_length=500,
        reward_function="trading_focused",  # Use trading-focused reward
        use_streaming=False
    )

    # Get observation and action dimensions by resetting environment first
    obs = env.reset()
    observation_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]

    action_dim_discrete = int(env.action_space.high[0]) + 1 # Number of discrete actions (0-4)
    action_dim_continuous = 1 # Quantity is a single continuous value

    logger.info(f"Environment dimensions: obs={observation_dim}, action_discrete={action_dim_discrete}, action_continuous={action_dim_continuous}")

    # Create agent
    if agent_type.upper() == "PPO":
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
    elif agent_type.upper() == "MOE":
        expert_configs = {
            'TrendAgent': {'lr': 0.001, 'hidden_dim': 64},
            'MeanReversionAgent': {'lr': 0.001, 'hidden_dim': 64},
            'VolatilityAgent': {'lr': 0.001, 'hidden_dim': 64},
            'ConsolidationAgent': {'lr': 0.001, 'hidden_dim': 64}
        }
        agent = MoEAgent(
            observation_dim=observation_dim,
            action_dim_discrete=action_dim_discrete,
            action_dim_continuous=action_dim_continuous,
            hidden_dim=64,
            expert_configs=expert_configs
        )
        logger.info(f"Created MoE agent with {len(expert_configs)} experts")
    elif agent_type.upper() == "MAML":
        # Use MoE agent with MAML meta-learning
        expert_configs = {
            'TrendAgent': {'lr': 0.001, 'hidden_dim': 64},
            'MeanReversionAgent': {'lr': 0.001, 'hidden_dim': 64},
            'VolatilityAgent': {'lr': 0.001, 'hidden_dim': 64}
        }
        agent = MoEAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=64,
            expert_configs=expert_configs
        )
        logger.info(f"Created MAML agent with {len(expert_configs)} experts for meta-learning")
    elif agent_type.upper() == "AUTONOMOUS":
        # Autonomous agents are handled by the autonomous trainer
        agent = None  # Will be created by autonomous trainer
        logger.info("Autonomous agent will be created by autonomous trainer")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    try:
        # Create trainer
        if agent_type.upper() == "MAML":
            # Use MAML meta-learning trainer
            trainer = Trainer(agent, num_episodes=num_episodes, log_interval=5)
            logger.info(f"Training {agent_type} agent with MAML meta-learning for {num_episodes} episodes...")

            # For simple training, create a single-symbol task list
            # Override the data_loader's sample_tasks method to return our specific symbol
            original_sample_tasks = data_loader.sample_tasks
            data_loader.sample_tasks = lambda n: [symbol] * min(n, 1)

            try:
                # Run MAML training
                trainer.meta_train(data_loader, INITIAL_CAPITAL,
                                  num_meta_iterations=max(1, num_episodes//10),  # 10% of episodes as meta-iterations
                                  num_inner_loop_steps=5,
                                  num_evaluation_steps=3,
                                  meta_batch_size=1)
            finally:
                # Restore original method
                data_loader.sample_tasks = original_sample_tasks
        elif agent_type.upper() == "AUTONOMOUS":
            # Use autonomous trainer
            from src.training.autonomous_trainer import run_autonomous_stage
            from src.training.sequence_manager import TrainingSequenceManager

            manager = TrainingSequenceManager()
            stage_config = manager.config['training_sequence']['stage_4_autonomous']

            logger.info(f"Training autonomous agents for {num_episodes} generations...")
            autonomous_results = run_autonomous_stage(stage_config)

            # Update results with autonomous training info
            results = {
                "symbol": symbol,
                "agent_type": agent_type,
                "generations_completed": autonomous_results.get('generation', 0),
                "best_fitness": autonomous_results.get('best_fitness', 0.0),
                "champion_path": autonomous_results.get('champion_path'),
                "status": "completed"
            }

            logger.info(f"Autonomous training completed. Best fitness: {results['best_fitness']:.4f}")
            return results
        else:
            # Use standard trainer
            trainer = Trainer(agent, num_episodes=num_episodes, log_interval=10)
            logger.info(f"Training {agent_type} agent for {num_episodes} episodes...")
            trainer.train(data_loader, symbol, INITIAL_CAPITAL)

        # Save the trained model (skip in testing mode)
        if not testing_mode:
            model_path = f"models/{symbol}_{agent_type.lower()}_model.pkl"
            os.makedirs("models", exist_ok=True)

            # Save agent state
            agent_state = {
                'type': agent_type,
                'observation_dim': observation_dim,
                'action_dim_discrete': action_dim_discrete,
                'action_dim_continuous': action_dim_continuous,
                'training_episodes': num_episodes,
                'symbol': symbol
            }

            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(agent_state, f)

            logger.info(f"Model saved to {model_path}")
        else:
            logger.info("🧪 Testing mode: Skipping model save")

        results = {
            "symbol": symbol,
            "agent_type": agent_type,
            "episodes_completed": num_episodes,
            "status": "completed"
        }

        logger.info(f"Simple training completed for {symbol}")
        return results

    except Exception as e:
        logger.error(f"Simple training failed for {symbol}: {e}")
        raise

def main():
    """Main entry point for RL training."""
    # Initialize error logger
    from src.utils.error_logger import error_logger

    parser = argparse.ArgumentParser(description="Run RL training for trading")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols to train on")
    parser.add_argument("--data-dir", default="data/final", help="Data directory")
    parser.add_argument("--algorithm", choices=["PPO", "IMPALA", "MoE", "MAML", "Autonomous"], default="Autonomous", help="RL algorithm (default: Autonomous for self-evolving agents)")
    parser.add_argument("--iterations", type=int, default=None, help="Number of training iterations/episodes (uses YAML config if not specified)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--mode", choices=["development", "production", "distributed"],
                       default="production", help="Training mode")
    parser.add_argument("--multi-symbol-mode", choices=["sequential", "parallel"],
                       default="sequential", help="Multi-symbol training mode")
    parser.add_argument("--local", action="store_true", help="Run in local mode for debugging")
    parser.add_argument("--simple", action="store_true", help="Use simple single-threaded training instead of parallel")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes for training (uses YAML config if not specified)")
    parser.add_argument("--sequence", action="store_true", default=True, help="Run complete training sequence: PPO -> MoE -> MAML (default)")
    parser.add_argument("--no-sequence", action="store_true", help="Disable sequence training and use single algorithm")
    parser.add_argument("--testing", action="store_true", help="Enable testing mode with minimal data and parameters")

    args = parser.parse_args()

    # Configure testing mode
    if args.testing:
        logger.info("🧪 TESTING MODE ENABLED")
        logger.info("Using minimal data and parameters for quick testing")
        # Override parameters for testing
        if args.episodes is None:
            args.episodes = 5  # Minimal episodes for testing
        if args.iterations is None:
            args.iterations = 3  # Minimal iterations for testing

    # Get available symbols
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = get_available_symbols(args.data_dir)
        if not symbols:
            logger.error("No symbols found. Please specify symbols or ensure data directory contains CSV files.")
            return
    
    # Create configuration
    config = get_recommended_config(symbols, args.mode)
    
    # Override with command line arguments
    if args.local:
        config["local_mode"] = True
        config["num_gpus"] = 0
    
    config["training_config"]["num_workers"] = args.workers
    
    # Validate configuration
    issues = ParallelTrainingConfig.validate_config(config)
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return
    
    logger.info(f"Configuration validated successfully")
    logger.info(f"Training {len(symbols)} symbols with {args.algorithm} algorithm")
    episodes_info = f"{args.episodes} episodes" if args.episodes else "YAML config episodes"
    logger.info(f"Using {args.workers} workers for {episodes_info}")
    
    try:
        # Override sequence if --no-sequence is specified
        if args.no_sequence:
            args.sequence = False

        if args.sequence:
            # Complete training sequence: PPO -> MoE -> MAML
            logger.info("Using complete training sequence: PPO -> MoE -> MAML -> Autonomous")

            # Universal model training for multiple symbols, single for one
            if len(symbols) > 1:
                logger.info("🚀 TRAINING UNIVERSAL MODEL on all symbols")
                logger.info(f"📊 Training on {len(symbols)} symbols: {', '.join(symbols)}")

                data_loader = DataLoader(args.data_dir)
                manager = TrainingSequenceManager()

                # Train universal model using all symbols data
                results = manager.run_universal_sequence(data_loader, symbols, episodes_override=args.episodes)
                all_success = all(r.success for r in results)
                logger.info(f"🎯 Universal model training completed: {'SUCCESS' if all_success else 'PARTIAL SUCCESS'}")
                logger.info(f"✅ Single model saved: models/universal_final_model.pth")

            elif len(symbols) == 1:
                # Single symbol sequence training
                data_loader = DataLoader(args.data_dir)
                manager = TrainingSequenceManager()
                results = manager.run_complete_sequence(data_loader, symbols[0], episodes_override=args.episodes)

                # Display final results
                all_success = all(r.success for r in results)
                logger.info(f"Training sequence completed: {'SUCCESS' if all_success else 'PARTIAL SUCCESS'}")

        elif args.simple:
            # Simple single-threaded training
            logger.info("Using simple single-threaded training mode")

            if len(symbols) == 1:
                # Single symbol simple training
                results = run_simple_training(
                    symbols[0], args.episodes, args.algorithm
                )
                logger.info(f"Simple training completed successfully for {symbols[0]}")
                logger.info(f"Episodes completed: {results['episodes_completed']}")
            else:
                # Multi-symbol simple training
                all_results = []
                for symbol in symbols:
                    try:
                        results = run_simple_training(symbol, args.episodes, args.algorithm, args.testing)
                        all_results.append(results)
                        logger.info(f"Completed training for {symbol}")
                    except Exception as e:
                        logger.error(f"Failed to train {symbol}: {e}")
                        continue

                logger.info(f"Simple training completed: {len(all_results)}/{len(symbols)} successful")

        else:
            # Parallel training mode
            logger.info("Using parallel training mode")

            if len(symbols) == 1:
                # Single symbol training
                results = run_single_symbol_training(
                    symbols[0], config, args.episodes, args.algorithm
                )
                logger.info(f"Training completed successfully")
                logger.info(f"Final results: {results['evaluation_results']['mean_reward']:.2f} ± {results['evaluation_results']['std_reward']:.2f}")

            else:
                # Multi-symbol training
                all_results = run_multi_symbol_training(
                    symbols, config, args.episodes, args.algorithm, args.multi_symbol_mode
                )

                # Summary statistics
                successful_trainings = len(all_results)
                if successful_trainings > 0:
                    mean_rewards = [r['evaluation_results']['mean_reward'] for r in all_results]
                    overall_mean = sum(mean_rewards) / len(mean_rewards)
                    logger.info(f"Multi-symbol training completed: {successful_trainings}/{len(symbols)} successful")
                    logger.info(f"Overall mean reward: {overall_mean:.2f}")
                else:
                    logger.error("All symbol trainings failed")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Write error summary
        from src.utils.error_logger import error_logger
        error_logger.write_summary()
        summary = error_logger.get_summary()
        if summary['total_errors'] > 0 or summary['total_warnings'] > 0:
            logger.info(f"📋 Error Summary: {summary['total_errors']} errors, {summary['total_warnings']} warnings logged to training_errors.txt")

if __name__ == "__main__":
    main()
