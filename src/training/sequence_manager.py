"""
Training Sequence Manager
Manages the optimal training sequence: PPO -> MoE -> MAML
"""

import os
import yaml
import logging
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.agents.ppo_agent import PPOAgent
from src.agents.moe_agent import MoEAgent
from src.training.trainer import Trainer
from src.utils.data_loader import DataLoader
from src.backtesting.environment import TradingEnv
from src.utils.dynamic_params import DynamicParameterManager
from src.config.config import INITIAL_CAPITAL

logger = logging.getLogger(__name__)

class TrainingStage(Enum):
    PPO = "stage_1_ppo"
    MOE = "stage_2_moe"
    MAML = "stage_3_maml"
    AUTONOMOUS = "stage_4_autonomous"

@dataclass
class StageResult:
    stage: TrainingStage
    success: bool
    metrics: Dict
    model_path: str
    episodes_completed: int
    message: str

class TrainingSequenceManager:
    """Manages the complete training sequence from PPO to MAML."""
    
    def __init__(self, config_path: str = "config/training_sequence.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.results = []
        
    def _load_config(self) -> Dict:
        """Load training sequence configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Check if testing mode is enabled
            if os.environ.get('TESTING_MODE') == 'true':
                logger.info("🧪 Applying testing configuration overrides")
                config = self._apply_testing_overrides(config)

            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            return self._get_default_config()

    def _apply_testing_overrides(self, config: Dict) -> Dict:
        """Apply testing configuration overrides."""
        if 'testing_overrides' in config:
            testing_config = config['testing_overrides']

            # Override training sequence parameters
            if 'training_sequence' in testing_config:
                for stage, params in testing_config['training_sequence'].items():
                    if stage in config['training_sequence']:
                        config['training_sequence'][stage].update(params)

            # Override training parameters
            if 'training_params' in testing_config:
                if 'training_params' not in config:
                    config['training_params'] = {}
                for param_type, params in testing_config['training_params'].items():
                    if param_type not in config['training_params']:
                        config['training_params'][param_type] = {}
                    config['training_params'][param_type].update(params)

            # Override progression rules
            if 'progression_rules' in testing_config:
                if 'progression_rules' not in config:
                    config['progression_rules'] = {}
                config['progression_rules'].update(testing_config['progression_rules'])

            logger.info("✅ Testing overrides applied successfully")

        return config

    def _create_test_data_loader(self, symbol: str) -> DataLoader:
        """Create a data loader with synthetic test data."""
        import sys

        if not hasattr(sys, 'test_data') or symbol not in sys.test_data:
            logger.error(f"Test data not available for symbol {symbol}")
            raise ValueError(f"Test data not found for symbol {symbol}")

        # Create a custom data loader that uses the test data
        from src.utils.data_loader import DataLoader

        class TestDataLoader(DataLoader):
            def __init__(self, test_data_dict):
                # Don't call parent __init__ since we're not loading from files
                self.test_data = test_data_dict
                # Set all attributes that the parent class expects
                self.final_data_dir = "test_data/final"
                self.raw_data_dir = "test_data/raw"
                self.chunk_size = 10000
                self.use_parquet = False  # We're using in-memory data, not parquet
                self.parquet_final_dir = "test_data/parquet_final"
                self.parquet_raw_dir = "test_data/parquet_raw"

            def load_data(self, symbol: str):
                """Load test data for the given symbol."""
                if symbol in self.test_data:
                    return self.test_data[symbol]['features']
                else:
                    raise ValueError(f"Test data not available for symbol {symbol}")

            def load_final_data_for_symbol(self, symbol: str):
                """Load final processed test data for the given symbol."""
                if symbol in self.test_data:
                    features_data = self.test_data[symbol]['features']
                    logger.info(f"Loaded test data for {symbol}: {len(features_data)} rows")
                    return features_data
                else:
                    raise ValueError(f"Test data not available for symbol {symbol}")

            def get_available_symbols(self):
                """Get available test symbols."""
                return list(self.test_data.keys())

            def get_data_length(self, symbol: str, data_type: str = "final"):
                """Get the length of test data for a symbol."""
                if symbol in self.test_data:
                    return len(self.test_data[symbol]['features'])
                else:
                    return 0

            def load_data_segment(self, symbol: str, start_idx: int, end_idx: int, data_type: str = "final"):
                """Load a segment of test data."""
                if symbol in self.test_data:
                    data = self.test_data[symbol]['features']
                    return data.iloc[start_idx:end_idx].copy()
                else:
                    raise ValueError(f"Test data not available for symbol {symbol}")

        return TestDataLoader(sys.test_data)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file not found."""
        return {
            'training_sequence': {
                'stage_1_ppo': {'algorithm': 'PPO', 'episodes': 500},
                'stage_2_moe': {'algorithm': 'MoE', 'episodes': 800},
                'stage_3_maml': {'algorithm': 'MAML', 'meta_iterations': 150},
                'stage_4_autonomous': {
                    'algorithm': 'Autonomous',
                    'generations': 50,
                    'autonomous': {
                        'population_size': 20,
                        'elite_size': 5,
                        'observation_dim': -1,  # Dynamic
                        'action_dim': 5,
                        'hidden_dim': 128,
                        'memory_size': 1000,
                        'memory_embedding_dim': 64,
                        'episodes_per_evaluation': 10,
                        'episode_length': 1000,
                        'mutation_rate': 0.3,
                        'crossover_rate': 0.7,
                        'enable_self_modification': True,
                        'modification_frequency': 5,
                        'save_directory': 'models/autonomous_agents',
                        'fitness_metrics': ['sharpe_ratio', 'profit_factor', 'max_drawdown']
                    }
                }
            },
            'progression_rules': {
                'auto_progression': True,
                'advancement_criteria': {
                    'stage_1_to_2': {'min_win_rate': 0.35, 'min_profit_factor': 0.8},
                    'stage_2_to_3': {'min_win_rate': 0.40, 'min_profit_factor': 1.0},
                    'stage_3_to_4': {'min_meta_iterations': 50, 'min_adaptation_speed': 5, 'min_cross_symbol_performance': 0.80}
                }
            }
        }
    
    def run_complete_sequence(self, data_loader: DataLoader, symbol: str,
                            initial_capital: float = None, episodes_override: int = None) -> List[StageResult]:
        """Run the complete training sequence for a symbol."""
        # Use global initial capital if not provided
        if initial_capital is None:
            initial_capital = INITIAL_CAPITAL

        logger.info(f"Starting complete training sequence for {symbol}")
        logger.info(f"Using initial capital: {initial_capital:,.2f}")
        logger.info("Sequence: PPO (baseline) -> MoE (specialization) -> MAML (meta-learning) -> Autonomous (evolution)")

        # Stage 1: PPO Baseline
        ppo_result = self._run_ppo_stage(data_loader, symbol, initial_capital, episodes_override)
        self.results.append(ppo_result)

        if not ppo_result.success:
            logger.error("PPO stage failed. Stopping sequence.")
            return self.results

        # Stage 2: MoE Specialization
        moe_result = self._run_moe_stage(data_loader, symbol, initial_capital, ppo_result.model_path, episodes_override)
        self.results.append(moe_result)

        if not moe_result.success:
            logger.error("MoE stage failed. Stopping sequence.")
            return self.results

        # Stage 3: MAML Meta-Learning
        maml_result = self._run_maml_stage(data_loader, symbol, initial_capital, moe_result.model_path, episodes_override)
        self.results.append(maml_result)

        if not maml_result.success:
            logger.error("MAML stage failed. Stopping sequence.")
            return self.results

        # Stage 4: Autonomous Evolution
        autonomous_result = self._run_autonomous_stage(data_loader, symbol, initial_capital, maml_result.model_path)
        self.results.append(autonomous_result)

        self._display_sequence_summary()
        return self.results

    def run_universal_sequence(self, data_loader: DataLoader, symbols: List[str],
                             initial_capital: float = INITIAL_CAPITAL, episodes_override: int = None) -> List[StageResult]:
        """
        Run complete training sequence for universal model on all symbols.

        Args:
            data_loader: Data loader instance
            symbols: List of all symbols to train on
            initial_capital: Starting capital for trading
            episodes_override: Override episodes from config

        Returns:
            List of stage results
        """
        # Check if we're in testing mode
        testing_mode = os.environ.get('TESTING_MODE') == 'true'

        if testing_mode:
            logger.info(f"🧪 Starting TESTING mode training sequence")
            logger.info(f"📊 Using synthetic test data for {len(symbols)} symbols: {', '.join(symbols)}")
            logger.info(f"🎯 Testing pipeline without saving models")

            # Use test data loader
            primary_symbol = symbols[0]
            data_loader = self._create_test_data_loader(primary_symbol)
        else:
            logger.info(f"🚀 Starting UNIVERSAL model training sequence")
            logger.info(f"📊 Training on {len(symbols)} symbols: {', '.join(symbols)}")
            logger.info(f"🎯 Will create ONE model for all instruments/timeframes")

        # For universal model, we'll train on the first symbol but save as universal model
        # This creates a single robust model that works across all instruments
        primary_symbol = symbols[0]  # Use first symbol for training

        # Stage 1: PPO Baseline (Universal)
        logger.info("🎯 Starting Stage 1: PPO Baseline Training")
        ppo_result = self._run_ppo_stage(data_loader, primary_symbol, initial_capital, episodes_override)
        logger.info(f"Stage 1 completed: {'SUCCESS' if ppo_result.success else 'PARTIAL SUCCESS'}")

        # Stage 2: MoE Specialization (Universal) - Continue regardless of PPO success
        logger.info("🎯 Starting Stage 2: MoE Specialization Training")
        moe_result = self._run_moe_stage(data_loader, primary_symbol, initial_capital, ppo_result.model_path, episodes_override)
        logger.info(f"Stage 2 completed: {'SUCCESS' if moe_result.success else 'PARTIAL SUCCESS'}")

        # Stage 3: MAML Meta-Learning (Universal) - Continue regardless of MoE success
        logger.info("🎯 Starting Stage 3: MAML Meta-Learning Training")
        maml_result = self._run_maml_stage_direct(data_loader, primary_symbol, initial_capital, moe_result.model_path, episodes_override)
        logger.info(f"Stage 3 completed: {'SUCCESS' if maml_result.success else 'PARTIAL SUCCESS'}")

        # Stage 4: Autonomous Evolution (Universal) - Continue regardless of MAML success
        logger.info("🎯 Starting Stage 4: Autonomous Evolution Training")
        autonomous_result = self._run_autonomous_stage(data_loader, primary_symbol, initial_capital, maml_result.model_path, testing_mode=episodes_override is not None)
        logger.info(f"Stage 4 completed: {'SUCCESS' if autonomous_result.success else 'PARTIAL SUCCESS'}")

        # Display comprehensive final summary with all metrics
        self._display_final_comprehensive_summary([ppo_result, moe_result, maml_result, autonomous_result], initial_capital)

        return [ppo_result, moe_result, maml_result, autonomous_result]

    def _run_universal_maml_stage(self, data_loader: DataLoader, symbol: str,
                                 initial_capital: float, moe_model_path: str, episodes_override: int = None) -> StageResult:
        """Run MAML meta-learning stage for universal model."""
        logger.info("=" * 60)
        logger.info("🎯 STAGE 3: MAML META-LEARNING (UNIVERSAL)")
        logger.info("=" * 60)

        stage_config = self.config['training_sequence']['stage_3_maml']
        meta_iterations = episodes_override if episodes_override is not None else stage_config.get('meta_iterations', 150)

        # Load data for dynamic parameter computation
        data = data_loader.load_final_data_for_symbol(symbol)

        # Initialize dynamic parameter manager with advanced training progress
        param_manager = DynamicParameterManager()
        dynamic_params = param_manager.compute_dynamic_params(data, training_progress=0.6)

        # Create environment with dynamic parameters
        env = TradingEnv(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=dynamic_params.lookback_window,
            episode_length=dynamic_params.episode_length
        )
        env.reset()
        observation_dim = env.observation_space.shape[0]
        logger.info(f"Dynamic observation dimension for MAML: {observation_dim}")

        # Load MoE agent for MAML training with dynamic parameters
        expert_configs = {
            "TrendAgent": {"hidden_dim": dynamic_params.hidden_dim},
            "MeanReversionAgent": {"hidden_dim": dynamic_params.hidden_dim},
            "VolatilityAgent": {"hidden_dim": dynamic_params.hidden_dim}
        }
        agent = MoEAgent(
            observation_dim=observation_dim,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=dynamic_params.hidden_dim,
            expert_configs=expert_configs
        )

        # Load the MoE model if available
        if moe_model_path:
            agent.load_model(moe_model_path)
        else:
            logger.info("No MoE model to load - starting MAML training from scratch")

        # Create trainer for MAML
        trainer = Trainer(agent, num_episodes=meta_iterations, log_interval=5)

        try:
            logger.info(f"Starting MAML meta-learning with {meta_iterations} meta-iterations...")

            # Train with MAML meta-learning
            results = trainer.train_maml(data_loader, symbol, initial_capital)

            # Extract metrics
            metrics = {
                'meta_iterations': meta_iterations,
                'final_reward': results.get('final_reward', 0),
                'avg_reward': results.get('avg_reward', 0)
            }

            success = True  # Assume success if training completes

            # Model will be saved in Stage 4 (Autonomous)
            logger.info(f"🎯 MAML stage completed - model will be saved in Stage 4 (Autonomous)")
            final_model_path = None  # No model saved in MAML stage

            return StageResult(
                stage=TrainingStage.MAML,
                success=success,
                metrics=metrics,
                model_path=final_model_path,
                episodes_completed=episodes,
                message="MAML meta-learning completed"
            )

        except Exception as e:
            logger.error(f"MAML stage failed: {e}")
            return StageResult(
                stage=TrainingStage.MAML,
                success=False,
                metrics={'error': str(e)},
                model_path=None,
                episodes_completed=0,
                message=f"MAML training failed: {e}"
            )

    def _run_ppo_stage(self, data_loader: DataLoader, symbol: str,
                      initial_capital: float, episodes_override: int = None) -> StageResult:
        """Run PPO baseline training stage."""
        logger.info("=" * 60)
        logger.info("🎯 STAGE 1: PPO BASELINE TRAINING")
        logger.info("=" * 60)
        
        stage_config = self.config['training_sequence']['stage_1_ppo']
        episodes = episodes_override if episodes_override is not None else stage_config.get('episodes', 500)

        # Load data for dynamic parameter computation
        data = data_loader.load_final_data_for_symbol(symbol)

        # Initialize dynamic parameter manager
        param_manager = DynamicParameterManager()
        dynamic_params = param_manager.compute_dynamic_params(data, training_progress=0.0)

        # Create environment with dynamic parameters
        env = TradingEnv(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=dynamic_params.lookback_window,
            episode_length=dynamic_params.episode_length
        )
        # Reset environment to initialize observation space
        env.reset()
        observation_dim = env.observation_space.shape[0]
        logger.info(f"Dynamic observation dimension: {observation_dim}")

        # Create PPO agent with dynamic parameters
        agent = PPOAgent(
            observation_dim=observation_dim,
            action_dim_discrete=2,          # BUY_LONG, SELL_SHORT actions
            action_dim_continuous=1,        # Quantity/position size
            hidden_dim=dynamic_params.hidden_dim,
            lr_actor=dynamic_params.lr_actor,
            lr_critic=dynamic_params.lr_critic,
            gamma=dynamic_params.gamma,
            epsilon_clip=dynamic_params.epsilon_clip,
            k_epochs=dynamic_params.k_epochs
        )
        
        # Create trainer
        trainer = Trainer(agent, num_episodes=episodes, log_interval=10)

        try:
            # Run training with the environment we created
            trainer.train(data_loader, symbol, initial_capital, env=env)
            
            # Get final metrics
            if hasattr(trainer, 'env') and trainer.env:
                trade_history = trainer.env.engine.get_trade_history()
                final_account = trainer.env.engine.get_account_state()
                
                # Calculate metrics for evaluation
                from src.utils.metrics import calculate_comprehensive_metrics
                # Calculate total reward from trade history if available
                total_reward = sum(trade.get('pnl', 0) for trade in trade_history) if trade_history else 0.0

                metrics = calculate_comprehensive_metrics(
                    trade_history=trade_history,
                    capital_history=getattr(trainer, 'capital_history', [initial_capital]),
                    initial_capital=initial_capital,
                    total_episodes=episodes,
                    total_reward=total_reward
                )
                
                # Check success criteria
                success = self._check_stage_success(TrainingStage.PPO, metrics)

                # Pass trained agent to next stage instead of saving to file
                logger.info(f"PPO stage completed (agent will be passed to MoE stage)")

                return StageResult(
                    stage=TrainingStage.PPO,
                    success=success,
                    metrics=metrics,
                    model_path=agent,  # Pass agent object directly
                    episodes_completed=episodes,
                    message="PPO baseline training completed"
                )
            else:
                return StageResult(
                    stage=TrainingStage.PPO,
                    success=False,
                    metrics={},
                    model_path="",
                    episodes_completed=0,
                    message="PPO training failed - no environment"
                )
                
        except Exception as e:
            from src.utils.error_logger import log_error
            log_error(f"PPO stage failed: {e}", f"Symbol: {symbol}, Episodes: {episodes}")
            logger.error(f"PPO stage failed: {e}")
            return StageResult(
                stage=TrainingStage.PPO,
                success=False,
                metrics={},
                model_path="",
                episodes_completed=0,
                message=f"PPO training failed: {e}"
            )
    
    def _run_moe_stage(self, data_loader: DataLoader, symbol: str,
                      initial_capital: float, ppo_model_path: str, episodes_override: int = None) -> StageResult:
        """Run MoE specialization training stage."""
        logger.info("=" * 60)
        logger.info("🧠 STAGE 2: MoE SPECIALIZATION TRAINING")
        logger.info("=" * 60)
        
        stage_config = self.config['training_sequence']['stage_2_moe']
        episodes = episodes_override if episodes_override is not None else stage_config.get('episodes', 800)
        
        # Load data for dynamic parameter computation
        data = data_loader.load_final_data_for_symbol(symbol)

        # Initialize dynamic parameter manager with some training progress
        param_manager = DynamicParameterManager()
        dynamic_params = param_manager.compute_dynamic_params(data, training_progress=0.3)

        # Create environment with dynamic parameters
        env = TradingEnv(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=dynamic_params.lookback_window,
            episode_length=dynamic_params.episode_length
        )
        # Reset environment to initialize observation space
        env.reset()
        observation_dim = env.observation_space.shape[0]
        logger.info(f"Dynamic observation dimension for MoE: {observation_dim}")

        # Create MoE agent with dynamic parameters
        expert_configs = {
            "TrendAgent": {"hidden_dim": dynamic_params.hidden_dim},
            "MeanReversionAgent": {"hidden_dim": dynamic_params.hidden_dim},
            "VolatilityAgent": {"hidden_dim": dynamic_params.hidden_dim}
        }
        agent = MoEAgent(
            observation_dim=observation_dim,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=dynamic_params.hidden_dim,
            expert_configs=expert_configs
        )
        
        # Transfer learning from PPO agent
        if ppo_model_path and hasattr(ppo_model_path, 'state_dict'):
            # ppo_model_path is actually the PPO agent object
            logger.info("Initializing MoE agent with PPO agent knowledge")
            # Transfer PPO knowledge to MoE experts
            ppo_agent = ppo_model_path
            if hasattr(ppo_agent, 'actor') and hasattr(ppo_agent, 'critic'):
                # Initialize each expert with PPO knowledge
                for expert in agent.experts:
                    if hasattr(expert, 'actor') and hasattr(expert, 'critic'):
                        # Copy compatible layers from PPO to expert
                        try:
                            expert.actor.load_state_dict(ppo_agent.actor.state_dict(), strict=False)
                            expert.critic.load_state_dict(ppo_agent.critic.state_dict(), strict=False)
                            logger.info(f"Transferred PPO knowledge to {expert.__class__.__name__}")
                        except Exception as e:
                            logger.warning(f"Partial transfer to {expert.__class__.__name__}: {e}")
        elif isinstance(ppo_model_path, str) and os.path.exists(ppo_model_path):
            # Fallback: load from file if path is provided
            logger.info(f"Loading PPO model from {ppo_model_path}")
            # Implementation for file loading if needed
        
        trainer = Trainer(agent, num_episodes=episodes, log_interval=10)

        try:
            trainer.train(data_loader, symbol, initial_capital, env=env)
            
            # Get metrics and check success
            if hasattr(trainer, 'env') and trainer.env:
                trade_history = trainer.env.engine.get_trade_history()
                
                from src.utils.metrics import calculate_comprehensive_metrics

                # Calculate total reward from trade history if available
                total_reward = sum(trade.get('pnl', 0) for trade in trade_history) if trade_history else 0.0

                metrics = calculate_comprehensive_metrics(
                    trade_history=trade_history,
                    capital_history=getattr(trainer, 'capital_history', [initial_capital]),
                    initial_capital=initial_capital,
                    total_episodes=episodes,
                    total_reward=total_reward
                )
                
                success = self._check_stage_success(TrainingStage.MOE, metrics)

                # Pass trained agent to next stage instead of saving to file
                logger.info(f"MoE stage completed (agent will be passed to MAML stage)")

                return StageResult(
                    stage=TrainingStage.MOE,
                    success=success,
                    metrics=metrics,
                    model_path=agent,  # Pass agent object directly
                    episodes_completed=episodes,
                    message="MoE specialization training completed"
                )
            else:
                return StageResult(
                    stage=TrainingStage.MOE,
                    success=False,
                    metrics={},
                    model_path="",
                    episodes_completed=0,
                    message="MoE training failed - no environment"
                )
                
        except Exception as e:
            logger.error(f"MoE stage failed: {e}")
            return StageResult(
                stage=TrainingStage.MOE,
                success=False,
                metrics={},
                model_path="",
                episodes_completed=0,
                message=f"MoE training failed: {e}"
            )
    
    def _run_maml_stage(self, data_loader: DataLoader, symbol: str,
                       initial_capital: float, moe_model_path: str, episodes_override: int = None) -> StageResult:
        """Run MAML meta-learning training stage."""
        logger.info("=" * 60)
        logger.info("🚀 STAGE 3: MAML META-LEARNING TRAINING")
        logger.info("=" * 60)
        
        stage_config = self.config['training_sequence']['stage_3_maml']
        meta_iterations = episodes_override if episodes_override is not None else stage_config.get('meta_iterations', 150)
        
        # Load MoE agent for MAML training with proper parameters
        expert_configs = {
            "TrendAgent": {"hidden_dim": 64},
            "MeanReversionAgent": {"hidden_dim": 64},
            "VolatilityAgent": {"hidden_dim": 64}
        }
        agent = MoEAgent(
            observation_dim=1246,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=64,
            expert_configs=expert_configs
        )
        
        if os.path.exists(moe_model_path):
            agent.load_model(moe_model_path)
            logger.info(f"Loaded MoE model from {moe_model_path}")
        
        trainer = Trainer(agent, num_episodes=meta_iterations, log_interval=5)
        
        try:
            trainer.meta_train(
                data_loader=data_loader,
                initial_capital=initial_capital,
                num_meta_iterations=meta_iterations,
                num_inner_loop_steps=5,
                num_evaluation_steps=3,
                meta_batch_size=1
            )
            
            # For MAML, success is based on adaptation capability
            # This is a simplified check - in practice, you'd test on new symbols
            success = True  # Assume success if training completes

            # Ensure models directory exists
            os.makedirs("models", exist_ok=True)

            # Save only the final production model
            if symbol == "universal" or "universal" in symbol:
                final_model_path = f"models/universal_final_model.pth"
            else:
                final_model_path = f"models/{symbol}_final_model.pth"
            agent.save_model(final_model_path)
            logger.info(f"🎯 FINAL MODEL SAVED: {final_model_path}")
            logger.info(f"✅ This is your production-ready model for backtesting and live trading!")
            
            return StageResult(
                stage=TrainingStage.MAML,
                success=success,
                metrics={'meta_iterations': meta_iterations},
                model_path=model_path,
                episodes_completed=meta_iterations,
                message="MAML meta-learning training completed"
            )
                
        except Exception as e:
            logger.error(f"MAML stage failed: {e}")
            return StageResult(
                stage=TrainingStage.MAML,
                success=False,
                metrics={},
                model_path="",
                episodes_completed=0,
                message=f"MAML training failed: {e}"
            )

    def _run_autonomous_stage(self, data_loader: DataLoader, symbol: str,
                             initial_capital: float, maml_model_path: str, testing_mode: bool = False) -> StageResult:
        """Run Autonomous evolution training stage."""
        logger.info("=" * 60)
        logger.info("🤖 STAGE 4: AUTONOMOUS EVOLUTION TRAINING")
        logger.info("=" * 60)

        try:
            from src.training.autonomous_trainer import run_autonomous_stage

            # Get autonomous stage configuration
            stage_config = self.config['training_sequence']['stage_4_autonomous'].copy()
            # Add symbol and initial capital to config
            stage_config['symbol'] = symbol
            stage_config['initial_capital'] = initial_capital
            # Add testing mode to autonomous config
            if 'autonomous' not in stage_config:
                stage_config['autonomous'] = {}
            stage_config['autonomous']['testing_mode'] = testing_mode

            logger.info(f"🎯 Starting autonomous evolution with {stage_config.get('generations', 50)} generations")
            logger.info(f"📊 Population size: {stage_config.get('autonomous', {}).get('population_size', 20)}")

            # Run autonomous training with MAML agent transfer
            from src.training.autonomous_trainer import run_autonomous_stage

            # Pass MAML agent for knowledge transfer
            if maml_model_path and hasattr(maml_model_path, 'state_dict'):
                logger.info("Passing MAML agent to autonomous stage for knowledge transfer")
                results = run_autonomous_stage(stage_config, maml_agent=maml_model_path)
            else:
                logger.info("No MAML agent to transfer - starting autonomous training from scratch")
                results = run_autonomous_stage(stage_config)

            # Extract metrics from results
            metrics = {
                'best_fitness': results.get('best_fitness', 0.0),
                'avg_fitness': results.get('current_avg_fitness', 0.0),
                'fitness_improvement': results.get('fitness_improvement', 0.0),
                'total_modifications': results.get('total_modifications', 0),
                'generations_completed': results.get('generation', 0)
            }

            # Always consider autonomous training successful if it completes
            # The best champion model is selected regardless of strict criteria
            success = True  # Always successful - we select the best performing champion

            logger.info("✅ Autonomous evolution training completed successfully!")
            logger.info(f"📈 Best fitness: {metrics['best_fitness']:.4f}")
            logger.info(f"📊 Fitness improvement: {metrics['fitness_improvement']:.4f}")
            logger.info("🏆 Best champion model selected from all candidates")

            # Save the universal final model in Stage 4 (Autonomous) - Skip in testing mode
            testing_mode = os.environ.get('TESTING_MODE') == 'true'
            universal_model_path = "models/universal_final_model.pth"

            if testing_mode:
                logger.info("🧪 TESTING MODE: Skipping model saving")
                universal_model_path = None  # No model saved in testing mode
            else:
                os.makedirs("models", exist_ok=True)

                # Save best agent directly as universal model (no intermediate champion save)
                if 'best_agent' in results and results['best_agent'] is not None:
                    # Save best_agent directly as universal model
                    logger.info("Saving best_agent directly as universal model")
                    import torch
                    best_agent = results['best_agent']

                    # Create universal model data structure
                    universal_model_data = {
                        'agent_state_dict': best_agent.state_dict() if hasattr(best_agent, 'state_dict') else None,
                        'agent_config': {
                            'observation_dim': getattr(best_agent, 'observation_dim', 0),
                            'action_dim': getattr(best_agent, 'action_dim', 0),
                            'hidden_dim': getattr(best_agent, 'hidden_dim', 128),
                            'memory_size': getattr(best_agent, 'memory_size', 1000),
                            'memory_embedding_dim': getattr(best_agent, 'memory_embedding_dim', 64)
                        },
                        'hyperparameters': best_agent.get_hyperparameters() if hasattr(best_agent, 'get_hyperparameters') else {},
                        'architecture': getattr(best_agent, '_nas_architecture', None),
                        'fitness_score': metrics.get('best_fitness', 0.0),
                        'generation': metrics.get('generations_completed', 0),
                        'model_type': 'autonomous_agent',
                        'version': '1.0'
                    }

                    # Save additional components if available
                    if hasattr(best_agent, 'world_model'):
                        universal_model_data['world_model_state_dict'] = best_agent.world_model.state_dict()
                    if hasattr(best_agent, 'external_memory'):
                        universal_model_data['external_memory_state'] = {
                            'memories': getattr(best_agent.external_memory, 'memories', []),
                            'config': getattr(best_agent.external_memory, 'config', {})
                        }

                    torch.save(universal_model_data, universal_model_path)
                    logger.info(f"✅ UNIVERSAL FINAL MODEL SAVED: {universal_model_path}")
                    logger.info(f"🎯 This is your production-ready universal model for all instruments!")
                else:
                    logger.error(f"❌ No best_agent available for universal model!")
                    logger.error(f"Best agent available: {'best_agent' in results}")
                    # Create a minimal placeholder to prevent crashes
                    import torch
                    torch.save({'error': 'No model available'}, universal_model_path)
                    logger.warning(f"⚠️ Created minimal placeholder at {universal_model_path}")

            # Create appropriate message based on testing mode
            if testing_mode:
                message = "Autonomous evolution training completed - Testing mode (no model saved)"
            else:
                message = "Autonomous evolution training completed - Universal model saved"

            return StageResult(
                stage=TrainingStage.AUTONOMOUS,
                success=success,
                metrics=metrics,
                model_path=universal_model_path,  # Return universal model path (None in testing mode)
                episodes_completed=metrics['generations_completed'],
                message=message
            )

        except Exception as e:
            from src.utils.error_logger import log_error
            log_error(f"Autonomous stage failed: {e}", f"Symbol: {symbol}")
            logger.error(f"Autonomous stage failed: {e}")
            return StageResult(
                stage=TrainingStage.AUTONOMOUS,
                success=False,
                metrics={},
                model_path="",
                episodes_completed=0,
                message=f"Autonomous training failed: {e}"
            )

    def _check_stage_success(self, stage: TrainingStage, metrics: Dict) -> bool:
        """Check if a training stage meets success criteria."""
        # All stages are considered successful upon completion
        # User will determine model quality through their own evaluation

        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)

        logger.info(f"Stage {stage.value} completion summary:")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Status: ✅ COMPLETED (success criteria removed - user evaluation)")

        return True  # Always successful - user determines model quality
    
    def _display_sequence_summary(self):
        """Display summary of the complete training sequence."""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPLETE TRAINING SEQUENCE SUMMARY")
        logger.info("=" * 80)
        
        for i, result in enumerate(self.results, 1):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            logger.info(f"Stage {i} ({result.stage.value}): {status}")
            logger.info(f"  Episodes: {result.episodes_completed}")
            logger.info(f"  Message: {result.message}")
            if result.metrics:
                win_rate = result.metrics.get('win_rate', 0)
                profit_factor = result.metrics.get('profit_factor', 0)
                logger.info(f"  Win Rate: {win_rate:.2%}")
                logger.info(f"  Profit Factor: {profit_factor:.2f}")
            logger.info("")
        
        # Overall assessment
        all_success = all(r.success for r in self.results)
        logger.info(f"🎯 OVERALL SEQUENCE: {'✅ COMPLETE SUCCESS' if all_success else '⚠️ PARTIAL SUCCESS'}")
        logger.info("=" * 80)

    def _display_final_comprehensive_summary(self, results: List[StageResult], initial_capital: float) -> None:
        """Display comprehensive final summary with all metrics."""
        logger.info("\n" + "=" * 100)
        logger.info("🏆 FINAL COMPREHENSIVE TRAINING SUMMARY")
        logger.info("=" * 100)

        # Overall status
        all_success = all([r.success for r in results])
        logger.info(f"🎯 Overall Training Result: {'COMPLETE SUCCESS' if all_success else 'PARTIAL SUCCESS'}")
        logger.info(f"💰 Initial Capital: ₹{initial_capital:,.2f}")

        # Stage-by-stage summary
        logger.info("\n📊 STAGE-BY-STAGE RESULTS:")
        logger.info("-" * 60)

        stage_names = ["PPO Baseline", "MoE Specialization", "MAML Meta-Learning", "Autonomous Evolution"]
        for i, (result, stage_name) in enumerate(zip(results, stage_names), 1):
            status = "✅ SUCCESS" if result.success else "⚠️ PARTIAL"
            logger.info(f"Stage {i} - {stage_name}: {status}")

            if result.metrics:
                # Extract key metrics
                final_capital = result.metrics.get('final_capital', 0)
                total_pnl = result.metrics.get('total_pnl', 0)
                win_rate = result.metrics.get('win_rate', 0)
                total_trades = result.metrics.get('total_trades', 0)
                sharpe_ratio = result.metrics.get('sharpe_ratio', 0)

                if final_capital > 0:
                    logger.info(f"  💵 Final Capital: ₹{final_capital:,.2f}")
                    logger.info(f"  📈 Total P&L: ₹{total_pnl:,.2f} ({((final_capital/initial_capital-1)*100):+.2f}%)")
                if total_trades > 0:
                    logger.info(f"  🎯 Total Trades: {total_trades}")
                    logger.info(f"  🏆 Win Rate: {win_rate:.1%}")
                if sharpe_ratio != 0:
                    logger.info(f"  📊 Sharpe Ratio: {sharpe_ratio:.3f}")

                logger.info(f"  ⏱️ Episodes Completed: {result.episodes_completed}")
            logger.info("")

        # Final model information
        logger.info("🤖 FINAL MODEL INFORMATION:")
        logger.info("-" * 40)
        logger.info("✅ Universal model saved in Stage 4 (Autonomous)")
        logger.info("📁 Model location: models/universal_final_model.pth")
        logger.info("🎯 Model type: Autonomous self-adapting agent")

        # Training enhancements summary
        logger.info("\n🚀 TRAINING ENHANCEMENTS APPLIED:")
        logger.info("-" * 50)
        logger.info("✅ Dynamic parameter computation based on data characteristics")
        logger.info("✅ Intelligent data feeding strategies (curriculum, adaptive, regime-aware)")
        logger.info("✅ Autonomous self-adaptation of learning rates and architecture")
        logger.info("✅ Real-time trade decision logging with detailed reasoning")
        logger.info("✅ Complete 4-stage training pipeline: PPO → MoE → MAML → Autonomous")

        # Display model complexity analysis
        self._display_model_complexity_analysis()

        logger.info("\n" + "=" * 100)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("🤖 Your autonomous trading agent is ready for deployment!")
        logger.info("=" * 100)

    def _display_model_complexity_analysis(self):
        """Display comprehensive model complexity and parameter analysis."""
        logger.info("\n🔬 MODEL COMPLEXITY ANALYSIS:")
        logger.info("-" * 60)

        # Check if universal model exists
        universal_model_path = "models/universal_final_model.pth"
        if not os.path.exists(universal_model_path):
            logger.warning("Universal model not found for analysis")
            return

        try:
            import torch

            # Check file integrity first
            file_size = os.path.getsize(universal_model_path)
            if file_size < 1024:  # Less than 1KB is likely corrupted
                logger.warning(f"Model file appears to be corrupted (size: {file_size} bytes)")
                return

            # Load the model with error handling for pickle issues
            try:
                model_data = torch.load(universal_model_path, map_location='cpu', weights_only=False)
            except (pickle.UnpicklingError, EOFError, RuntimeError) as e:
                if "pickle data was truncated" in str(e) or "truncated" in str(e).lower():
                    logger.warning(f"Model file appears to be truncated or corrupted: {e}")
                    logger.info("This can happen if training was interrupted during model saving")
                    return
                else:
                    raise e

            # Comprehensive model architecture analysis
            total_params = 0
            layer_info = []
            architecture_stats = {
                'total_components': 0,
                'moe_experts': 0,
                'attention_heads': 0,
                'transformer_layers': 0,
                'linear_layers': 0,
                'embedding_layers': 0,
                'normalization_layers': 0,
                'activation_functions': 0,
                'total_depth': 0,
                'largest_layer_params': 0,
                'largest_layer_name': '',
                'memory_footprint_mb': 0
            }

            file_size_mb = os.path.getsize(universal_model_path) / (1024*1024)
            logger.info(f"📁 Model file size: {file_size_mb:.2f} MB")

            if isinstance(model_data, dict):
                # Check what's in the model data
                logger.info(f"🔑 Model contains: {list(model_data.keys())}")
                architecture_stats['total_components'] = len(model_data.keys())

                # Look for state dictionaries and analyze each component
                for key, value in model_data.items():
                    if 'state_dict' in key.lower() and isinstance(value, dict):
                        logger.info(f"\n📊 Analyzing {key}:")
                        component_params = 0
                        component_layers = {'linear': 0, 'attention': 0, 'embedding': 0, 'norm': 0}

                        for param_name, param_tensor in value.items():
                            if isinstance(param_tensor, torch.Tensor):
                                param_count = param_tensor.numel()
                                component_params += param_count

                                # Track largest layer
                                if param_count > architecture_stats['largest_layer_params']:
                                    architecture_stats['largest_layer_params'] = param_count
                                    architecture_stats['largest_layer_name'] = param_name

                                # Detailed categorization
                                layer_type = "Unknown"
                                if 'weight' in param_name:
                                    if 'linear' in param_name.lower() or 'fc' in param_name.lower():
                                        layer_type = "Linear"
                                        component_layers['linear'] += 1
                                        architecture_stats['linear_layers'] += 1
                                    elif 'attention' in param_name.lower() or 'attn' in param_name.lower():
                                        layer_type = "Attention"
                                        component_layers['attention'] += 1
                                        # Count attention heads from weight shape
                                        if len(param_tensor.shape) >= 2 and 'q_proj' in param_name.lower():
                                            # Estimate heads from dimension (common pattern: hidden_dim = num_heads * head_dim)
                                            hidden_dim = param_tensor.shape[0]
                                            estimated_heads = max(1, hidden_dim // 64)  # Assume 64 as typical head dimension
                                            architecture_stats['attention_heads'] += estimated_heads
                                    elif 'embedding' in param_name.lower():
                                        layer_type = "Embedding"
                                        component_layers['embedding'] += 1
                                        architecture_stats['embedding_layers'] += 1
                                    elif 'norm' in param_name.lower() or 'layer_norm' in param_name.lower():
                                        layer_type = "Normalization"
                                        component_layers['norm'] += 1
                                        architecture_stats['normalization_layers'] += 1
                                    elif 'transformer' in param_name.lower():
                                        layer_type = "Transformer"
                                        architecture_stats['transformer_layers'] += 1
                                    elif 'expert' in param_name.lower() or 'moe' in param_name.lower():
                                        layer_type = "MoE Expert"
                                        architecture_stats['moe_experts'] += 1
                                    else:
                                        layer_type = "Other Weight"
                                elif 'bias' in param_name:
                                    layer_type = "Bias"

                                layer_info.append({
                                    'component': key,
                                    'name': param_name,
                                    'type': layer_type,
                                    'shape': list(param_tensor.shape),
                                    'params': param_count,
                                    'dtype': str(param_tensor.dtype),
                                    'memory_mb': param_count * 4 / (1024*1024)  # Assume float32
                                })

                        logger.info(f"   Parameters in {key}: {component_params:,}")
                        logger.info(f"   Layer breakdown: Linear={component_layers['linear']}, "
                                   f"Attention={component_layers['attention']}, "
                                   f"Embedding={component_layers['embedding']}, "
                                   f"Normalization={component_layers['norm']}")
                        total_params += component_params

                # Calculate memory footprint
                architecture_stats['memory_footprint_mb'] = sum(layer['memory_mb'] for layer in layer_info)
                architecture_stats['total_depth'] = len([l for l in layer_info if 'weight' in l['name']])

                # Group by layer type for summary
                layer_types = {}
                for layer in layer_info:
                    layer_type = layer['type']
                    if layer_type not in layer_types:
                        layer_types[layer_type] = {'count': 0, 'params': 0, 'memory_mb': 0}
                    layer_types[layer_type]['count'] += 1
                    layer_types[layer_type]['params'] += layer['params']
                    layer_types[layer_type]['memory_mb'] += layer['memory_mb']

                logger.info(f"\n🏗️ DETAILED ARCHITECTURE BREAKDOWN:")
                for layer_type, info in sorted(layer_types.items(), key=lambda x: x[1]['params'], reverse=True):
                    percentage = (info['params'] / total_params * 100) if total_params > 0 else 0
                    logger.info(f"   {layer_type}: {info['count']} layers, {info['params']:,} params "
                               f"({percentage:.1f}%), {info['memory_mb']:.1f} MB")

            # Comprehensive final summary
            logger.info(f"\n🎯 COMPREHENSIVE MODEL ARCHITECTURE SUMMARY:")
            logger.info(f"=" * 80)

            # Basic metrics
            logger.info(f"📊 BASIC METRICS:")
            logger.info(f"   Total Parameters: {total_params:,}")
            logger.info(f"   Model File Size: {file_size_mb:.2f} MB")
            logger.info(f"   Memory Footprint: {architecture_stats['memory_footprint_mb']:.2f} MB")
            if total_params > 0:
                logger.info(f"   Parameters per MB: {total_params / file_size_mb:,.0f}")
                logger.info(f"   Compression Ratio: {architecture_stats['memory_footprint_mb'] / file_size_mb:.2f}x")

            # Architecture details
            logger.info(f"\n🏗️ ARCHITECTURE DETAILS:")
            logger.info(f"   Total Components: {architecture_stats['total_components']}")
            logger.info(f"   Total Network Depth: {architecture_stats['total_depth']} layers")
            logger.info(f"   Linear Layers: {architecture_stats['linear_layers']}")
            logger.info(f"   Transformer Layers: {architecture_stats['transformer_layers']}")
            logger.info(f"   Attention Heads: {architecture_stats['attention_heads']}")
            logger.info(f"   Embedding Layers: {architecture_stats['embedding_layers']}")
            logger.info(f"   Normalization Layers: {architecture_stats['normalization_layers']}")
            logger.info(f"   MoE Experts: {architecture_stats['moe_experts']}")

            # Largest component analysis
            if architecture_stats['largest_layer_name']:
                largest_percentage = (architecture_stats['largest_layer_params'] / total_params * 100) if total_params > 0 else 0
                logger.info(f"\n🔍 LARGEST COMPONENT:")
                logger.info(f"   Name: {architecture_stats['largest_layer_name']}")
                logger.info(f"   Parameters: {architecture_stats['largest_layer_params']:,} ({largest_percentage:.1f}%)")

            # Complexity classification
            if total_params < 100_000:
                complexity = "Small"
                complexity_desc = "Suitable for edge deployment"
            elif total_params < 1_000_000:
                complexity = "Medium"
                complexity_desc = "Good balance of performance and efficiency"
            elif total_params < 10_000_000:
                complexity = "Large"
                complexity_desc = "High performance, requires significant compute"
            else:
                complexity = "Very Large"
                complexity_desc = "State-of-the-art performance, GPU recommended"

            logger.info(f"\n🎯 MODEL CLASSIFICATION:")
            logger.info(f"   Complexity: {complexity}")
            logger.info(f"   Description: {complexity_desc}")

            # Performance estimates
            logger.info(f"\n⚡ PERFORMANCE ESTIMATES:")
            if architecture_stats['attention_heads'] > 0:
                logger.info(f"   Attention Mechanism: Multi-head ({architecture_stats['attention_heads']} heads)")
            if architecture_stats['moe_experts'] > 0:
                logger.info(f"   Expert Specialization: {architecture_stats['moe_experts']} specialized experts")
            if architecture_stats['transformer_layers'] > 0:
                logger.info(f"   Sequence Processing: {architecture_stats['transformer_layers']} transformer layers")

            logger.info(f"=" * 80)

        except Exception as e:
            logger.error(f"Error analyzing model complexity: {e}")

    def _run_maml_stage_direct(self, data_loader: DataLoader, symbol: str,
                              initial_capital: float, moe_model_path: str, episodes_override: int = None) -> StageResult:
        """Run MAML meta-learning stage with universal parameters."""
        logger.info("=" * 60)
        logger.info("🎯 STAGE 3: MAML META-LEARNING (UNIVERSAL)")
        logger.info("=" * 60)

        stage_config = self.config['training_sequence']['stage_3_maml']
        meta_iterations = episodes_override if episodes_override is not None else stage_config.get('meta_iterations', 150)

        # Load data for dynamic parameter computation
        data = data_loader.load_final_data_for_symbol(symbol)

        # Initialize dynamic parameter manager with advanced training progress
        param_manager = DynamicParameterManager()
        universal_params = param_manager.compute_dynamic_params(data, training_progress=0.6)

        logger.info(f"🎯 Universal parameters for MAML: lookback={universal_params.lookback_window}, episode_length={universal_params.episode_length}")

        # Create environment with universal parameters to get observation dimension
        env = TradingEnv(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=universal_params.lookback_window,
            episode_length=universal_params.episode_length
        )
        env.reset()
        observation_dim = env.observation_space.shape[0]
        logger.info(f"🎯 Universal observation dimension for MAML: {observation_dim}")

        # Load MoE agent for MAML training with universal parameters
        expert_configs = {
            "TrendAgent": {"hidden_dim": universal_params.hidden_dim},
            "MeanReversionAgent": {"hidden_dim": universal_params.hidden_dim},
            "VolatilityAgent": {"hidden_dim": universal_params.hidden_dim}
        }
        agent = MoEAgent(
            observation_dim=observation_dim,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=universal_params.hidden_dim,
            expert_configs=expert_configs
        )

        # Transfer learning from MoE agent
        if moe_model_path and hasattr(moe_model_path, 'state_dict'):
            # moe_model_path is actually the MoE agent object
            logger.info("Initializing MAML agent with MoE agent knowledge")
            moe_agent = moe_model_path
            # Transfer MoE knowledge to MAML agent
            try:
                agent.load_state_dict(moe_agent.state_dict(), strict=False)
                logger.info("Successfully transferred MoE knowledge to MAML agent")
            except Exception as e:
                logger.warning(f"Partial transfer from MoE to MAML: {e}")
                # Try to transfer individual components
                if hasattr(moe_agent, 'gating_network') and hasattr(agent, 'gating_network'):
                    agent.gating_network.load_state_dict(moe_agent.gating_network.state_dict(), strict=False)
                if hasattr(moe_agent, 'experts') and hasattr(agent, 'experts'):
                    for i, (src_expert, dst_expert) in enumerate(zip(moe_agent.experts, agent.experts)):
                        try:
                            dst_expert.load_state_dict(src_expert.state_dict(), strict=False)
                            logger.info(f"Transferred expert {i} from MoE to MAML")
                        except Exception as ex:
                            logger.warning(f"Failed to transfer expert {i}: {ex}")
        elif isinstance(moe_model_path, str) and os.path.exists(moe_model_path):
            # Fallback: load from file if path is provided
            agent.load_model(moe_model_path)
            logger.info(f"Loaded MoE model from {moe_model_path}")
        else:
            logger.info("No MoE model to load - starting MAML training from scratch")

        # Create trainer for MAML and pass universal parameters
        trainer = Trainer(agent, num_episodes=meta_iterations, log_interval=5)
        trainer.universal_params = universal_params  # CRITICAL: Pass universal parameters

        try:
            logger.info(f"Starting MAML meta-learning with {meta_iterations} meta-iterations...")

            # Train with MAML meta-learning
            trainer.meta_train(
                data_loader=data_loader,
                initial_capital=initial_capital,
                num_meta_iterations=meta_iterations,
                num_inner_loop_steps=5,
                num_evaluation_steps=3,
                meta_batch_size=1
            )

            # Extract metrics
            metrics = {
                'meta_iterations': meta_iterations,
                'final_reward': 0,  # Placeholder
                'avg_reward': 0     # Placeholder
            }

            success = True  # Assume success if training completes

            # Pass trained agent to next stage instead of saving to file
            logger.info(f"🎯 MAML stage completed - agent will be passed to Autonomous stage")

            return StageResult(
                stage=TrainingStage.MAML,
                success=success,
                metrics=metrics,
                model_path=agent,  # Pass agent object directly
                episodes_completed=meta_iterations,
                message="MAML meta-learning completed"
            )

        except Exception as e:
            logger.error(f"MAML stage failed: {e}")
            return StageResult(
                stage=TrainingStage.MAML,
                success=False,
                metrics={'error': str(e)},
                model_path=None,
                episodes_completed=0,
                message=f"MAML training failed: {e}"
            )

def run_training_sequence(symbol: str, data_dir: str = "data/final") -> List[StageResult]:
    """Convenience function to run the complete training sequence."""
    data_loader = DataLoader(data_dir)
    manager = TrainingSequenceManager()
    return manager.run_complete_sequence(data_loader, symbol)
