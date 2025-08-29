#!/usr/bin/env python3
"""
HRM Trading Model Training Script - Production Grade
Uses real market data from data/final directory with proper instrument/timeframe embeddings
"""

import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.models.hrm_trainer import HRMTrainer, HRMLossFunction
from src.models.hrm_trading_environment import HRMTradingEnvironment
from src.utils.data_loader import DataLoader
from src.utils.instruments_loader import get_instruments_loader
from src.env.trading_mode import TradingMode

# Configure basic logging to console only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HRMTrainingPipeline:
    """Production-grade HRM training pipeline with real market data"""
    
    def __init__(self, 
                 config_path: str = "config/hrm_config.yaml",
                 data_path: str = "data/final",
                 device: str = None):
        
        self.config_path = config_path
        self.data_path = data_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize instruments loader
        self.instruments_loader = get_instruments_loader()
        
        # Initialize components
        self.data_loader = None
        self.trainer = None
        self.available_instruments = []
        
        # Validate data directory (this populates available_instruments)
        self._validate_data_directory()
        
        logger.info(f"HRM Training Pipeline initialized on {self.device}")
        
    def _load_config(self) -> dict:
        """Load training configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _validate_data_directory(self):
        """Validate and discover available market data"""
        data_dir = Path(self.data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Find available feature files
        feature_files = list(data_dir.glob("features_*.csv"))
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {data_dir}")
        
        # Extract instrument names from filenames
        instrument_symbols = set()  # Use set to avoid duplicates
        for file in feature_files:
            # Extract symbol from filename like "features_Bank_Nifty_5.csv"
            filename = file.stem
            if filename.startswith("features_"):
                symbol_part = filename.replace("features_", "")
                # Extract base symbol (remove timeframe suffix if present)
                base_symbol = "_".join(symbol_part.split("_")[:-1]) if "_" in symbol_part else symbol_part
                instrument_symbols.add(base_symbol)
        
        instrument_symbols = list(instrument_symbols)  # Convert back to list
        
        # Validate instruments against instruments.yaml
        valid_instruments = []
        logger.info(f"Validating {len(instrument_symbols)} instrument symbols: {instrument_symbols}")
        
        for symbol in instrument_symbols:
            instrument_id = self.instruments_loader.get_instrument_id(symbol)
            if instrument_id is not None:
                valid_instruments.append(symbol)
                logger.info(f"Found valid instrument: {symbol} (ID: {instrument_id})")
            else:
                logger.warning(f"Instrument {symbol} not found in instruments.yaml")
        
        if not valid_instruments:
            raise ValueError("No valid instruments found that match instruments.yaml")
        
        self.available_instruments = valid_instruments
        logger.info(f"Found {len(valid_instruments)} valid instruments for training")
    
    def _find_actual_data_file(self, instrument_symbol: str, timeframe: str) -> str:
        """Find the actual data file with instrument+timeframe combination"""
        data_dir = Path(self.data_path)
        
        # Look for files matching the pattern: features_{instrument}_{timeframe}.csv
        expected_filename = f"features_{instrument_symbol}_{timeframe}.csv"
        expected_path = data_dir / expected_filename
        
        if expected_path.exists():
            logger.info(f"Found data file: {expected_filename}")
            return f"{instrument_symbol}_{timeframe}"
        
        # If exact match not found, list available files for this instrument
        available_files = list(data_dir.glob(f"features_{instrument_symbol}_*.csv"))
        if available_files:
            logger.warning(f"Exact file {expected_filename} not found. Available files:")
            for file in available_files:
                logger.warning(f"  {file.name}")
            
            # Use the first available file for this instrument
            first_file = available_files[0]
            # Extract symbol_timeframe from filename
            actual_symbol = first_file.stem.replace("features_", "")
            logger.info(f"Using available file: {first_file.name} -> symbol: {actual_symbol}")
            return actual_symbol
        
        raise FileNotFoundError(f"No data files found for instrument {instrument_symbol} with timeframe {timeframe}")
    
    def setup_training(self, instrument_symbol: str = None, timeframe: str = "5"):
        """Setup training for a specific instrument or auto-select"""
        
        logger.info(f"setup_training called with: {instrument_symbol}, timeframe: {timeframe}")
        logger.info(f"Available instruments: {self.available_instruments}")
        
        if not self.available_instruments:
            raise ValueError("No instruments available for training")
        
        if instrument_symbol is None:
            # Auto-select first available instrument
            instrument_symbol = self.available_instruments[0]
            logger.info(f"Auto-selected instrument: {instrument_symbol}")
        elif instrument_symbol not in self.available_instruments:
            raise ValueError(f"Instrument {instrument_symbol} not available. Available: {self.available_instruments}")
        
        # Find the actual data file with timeframe suffix
        actual_symbol_with_timeframe = self._find_actual_data_file(instrument_symbol, timeframe)
        
        # Initialize data loader
        self.data_loader = DataLoader(final_data_dir=self.data_path)
        
        # Initialize trainer with proper configuration
        self.trainer = HRMTrainer(
            config_path=self.config_path,
            data_path=self.data_path,
            device=str(self.device)
        )
        
        # Setup training environment with real instrument+timeframe
        self.trainer.setup_training(symbol=actual_symbol_with_timeframe)
        
        logger.info(f"Training setup completed for {instrument_symbol}")
        
        # Log model summary
        model_summary = self.trainer.model.get_model_summary()
        logger.info("HRM Model Summary:")
        for key, value in model_summary.items():
            logger.info(f"  {key}: {value}")
    
    def train(self, 
              epochs: int = 100,
              instrument_symbol: str = None,
              save_frequency: int = 25,
              log_frequency: int = 5,
              validation_frequency: int = 10):
        """Run training with production-grade monitoring"""
        
        # Setup training
        self.setup_training(instrument_symbol)
        
        logger.info(f"Starting HRM training for {epochs} epochs on {instrument_symbol or 'auto-selected'}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Total model parameters: {sum(p.numel() for p in self.trainer.model.parameters()):,}")
        
        try:
            # Run training
            training_history = self.trainer.train(
                episodes=epochs,
                save_frequency=save_frequency,
                log_frequency=log_frequency
            )
            
            # Final evaluation
            logger.info("Running final evaluation...")
            eval_results = self.trainer.evaluate(episodes=20)
            
            # Save training results
            self._save_training_results(training_history, eval_results)
            
            logger.info("Training completed successfully!")
            return training_history, eval_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _save_training_results(self, training_history, eval_results):
        """Save training results and model checkpoints"""
        
        results_dir = Path("training_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        history_file = results_dir / f"hrm_training_history_{timestamp}.json"
        import json
        with open(history_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json_history = []
            for episode in training_history:
                episode_json = {}
                for key, value in episode.items():
                    if isinstance(value, dict):
                        episode_json[key] = {k: convert_numpy(v) for k, v in value.items()}
                    else:
                        episode_json[key] = convert_numpy(value)
                json_history.append(episode_json)
            
            json.dump({
                'training_history': json_history,
                'evaluation_results': {k: convert_numpy(v) for k, v in eval_results.items()},
                'config': self.config,
                'timestamp': timestamp
            }, f, indent=2)
        
        logger.info(f"Training results saved to {history_file}")
    
    def validate_checkpoints(self):
        """Validate that model checkpoints can be loaded and used"""
        try:
            checkpoint_dir = Path("checkpoints/hrm")
            model_dir = Path("models/hrm")
            
            # Check for checkpoints
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            models = list(model_dir.glob("*.pt"))
            
            logger.info(f"Found {len(checkpoints)} checkpoints and {len(models)} saved models")
            
            if models:
                # Test loading the best model
                best_model_path = model_dir / "hrm_best_model.pt"
                if best_model_path.exists():
                    model_data = torch.load(best_model_path, map_location=self.device)
                    logger.info(f"Best model performance: {model_data.get('performance', 'N/A')}")
                    logger.info("Model checkpoints are valid and ready for live trading")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False


def main():
    """Main training entry point"""
    
    parser = argparse.ArgumentParser(description="HRM Trading Model Training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--instrument", type=str, help="Specific instrument to train on")
    parser.add_argument("--device", type=str, help="Training device (cuda/cpu)")
    parser.add_argument("--config", type=str, default="config/hrm_config.yaml", help="Config file path")
    parser.add_argument("--data", type=str, default="data/final", help="Data directory path")
    parser.add_argument("--test-only", action="store_true", help="Only run test training (1-2 epochs)")
    
    args = parser.parse_args()
    
    
    try:
        # Initialize training pipeline
        pipeline = HRMTrainingPipeline(
            config_path=args.config,
            data_path=args.data,
            device=args.device
        )
        
        # Determine number of epochs
        epochs = 2 if args.test_only else args.epochs
        
        logger.info(f"Starting {'test' if args.test_only else 'full'} training with {epochs} epochs")
        
        # Run training
        training_history, eval_results = pipeline.train(
            epochs=epochs,
            instrument_symbol=args.instrument,
            save_frequency=max(1, epochs // 4),  # Save 4 times during training
            log_frequency=max(1, epochs // 20)   # Log 20 times during training
        )
        
        # Validate checkpoints
        if pipeline.validate_checkpoints():
            logger.info("✓ Training completed and model is ready for live trading")
        else:
            logger.warning("⚠ Training completed but checkpoint validation failed")
        
        # Print summary
        print("\\n" + "="*80)
        print("HRM TRAINING COMPLETED")
        print("="*80)
        print(f"Epochs trained: {len(training_history)}")
        print(f"Final performance: {eval_results.get('mean_episode_reward', 'N/A'):.4f}")
        print(f"Best episode reward: {eval_results.get('best_episode', 'N/A'):.4f}")
        print(f"Model checkpoints: checkpoints/hrm/")
        print(f"Best model: models/hrm/hrm_best_model.pt")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()