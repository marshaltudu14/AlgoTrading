"""
Training script for the TradingTransformer model.
Runs the full training pipeline: behavioral cloning, reward modeling, and RL fine-tuning.
"""
import os
import argparse
import logging
import torch
from models.training import train_full_pipeline
from config import (
    INSTRUMENTS, 
    TIMEFRAMES, 
    TRANSFORMER_CONFIG, 
    TRAINING_CONFIG, 
    RLHF_CONFIG,
    DEVICE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the TradingTransformer model")
    
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=list(INSTRUMENTS.keys()),
        help="List of instruments to train on"
    )
    
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=TIMEFRAMES,
        help="List of timeframes to train on"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory to save models"
    )
    
    parser.add_argument(
        "--bc_epochs",
        type=int,
        default=TRAINING_CONFIG["bc_epochs"],
        help="Number of epochs for behavioral cloning"
    )
    
    parser.add_argument(
        "--rm_epochs",
        type=int,
        default=TRAINING_CONFIG["rm_epochs"],
        help="Number of epochs for reward modeling"
    )
    
    parser.add_argument(
        "--rl_epochs",
        type=int,
        default=TRAINING_CONFIG["rl_epochs"],
        help="Number of epochs for RL fine-tuning"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--rlhf_weight",
        type=float,
        default=RLHF_CONFIG["reward_lr"],
        help="Learning rate for reward model"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=TRANSFORMER_CONFIG["hidden_dim"],
        help="Hidden dimension for transformer model"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=TRANSFORMER_CONFIG["num_layers"],
        help="Number of layers for transformer model"
    )
    
    parser.add_argument(
        "--num_heads",
        type=int,
        default=TRANSFORMER_CONFIG["num_heads"],
        help="Number of attention heads for transformer model"
    )
    
    parser.add_argument(
        "--skip_bc",
        action="store_true",
        help="Skip behavioral cloning stage"
    )
    
    parser.add_argument(
        "--skip_rm",
        action="store_true",
        help="Skip reward modeling stage"
    )
    
    parser.add_argument(
        "--skip_rl",
        action="store_true",
        help="Skip RL fine-tuning stage"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Update configurations based on command line arguments
    transformer_config = TRANSFORMER_CONFIG.copy()
    transformer_config["hidden_dim"] = args.hidden_dim
    transformer_config["num_layers"] = args.num_layers
    transformer_config["num_heads"] = args.num_heads
    
    training_config = TRAINING_CONFIG.copy()
    training_config["bc_epochs"] = args.bc_epochs
    training_config["rm_epochs"] = args.rm_epochs
    training_config["rl_epochs"] = args.rl_epochs
    training_config["batch_size"] = args.batch_size
    training_config["learning_rate"] = args.learning_rate
    
    rlhf_config = RLHF_CONFIG.copy()
    rlhf_config["reward_lr"] = args.rlhf_weight
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Training on instruments: {args.instruments}")
    logger.info(f"Training on timeframes: {args.timeframes}")
    logger.info(f"Transformer config: {transformer_config}")
    logger.info(f"Training config: {training_config}")
    logger.info(f"RLHF config: {rlhf_config}")
    
    # Train the model
    model, reward_model = train_full_pipeline(
        instruments=args.instruments,
        timeframes=args.timeframes,
        transformer_config=transformer_config,
        training_config=training_config,
        rlhf_config=rlhf_config,
        device=DEVICE,
        save_dir=args.save_dir
    )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
