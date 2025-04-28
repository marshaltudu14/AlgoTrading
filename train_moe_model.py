"""
Training script for the Mixture of Experts (MoE) model.
Implements the full training pipeline with anti-overfitting measures.
"""
import os
import argparse
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Any

from models.moe_transformer import MoETransformer, create_moe_transformer
from models.datasets import (
    create_bc_dataloaders,
    create_rm_dataloaders,
    create_rl_dataset
)
from models.training import (
    train_behavioral_cloning,
    train_reward_model,
    train_rl_finetuning
)
from config import (
    INSTRUMENTS, 
    TIMEFRAMES, 
    MOE_CONFIG, 
    TRAINING_CONFIG, 
    RLHF_CONFIG,
    OVERFITTING_CONFIG,
    DEVICE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("moe_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the MoE model")
    
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
        "--num_experts",
        type=int,
        default=MOE_CONFIG["num_experts"],
        help="Number of experts in the MoE model"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=MOE_CONFIG["k"],
        help="Number of experts to use for each input"
    )
    
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=MOE_CONFIG["hidden_dim"],
        help="Hidden dimension for the model"
    )
    
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=OVERFITTING_CONFIG["ensemble_size"],
        help="Number of models in the ensemble"
    )
    
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        help="Use cross-validation"
    )
    
    parser.add_argument(
        "--balance_instruments",
        action="store_true",
        help="Balance data across instruments"
    )
    
    parser.add_argument(
        "--balance_signals",
        action="store_true",
        help="Balance data across signal types"
    )
    
    parser.add_argument(
        "--use_enhanced_features",
        action="store_true",
        help="Use enhanced features"
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


def train_moe_model(args):
    """
    Train the MoE model with the specified arguments.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting MoE model training...")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Update configurations based on command line arguments
    moe_config = MOE_CONFIG.copy()
    moe_config["num_experts"] = args.num_experts
    moe_config["k"] = args.k
    moe_config["hidden_dim"] = args.hidden_dim
    
    training_config = TRAINING_CONFIG.copy()
    training_config["bc_epochs"] = args.bc_epochs
    training_config["rm_epochs"] = args.rm_epochs
    training_config["rl_epochs"] = args.rl_epochs
    training_config["batch_size"] = args.batch_size
    training_config["learning_rate"] = args.learning_rate
    training_config["balance_instruments"] = args.balance_instruments
    training_config["balance_signals"] = args.balance_signals
    
    # Log configuration
    logger.info(f"Training on instruments: {args.instruments}")
    logger.info(f"Training on timeframes: {args.timeframes}")
    logger.info(f"MoE config: {moe_config}")
    logger.info(f"Training config: {training_config}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    # Create data loaders for behavioral cloning
    bc_train_loader, bc_val_loader = create_bc_dataloaders(
        instruments=args.instruments,
        timeframes=args.timeframes,
        batch_size=args.batch_size,
        validation_split=training_config["validation_split"]
    )
    
    # Create data loaders for reward modeling
    rm_train_loader, rm_val_loader = create_rm_dataloaders(
        instruments=args.instruments,
        timeframes=args.timeframes,
        batch_size=args.batch_size,
        validation_split=training_config["validation_split"]
    )
    
    # Create dataset for RL fine-tuning
    rl_dataset = create_rl_dataset(
        instruments=args.instruments,
        timeframes=args.timeframes
    )
    
    # Get state dimension from a sample
    sample_batch = next(iter(bc_train_loader))
    state_dim = sample_batch['features'].shape[2]
    
    # Train ensemble of models
    ensemble_models = []
    ensemble_reward_models = []
    
    for i in range(args.ensemble_size):
        logger.info(f"Training ensemble model {i+1}/{args.ensemble_size}...")
        
        # Create MoE model
        model = create_moe_transformer(
            config=moe_config,
            state_dim=state_dim,
            num_instruments=len(args.instruments),
            num_timeframes=len(args.timeframes),
            action_dim=3  # hold, buy, sell
        )
        
        # Create reward model
        from models.trading_transformer import create_reward_model
        reward_model = create_reward_model(
            config=RLHF_CONFIG,
            state_dim=state_dim,
            action_dim=3  # hold, buy, sell
        )
        
        # Step 1: Behavioral Cloning
        if not args.skip_bc:
            logger.info(f"Step 1: Behavioral Cloning (Ensemble {i+1})")
            model = train_behavioral_cloning(
                model=model,
                train_loader=bc_train_loader,
                val_loader=bc_val_loader,
                config=training_config,
                device=DEVICE
            )
            
            # Save BC model
            bc_model_path = os.path.join(args.save_dir, f'bc_model_ensemble_{i+1}.pt')
            torch.save(model.state_dict(), bc_model_path)
            logger.info(f"Saved BC model to {bc_model_path}")
        
        # Step 2: Reward Modeling
        if not args.skip_rm:
            logger.info(f"Step 2: Reward Modeling (Ensemble {i+1})")
            reward_model = train_reward_model(
                reward_model=reward_model,
                train_loader=rm_train_loader,
                val_loader=rm_val_loader,
                config=RLHF_CONFIG,
                device=DEVICE
            )
            
            # Save reward model
            reward_model_path = os.path.join(args.save_dir, f'reward_model_ensemble_{i+1}.pt')
            torch.save(reward_model.state_dict(), reward_model_path)
            logger.info(f"Saved reward model to {reward_model_path}")
        
        # Step 3: RL Fine-tuning
        if not args.skip_rl:
            logger.info(f"Step 3: RL Fine-tuning (Ensemble {i+1})")
            
            # Create a new model for RL fine-tuning
            rl_model = create_moe_transformer(
                config=moe_config,
                state_dim=state_dim,
                num_instruments=len(args.instruments),
                num_timeframes=len(args.timeframes),
                action_dim=3  # hold, buy, sell
            )
            
            # Load BC weights
            rl_model.load_state_dict(model.state_dict())
            
            # Fine-tune with RL
            model = train_rl_finetuning(
                model=rl_model,
                bc_model=model,
                reward_model=reward_model,
                dataset=rl_dataset,
                config=training_config,
                device=DEVICE
            )
        
        # Save final model
        model_path = os.path.join(args.save_dir, f'moe_model_ensemble_{i+1}.pt')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved final model to {model_path}")
        
        # Add to ensemble
        ensemble_models.append(model)
        ensemble_reward_models.append(reward_model)
    
    # Save ensemble metadata
    ensemble_metadata = {
        'ensemble_size': args.ensemble_size,
        'instruments': args.instruments,
        'timeframes': args.timeframes,
        'moe_config': moe_config,
        'training_config': {k: v for k, v in training_config.items() if not callable(v)},
        'model_paths': [f'moe_model_ensemble_{i+1}.pt' for i in range(args.ensemble_size)],
        'reward_model_paths': [f'reward_model_ensemble_{i+1}.pt' for i in range(args.ensemble_size)]
    }
    
    import json
    with open(os.path.join(args.save_dir, 'ensemble_metadata.json'), 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    logger.info("MoE model training completed successfully")
    
    return ensemble_models, ensemble_reward_models


def main():
    """Main function."""
    args = parse_args()
    train_moe_model(args)


if __name__ == "__main__":
    main()
