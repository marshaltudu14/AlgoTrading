"""
Evaluation script for the Mixture of Experts (MoE) model.
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
import json

from models.moe_transformer import MoETransformer, create_moe_transformer
from envs.trading_env import TradingEnv
from config import (
    INSTRUMENTS, 
    TIMEFRAMES, 
    MOE_CONFIG, 
    DEVICE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("moe_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the MoE model")
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing trained models"
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
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--use_ensemble",
        action="store_true",
        help="Use ensemble of models"
    )
    
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=3,
        help="Number of models in the ensemble"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic predictions"
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate performance plots"
    )
    
    parser.add_argument(
        "--use_enhanced_features",
        action="store_true",
        help="Use enhanced features"
    )
    
    return parser.parse_args()


class MoEEnsemble:
    """
    Ensemble of MoE models for more robust predictions.
    """
    def __init__(
        self,
        models: List[MoETransformer],
        device: torch.device = DEVICE
    ):
        """
        Initialize the ensemble.
        
        Args:
            models: List of MoE models
            device: Device to run inference on
        """
        self.models = models
        self.device = device
        
        # Set all models to eval mode
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict(
        self,
        states: torch.Tensor,
        instrument_id: torch.Tensor,
        timeframe_id: torch.Tensor,
        deterministic: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get ensemble prediction.
        
        Args:
            states: Input states
            instrument_id: Instrument IDs
            timeframe_id: Timeframe IDs
            deterministic: Whether to use deterministic prediction
            
        Returns:
            Tuple of (actions, extra_info)
        """
        # Get predictions from all models
        all_actions = []
        all_probs = []
        all_values = []
        all_risks = []
        
        with torch.no_grad():
            for model in self.models:
                actions, extra_info = model.get_action(
                    states, 
                    instrument_id, 
                    timeframe_id, 
                    deterministic=deterministic
                )
                
                all_actions.append(actions)
                all_probs.append(extra_info['action_probs'])
                all_values.append(extra_info['state_values'])
                all_risks.append(extra_info['risk_assessment'])
        
        # Stack predictions
        actions_stack = torch.stack(all_actions)
        probs_stack = torch.stack(all_probs)
        values_stack = torch.stack(all_values)
        risks_stack = torch.stack(all_risks)
        
        # Compute ensemble prediction
        if deterministic:
            # Majority vote
            actions, _ = torch.mode(actions_stack, dim=0)
        else:
            # Average probabilities
            avg_probs = torch.mean(probs_stack, dim=0)
            dist = torch.distributions.Categorical(avg_probs)
            actions = dist.sample()
        
        # Average other metrics
        avg_values = torch.mean(values_stack, dim=0)
        avg_risks = torch.mean(risks_stack, dim=0)
        
        # Compute uncertainty
        action_uncertainty = torch.std(probs_stack, dim=0)
        value_uncertainty = torch.std(values_stack, dim=0)
        risk_uncertainty = torch.std(risks_stack, dim=0)
        
        # Create extra info
        extra_info = {
            'action_probs': torch.mean(probs_stack, dim=0),
            'state_values': avg_values,
            'risk_assessment': avg_risks,
            'action_uncertainty': action_uncertainty,
            'value_uncertainty': value_uncertainty,
            'risk_uncertainty': risk_uncertainty
        }
        
        return actions, extra_info


def load_ensemble_models(
    model_dir: str,
    ensemble_size: int,
    state_dim: int,
    num_instruments: int,
    num_timeframes: int
) -> List[MoETransformer]:
    """
    Load ensemble of MoE models.
    
    Args:
        model_dir: Directory containing trained models
        ensemble_size: Number of models in the ensemble
        state_dim: Dimension of the state space
        num_instruments: Number of instruments
        num_timeframes: Number of timeframes
        
    Returns:
        List of MoE models
    """
    models = []
    
    # Check if ensemble metadata exists
    metadata_path = os.path.join(model_dir, 'ensemble_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Use metadata to load models
        moe_config = metadata.get('moe_config', MOE_CONFIG)
        model_paths = metadata.get('model_paths', [])
        
        # Limit to specified ensemble size
        model_paths = model_paths[:ensemble_size]
        
        for model_path in model_paths:
            # Create model
            model = create_moe_transformer(
                config=moe_config,
                state_dim=state_dim,
                num_instruments=num_instruments,
                num_timeframes=num_timeframes,
                action_dim=3  # hold, buy, sell
            )
            
            # Load weights
            full_path = os.path.join(model_dir, model_path)
            if os.path.exists(full_path):
                model.load_state_dict(torch.load(full_path, map_location=DEVICE))
                models.append(model)
            else:
                logger.warning(f"Model file not found: {full_path}")
    else:
        # Load models based on naming convention
        for i in range(1, ensemble_size + 1):
            model_path = os.path.join(model_dir, f'moe_model_ensemble_{i}.pt')
            
            if os.path.exists(model_path):
                # Create model
                model = create_moe_transformer(
                    config=MOE_CONFIG,
                    state_dim=state_dim,
                    num_instruments=num_instruments,
                    num_timeframes=num_timeframes,
                    action_dim=3  # hold, buy, sell
                )
                
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                models.append(model)
            else:
                logger.warning(f"Model file not found: {model_path}")
    
    logger.info(f"Loaded {len(models)} models for ensemble")
    return models


def evaluate_model(args):
    """
    Evaluate the MoE model on historical data.
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting MoE model evaluation...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results
    records = []
    
    # Evaluate on each instrument and timeframe
    for instrument in args.instruments:
        logger.info(f"Evaluating instrument: {instrument}")
        
        for timeframe in args.timeframes:
            logger.info(f"  Timeframe: {timeframe}...")
            
            # Create environment
            env = TradingEnv(
                instrument=instrument,
                timeframe=timeframe,
                include_position_features=True,
                use_enhanced_features=args.use_enhanced_features
            )
            
            # Reset environment
            obs, _ = env.reset()
            
            # Get state dimension
            state_dim = obs.shape[1]
            
            # Load models
            if args.use_ensemble:
                # Load ensemble of models
                models = load_ensemble_models(
                    model_dir=args.model_dir,
                    ensemble_size=args.ensemble_size,
                    state_dim=state_dim,
                    num_instruments=len(args.instruments),
                    num_timeframes=len(args.timeframes)
                )
                
                # Create ensemble
                model = MoEEnsemble(models)
            else:
                # Load single model
                model_path = os.path.join(args.model_dir, 'moe_model.pt')
                
                if not os.path.exists(model_path):
                    # Try ensemble model 1
                    model_path = os.path.join(args.model_dir, 'moe_model_ensemble_1.pt')
                
                if not os.path.exists(model_path):
                    logger.error(f"Model file not found: {model_path}")
                    continue
                
                # Create model
                model = create_moe_transformer(
                    config=MOE_CONFIG,
                    state_dim=state_dim,
                    num_instruments=len(args.instruments),
                    num_timeframes=len(args.timeframes),
                    action_dim=3  # hold, buy, sell
                )
                
                # Load weights
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()
            
            # Initialize metrics
            done = False
            steps = 0
            actions = []
            capitals = [env.capital]
            positions = [env.position]
            
            # Get instrument and timeframe IDs
            from config import INSTRUMENT_IDS, TIMEFRAME_IDS
            instrument_id = INSTRUMENT_IDS.get(instrument, 0)
            timeframe_id = TIMEFRAME_IDS.get(timeframe, 0)
            
            # Run episode
            while not done:
                # Get action from model
                features = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                instrument_id_tensor = torch.tensor([instrument_id], dtype=torch.long).to(DEVICE)
                timeframe_id_tensor = torch.tensor([timeframe_id], dtype=torch.long).to(DEVICE)
                
                # Get action
                if args.use_ensemble:
                    action, extra_info = model.predict(
                        features,
                        instrument_id_tensor,
                        timeframe_id_tensor,
                        deterministic=args.deterministic
                    )
                    action = action.item()
                else:
                    with torch.no_grad():
                        action, extra_info = model.get_action(
                            features,
                            instrument_id_tensor,
                            timeframe_id_tensor,
                            deterministic=args.deterministic
                        )
                        action = action.item()
                
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
            if args.plot:
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
                plt.savefig(os.path.join(args.output_dir, f"{instrument}_{timeframe}.png"))
                plt.close()
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save results
    df.to_csv(os.path.join(args.output_dir, "evaluation_results.csv"), index=False)
    
    # Print summary
    logger.info("\nEvaluation summary:")
    logger.info(f"Average return: {df['returns'].mean():.2f}%")
    logger.info(f"Average max drawdown: {df['max_drawdown'].mean():.2f}")
    logger.info(f"Average trade count: {df['trade_count'].mean():.2f}")
    logger.info(f"Average win rate: {df['win_rate'].mean():.2f}")
    logger.info(f"Average hold percentage: {df['hold_pct'].mean():.2f}%")
    logger.info(f"Average buy percentage: {df['buy_pct'].mean():.2f}%")
    logger.info(f"Average sell percentage: {df['sell_pct'].mean():.2f}%")
    
    # Generate summary plot
    if args.plot:
        plt.figure(figsize=(12, 8))
        
        # Plot returns by instrument
        plt.subplot(2, 2, 1)
        df.groupby('instrument')['returns'].mean().plot(kind='bar')
        plt.title("Average Returns by Instrument")
        plt.ylabel("Returns (%)")
        plt.grid(True)
        
        # Plot win rate by instrument
        plt.subplot(2, 2, 2)
        df.groupby('instrument')['win_rate'].mean().plot(kind='bar')
        plt.title("Average Win Rate by Instrument")
        plt.ylabel("Win Rate")
        plt.grid(True)
        
        # Plot trade count by instrument
        plt.subplot(2, 2, 3)
        df.groupby('instrument')['trade_count'].mean().plot(kind='bar')
        plt.title("Average Trade Count by Instrument")
        plt.ylabel("Trade Count")
        plt.grid(True)
        
        # Plot action distribution
        plt.subplot(2, 2, 4)
        action_data = df[['hold_pct', 'buy_pct', 'sell_pct']].mean()
        action_data.plot(kind='pie', autopct='%1.1f%%')
        plt.title("Action Distribution")
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "summary.png"))
        plt.close()
    
    logger.info("MoE model evaluation completed successfully")
    
    return df


def main():
    """Main function."""
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
