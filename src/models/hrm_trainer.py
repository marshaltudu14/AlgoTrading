import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json

from src.models.hrm import HRMTradingAgent, HRMCarry
from src.models.hrm_trading_environment import HRMTradingEnvironment, HRMTradingWrapper
from src.utils.data_loader import DataLoader
from src.env.trading_mode import TradingMode

logger = logging.getLogger(__name__)


class HRMLossFunction:
    """Multi-component loss function for HRM trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loss_weights = config['training']['loss_weights']
        
    def calculate_loss(self, 
                      outputs: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor],
                      segment_rewards: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate multi-component HRM loss"""
        
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Strategic loss (H-module decisions)
        if 'regime_probabilities' in outputs and 'true_regime' in targets:
            regime_loss = nn.CrossEntropyLoss()(
                outputs['regime_probabilities'], 
                targets['true_regime']
            )
            losses['strategic_loss'] = regime_loss
            total_loss = total_loss + self.loss_weights['strategic_loss'] * regime_loss
        
        # Tactical loss (L-module decisions)  
        if 'action_logits' in outputs and 'true_action' in targets:
            action_loss = nn.CrossEntropyLoss()(
                outputs['action_logits'],
                targets['true_action']
            )
            losses['tactical_loss'] = action_loss
            total_loss = total_loss + self.loss_weights['tactical_loss'] * action_loss
        
        # Quantity prediction loss
        if 'quantity' in outputs and 'true_quantity' in targets:
            quantity_loss = nn.MSELoss()(
                outputs['quantity'],
                targets['true_quantity']
            )
            losses['quantity_loss'] = quantity_loss
            total_loss = total_loss + 0.1 * quantity_loss
        
        # ACT loss (adaptive computation time)
        if 'halt_logits' in outputs and 'continue_logits' in outputs:
            # Reward-based ACT loss
            q_targets = self._calculate_act_targets(segment_rewards)
            
            halt_loss = nn.MSELoss()(
                outputs['halt_logits'].squeeze(),
                q_targets['halt']
            )
            continue_loss = nn.MSELoss()(
                outputs['continue_logits'].squeeze(),
                q_targets['continue']
            )
            
            act_loss = halt_loss + continue_loss
            losses['act_loss'] = act_loss
            total_loss = total_loss + self.loss_weights['act_loss'] * act_loss
        
        # Performance-based loss
        if segment_rewards:
            performance_loss = -torch.tensor(np.mean(segment_rewards))  # Maximize rewards
            losses['performance_loss'] = performance_loss
            total_loss = total_loss + self.loss_weights['performance_loss'] * performance_loss
        
        # Convert loss values to float for logging
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_values['total_loss'] = total_loss.item()
        
        return total_loss, loss_values
    
    def _calculate_act_targets(self, segment_rewards: List[float]) -> Dict[str, torch.Tensor]:
        """Calculate Q-learning targets for ACT"""
        
        if not segment_rewards:
            return {
                'halt': torch.tensor([0.0]),
                'continue': torch.tensor([0.0])
            }
        
        # Simple reward-based targets
        cumulative_reward = sum(segment_rewards)
        
        # Encourage halting when rewards are good, continuing when they're not
        if cumulative_reward > 0:
            halt_target = torch.sigmoid(torch.tensor([cumulative_reward]))
            continue_target = 1 - halt_target
        else:
            halt_target = torch.sigmoid(torch.tensor([cumulative_reward]))
            continue_target = 1 - halt_target
            
        return {
            'halt': halt_target,
            'continue': continue_target
        }


class HRMTrainer:
    """Trainer for HRM Trading Agent"""
    
    def __init__(self, 
                 config_path: str = "config/hrm_config.yaml",
                 data_path: str = "data/final",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = torch.device(device)
        self.config_path = config_path
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.data_loader = DataLoader(final_data_dir=data_path)
        self.loss_function = HRMLossFunction(self.config)
        
        # Training state
        self.current_episode = 0
        self.best_performance = float('-inf')
        self.training_history = []
        
        # Model and optimizer will be initialized during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, symbol: str = "Bank_Nifty_5"):
        """Setup training environment and model"""
        
        # Create training environment
        self.env = HRMTradingEnvironment(
            data_loader=self.data_loader,
            symbol=symbol,
            initial_capital=100000.0,
            mode=TradingMode.TRAINING,
            hrm_config_path=self.config_path,
            device=self.device
        )
        
        # Reset environment to initialize observation space
        _ = self.env.reset()
        
        # Model is initialized automatically in the environment
        self.model = self.env.hrm_agent
        
        # Setup optimizer
        training_config = self.config['training']
        
        # Different learning rates for different components
        param_groups = []
        
        # Strategic parameters (H-module) - slower learning
        strategic_params = list(self.model.high_level_module.parameters())
        param_groups.append({
            'params': strategic_params,
            'lr': training_config['learning_rates']['strategic_lr']
        })
        
        # Tactical parameters (L-module) - faster learning  
        tactical_params = list(self.model.low_level_module.parameters())
        param_groups.append({
            'params': tactical_params,
            'lr': training_config['learning_rates']['tactical_lr']
        })
        
        # ACT parameters - fast adaptation
        act_params = list(self.model.act_module.parameters())
        param_groups.append({
            'params': act_params,
            'lr': training_config['learning_rates']['act_lr']
        })
        
        # Deep supervision parameters
        ds_params = list(self.model.deep_supervision.parameters())
        param_groups.append({
            'params': ds_params,
            'lr': training_config['learning_rates']['base_lr']
        })
        
        # Create optimizer
        if training_config['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                param_groups,
                weight_decay=training_config['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(param_groups)
        
        # Learning rate scheduler
        if training_config['lr_schedule'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config.get('total_episodes', 1000),
                eta_min=training_config['learning_rates']['base_lr'] * training_config['min_lr_ratio']
            )
        
        logger.info("Training setup completed")
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode"""
        
        # Reset environment
        observation = self.env.reset()
        
        episode_metrics = {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'steps': 0,
            'hierarchical_metrics': {}
        }
        
        done = False
        step_count = 0
        
        while not done and step_count < 1000:  # Max steps per episode
            
            # Environment step (HRM handles the reasoning internally)
            observation, reward, done, info = self.env.step()
            
            episode_metrics['total_reward'] += reward
            step_count += 1
            
            # Extract loss information if available in info
            if 'segment_rewards' in info:
                # Create dummy targets for loss calculation
                # In practice, these would come from expert demonstrations or previous experience
                targets = self._create_dummy_targets(info)
                
                # Get the last outputs from the environment
                # This is a simplified version - real implementation would need proper output tracking
                if hasattr(self.env, 'current_carry') and hasattr(self.env.hrm_agent, '_last_outputs'):
                    outputs = getattr(self.env.hrm_agent, '_last_outputs', {})
                    
                    if outputs:
                        loss, loss_components = self.loss_function.calculate_loss(
                            outputs, 
                            targets, 
                            info['segment_rewards']
                        )
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        if self.config['training']['gradient_clip'] > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config['training']['gradient_clip']
                            )
                        
                        self.optimizer.step()
                        
                        episode_metrics['total_loss'] += loss.item()
        
        episode_metrics['steps'] = step_count
        episode_metrics['avg_reward'] = episode_metrics['total_reward'] / max(step_count, 1)
        episode_metrics['avg_loss'] = episode_metrics['total_loss'] / max(step_count, 1)
        
        # Get hierarchical performance metrics
        episode_metrics['hierarchical_metrics'] = self.env.get_hierarchical_performance_metrics()
        
        return episode_metrics
    
    def _create_dummy_targets(self, info: Dict) -> Dict[str, torch.Tensor]:
        """Create dummy targets for training - replace with real targets in production"""
        
        targets = {}
        
        # Dummy regime target (would be derived from market analysis)
        targets['true_regime'] = torch.randint(0, 5, (1,)).to(self.device)
        
        # Dummy action target (would be from expert demonstrations)
        targets['true_action'] = torch.randint(0, 5, (1,)).to(self.device)
        
        # Dummy quantity target
        targets['true_quantity'] = torch.rand(1, 1).to(self.device)
        
        return targets
    
    def train(self, 
              episodes: int = 1000, 
              symbol: str = None,
              save_frequency: int = 100,
              log_frequency: int = 10):
        """Main training loop"""
        
        # Setup training only if not already done
        if self.model is None:
            if symbol is None:
                raise ValueError("Symbol must be provided for initial setup")
            self.setup_training(symbol)
        
        logger.info(f"Starting HRM training for {episodes} episodes")
        
        for episode in range(episodes):
            self.current_episode = episode
            
            # Train episode
            metrics = self.train_episode()
            self.training_history.append(metrics)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            if episode % log_frequency == 0:
                self._log_training_progress(episode, metrics)
            
            # Save checkpoint (including first episode)
            if episode % save_frequency == 0:
                self._save_checkpoint(episode)
            
            # Check for improvement and save best model
            if metrics['avg_reward'] > self.best_performance:
                self.best_performance = metrics['avg_reward']
                self._save_best_model()
        
        logger.info("Training completed")
        return self.training_history
    
    def _log_training_progress(self, episode: int, metrics: Dict):
        """Log training progress"""
        
        logger.info(f"Episode {episode:4d} | "
                   f"Reward: {metrics['avg_reward']:6.3f} | "
                   f"Loss: {metrics['avg_loss']:6.3f} | "
                   f"Steps: {metrics['steps']:3d}")
        
        # Log hierarchical metrics if available
        h_metrics = metrics.get('hierarchical_metrics', {})
        if h_metrics:
            for category, values in h_metrics.items():
                if isinstance(values, dict):
                    for metric, value in values.items():
                        logger.debug(f"  {category}.{metric}: {value}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        
        checkpoint_dir = Path("checkpoints/hrm")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_performance': self.best_performance,
            'config': self.config,
            'training_history': self.training_history[-100:]  # Keep last 100 episodes
        }
        
        filepath = checkpoint_dir / f"hrm_checkpoint_episode_{episode}.pt"
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def _save_best_model(self):
        """Save the best performing model"""
        
        model_dir = Path("models/hrm")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'performance': self.best_performance,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = model_dir / "hrm_best_model.pt"
        torch.save(model_data, filepath)
        logger.info(f"Best model saved: {filepath} (performance: {self.best_performance:.4f})")
    
    def evaluate(self, 
                episodes: int = 10, 
                symbol: str = "Bank_Nifty_5") -> Dict[str, float]:
        """Evaluate trained model"""
        
        self.model.eval()
        
        evaluation_results = []
        
        with torch.no_grad():
            for episode in range(episodes):
                observation = self.env.reset()
                
                episode_reward = 0.0
                step_count = 0
                done = False
                
                while not done and step_count < 1000:
                    observation, reward, done, info = self.env.step()
                    episode_reward += reward
                    step_count += 1
                
                evaluation_results.append({
                    'episode_reward': episode_reward,
                    'steps': step_count,
                    'avg_reward': episode_reward / max(step_count, 1)
                })
        
        # Calculate aggregate statistics
        total_rewards = [r['episode_reward'] for r in evaluation_results]
        avg_rewards = [r['avg_reward'] for r in evaluation_results]
        
        results = {
            'mean_episode_reward': np.mean(total_rewards),
            'std_episode_reward': np.std(total_rewards),
            'mean_avg_reward': np.mean(avg_rewards),
            'std_avg_reward': np.std(avg_rewards),
            'best_episode': max(total_rewards),
            'worst_episode': min(total_rewards)
        }
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return results
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load training checkpoint and resume training"""
        
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.model is None:
            raise ValueError("Model must be initialized before loading checkpoint")
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.best_performance = checkpoint['best_performance']
        self.current_episode = checkpoint['episode']
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from episode {self.current_episode}")
        logger.info(f"Best performance so far: {self.best_performance:.4f}")
        
        return self.current_episode


def main():
    """Main function to run HRM training"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize trainer
    trainer = HRMTrainer(
        config_path="config/hrm_config.yaml",
        data_path="data/final",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    try:
        # Train the model
        history = trainer.train(
            episodes=500,
            symbol="Bank_Nifty_5",
            save_frequency=50,
            log_frequency=5
        )
        
        # Evaluate the model
        results = trainer.evaluate(episodes=20)
        
        logger.info("HRM training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()