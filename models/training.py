"""
Training pipeline for the TradingTransformer model.
Implements behavioral cloning, reward modeling, and RL fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm

from models.trading_transformer import (
    TradingTransformer, 
    RewardModel, 
    create_trading_transformer, 
    create_reward_model
)
from models.datasets import (
    create_bc_dataloaders,
    create_rm_dataloaders,
    create_rl_dataset
)
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


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss doesn't improve for a given number of epochs.
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in validation loss to be considered an improvement
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


def train_behavioral_cloning(
    model: TradingTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device = DEVICE
) -> TradingTransformer:
    """
    Train the model using behavioral cloning.
    
    Args:
        model: TradingTransformer model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained model
    """
    logger.info("Starting behavioral cloning training...")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Training loop
    for epoch in range(config['bc_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['bc_epochs']} [Train]"):
            # Move batch to device
            features = batch['features'].to(device)
            instrument_id = batch['instrument_id'].to(device)
            timeframe_id = batch['timeframe_id'].to(device)
            target_action = batch['target_action'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(features, instrument_id, timeframe_id)
            action_logits = outputs['action_logits']
            
            # Compute loss
            loss = F.cross_entropy(action_logits, target_action)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compute accuracy
            preds = torch.argmax(action_logits, dim=1)
            acc = (preds == target_action).float().mean().item()
            
            # Update metrics
            train_loss += loss.item()
            train_acc += acc
            
        # Compute average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['bc_epochs']} [Val]"):
                # Move batch to device
                features = batch['features'].to(device)
                instrument_id = batch['instrument_id'].to(device)
                timeframe_id = batch['timeframe_id'].to(device)
                target_action = batch['target_action'].to(device)
                
                # Forward pass
                outputs = model(features, instrument_id, timeframe_id)
                action_logits = outputs['action_logits']
                
                # Compute loss
                loss = F.cross_entropy(action_logits, target_action)
                
                # Compute accuracy
                preds = torch.argmax(action_logits, dim=1)
                acc = (preds == target_action).float().mean().item()
                
                # Update metrics
                val_loss += loss.item()
                val_acc += acc
                
        # Compute average metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['bc_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Check early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Behavioral cloning training completed")
    return model


def train_reward_model(
    reward_model: RewardModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device = DEVICE
) -> RewardModel:
    """
    Train the reward model using preference pairs.
    
    Args:
        reward_model: RewardModel to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained reward model
    """
    logger.info("Starting reward model training...")
    
    # Move model to device
    reward_model = reward_model.to(device)
    
    # Create optimizer
    optimizer = optim.Adam(
        reward_model.parameters(),
        lr=config['reward_lr'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Create early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Training loop
    for epoch in range(config['rm_epochs']):
        # Training phase
        reward_model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['rm_epochs']} [Train]"):
            # Move batch to device
            state = batch['state'].to(device)
            preferred_action = batch['preferred_action'].to(device)
            non_preferred_action = batch['non_preferred_action'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass for both actions
            preferred_reward = reward_model(state, preferred_action)
            non_preferred_reward = reward_model(state, non_preferred_action)
            
            # Add noise to preferences for robustness
            if config['preference_noise'] > 0:
                noise = torch.randn_like(preferred_reward) * config['preference_noise']
                preferred_reward = preferred_reward + noise
                
                noise = torch.randn_like(non_preferred_reward) * config['preference_noise']
                non_preferred_reward = non_preferred_reward + noise
            
            # Compute loss (Bradley-Terry model)
            logits = preferred_reward - non_preferred_reward
            loss = F.binary_cross_entropy_with_logits(
                logits,
                torch.ones_like(logits)
            )
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compute accuracy
            acc = (preferred_reward > non_preferred_reward).float().mean().item()
            
            # Update metrics
            train_loss += loss.item()
            train_acc += acc
            
        # Compute average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase
        reward_model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['rm_epochs']} [Val]"):
                # Move batch to device
                state = batch['state'].to(device)
                preferred_action = batch['preferred_action'].to(device)
                non_preferred_action = batch['non_preferred_action'].to(device)
                
                # Forward pass for both actions
                preferred_reward = reward_model(state, preferred_action)
                non_preferred_reward = reward_model(state, non_preferred_action)
                
                # Compute loss (Bradley-Terry model)
                logits = preferred_reward - non_preferred_reward
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    torch.ones_like(logits)
                )
                
                # Compute accuracy
                acc = (preferred_reward > non_preferred_reward).float().mean().item()
                
                # Update metrics
                val_loss += loss.item()
                val_acc += acc
                
        # Compute average metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['rm_epochs']} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        
        # Check early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info("Reward model training completed")
    return reward_model


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for RL fine-tuning.
    Implements PPO with KL constraint to the behavioral cloning policy.
    """
    def __init__(
        self,
        model: TradingTransformer,
        bc_model: TradingTransformer,
        reward_model: RewardModel,
        config: Dict[str, Any],
        device: torch.device = DEVICE
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            model: TradingTransformer model to train
            bc_model: Behavioral cloning model to constrain against
            reward_model: Reward model for RLHF
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.bc_model = bc_model.to(device)
        self.reward_model = reward_model.to(device)
        self.config = config
        self.device = device
        
        # Set BC model to eval mode
        self.bc_model.eval()
        self.reward_model.eval()
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.kl_weight = config['kl_weight']
        self.exploration_weight = config['exploration_weight']
        self.sl_penalty_weight = config['sl_penalty_weight']
        
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> torch.Tensor:
        """
        Compute generalized advantage estimates (GAE).
        
        Args:
            rewards: Rewards tensor of shape [batch_size, T]
            values: Value estimates of shape [batch_size, T]
            dones: Done flags of shape [batch_size, T]
            gamma: Discount factor
            lam: GAE parameter
            
        Returns:
            Advantages tensor of shape [batch_size, T]
        """
        batch_size, T = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(T-1)):
            # Compute delta
            delta = rewards[:, t] + gamma * values[:, t+1] * (1 - dones[:, t]) - values[:, t]
            
            # Compute GAE
            advantages[:, t] = last_gae = delta + gamma * lam * (1 - dones[:, t]) * last_gae
            
        # Compute returns
        returns = advantages + values[:, :-1]
        
        return advantages, returns
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        instrument_id: torch.Tensor,
        timeframe_id: torch.Tensor
    ) -> Dict[str, float]:
        """
        Perform a single PPO training step.
        
        Args:
            states: States tensor of shape [batch_size, T, state_dim]
            actions: Actions tensor of shape [batch_size, T]
            old_log_probs: Log probabilities of actions under old policy of shape [batch_size, T]
            advantages: Advantages tensor of shape [batch_size, T]
            returns: Returns tensor of shape [batch_size, T]
            instrument_id: Instrument IDs of shape [batch_size]
            timeframe_id: Timeframe IDs of shape [batch_size]
            
        Returns:
            Dictionary of metrics
        """
        # Reshape tensors
        batch_size, T, _ = states.shape
        flat_states = states.reshape(batch_size * T, -1)
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        
        # Repeat instrument and timeframe IDs for each timestep
        flat_instrument_id = instrument_id.repeat_interleave(T)
        flat_timeframe_id = timeframe_id.repeat_interleave(T)
        
        # Forward pass
        outputs = self.model(flat_states, flat_instrument_id, flat_timeframe_id)
        action_logits = outputs['action_logits']
        values = outputs['state_values'].squeeze(-1)
        
        # Compute log probabilities
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(flat_actions)
        
        # Compute entropy
        entropy = dist.entropy().mean()
        
        # Compute KL divergence with BC policy
        with torch.no_grad():
            bc_outputs = self.bc_model(flat_states, flat_instrument_id, flat_timeframe_id)
            bc_action_probs = F.softmax(bc_outputs['action_logits'], dim=-1)
        
        kl_div = (action_probs * (torch.log(action_probs) - torch.log(bc_action_probs))).sum(dim=-1).mean()
        
        # Compute ratio and clipped ratio
        ratio = torch.exp(log_probs - flat_old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # Compute policy loss
        policy_loss = -torch.min(
            ratio * flat_advantages,
            clipped_ratio * flat_advantages
        ).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, flat_returns)
        
        # Compute exploration bonus
        exploration_bonus = self.exploration_weight * entropy
        
        # Compute KL penalty
        kl_penalty = self.kl_weight * kl_div
        
        # Compute total loss
        total_loss = policy_loss + self.value_coef * value_loss - exploration_bonus + kl_penalty
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Return metrics
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl_div': kl_div.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(
        self,
        dataset,
        batch_size: int = 64,
        epochs: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataset: Dataset to train on
            batch_size: Batch size
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary of metrics
        """
        # Set model to train mode
        self.model.train()
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'kl_div': 0.0,
            'total_loss': 0.0
        }
        
        # Training loop
        for batch in tqdm(data_loader, desc="Training PPO"):
            # Move batch to device
            features = batch['features'].to(self.device)
            instrument_id = batch['instrument_id'].to(self.device)
            timeframe_id = batch['timeframe_id'].to(self.device)
            
            batch_size, seq_len, _ = features.shape
            
            # Generate trajectories
            with torch.no_grad():
                # Get BC policy actions and log probs
                bc_outputs = self.bc_model(features, instrument_id, timeframe_id)
                bc_action_logits = bc_outputs['action_logits']
                bc_action_probs = F.softmax(bc_action_logits, dim=-1)
                bc_dist = torch.distributions.Categorical(bc_action_probs)
                
                # Sample actions from BC policy
                actions = bc_dist.sample()
                old_log_probs = bc_dist.log_prob(actions)
                
                # Get values from current policy
                outputs = self.model(features, instrument_id, timeframe_id)
                values = outputs['state_values'].squeeze(-1)
                
                # Get risk assessment
                risk_assessment = self.model(features, instrument_id, timeframe_id, return_risk=True)['risk_assessment'].squeeze(-1)
                
                # Compute rewards using reward model
                rewards = torch.zeros(batch_size, seq_len, device=self.device)
                
                for t in range(seq_len):
                    state = features[:, t]
                    action = actions[:, t]
                    
                    # Get reward from reward model
                    with torch.no_grad():
                        reward = self.reward_model(state, action)
                    
                    # Add penalty for high-risk actions
                    risk = risk_assessment[:, t]
                    sl_penalty = self.sl_penalty_weight * risk * (action == 1).float()
                    
                    rewards[:, t] = reward - sl_penalty
                
                # Compute advantages
                dones = torch.zeros(batch_size, seq_len, device=self.device)
                dones[:, -1] = 1  # Last step is done
                
                advantages, returns = self.compute_advantages(
                    rewards, 
                    values, 
                    dones
                )
            
            # Train for multiple epochs on this batch
            for _ in range(epochs):
                batch_metrics = self.train_step(
                    features,
                    actions,
                    old_log_probs,
                    advantages,
                    returns,
                    instrument_id,
                    timeframe_id
                )
                
                # Update metrics
                for k, v in batch_metrics.items():
                    metrics[k] += v
        
        # Compute average metrics
        for k in metrics:
            metrics[k] /= len(data_loader) * epochs
        
        return metrics


def train_rl_finetuning(
    model: TradingTransformer,
    bc_model: TradingTransformer,
    reward_model: RewardModel,
    dataset,
    config: Dict[str, Any],
    device: torch.device = DEVICE
) -> TradingTransformer:
    """
    Fine-tune the model using RL.
    
    Args:
        model: TradingTransformer model to train
        bc_model: Behavioral cloning model to constrain against
        reward_model: Reward model for RLHF
        dataset: Dataset to train on
        config: Training configuration
        device: Device to train on
        
    Returns:
        Trained model
    """
    logger.info("Starting RL fine-tuning...")
    
    # Create PPO trainer
    trainer = PPOTrainer(
        model=model,
        bc_model=bc_model,
        reward_model=reward_model,
        config=config,
        device=device
    )
    
    # Training loop
    for epoch in range(config['rl_epochs']):
        # Train for one epoch
        metrics = trainer.train_epoch(
            dataset=dataset,
            batch_size=config['batch_size'],
            epochs=10  # Number of PPO epochs per batch
        )
        
        # Log metrics
        logger.info(
            f"Epoch {epoch+1}/{config['rl_epochs']} - "
            f"Policy Loss: {metrics['policy_loss']:.4f}, "
            f"Value Loss: {metrics['value_loss']:.4f}, "
            f"Entropy: {metrics['entropy']:.4f}, "
            f"KL Div: {metrics['kl_div']:.4f}, "
            f"Total Loss: {metrics['total_loss']:.4f}"
        )
    
    logger.info("RL fine-tuning completed")
    return model


def train_full_pipeline(
    instruments: List[str] = list(INSTRUMENTS.keys()),
    timeframes: List[int] = TIMEFRAMES,
    transformer_config: Dict[str, Any] = TRANSFORMER_CONFIG,
    training_config: Dict[str, Any] = TRAINING_CONFIG,
    rlhf_config: Dict[str, Any] = RLHF_CONFIG,
    device: torch.device = DEVICE,
    save_dir: str = 'models'
) -> Tuple[TradingTransformer, RewardModel]:
    """
    Train the full pipeline: behavioral cloning, reward modeling, and RL fine-tuning.
    
    Args:
        instruments: List of instruments to train on
        timeframes: List of timeframes to train on
        transformer_config: Configuration for the transformer model
        training_config: Configuration for training
        rlhf_config: Configuration for RLHF
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        Tuple of (trained_model, reward_model)
    """
    logger.info("Starting full training pipeline...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create data loaders for behavioral cloning
    logger.info("Creating data loaders for behavioral cloning...")
    bc_train_loader, bc_val_loader = create_bc_dataloaders(
        instruments=instruments,
        timeframes=timeframes,
        batch_size=training_config['batch_size'],
        validation_split=training_config['validation_split']
    )
    
    # Create data loaders for reward modeling
    logger.info("Creating data loaders for reward modeling...")
    rm_train_loader, rm_val_loader = create_rm_dataloaders(
        instruments=instruments,
        timeframes=timeframes,
        batch_size=training_config['batch_size'],
        validation_split=training_config['validation_split']
    )
    
    # Create dataset for RL fine-tuning
    logger.info("Creating dataset for RL fine-tuning...")
    rl_dataset = create_rl_dataset(
        instruments=instruments,
        timeframes=timeframes
    )
    
    # Create models
    logger.info("Creating models...")
    
    # Get state dimension from a sample
    sample_batch = next(iter(bc_train_loader))
    state_dim = sample_batch['features'].shape[2]
    
    # Create transformer model
    model = create_trading_transformer(
        config=transformer_config,
        state_dim=state_dim,
        num_instruments=len(instruments),
        num_timeframes=len(timeframes),
        action_dim=3  # hold, buy, sell
    )
    
    # Create reward model
    reward_model = create_reward_model(
        config=rlhf_config,
        state_dim=state_dim,
        action_dim=3  # hold, buy, sell
    )
    
    # Step 1: Behavioral Cloning
    logger.info("Step 1: Behavioral Cloning")
    bc_model = train_behavioral_cloning(
        model=model,
        train_loader=bc_train_loader,
        val_loader=bc_val_loader,
        config=training_config,
        device=device
    )
    
    # Save BC model
    bc_model_path = os.path.join(save_dir, 'bc_model.pt')
    torch.save(bc_model.state_dict(), bc_model_path)
    logger.info(f"Saved BC model to {bc_model_path}")
    
    # Step 2: Reward Modeling
    logger.info("Step 2: Reward Modeling")
    trained_reward_model = train_reward_model(
        reward_model=reward_model,
        train_loader=rm_train_loader,
        val_loader=rm_val_loader,
        config=rlhf_config,
        device=device
    )
    
    # Save reward model
    reward_model_path = os.path.join(save_dir, 'reward_model.pt')
    torch.save(trained_reward_model.state_dict(), reward_model_path)
    logger.info(f"Saved reward model to {reward_model_path}")
    
    # Step 3: RL Fine-tuning
    logger.info("Step 3: RL Fine-tuning")
    
    # Create a new model for RL fine-tuning
    rl_model = create_trading_transformer(
        config=transformer_config,
        state_dim=state_dim,
        num_instruments=len(instruments),
        num_timeframes=len(timeframes),
        action_dim=3  # hold, buy, sell
    )
    
    # Load BC weights
    rl_model.load_state_dict(bc_model.state_dict())
    
    # Fine-tune with RL
    trained_model = train_rl_finetuning(
        model=rl_model,
        bc_model=bc_model,
        reward_model=trained_reward_model,
        dataset=rl_dataset,
        config=training_config,
        device=device
    )
    
    # Save final model
    model_path = os.path.join(save_dir, 'trading_transformer.pt')
    torch.save(trained_model.state_dict(), model_path)
    logger.info(f"Saved final model to {model_path}")
    
    logger.info("Full training pipeline completed")
    return trained_model, trained_reward_model


if __name__ == "__main__":
    # Train the full pipeline
    model, reward_model = train_full_pipeline()
