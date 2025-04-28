"""
Trading Transformer: A custom PyTorch model for algorithmic trading based on transformer architecture.
Implements a Decision Transformer approach with market regime detection and task embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds positional information to input embeddings.
    """
    def __init__(self, d_model: int, max_seq_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PrototypicalNetwork(nn.Module):
    """
    Market regime detection using prototypical networks.
    Learns prototypes for different market regimes and computes similarity.
    """
    def __init__(self, input_dim: int, num_regimes: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_regimes = num_regimes
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable prototypes for each regime
        self.prototypes = nn.Parameter(torch.randn(num_regimes, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
        
        Returns:
            Regime embedding of shape [batch_size, hidden_dim]
        """
        # Encode the input sequence
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(batch_size * seq_len, -1)
        encoded = self.encoder(x_flat).view(batch_size, seq_len, -1)
        
        # Compute mean embedding across sequence
        mean_embedding = encoded.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Compute distances to prototypes
        dists = torch.cdist(mean_embedding.unsqueeze(1), self.prototypes.unsqueeze(0))
        dists = dists.squeeze(1)  # [batch_size, num_regimes]
        
        # Convert distances to similarities (softmax)
        similarities = F.softmax(-dists, dim=1)  # [batch_size, num_regimes]
        
        # Weighted sum of prototypes based on similarities
        regime_embedding = torch.matmul(similarities, self.prototypes)  # [batch_size, hidden_dim]
        
        return regime_embedding


class TradingTransformer(nn.Module):
    """
    Main model architecture combining transformer encoder, market regime detection,
    and task embedding for trading decisions.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        num_instruments: int,
        num_timeframes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        num_regimes: int = 3
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Input projection to hidden dimension
        self.input_projection = nn.Linear(state_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Market regime detection
        self.regime_detector = PrototypicalNetwork(hidden_dim, num_regimes, hidden_dim)
        
        # Task embedding (instrument + timeframe)
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim // 2)
        self.timeframe_embedding = nn.Embedding(num_timeframes, hidden_dim // 2)
        
        # Policy head (action prediction)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head (for RL)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Risk assessment head (predicts probability of SL hit)
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        states: torch.Tensor, 
        instrument_id: torch.Tensor, 
        timeframe_id: torch.Tensor,
        return_risk: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            states: Input states of shape [batch_size, seq_len, state_dim]
            instrument_id: Instrument IDs of shape [batch_size]
            timeframe_id: Timeframe IDs of shape [batch_size]
            return_risk: Whether to return risk assessment
            
        Returns:
            Dictionary containing action logits, state values, and optionally risk assessment
        """
        batch_size, seq_len, _ = states.shape
        
        # Project input to hidden dimension
        x = self.input_projection(states)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(x)
        
        # Detect market regime
        regime_embedding = self.regime_detector(encoded)
        
        # Get task embeddings
        instr_emb = self.instrument_embedding(instrument_id)
        tf_emb = self.timeframe_embedding(timeframe_id)
        task_emb = torch.cat([instr_emb, tf_emb], dim=-1)
        
        # Combine embeddings
        combined = torch.cat([regime_embedding, task_emb], dim=-1)
        
        # Generate outputs
        action_logits = self.policy_head(combined)
        state_values = self.value_head(combined)
        
        outputs = {
            'action_logits': action_logits,
            'state_values': state_values
        }
        
        if return_risk:
            risk_assessment = self.risk_head(combined)
            outputs['risk_assessment'] = risk_assessment
            
        return outputs
    
    def get_action(
        self, 
        states: torch.Tensor, 
        instrument_id: torch.Tensor, 
        timeframe_id: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get actions from the model for inference.
        
        Args:
            states: Input states
            instrument_id: Instrument IDs
            timeframe_id: Timeframe IDs
            deterministic: Whether to sample deterministically
            
        Returns:
            Tuple of (actions, extra_info)
        """
        outputs = self.forward(states, instrument_id, timeframe_id, return_risk=True)
        
        action_logits = outputs['action_logits']
        action_probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from the distribution
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()
        
        # Get risk assessment
        risk = outputs['risk_assessment']
        
        # If risk is too high, consider holding instead
        high_risk_mask = (risk > 0.7).squeeze(-1)
        actions_buy_mask = (actions == 1)  # Buy action
        
        # Convert high-risk buy actions to hold
        actions = torch.where(
            high_risk_mask & actions_buy_mask,
            torch.zeros_like(actions),  # Hold action
            actions
        )
        
        extra_info = {
            'action_probs': action_probs,
            'state_values': outputs['state_values'],
            'risk_assessment': risk
        }
        
        return actions, extra_info


class RewardModel(nn.Module):
    """
    Neural reward model for RLHF that learns to predict rewards from state-action pairs.
    Used to provide reward signals during RL fine-tuning.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.action_dim = action_dim
        
        # Create layers
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reward model.
        
        Args:
            states: Input states of shape [batch_size, state_dim]
            actions: Actions of shape [batch_size] or [batch_size, 1]
            
        Returns:
            Predicted rewards of shape [batch_size]
        """
        # One-hot encode actions
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)
            
        onehot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        
        # If actions has shape [batch_size, 1, action_dim], flatten it
        if onehot.dim() > 2:
            onehot = onehot.squeeze(1)
            
        # Concatenate states and one-hot actions
        x = torch.cat([states, onehot], dim=-1)
        
        # Predict rewards
        return self.net(x).squeeze(-1)


# Utility function to create the model
def create_trading_transformer(config, state_dim, num_instruments, num_timeframes, action_dim=3):
    """
    Create a TradingTransformer model with the given configuration.
    
    Args:
        config: Model configuration dictionary
        state_dim: Dimension of the state space
        num_instruments: Number of instruments
        num_timeframes: Number of timeframes
        action_dim: Dimension of the action space (default: 3 for hold/buy/sell)
        
    Returns:
        Initialized TradingTransformer model
    """
    return TradingTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_instruments=num_instruments,
        num_timeframes=num_timeframes,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 50),
        num_regimes=config.get('num_regimes', 3)
    )


def create_reward_model(config, state_dim, action_dim=3):
    """
    Create a RewardModel with the given configuration.
    
    Args:
        config: RLHF configuration dictionary
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        
    Returns:
        Initialized RewardModel
    """
    return RewardModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=config.get('reward_hidden_dim', 64),
        num_layers=config.get('reward_layers', 2)
    )
