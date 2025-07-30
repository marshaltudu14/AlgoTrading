"""
Transformer-based World Model for Autonomous Trading Agent

This module provides a sophisticated world model that can predict future market states
and provide policy recommendations. The model serves as the "brain" of the autonomous
agent, enabling it to simulate and reason about potential future scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union
import math

from src.models.core_transformer import CoreTransformer


class TransformerWorldModel(nn.Module):
    """
    Transformer-based World Model for autonomous trading agents.
    
    This model serves as the core reasoning engine for the autonomous agent,
    providing both predictive capabilities (what will happen) and policy
    recommendations (what should be done).
    
    The model has two main output heads:
    1. Prediction Head: Predicts future market states (OHLCV, regime changes, etc.)
    2. Policy Head: Outputs action probability distributions
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        prediction_horizon: int = 5,
        market_features: int = 5,  # OHLCV
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
        predict_market_regime: bool = True,
        num_market_regimes: int = 4  # Trending, Ranging, Volatile, Consolidation
    ):
        """
        Initialize the TransformerWorldModel.
        
        Args:
            input_dim: Dimension of input market features
            action_dim: Number of possible trading actions
            prediction_horizon: Number of future time steps to predict
            market_features: Number of market features to predict (e.g., OHLCV = 5)
            num_heads: Number of attention heads in transformer
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension of transformer
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            predict_market_regime: Whether to predict market regime changes
            num_market_regimes: Number of market regimes to classify
        """
        super(TransformerWorldModel, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.prediction_horizon = prediction_horizon
        self.market_features = market_features
        self.hidden_dim = hidden_dim
        self.predict_market_regime = predict_market_regime
        self.num_market_regimes = num_market_regimes
        
        # Core transformer backbone
        self.transformer = CoreTransformer(
            input_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=hidden_dim,
            output_dim=hidden_dim,  # Output to hidden_dim for shared representation
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_positional_encoding=True
        )
        
        # Prediction Head Components
        self.prediction_head = nn.ModuleDict()
        
        # Market state prediction (OHLCV for next N steps)
        self.prediction_head['market_state'] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, prediction_horizon * market_features)
        )
        
        # Market regime prediction (if enabled)
        if predict_market_regime:
            self.prediction_head['market_regime'] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, num_market_regimes)
            )
        
        # Volatility prediction
        self.prediction_head['volatility'] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, prediction_horizon)
        )
        
        # Policy Head Components
        self.policy_head = nn.ModuleDict()
        
        # Action probability distribution
        self.policy_head['actions'] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Value estimation (for policy evaluation)
        self.policy_head['value'] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Confidence estimation (how certain the model is about its predictions)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the world model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing outputs from both prediction and policy heads:
            - 'predictions': Dict with market predictions
            - 'policy': Dict with policy outputs
            - 'confidence': Model confidence in predictions
            - 'attention_weights': (optional) Attention weights
        """
        batch_size = x.shape[0]
        
        # Handle single timestep input
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # Get shared representation from transformer
        shared_repr = self.transformer(x, mask=mask)  # (batch_size, hidden_dim)
        
        # Prediction Head Outputs
        predictions = {}
        
        # Market state prediction (reshape to (batch_size, prediction_horizon, market_features))
        market_state_flat = self.prediction_head['market_state'](shared_repr)
        predictions['market_state'] = market_state_flat.view(
            batch_size, self.prediction_horizon, self.market_features
        )
        
        # Market regime prediction (if enabled)
        if self.predict_market_regime:
            predictions['market_regime'] = F.softmax(
                self.prediction_head['market_regime'](shared_repr), dim=-1
            )
        
        # Volatility prediction
        predictions['volatility'] = torch.relu(
            self.prediction_head['volatility'](shared_repr)
        )  # Ensure positive volatility
        
        # Policy Head Outputs
        policy = {}
        policy['actions'] = self.policy_head['actions'](shared_repr)
        policy['value'] = self.policy_head['value'](shared_repr)
        
        # Confidence estimation
        confidence = self.confidence_head(shared_repr)
        
        # Prepare output dictionary
        output = {
            'predictions': predictions,
            'policy': policy,
            'confidence': confidence
        }
        
        # Add attention weights if requested
        if return_attention:
            output['attention_weights'] = self.transformer.get_attention_weights(x, mask)
        
        return output
    
    def predict_future_states(
        self, 
        x: torch.Tensor, 
        num_steps: int = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future market states for "what-if" scenario analysis.
        
        Args:
            x: Current market state tensor
            num_steps: Number of future steps to predict (defaults to prediction_horizon)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing future state predictions
        """
        if num_steps is None:
            num_steps = self.prediction_horizon
        
        with torch.no_grad():
            output = self.forward(x, mask=mask)
            
            # Extract relevant predictions
            future_states = {
                'market_state': output['predictions']['market_state'][:, :num_steps, :],
                'volatility': output['predictions']['volatility'][:, :num_steps],
                'confidence': output['confidence']
            }
            
            if self.predict_market_regime:
                future_states['market_regime'] = output['predictions']['market_regime']
            
            return future_states
    
    def simulate_action_outcomes(
        self, 
        x: torch.Tensor, 
        actions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate the outcomes of different actions for decision making.
        
        Args:
            x: Current market state tensor
            actions: Tensor of actions to simulate (batch_size, num_actions)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing simulated outcomes for each action
        """
        batch_size = x.shape[0]
        num_actions = actions.shape[1] if len(actions.shape) > 1 else 1
        
        # Get base predictions
        base_output = self.forward(x, mask=mask)
        
        # For now, return the same predictions for all actions
        # In a more sophisticated implementation, this would condition on actions
        simulated_outcomes = {
            'predicted_returns': base_output['predictions']['market_state'][:, :, 3],  # Close prices
            'predicted_volatility': base_output['predictions']['volatility'],
            'action_values': base_output['policy']['value'].expand(batch_size, num_actions),
            'confidence': base_output['confidence'].expand(batch_size, num_actions)
        }
        
        return simulated_outcomes
    
    def get_config(self) -> Dict:
        """Get model configuration for saving/loading."""
        return {
            'input_dim': self.input_dim,
            'action_dim': self.action_dim,
            'prediction_horizon': self.prediction_horizon,
            'market_features': self.market_features,
            'hidden_dim': self.hidden_dim,
            'predict_market_regime': self.predict_market_regime,
            'num_market_regimes': self.num_market_regimes
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> 'TransformerWorldModel':
        """Create model from configuration dictionary."""
        return cls(**config)
