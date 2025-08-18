#!/usr/bin/env python3
"""
Dynamic parameter management for adaptive training.
All parameters are computed dynamically based on data characteristics and training progress.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DynamicParams:
    """Container for dynamically computed parameters."""
    # Data-dependent parameters
    observation_dim: int
    feature_count: int
    sequence_length: int
    data_complexity: float
    
    # Model architecture parameters
    hidden_dim: int
    num_heads: int
    num_layers: int
    dropout: float
    
    # Training parameters
    lr_actor: float
    lr_critic: float
    batch_size: int
    k_epochs: int
    epsilon_clip: float
    gamma: float
    
    # Environment parameters
    episode_length: int
    lookback_window: int
    
    # Adaptive parameters
    gradient_clip_norm: float
    exploration_rate: float
    reward_scale: float

class DynamicParameterManager:
    """Manages dynamic parameter computation based on data and training progress."""
    
    def __init__(self):
        self.training_history = []
        self.performance_metrics = []
        
    def compute_data_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Compute characteristics of the training data."""
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Basic statistics
        volatility = numeric_data.std().mean()
        complexity = self._compute_data_complexity(numeric_data)
        correlation_strength = self._compute_correlation_strength(numeric_data)
        trend_strength = self._compute_trend_strength(numeric_data)
        
        return {
            'volatility': volatility,
            'complexity': complexity,
            'correlation_strength': correlation_strength,
            'trend_strength': trend_strength,
            'feature_count': len(numeric_data.columns),
            'sample_count': len(data)
        }
    
    def _compute_data_complexity(self, data: pd.DataFrame) -> float:
        """Compute data complexity score (0-1)."""
        try:
            # Use coefficient of variation as complexity measure
            cv_scores = []
            for col in data.columns:
                if data[col].std() > 0:
                    cv = data[col].std() / (abs(data[col].mean()) + 1e-8)
                    cv_scores.append(cv)
            
            if cv_scores:
                complexity = np.mean(cv_scores)
                # Normalize to 0-1 range
                return min(1.0, complexity / 2.0)
            return 0.5
        except:
            return 0.5
    
    def _compute_correlation_strength(self, data: pd.DataFrame) -> float:
        """Compute average correlation strength."""
        try:
            corr_matrix = data.corr().abs()
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.where(mask).stack()
            return correlations.mean() if len(correlations) > 0 else 0.0
        except:
            return 0.0
    
    def _compute_trend_strength(self, data: pd.DataFrame) -> float:
        """Compute trend strength in the data."""
        try:
            trend_scores = []
            for col in data.columns:
                if len(data[col]) > 10:
                    # Simple trend detection using linear regression slope
                    x = np.arange(len(data[col]))
                    slope = np.polyfit(x, data[col], 1)[0]
                    trend_scores.append(abs(slope))
            
            return np.mean(trend_scores) if trend_scores else 0.0
        except:
            return 0.0
    
    def compute_dynamic_params(self, data: pd.DataFrame, training_progress: float = 0.0) -> DynamicParams:
        """Compute all dynamic parameters based on data and training progress."""
        
        # Get data characteristics
        data_chars = self.compute_data_characteristics(data)
        
        # Compute observation dimension
        observation_dim = self._compute_observation_dim(data_chars)
        
        # Compute model architecture parameters
        hidden_dim = self._compute_hidden_dim(data_chars)
        num_heads = self._compute_num_heads(hidden_dim)
        num_layers = self._compute_num_layers(data_chars)
        dropout = self._compute_dropout(data_chars, training_progress)
        
        # Compute training parameters
        lr_actor, lr_critic = self._compute_learning_rates(data_chars, training_progress)
        batch_size = self._compute_batch_size(data_chars)
        k_epochs = self._compute_k_epochs(data_chars, training_progress)
        epsilon_clip = self._compute_epsilon_clip(training_progress)
        gamma = self._compute_gamma(data_chars)
        
        # Compute environment parameters
        episode_length = self._compute_episode_length(data_chars)
        lookback_window = self._compute_lookback_window(data_chars)
        
        # Compute adaptive parameters
        gradient_clip_norm = self._compute_gradient_clip_norm(data_chars)
        exploration_rate = self._compute_exploration_rate(training_progress)
        reward_scale = self._compute_reward_scale(data_chars)
        
        params = DynamicParams(
            observation_dim=observation_dim,
            feature_count=data_chars['feature_count'],
            sequence_length=len(data),
            data_complexity=data_chars['complexity'],
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            batch_size=batch_size,
            k_epochs=k_epochs,
            epsilon_clip=epsilon_clip,
            gamma=gamma,
            episode_length=episode_length,
            lookback_window=lookback_window,
            gradient_clip_norm=gradient_clip_norm,
            exploration_rate=exploration_rate,
            reward_scale=reward_scale
        )
        
        logger.info(f"🔧 Dynamic parameters computed:")
        logger.info(f"   Data complexity: {data_chars['complexity']:.3f}")
        logger.info(f"   Observation dim: {observation_dim}")
        logger.info(f"   Hidden dim: {hidden_dim}")
        logger.info(f"   Learning rates: actor={lr_actor:.6f}, critic={lr_critic:.6f}")
        logger.info(f"   Architecture: {num_layers} layers, {num_heads} heads")
        
        return params
    
    def _compute_observation_dim(self, data_chars: Dict) -> int:
        """Compute observation dimension based on data."""
        # This will be set by the environment based on actual observation
        return data_chars['feature_count']
    
    def _compute_hidden_dim(self, data_chars: Dict) -> int:
        """Compute hidden dimension based on data complexity."""
        base_dim = 32
        complexity_factor = 1 + data_chars['complexity']
        feature_factor = min(2.0, data_chars['feature_count'] / 50.0)
        
        hidden_dim = int(base_dim * complexity_factor * feature_factor)
        # Ensure it's a multiple of num_heads for attention
        return max(32, (hidden_dim // 8) * 8)
    
    def _compute_num_heads(self, hidden_dim: int) -> int:
        """Compute number of attention heads."""
        # Ensure hidden_dim is divisible by num_heads
        possible_heads = [4, 8, 16]
        for heads in possible_heads:
            if hidden_dim % heads == 0:
                return heads
        return 4  # fallback
    
    def _compute_num_layers(self, data_chars: Dict) -> int:
        """Compute number of transformer layers."""
        base_layers = 2
        complexity_bonus = int(data_chars['complexity'] * 4)
        return min(6, base_layers + complexity_bonus)
    
    def _compute_dropout(self, data_chars: Dict, training_progress: float) -> float:
        """Compute dropout rate."""
        base_dropout = 0.1
        complexity_factor = data_chars['complexity'] * 0.2
        # Reduce dropout as training progresses
        progress_factor = training_progress * 0.05
        return max(0.05, base_dropout + complexity_factor - progress_factor)
    
    def _compute_learning_rates(self, data_chars: Dict, training_progress: float) -> Tuple[float, float]:
        """Compute adaptive learning rates."""
        # Base learning rates (conservative for stability)
        base_lr_actor = 1e-5
        base_lr_critic = 1e-4
        
        # Adjust based on data complexity
        complexity_factor = 1.0 / (1.0 + data_chars['complexity'])
        
        # Adjust based on training progress (learning rate decay)
        progress_factor = max(0.1, 1.0 - training_progress * 0.5)
        
        lr_actor = base_lr_actor * complexity_factor * progress_factor
        lr_critic = base_lr_critic * complexity_factor * progress_factor
        
        return lr_actor, lr_critic
    
    def _compute_batch_size(self, data_chars: Dict) -> int:
        """Compute batch size based on data size."""
        sample_count = data_chars['sample_count']
        if sample_count < 100:
            return 8
        elif sample_count < 500:
            return 16
        elif sample_count < 1000:
            return 32
        else:
            return 64
    
    def _compute_k_epochs(self, data_chars: Dict, training_progress: float) -> int:
        """Compute number of training epochs."""
        base_epochs = 2
        # More epochs for complex data, fewer as training progresses
        complexity_bonus = int(data_chars['complexity'] * 2)
        progress_penalty = int(training_progress * 2)
        return max(1, base_epochs + complexity_bonus - progress_penalty)
    
    def _compute_epsilon_clip(self, training_progress: float) -> float:
        """Compute clipping parameter for policy updates."""
        base_clip = 0.2
        # Reduce clipping as training progresses for more conservative updates
        return max(0.05, base_clip * (1.0 - training_progress * 0.5))
    
    def _compute_gamma(self, data_chars: Dict) -> float:
        """Compute discount factor."""
        # Higher gamma for more stable/trending data
        base_gamma = 0.99
        trend_bonus = data_chars['trend_strength'] * 0.005
        return min(0.999, base_gamma + trend_bonus)
    
    def _compute_episode_length(self, data_chars: Dict) -> int:
        """Compute episode length."""
        sample_count = data_chars['sample_count']
        # Use a fraction of available data
        return min(500, max(50, sample_count // 4))
    
    def _compute_lookback_window(self, data_chars: Dict) -> int:
        """Compute lookback window."""
        # Base on data complexity and correlation
        base_window = 20
        complexity_factor = data_chars['complexity'] * 30
        correlation_factor = data_chars['correlation_strength'] * 20
        return int(base_window + complexity_factor + correlation_factor)
    
    def _compute_gradient_clip_norm(self, data_chars: Dict) -> float:
        """Compute gradient clipping norm."""
        # More aggressive clipping for complex data
        base_norm = 1.0
        complexity_factor = data_chars['complexity'] * 0.5
        return max(0.5, base_norm - complexity_factor)
    
    def _compute_exploration_rate(self, training_progress: float) -> float:
        """Compute exploration rate."""
        # Start high, decay over time
        return max(0.01, 0.3 * (1.0 - training_progress))
    
    def _compute_reward_scale(self, data_chars: Dict) -> float:
        """Compute reward scaling factor."""
        # Scale based on data volatility
        base_scale = 1.0
        volatility_factor = 1.0 / (1.0 + data_chars['volatility'])
        return base_scale * volatility_factor
    
    def update_training_progress(self, metrics: Dict[str, Any]):
        """Update training progress and adapt parameters."""
        self.training_history.append(metrics)
        self.performance_metrics.append(metrics.get('performance', 0.0))
        
    def get_training_progress(self) -> float:
        """Get current training progress (0-1)."""
        if not self.performance_metrics:
            return 0.0
        
        # Simple progress based on performance improvement
        if len(self.performance_metrics) < 2:
            return 0.0
        
        recent_performance = np.mean(self.performance_metrics[-10:])
        initial_performance = np.mean(self.performance_metrics[:5]) if len(self.performance_metrics) >= 5 else self.performance_metrics[0]
        
        if initial_performance == 0:
            return 0.5
        
        improvement = (recent_performance - initial_performance) / abs(initial_performance)
        return min(1.0, max(0.0, improvement))
