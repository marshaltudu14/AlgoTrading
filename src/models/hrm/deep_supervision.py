"""
Deep Supervision Module
Manages multi-horizon learning with different time scales
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class DeepSupervision(nn.Module):
    """
    Deep supervision mechanism for multi-horizon learning
    
    Key concepts:
    - Multiple supervision segments with different time horizons
    - Segment 1: Immediate rewards (1-step)
    - Segment 2: Short-term performance (5-10 steps)
    - Segment 3: Medium-term results (20-50 steps) 
    - Segment 4: Long-term strategy (100+ steps)
    """
    
    def __init__(self,
                 num_segments: int = 4,
                 segment_weights: Optional[List[float]] = None,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.num_segments = num_segments
        self.segment_weights = segment_weights or [0.4, 0.3, 0.2, 0.1]
        self.hidden_dim = hidden_dim
        
        # Ensure segment weights sum to 1
        total_weight = sum(self.segment_weights)
        self.segment_weights = [w / total_weight for w in self.segment_weights]
        
        # Segment-specific value estimators
        self.segment_value_heads = nn.ModuleList([
            self._create_value_head(hidden_dim, f"segment_{i}")
            for i in range(num_segments)
        ])
        
        # Multi-horizon reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_segments)  # Predict rewards for each horizon
        )
        
        # Temporal consistency enforcer
        self.consistency_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def _create_value_head(self, hidden_dim: int, name: str) -> nn.Module:
        """Create value estimation head for a specific segment"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self,
                strategic_hidden: torch.Tensor,
                tactical_hidden: torch.Tensor,
                segment_idx: int) -> Dict[str, torch.Tensor]:
        """
        Forward pass for specific supervision segment
        
        Args:
            strategic_hidden: H-module hidden state
            tactical_hidden: L-module hidden state  
            segment_idx: Current supervision segment (0 to num_segments-1)
            
        Returns:
            Supervision outputs for the current segment
        """
        
        # Combine strategic and tactical information
        combined_hidden = strategic_hidden + tactical_hidden  # Residual combination
        
        # Segment-specific value estimation
        segment_value = self.segment_value_heads[segment_idx](combined_hidden)
        
        # Multi-horizon reward prediction
        predicted_rewards = self.reward_predictor(combined_hidden)
        
        # Current segment reward prediction
        current_segment_reward = predicted_rewards[:, segment_idx:segment_idx+1]
        
        return {
            'segment_value': segment_value,
            'predicted_rewards': predicted_rewards,
            'current_segment_reward': current_segment_reward,
            'segment_weight': self.segment_weights[segment_idx]
        }
    
    def compute_supervision_loss(self,
                                outputs_history: List[Dict[str, torch.Tensor]], 
                                actual_rewards: List[float],
                                actions_taken: List[int]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-horizon supervision loss
        
        Args:
            outputs_history: List of model outputs for each segment
            actual_rewards: List of actual rewards received
            actions_taken: List of actions taken in each segment
            
        Returns:
            Dictionary of losses and metrics
        """
        
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # Immediate reward supervision (Segment 0)
        if len(actual_rewards) > 0 and len(outputs_history) > 0:
            immediate_reward = torch.tensor(actual_rewards[0]).float()
            if 'current_segment_reward' in outputs_history[0]:
                immediate_loss = nn.MSELoss()(
                    outputs_history[0]['current_segment_reward'].squeeze(),
                    immediate_reward.expand_as(outputs_history[0]['current_segment_reward'].squeeze())
                )
                losses['immediate_loss'] = immediate_loss
                total_loss = total_loss + self.segment_weights[0] * immediate_loss
        
        # Short-term supervision (Segment 1) - GPU optimized
        if len(actual_rewards) >= 5:
            if torch.cuda.is_available():
                # GPU: Use tensor operations directly
                rewards_tensor = torch.tensor(actual_rewards[:5], dtype=torch.float32, device='cuda')
                short_term_reward = torch.mean(rewards_tensor)
            else:
                # CPU: Use numpy when GPU unavailable
                short_term_reward = torch.tensor(np.mean(actual_rewards[:5])).float()
            
            if len(outputs_history) > 1 and 'current_segment_reward' in outputs_history[1]:
                short_term_loss = nn.MSELoss()(
                    outputs_history[1]['current_segment_reward'].squeeze(),
                    short_term_reward.expand_as(outputs_history[1]['current_segment_reward'].squeeze())
                )
                losses['short_term_loss'] = short_term_loss
                total_loss = total_loss + self.segment_weights[1] * short_term_loss
        
        # Medium-term supervision (Segment 2) - GPU optimized
        if len(actual_rewards) >= 20:
            if torch.cuda.is_available():
                # GPU: Use tensor operations directly
                rewards_tensor = torch.tensor(actual_rewards[:20], dtype=torch.float32, device='cuda')
                medium_term_reward = torch.mean(rewards_tensor)
            else:
                # CPU: Use numpy when GPU unavailable
                medium_term_reward = torch.tensor(np.mean(actual_rewards[:20])).float()
            
            if len(outputs_history) > 2 and 'current_segment_reward' in outputs_history[2]:
                medium_term_loss = nn.MSELoss()(
                    outputs_history[2]['current_segment_reward'].squeeze(),
                    medium_term_reward.expand_as(outputs_history[2]['current_segment_reward'].squeeze())
                )
                losses['medium_term_loss'] = medium_term_loss
                total_loss = total_loss + self.segment_weights[2] * medium_term_loss
        
        # Long-term supervision (Segment 3) - GPU optimized
        if len(actual_rewards) >= 50:
            if torch.cuda.is_available():
                # GPU: Use tensor operations directly
                rewards_tensor = torch.tensor(actual_rewards, dtype=torch.float32, device='cuda')
                long_term_reward = torch.mean(rewards_tensor)
            else:
                # CPU: Use numpy when GPU unavailable
                long_term_reward = torch.tensor(np.mean(actual_rewards)).float()
            if len(outputs_history) > 3 and 'current_segment_reward' in outputs_history[3]:
                long_term_loss = nn.MSELoss()(
                    outputs_history[3]['current_segment_reward'].squeeze(),
                    long_term_reward.expand_as(outputs_history[3]['current_segment_reward'].squeeze())
                )
                losses['long_term_loss'] = long_term_loss
                total_loss = total_loss + self.segment_weights[3] * long_term_loss
        
        # Temporal consistency loss
        if len(outputs_history) >= 2:
            consistency_loss = self._compute_consistency_loss(outputs_history)
            losses['consistency_loss'] = consistency_loss
            total_loss = total_loss + 0.1 * consistency_loss
        
        # Value estimation loss (for each segment's value head)
        for i, outputs in enumerate(outputs_history):
            if 'segment_value' in outputs and i < len(actual_rewards):
                # Simple value target: discounted future rewards
                value_target = self._compute_value_target(actual_rewards[i:], gamma=0.95)
                value_loss = nn.MSELoss()(
                    outputs['segment_value'].squeeze(),
                    value_target.expand_as(outputs['segment_value'].squeeze())
                )
                losses[f'value_loss_segment_{i}'] = value_loss
                total_loss = total_loss + 0.1 * value_loss
        
        losses['total_supervision_loss'] = total_loss
        return losses
    
    def _compute_consistency_loss(self, outputs_history: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute temporal consistency loss between segments"""
        
        consistency_losses = []
        
        for i in range(len(outputs_history) - 1):
            if ('predicted_rewards' in outputs_history[i] and 
                'predicted_rewards' in outputs_history[i+1]):
                
                # Rewards should be consistent across segments
                current_rewards = outputs_history[i]['predicted_rewards']
                next_rewards = outputs_history[i+1]['predicted_rewards']
                
                # Consistency: predictions should not change drastically
                consistency = torch.mean(torch.abs(current_rewards - next_rewards))
                consistency_losses.append(consistency)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        else:
            return torch.tensor(0.0, requires_grad=True)
    
    def _compute_value_target(self, future_rewards: List[float], gamma: float = 0.95) -> torch.Tensor:
        """Compute discounted value target from future rewards"""
        
        if not future_rewards:
            return torch.tensor(0.0)
        
        discounted_return = 0.0
        for i, reward in enumerate(future_rewards):
            discounted_return += (gamma ** i) * reward
            
        return torch.tensor(discounted_return).float()
    
    def get_supervision_schedule(self, 
                               episode: int, 
                               total_episodes: int,
                               curriculum_strategy: str = "progressive") -> List[int]:
        """
        Get supervision schedule based on training progress
        
        Args:
            episode: Current episode
            total_episodes: Total training episodes
            curriculum_strategy: How to schedule supervision complexity
            
        Returns:
            List of active segment indices for this episode
        """
        
        progress = episode / total_episodes
        
        if curriculum_strategy == "progressive":
            # Start with immediate supervision, gradually add longer horizons
            if progress < 0.25:
                return [0]  # Only immediate supervision
            elif progress < 0.5:
                return [0, 1]  # Immediate + short-term
            elif progress < 0.75:
                return [0, 1, 2]  # Up to medium-term
            else:
                return [0, 1, 2, 3]  # All segments
                
        elif curriculum_strategy == "alternating":
            # Alternate between different supervision patterns
            cycle = episode % 4
            if cycle == 0:
                return [0, 1]
            elif cycle == 1:
                return [1, 2]  
            elif cycle == 2:
                return [2, 3]
            else:
                return [0, 3]  # Immediate and long-term
                
        elif curriculum_strategy == "random":
            # Randomly select segments
            num_active = np.random.randint(1, self.num_segments + 1)
            return sorted(np.random.choice(self.num_segments, num_active, replace=False))
            
        else:  # "all"
            return list(range(self.num_segments))
    
    def analyze_supervision_effectiveness(self,
                                        supervision_losses: Dict[str, List[float]],
                                        actual_performance: List[float]) -> Dict[str, float]:
        """
        Analyze effectiveness of different supervision segments
        
        Args:
            supervision_losses: Dictionary of losses for each segment over time
            actual_performance: Actual trading performance over time
            
        Returns:
            Analysis of supervision effectiveness
        """
        
        analysis = {}
        
        # Correlation between supervision losses and performance
        for segment_name, losses in supervision_losses.items():
            if len(losses) == len(actual_performance) and len(losses) > 10:
                correlation = np.corrcoef(losses, actual_performance)[0, 1]
                analysis[f'{segment_name}_performance_correlation'] = correlation
        
        # Learning speed analysis
        for segment_name, losses in supervision_losses.items():
            if len(losses) > 50:
                early_loss = np.mean(losses[:20])
                late_loss = np.mean(losses[-20:])
                improvement = (early_loss - late_loss) / early_loss if early_loss > 0 else 0
                analysis[f'{segment_name}_improvement_rate'] = improvement
        
        # Segment importance (based on loss magnitudes)
        loss_magnitudes = {name: np.mean(losses) for name, losses in supervision_losses.items() if losses}
        total_loss = sum(loss_magnitudes.values())
        
        for name, magnitude in loss_magnitudes.items():
            analysis[f'{name}_relative_importance'] = magnitude / total_loss if total_loss > 0 else 0
        
        return analysis