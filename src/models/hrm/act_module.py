"""
Adaptive Computation Time (ACT) Module
Decides when to halt computation based on market complexity and confidence
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class AdaptiveComputationTime(nn.Module):
    """
    ACT mechanism that decides when to halt computation
    
    Key principles:
    - More computation for complex/uncertain market conditions
    - Less computation for clear/simple market conditions
    - Balances accuracy vs computational efficiency
    """
    
    def __init__(self, 
                 strategic_hidden_dim: int,
                 tactical_hidden_dim: int,
                 regime_dim: int = 5,
                 performance_dim: int = 4):
        super().__init__()
        
        self.strategic_hidden_dim = strategic_hidden_dim
        self.tactical_hidden_dim = tactical_hidden_dim
        self.regime_dim = regime_dim
        self.performance_dim = performance_dim
        
        # Combine all information sources
        total_input_dim = (strategic_hidden_dim + 
                          tactical_hidden_dim + 
                          regime_dim + 
                          performance_dim)
        
        # Information fusion network
        self.fusion_network = nn.Sequential(
            nn.Linear(total_input_dim, strategic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(strategic_hidden_dim, strategic_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Halt decision network
        self.halt_network = nn.Sequential(
            nn.Linear(strategic_hidden_dim // 2, strategic_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(strategic_hidden_dim // 4, 2)  # halt_logits, continue_logits
        )
        
        # Market complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(strategic_hidden_dim // 2, strategic_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(strategic_hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(strategic_hidden_dim // 2, strategic_hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(strategic_hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                strategic_hidden: torch.Tensor,
                tactical_hidden: torch.Tensor,
                regime_probs: torch.Tensor,
                performance_metrics: torch.Tensor,
                step_count: int,
                max_steps: int) -> Dict[str, torch.Tensor]:
        """
        Decide whether to halt or continue computation
        
        Args:
            strategic_hidden: H-module hidden state
            tactical_hidden: L-module hidden state  
            regime_probs: Market regime probabilities
            performance_metrics: Recent performance metrics
            step_count: Current computation step
            max_steps: Maximum allowed steps
            
        Returns:
            Dictionary with halt decision and related information
        """
        
        # Combine all information
        combined_info = torch.cat([
            strategic_hidden,
            tactical_hidden, 
            regime_probs,
            performance_metrics
        ], dim=-1)
        
        # Fuse information
        fused_repr = self.fusion_network(combined_info)
        
        # Generate halt/continue logits
        halt_continue_logits = self.halt_network(fused_repr)
        halt_logits = halt_continue_logits[:, 0:1]
        continue_logits = halt_continue_logits[:, 1:2]
        
        # Estimate market complexity and model confidence
        market_complexity = self.complexity_estimator(fused_repr)
        model_confidence = self.confidence_estimator(fused_repr)
        
        # Adaptive halting logic
        halt_probability = self._compute_halt_probability(
            halt_logits, continue_logits, market_complexity, 
            model_confidence, step_count, max_steps
        )
        
        # Q-learning targets for training
        q_targets = self._compute_q_targets(
            market_complexity, model_confidence, step_count, max_steps
        )
        
        return {
            'halt_logits': halt_logits,
            'continue_logits': continue_logits,
            'halt_probability': halt_probability,
            'market_complexity': market_complexity,
            'model_confidence': model_confidence,
            'q_halt_target': q_targets['halt'],
            'q_continue_target': q_targets['continue'],
            'should_halt': halt_probability > 0.5
        }
    
    def _compute_halt_probability(self,
                                 halt_logits: torch.Tensor,
                                 continue_logits: torch.Tensor,
                                 market_complexity: torch.Tensor,
                                 model_confidence: torch.Tensor,
                                 step_count: int,
                                 max_steps: int) -> torch.Tensor:
        """Compute probability of halting based on multiple factors"""
        
        # Base halt probability from logits
        base_halt_prob = torch.sigmoid(halt_logits - continue_logits)
        
        # Complexity adjustment: more complex markets need more computation
        complexity_adjustment = 1.0 - market_complexity  # High complexity -> lower halt prob
        
        # Confidence adjustment: high confidence allows earlier halt
        confidence_adjustment = model_confidence
        
        # Step count pressure: increase halt probability as steps increase
        step_pressure = torch.tensor(step_count / max_steps).to(halt_logits.device)
        
        # Combine factors
        adjusted_halt_prob = base_halt_prob * confidence_adjustment * complexity_adjustment
        
        # Apply step pressure
        final_halt_prob = adjusted_halt_prob + step_pressure * 0.3
        
        return torch.clamp(final_halt_prob, 0.0, 1.0)
    
    def _compute_q_targets(self,
                          market_complexity: torch.Tensor,
                          model_confidence: torch.Tensor, 
                          step_count: int,
                          max_steps: int) -> Dict[str, torch.Tensor]:
        """Compute Q-learning targets for ACT training"""
        
        # Reward for halting: higher for confident predictions in simple markets
        halt_reward = model_confidence * (1.0 - market_complexity)
        
        # Penalty for using too many steps
        step_penalty = torch.tensor(step_count / max_steps).to(market_complexity.device) * 0.1
        
        # Reward for continuing: higher when more computation is needed
        continue_reward = market_complexity * (1.0 - model_confidence) - step_penalty
        
        # Q-targets (discounted future rewards)
        gamma = 0.95  # Discount factor
        
        q_halt = halt_reward
        q_continue = continue_reward + gamma * torch.max(halt_reward, continue_reward)
        
        return {
            'halt': q_halt,
            'continue': q_continue
        }
    
    def should_halt(self,
                   outputs: Dict[str, torch.Tensor],
                   step_count: int,
                   max_steps: int,
                   exploration_prob: float = 0.1) -> bool:
        """
        Make final halt decision
        
        Args:
            outputs: ACT outputs from forward pass
            step_count: Current step count
            max_steps: Maximum allowed steps
            exploration_prob: Probability of random exploration
            
        Returns:
            Whether to halt computation
        """
        
        # Force halt at max steps
        if step_count >= max_steps:
            return True
        
        # Exploration: sometimes halt randomly to encourage diversity
        if self.training and torch.rand(1).item() < exploration_prob:
            return torch.rand(1).item() < 0.5
        
        # Use model's halt decision
        return outputs['should_halt'].item() if outputs['should_halt'].numel() == 1 else outputs['should_halt'][0].item()
    
    def get_computation_stats(self, 
                             halt_decisions: List[bool],
                             complexities: List[float],
                             confidences: List[float]) -> Dict[str, float]:
        """
        Analyze computation efficiency statistics
        
        Args:
            halt_decisions: List of halt decisions over episodes
            complexities: List of market complexities
            confidences: List of model confidences
            
        Returns:
            Computation efficiency statistics
        """
        
        if not halt_decisions:
            return {}
        
        avg_steps = sum([i for i, halt in enumerate(halt_decisions) if halt]) / len(halt_decisions)
        halt_rate = sum(halt_decisions) / len(halt_decisions)
        
        # Correlation between complexity and computation time
        if len(complexities) == len(halt_decisions):
            complex_episodes = [i for i, (halt, comp) in enumerate(zip(halt_decisions, complexities)) if comp > 0.7]
            simple_episodes = [i for i, (halt, comp) in enumerate(zip(halt_decisions, complexities)) if comp < 0.3]
            
            avg_steps_complex = np.mean(complex_episodes) if complex_episodes else 0
            avg_steps_simple = np.mean(simple_episodes) if simple_episodes else 0
        else:
            avg_steps_complex = avg_steps_simple = 0
        
        return {
            'average_steps': avg_steps,
            'halt_rate': halt_rate,
            'avg_steps_complex_markets': avg_steps_complex,
            'avg_steps_simple_markets': avg_steps_simple,
            'computation_efficiency': 1.0 - (avg_steps / len(halt_decisions)) if halt_decisions else 0,
            'adaptive_behavior': abs(avg_steps_complex - avg_steps_simple)
        }