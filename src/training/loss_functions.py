"""
HRM Loss Function Components
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class HRMLossFunction:
    """Multi-component loss function for HRM trading"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.loss_weights = config['training']['loss_weights']
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
    def calculate_loss(self, 
                      outputs: Dict[str, torch.Tensor], 
                      targets: Dict[str, torch.Tensor],
                      segment_rewards: List[float]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate multi-component HRM loss with expert targets using mixed precision"""
        
        losses = {}
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # ONLY market understanding guidance loss (help model interpret market context)
        if 'market_understanding_outputs' in outputs and 'market_understanding' in targets:
            if targets['market_understanding'].numel() > 0:
                model_output = outputs['market_understanding_outputs']
                target_tensor = targets['market_understanding']
                
                # Market understanding loss to guide market context interpretation
                understanding_loss = nn.MSELoss()(model_output, target_tensor)
                losses['market_understanding_loss'] = understanding_loss
                total_loss = total_loss + self.loss_weights.get('tactical_loss', 1.0) * understanding_loss
        
        # ACT loss - use the Q-learning targets from the ACT module
        if 'halt_logits' in outputs and 'continue_logits' in outputs and 'q_halt_target' in outputs:
            # Ensure proper tensor shapes for MSE loss
            halt_logits = outputs['halt_logits']
            continue_logits = outputs['continue_logits']
            halt_target = outputs['q_halt_target'].detach()
            continue_target = outputs['q_continue_target'].detach()
            
            # Reshape tensors to ensure compatibility
            if halt_logits.dim() > halt_target.dim():
                halt_logits = halt_logits.view_as(halt_target)
            elif halt_target.dim() > halt_logits.dim():
                halt_target = halt_target.view_as(halt_logits)
                
            if continue_logits.dim() > continue_target.dim():
                continue_logits = continue_logits.view_as(continue_target)
            elif continue_target.dim() > continue_logits.dim():
                continue_target = continue_target.view_as(continue_logits)
            
            halt_loss = nn.MSELoss()(halt_logits, halt_target)
            continue_loss = nn.MSELoss()(continue_logits, continue_target)
            
            act_loss = halt_loss + continue_loss
            losses['act_loss'] = act_loss
            total_loss = total_loss + self.loss_weights['act_loss'] * act_loss
        
        # Action entropy regularization for exploration (encourage diverse action selection)
        if 'action_probabilities' in outputs:
            action_probs = outputs['action_probabilities']
            # Calculate entropy: -sum(p * log(p))
            entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1)
            # We want to maximize entropy (encourage exploration), so minimize negative entropy
            entropy_loss = -torch.mean(entropy)
            losses['entropy_loss'] = entropy_loss
            total_loss = total_loss + self.loss_weights.get('entropy_loss', 0.01) * entropy_loss
        
        # Performance-based loss (primary signal) - GPU optimized when available
        if segment_rewards:
            if total_loss.device.type == 'cuda':
                # GPU: Use tensor operations for better GPU utilization
                rewards_tensor = torch.tensor(segment_rewards, device=total_loss.device, dtype=torch.float32)
                performance_loss = -torch.mean(rewards_tensor)
            else:
                # CPU: Use standard numpy approach (more efficient on CPU)
                performance_loss = -torch.tensor(np.mean(segment_rewards))
            losses['performance_loss'] = performance_loss
            total_loss = total_loss + self.loss_weights['performance_loss'] * performance_loss
        
        # Convert loss values to float for logging
        loss_values = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        loss_values['total_loss'] = total_loss.item()
        
        return total_loss, loss_values