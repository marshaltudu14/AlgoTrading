"""
HRM Output Heads

Implements the output prediction heads for the HRM:
- PolicyHead: Discrete action prediction
- QuantityHead: Continuous position sizing
- ValueHead: State value estimation
- QHead: Q-learning for ACT mechanism

Based on the HRM research paper's output methodology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple


class PolicyHead(nn.Module):
    """Policy head for discrete action prediction"""
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.fc = nn.Linear(input_dim, action_dim, bias=False)

    def forward(self, z_h):
        return self.fc(z_h)


class QuantityHead(nn.Module):
    """Quantity head for continuous position sizing"""
    def __init__(self, input_dim: int, quantity_min: float = 1.0, quantity_max: float = 100000.0):
        super().__init__()
        self.input_dim = input_dim
        self.quantity_min = quantity_min
        self.quantity_max = quantity_max
        self.fc = nn.Linear(input_dim, 1, bias=False)

    def forward(self, z_h):
        raw_output = torch.sigmoid(self.fc(z_h))
        # Scale to quantity range
        quantity = self.quantity_min + raw_output * (self.quantity_max - self.quantity_min)
        return quantity.squeeze(-1)


class ValueHead(nn.Module):
    """Value head for state value estimation"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 1, bias=False)

    def forward(self, z_h):
        return self.fc(z_h).squeeze(-1)


class QHead(nn.Module):
    """Q-head for ACT halting mechanism"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 2, bias=False)  # halt, continue

    def forward(self, z_h):
        return torch.sigmoid(self.fc(z_h))


class OutputProcessor:
    """
    Orchestrates output generation from HRM's final H-module state.
    
    Handles all output heads and provides unified interface for action generation
    with comprehensive error handling and validation.
    """
    
    def __init__(
        self,
        policy_head: PolicyHead,
        quantity_head: QuantityHead,
        value_head: ValueHead = None,
        q_head: QHead = None
    ):
        self.policy_head = policy_head
        self.quantity_head = quantity_head
        self.value_head = value_head
        self.q_head = q_head
    
    def generate_outputs(self, z_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate all outputs from final H-module state with error handling.
        
        Args:
            z_h: Final H-module state [batch_size, hidden_dim]
            
        Returns:
            Dictionary containing all output predictions
        """
        batch_size = z_h.size(0)
        device = z_h.device
        outputs = {}
        
        # Policy head (required)
        try:
            outputs['action_type'] = self.policy_head(z_h)
        except Exception as e:
            # Fallback: uniform random policy
            outputs['action_type'] = torch.zeros(
                batch_size, self.policy_head.action_dim, device=device
            )
        
        # Quantity head (required)
        try:
            outputs['quantity'] = self.quantity_head(z_h)
        except Exception as e:
            # Fallback: minimum quantity
            outputs['quantity'] = torch.full(
                (batch_size,), self.quantity_head.quantity_min, device=device
            )
        
        # Value head (optional)
        if self.value_head is not None:
            try:
                outputs['value'] = self.value_head(z_h)
            except Exception as e:
                outputs['value'] = torch.zeros(batch_size, device=device)
        
        # Q-head (optional)
        if self.q_head is not None:
            try:
                outputs['q_values'] = self.q_head(z_h)
            except Exception as e:
                outputs['q_values'] = torch.zeros(batch_size, 2, device=device)  # halt, continue
        
        return outputs
    
    def sample_action(
        self, 
        outputs: Dict[str, torch.Tensor], 
        available_capital: float = None,
        current_position_quantity: float = None,
        current_price: float = None,
        use_sampling: bool = True
    ) -> Tuple[int, float]:
        """
        Sample action from policy outputs with trading constraints.
        
        Args:
            outputs: Output dictionary from generate_outputs()
            available_capital: Available capital for position sizing
            current_position_quantity: Current position (+ for long, - for short, 0 for none)
            current_price: Current market price
            use_sampling: Whether to use probabilistic sampling or deterministic argmax
            
        Returns:
            Tuple of (action_type, quantity)
        """
        try:
            action_logits = outputs['action_type']
            
            # Validate logits
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                action_probs = torch.ones_like(action_logits) / action_logits.size(-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
            
            # Apply action masking if trading context provided
            if all(v is not None for v in [available_capital, current_position_quantity, current_price]):
                action_probs = self._apply_trading_mask(
                    action_probs, available_capital, current_position_quantity, current_price
                )
            
            # Sample or select deterministically
            if use_sampling:
                try:
                    action_type = torch.multinomial(action_probs, 1).item()
                except Exception:
                    action_type = torch.argmax(action_probs, dim=-1).item()
            else:
                action_type = torch.argmax(action_probs, dim=-1).item()
            
            # Validate action type
            if action_type < 0 or action_type >= self.policy_head.action_dim:
                action_type = 4  # HOLD as fallback
            
            # Extract quantity
            quantity = outputs['quantity'].item()
            
            # Validate quantity
            if torch.isnan(torch.tensor(quantity)) or torch.isinf(torch.tensor(quantity)):
                quantity = self.quantity_head.quantity_min
            
            quantity = max(self.quantity_head.quantity_min, 
                          min(quantity, self.quantity_head.quantity_max))
            
            return action_type, quantity
            
        except Exception as e:
            # Emergency fallback
            return 4, self.quantity_head.quantity_min  # HOLD, minimum quantity
    
    def _apply_trading_mask(
        self, 
        action_probs: torch.Tensor,
        available_capital: float,
        current_position_quantity: float,
        current_price: float
    ) -> torch.Tensor:
        """Apply trading constraints to action probabilities"""
        # Action types: 0=BUY_LONG, 1=SELL_SHORT, 2=CLOSE_LONG, 3=CLOSE_SHORT, 4=HOLD
        mask = torch.ones_like(action_probs, dtype=torch.bool)
        
        # If no capital available, mask opening positions
        max_affordable = available_capital / current_price if current_price > 0 else 0
        if max_affordable <= 0:
            mask[0, 0] = False  # BUY_LONG
            mask[0, 1] = False  # SELL_SHORT
        
        # Mask closing actions based on current position
        if current_position_quantity <= 0:
            mask[0, 2] = False  # CLOSE_LONG (not holding long position)
        if current_position_quantity >= 0:
            mask[0, 3] = False  # CLOSE_SHORT (not holding short position)
        
        # If already in position, prevent opening new ones
        if current_position_quantity != 0:
            mask[0, 0] = False  # BUY_LONG
            mask[0, 1] = False  # SELL_SHORT
        
        # Apply mask by setting masked actions to very low probability
        action_probs = torch.where(
            mask, action_probs, torch.tensor(1e-9, device=action_probs.device)
        )
        
        # Renormalize
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        return action_probs
    
    def get_head_stats(self) -> Dict[str, Any]:
        """Get statistics about output heads"""
        stats = {
            'policy_head_params': sum(p.numel() for p in self.policy_head.parameters()),
            'quantity_head_params': sum(p.numel() for p in self.quantity_head.parameters()),
            'action_dim': self.policy_head.action_dim,
            'quantity_range': (self.quantity_head.quantity_min, self.quantity_head.quantity_max),
            'has_value_head': self.value_head is not None,
            'has_q_head': self.q_head is not None
        }
        
        if self.value_head is not None:
            stats['value_head_params'] = sum(p.numel() for p in self.value_head.parameters())
        
        if self.q_head is not None:
            stats['q_head_params'] = sum(p.numel() for p in self.q_head.parameters())
        
        return stats