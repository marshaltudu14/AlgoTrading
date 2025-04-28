"""
Mixture of Experts (MoE) Transformer for algorithmic trading.
Combines multiple specialized expert networks with a gating mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List, Any

from models.trading_transformer import (
    PositionalEncoding, 
    PrototypicalNetwork, 
    TradingTransformer
)


class ExpertNetwork(nn.Module):
    """
    Expert network for the Mixture of Experts model.
    Each expert specializes in a different market regime or pattern.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Create layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the expert network."""
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    Gating network for the Mixture of Experts model.
    Routes inputs to the appropriate experts based on input features.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
        noisy_gating: bool = True,
        k: int = 2  # Top-k experts to use (sparse gating)
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.k = min(k, num_experts)  # Can't select more experts than we have
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # Noise for exploration (optional)
        if noisy_gating:
            self.noise_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the gating network.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (routing_weights, expert_indices)
            - routing_weights: Tensor of shape [batch_size, k] with weights for each selected expert
            - expert_indices: Tensor of shape [batch_size, k] with indices of selected experts
        """
        # Get raw gating scores
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        
        # Add noise for exploration if enabled
        if self.noisy_gating and self.training:
            noise = torch.randn_like(gate_logits) * self.noise_scale
            gate_logits = gate_logits + noise
        
        # Get top-k experts
        routing_weights, expert_indices = torch.topk(
            F.softmax(gate_logits, dim=-1), 
            k=self.k, 
            dim=-1
        )
        
        # Normalize weights to sum to 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        return routing_weights, expert_indices


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    Combines multiple expert networks with a gating mechanism.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        expert_layers: int = 2,
        gate_hidden_dim: int = 64,
        k: int = 2,
        dropout: float = 0.1,
        noisy_gating: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=input_dim,
                hidden_dim=expert_hidden_dim,
                output_dim=output_dim,
                num_layers=expert_layers,
                dropout=dropout
            )
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gating = GatingNetwork(
            input_dim=input_dim,
            hidden_dim=gate_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
            noisy_gating=noisy_gating,
            k=k
        )
        
        # Load balancing loss coefficient
        self.balance_coef = 0.01
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (output, aux_loss)
            - output: Tensor of shape [batch_size, output_dim]
            - aux_loss: Dictionary of auxiliary losses
        """
        batch_size = x.shape[0]
        
        # Get routing weights and expert indices
        routing_weights, expert_indices = self.gating(x)  # [batch_size, k], [batch_size, k]
        
        # Initialize output
        output = torch.zeros(batch_size, self.output_dim, device=x.device)
        
        # Expert utilization for load balancing
        expert_usage = torch.zeros(self.num_experts, device=x.device)
        
        # Route inputs to experts
        for i in range(self.k):
            # Get expert indices and weights for this slot
            indices = expert_indices[:, i]  # [batch_size]
            weights = routing_weights[:, i].unsqueeze(1)  # [batch_size, 1]
            
            # Update expert usage
            for expert_idx in range(self.num_experts):
                expert_usage[expert_idx] += (indices == expert_idx).float().mean()
            
            # Process each expert
            for expert_idx in range(self.num_experts):
                # Find samples routed to this expert
                mask = (indices == expert_idx)
                if not mask.any():
                    continue
                
                # Get inputs for this expert
                expert_inputs = x[mask]
                
                # Get expert outputs
                expert_outputs = self.experts[expert_idx](expert_inputs)
                
                # Weight outputs and add to result
                weighted_outputs = expert_outputs * weights[mask]
                output[mask] += weighted_outputs
        
        # Compute load balancing loss
        # We want each expert to be used equally
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        balance_loss = F.mse_loss(expert_usage, target_usage) * self.balance_coef
        
        # Return output and auxiliary losses
        aux_losses = {
            'balance_loss': balance_loss,
            'expert_usage': expert_usage
        }
        
        return output, aux_losses


class MoETransformer(nn.Module):
    """
    Mixture of Experts Transformer for algorithmic trading.
    Extends the TradingTransformer with MoE layers for better specialization.
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
        num_regimes: int = 3,
        num_experts: int = 4,
        expert_hidden_dim: int = 128,
        k: int = 2,
        noisy_gating: bool = True
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
        
        # Combined embedding dimension
        combined_dim = hidden_dim * 2  # regime_embedding + task_embedding
        
        # MoE policy head
        self.policy_moe = MoELayer(
            input_dim=combined_dim,
            output_dim=action_dim,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            noisy_gating=noisy_gating
        )
        
        # MoE value head
        self.value_moe = MoELayer(
            input_dim=combined_dim,
            output_dim=1,
            num_experts=num_experts,
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            noisy_gating=noisy_gating
        )
        
        # Risk assessment head
        self.risk_moe = MoELayer(
            input_dim=combined_dim,
            output_dim=1,
            num_experts=num_experts // 2,  # Fewer experts for risk
            expert_hidden_dim=expert_hidden_dim,
            k=k,
            dropout=dropout,
            noisy_gating=noisy_gating
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
            Dictionary containing action logits, state values, auxiliary losses, and optionally risk assessment
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
        
        # Generate outputs using MoE layers
        action_logits, policy_aux = self.policy_moe(combined)
        state_values, value_aux = self.value_moe(combined)
        
        # Combine auxiliary losses
        aux_losses = {
            'policy_balance_loss': policy_aux['balance_loss'],
            'value_balance_loss': value_aux['balance_loss'],
            'policy_expert_usage': policy_aux['expert_usage'],
            'value_expert_usage': value_aux['expert_usage']
        }
        
        outputs = {
            'action_logits': action_logits,
            'state_values': state_values,
            'aux_losses': aux_losses
        }
        
        if return_risk:
            risk_assessment, risk_aux = self.risk_moe(combined)
            risk_assessment = torch.sigmoid(risk_assessment)  # Convert to probability
            outputs['risk_assessment'] = risk_assessment
            outputs['aux_losses']['risk_balance_loss'] = risk_aux['balance_loss']
            outputs['aux_losses']['risk_expert_usage'] = risk_aux['expert_usage']
            
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
            'risk_assessment': risk,
            'expert_usage': {
                'policy': outputs['aux_losses']['policy_expert_usage'],
                'value': outputs['aux_losses']['value_expert_usage'],
                'risk': outputs['aux_losses']['risk_expert_usage']
            }
        }
        
        return actions, extra_info


# Utility function to create the model
def create_moe_transformer(
    config: Dict[str, Any],
    state_dim: int,
    num_instruments: int,
    num_timeframes: int,
    action_dim: int = 3
) -> MoETransformer:
    """
    Create a MoETransformer model with the given configuration.
    
    Args:
        config: Model configuration dictionary
        state_dim: Dimension of the state space
        num_instruments: Number of instruments
        num_timeframes: Number of timeframes
        action_dim: Dimension of the action space (default: 3 for hold/buy/sell)
        
    Returns:
        Initialized MoETransformer model
    """
    return MoETransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        num_instruments=num_instruments,
        num_timeframes=num_timeframes,
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.1),
        max_seq_len=config.get('max_seq_len', 50),
        num_regimes=config.get('num_regimes', 3),
        num_experts=config.get('num_experts', 4),
        expert_hidden_dim=config.get('expert_hidden_dim', 128),
        k=config.get('k', 2),
        noisy_gating=config.get('noisy_gating', True)
    )
