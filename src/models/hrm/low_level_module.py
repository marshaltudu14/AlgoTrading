"""
Low-Level Module (L-Module) for Tactical Execution
Focuses on short-term patterns and precise entry/exit timing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional

from .base_components import HierarchicalReasoningModule


class LowLevelModule(HierarchicalReasoningModule):
    """
    Tactical execution module for short-term trading decisions
    
    Key characteristics:
    - Sees 15 candles for immediate market context
    - Updates every step for quick reaction
    - Focuses on precise entry/exit timing, stop placement
    - Receives strategic context from H-module
    """
    
    def __init__(self, 
                 feature_dim: int,  # Number of features per candle
                 lookback_window: int,  # Number of candles to look back (15)
                 hidden_dim: int, 
                 num_layers: int, 
                 num_heads: int,
                 strategic_context_dim: int,  # Dimension of context from H-module
                 dropout: float = 0.1):
        
        # Input dimension is feature_dim per timestep (not flattened)
        # Base module expects [batch, seq, features] format
        input_dim = feature_dim
        
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        
        self.feature_dim = feature_dim
        self.lookback_window = lookback_window
        self.strategic_context_dim = strategic_context_dim
        
        # Strategic context integration
        self.context_integrator = nn.Sequential(
            nn.Linear(strategic_context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Tactical output heads
        self._init_output_heads(hidden_dim)
        
    def _init_output_heads(self, hidden_dim: int):
        """Initialize tactical output heads"""
        
        # Trading action prediction - get number of actions from config
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        self.num_actions = config.get('actions', {}).get('num_actions', 5)
        self.action_names = config.get('actions', {}).get('action_names', 
            ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"])
        
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.num_actions)
        )
        
        
        # Tactical confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Entry timing quality (how good is current moment for entry)
        self.entry_timing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Exit timing quality (how urgent is it to exit current position)
        self.exit_urgency = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Dynamic stop-loss and take-profit adjustments (as factors to multiply base levels)
        self.risk_adjustments = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # stop_loss_factor, take_profit_factor
            nn.Sigmoid()
        )
        
        # Execution quality assessment (expected slippage, market impact)
        self.execution_quality = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # slippage_estimate, market_impact_estimate
        )
        
    def forward(self, 
                market_data: torch.Tensor, 
                strategic_context: torch.Tensor,
                previous_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for tactical reasoning
        
        Args:
            market_data: [batch_size, lookback_window, feature_dim] - recent market data sequence
            strategic_context: Strategic context from H-module
            previous_state: Previous L-module state for continuity
            
        Returns:
            hidden_state: Updated hidden state for this module
            outputs: Dictionary of tactical outputs
        """
        
        # Market data is already in sequence format
        sequence_data = market_data
        
        # Process strategic context
        integrated_context = self.context_integrator(strategic_context)
        
        # Base forward pass with strategic context
        hidden = super().forward(sequence_data, integrated_context)
        
        # Add previous state for temporal continuity
        if previous_state is not None:
            if len(hidden.shape) == 3:  # Sequence output
                hidden = hidden + previous_state.unsqueeze(1).expand_as(hidden)
            else:
                hidden = hidden + previous_state
        
        # Extract tactical representation (use last timestep for decisions)
        if len(hidden.shape) == 3:  # [batch, seq_len, hidden]
            tactical_repr = hidden[:, -1, :]  # Use last timestep for immediate decisions
        else:
            tactical_repr = hidden
            
        # Generate tactical outputs
        action_logits = self.action_head(tactical_repr)
        confidence = self.confidence_head(tactical_repr)
        entry_timing = self.entry_timing(tactical_repr)
        exit_urgency = self.exit_urgency(tactical_repr)
        risk_adjustments = self.risk_adjustments(tactical_repr)
        execution_estimates = self.execution_quality(tactical_repr)
        
        # Package outputs
        outputs = {
            'action_logits': action_logits,
            'action_probabilities': F.softmax(action_logits, dim=-1),
            'confidence': confidence,
            'entry_timing_quality': entry_timing,
            'exit_urgency': exit_urgency,
            'stop_loss_factor': risk_adjustments[:, 0:1],
            'take_profit_factor': risk_adjustments[:, 1:2],
            'expected_slippage': execution_estimates[:, 0:1],
            'market_impact': execution_estimates[:, 1:2],
            'tactical_hidden': tactical_repr
        }
        
        return hidden, outputs
    
    def extract_trading_decision(self, 
                               outputs: Dict[str, torch.Tensor],
                               strategic_outputs: Dict[str, torch.Tensor],
                               current_position: float = 0.0,
                               available_capital: float = 10000.0) -> Dict[str, any]:
        """
        Extract final trading decision from tactical outputs
        
        Args:
            outputs: L-module tactical outputs
            strategic_outputs: H-module strategic outputs for context
            current_position: Current position size
            available_capital: Available capital for trading
            
        Returns:
            Trading decision dictionary
        """
        
        # Get action probabilities
        action_probs = outputs['action_probabilities']
        
        # Sample action based on probabilities (can be deterministic in production)
        if self.training:
            action_idx = torch.multinomial(action_probs, 1).squeeze(-1)
        else:
            action_idx = torch.argmax(action_probs, dim=-1)
        
        # Action mapping using configured action names
        action_name = self.action_names[action_idx.item()] if action_idx.item() < len(self.action_names) else 'UNKNOWN'
        
        # Use a fixed quantity of 1 since we no longer predict quantity
        confidence = outputs['confidence'].item()
        final_quantity = 1.0  # Always use 1 lot
        
        # Risk management levels
        stop_loss_factor = outputs['stop_loss_factor'].item()
        take_profit_factor = outputs['take_profit_factor'].item()
        
        # Execution quality
        expected_slippage = outputs['expected_slippage'].item()
        market_impact = outputs['market_impact'].item()
        
        return {
            'action': action_name,
            'action_idx': action_idx.item(),
            'quantity': final_quantity,
            'confidence': confidence,
            'entry_timing_quality': outputs['entry_timing_quality'].item(),
            'exit_urgency': outputs['exit_urgency'].item(),
            'stop_loss_factor': stop_loss_factor,
            'take_profit_factor': take_profit_factor,
            'expected_slippage': expected_slippage,
            'market_impact': market_impact,
            'action_probabilities': action_probs.detach().cpu().numpy().tolist()
        }
    
    def should_act(self, 
                   outputs: Dict[str, torch.Tensor],
                   current_position: float,
                   min_confidence: float = 0.6) -> bool:
        """
        Determine if the model is confident enough to take action
        
        Args:
            outputs: L-module tactical outputs
            current_position: Current position size
            min_confidence: Minimum confidence threshold
            
        Returns:
            Whether to act on the model's suggestion
        """
        
        confidence = outputs['confidence'].item()
        entry_quality = outputs['entry_timing_quality'].item()
        exit_urgency = outputs['exit_urgency'].item()
        
        # For entries, require both confidence and good timing
        if current_position == 0:
            return confidence > min_confidence and entry_quality > 0.7
        
        # For exits, consider urgency
        else:
            return confidence > min_confidence * 0.8 or exit_urgency > 0.8