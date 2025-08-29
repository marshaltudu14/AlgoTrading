"""
Main HRM Trading Agent - Orchestrates all components
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import logging

from .high_level_module import HighLevelModule, MarketRegime
from .low_level_module import LowLevelModule  
from .act_module import AdaptiveComputationTime
from .deep_supervision import DeepSupervision
from ...utils.instruments_loader import get_instruments_loader

logger = logging.getLogger(__name__)


@dataclass
class HRMTradingState:
    """Internal state for HRM trading agent"""
    z_H: torch.Tensor  # High-level (strategic) hidden state
    z_L: torch.Tensor  # Low-level (tactical) hidden state
    step_count: int    # Current computation step
    segment_count: int # Current deep supervision segment


@dataclass
class HRMCarry:
    """Carry state between HRM forward passes"""
    inner_state: HRMTradingState
    halted: torch.Tensor  # Whether computation has halted for this batch
    performance_history: List[float]  # Recent performance for ACT


class HRMTradingAgent(nn.Module):
    """
    Complete HRM agent for algorithmic trading
    
    Orchestrates:
    - High-level strategic reasoning (100 candles)
    - Low-level tactical execution (15 candles)  
    - Adaptive computation time
    - Deep supervision
    """
    
    def __init__(self, 
                 feature_dim: int,  # Features per candle
                 h_lookback: int = 100,  # H-module lookback
                 l_lookback: int = 15,   # L-module lookback
                 hidden_dim: int = 256,
                 H_layers: int = 4,
                 L_layers: int = 3,
                 num_heads: int = 8,
                 H_cycles: int = 2,
                 L_cycles: int = 5,
                 halt_max_steps: int = 8,
                 halt_exploration_prob: float = 0.1,
                 dropout: float = 0.1,
                 device: str = "cuda",
                 use_market_embeddings: bool = True):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.h_lookback = h_lookback
        self.l_lookback = l_lookback
        self.hidden_dim = hidden_dim
        self.H_layers = H_layers
        self.L_layers = L_layers
        self.num_heads = num_heads
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        self.halt_max_steps = halt_max_steps
        self.halt_exploration_prob = halt_exploration_prob
        self.device = torch.device(device)
        self.use_market_embeddings = use_market_embeddings
        
        # Load instruments configuration if using embeddings
        self.instruments_loader = get_instruments_loader() if use_market_embeddings else None
        
        # Initialize modules
        self._init_modules(dropout)
        
        # Move to device
        self.to(self.device)
        
        logger.info(f"HRM Agent initialized: H-lookback={h_lookback}, L-lookback={l_lookback}, "
                   f"Hidden={hidden_dim}, H-cycles={H_cycles}, L-cycles={L_cycles}")
    
    def _init_modules(self, dropout: float):
        """Initialize all HRM components"""
        
        # High-level strategic module (sees 100 candles)
        self.high_level_module = HighLevelModule(
            feature_dim=self.feature_dim,
            lookback_window=self.h_lookback,
            hidden_dim=self.hidden_dim,
            num_layers=self.H_layers,
            num_heads=self.num_heads,
            instruments_loader=self.instruments_loader,
            dropout=dropout
        )
        
        # Low-level tactical module (sees 15 candles)
        # Strategic context is just the hidden representation from H-module
        strategic_context_dim = self.hidden_dim
        self.low_level_module = LowLevelModule(
            feature_dim=self.feature_dim,
            lookback_window=self.l_lookback, 
            hidden_dim=self.hidden_dim,
            num_layers=self.L_layers,
            num_heads=self.num_heads,
            strategic_context_dim=strategic_context_dim,
            dropout=dropout
        )
        
        # Adaptive Computation Time
        self.act_module = AdaptiveComputationTime(
            strategic_hidden_dim=self.hidden_dim,
            tactical_hidden_dim=self.hidden_dim,
            regime_dim=len(MarketRegime),
            performance_dim=4
        )
        
        # Deep Supervision
        self.deep_supervision = DeepSupervision(
            num_segments=4,
            segment_weights=[0.4, 0.3, 0.2, 0.1],
            hidden_dim=self.hidden_dim
        )
        
        # Initialize hidden states
        self.register_buffer('initial_H_state', torch.randn(1, self.hidden_dim) * 0.1)
        self.register_buffer('initial_L_state', torch.randn(1, self.hidden_dim) * 0.1)
        
    def create_initial_carry(self, batch_size: int) -> HRMCarry:
        """Create initial carry state for a batch"""
        
        initial_state = HRMTradingState(
            z_H=self.initial_H_state.expand(batch_size, -1).clone(),
            z_L=self.initial_L_state.expand(batch_size, -1).clone(),
            step_count=0,
            segment_count=0
        )
        
        return HRMCarry(
            inner_state=initial_state,
            halted=torch.ones(batch_size, dtype=torch.bool, device=self.device),  # Start halted for reset
            performance_history=[]
        )
    
    def prepare_hierarchical_input(self, 
                                  full_observation: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare different inputs for H and L modules from full observation
        
        Args:
            full_observation: [batch_size, total_features] where total_features = max_lookback * feature_dim + account_features
            
        Returns:
            h_input: Strategic input (H lookback candles)
            l_input: Tactical input (L lookback candles)
        """
        
        batch_size = full_observation.size(0)
        
        # Account features are at the end (6 features)
        account_features = 6
        market_features_dim = full_observation.size(1) - account_features
        
        # Extract market features and account features
        market_features = full_observation[:, :market_features_dim]
        account_state = full_observation[:, market_features_dim:]
        
        # Calculate actual lookback from market features
        max_lookback = market_features_dim // self.feature_dim
        
        # Ensure we only use features that divide evenly
        usable_features = max_lookback * self.feature_dim
        market_features_clean = market_features[:, :usable_features]
        
        # Reshape market features to [batch_size, max_lookback, feature_dim]
        reshaped = market_features_clean.view(batch_size, max_lookback, self.feature_dim)
        
        # Extract H-module input (strategic lookback) - keep as 3D sequence
        h_candles = min(self.h_lookback, max_lookback)
        h_sequence = reshaped[:, :h_candles, :]  # [batch, h_candles, feature_dim]
        
        # Extract L-module input (tactical lookback) - keep as 3D sequence
        l_candles = min(self.l_lookback, max_lookback)
        l_sequence = reshaped[:, :l_candles, :]  # [batch, l_candles, feature_dim]
        
        # H and L modules will process market features only
        # Account state will be handled separately in the trading decision logic
        h_input = h_sequence
        l_input = l_sequence
        
        # Store account state for later use
        self._current_account_state = account_state
        
        return h_input, l_input
    
    def forward(self, 
                carry: HRMCarry, 
                observation: torch.Tensor, 
                training: bool = True,
                instrument_id: Optional[torch.Tensor] = None,
                timeframe_id: Optional[torch.Tensor] = None) -> Tuple[HRMCarry, Dict[str, torch.Tensor]]:
        """
        Main forward pass with hierarchical reasoning
        
        Args:
            carry: Current HRM state
            observation: Market observation [batch_size, obs_dim]
            training: Whether in training mode
            
        Returns:
            new_carry: Updated HRM state
            outputs: All module outputs
        """
        
        batch_size = observation.size(0)
        
        # Reset states for halted sequences
        reset_mask = carry.halted.unsqueeze(-1)
        carry.inner_state.z_H = torch.where(reset_mask, 
                                           self.initial_H_state.expand_as(carry.inner_state.z_H), 
                                           carry.inner_state.z_H)
        carry.inner_state.z_L = torch.where(reset_mask,
                                           self.initial_L_state.expand_as(carry.inner_state.z_L),
                                           carry.inner_state.z_L)
        carry.inner_state.step_count = 0 if carry.halted.any() else carry.inner_state.step_count
        
        # Prepare hierarchical inputs
        h_input, l_input = self.prepare_hierarchical_input(observation)
        
        # Hierarchical reasoning with one-step gradient approximation
        z_H, z_L = carry.inner_state.z_H, carry.inner_state.z_L
        
        # Most computation without gradients for efficiency
        with torch.no_grad():
            for h_cycle in range(self.H_cycles):
                # L-module cycles within each H-cycle
                for l_cycle in range(self.L_cycles):
                    if not (h_cycle == self.H_cycles - 1 and l_cycle == self.L_cycles - 1):
                        # Get strategic context (use H-module hidden state)
                        temp_h_hidden, _ = self.high_level_module(h_input, None, instrument_id, timeframe_id)
                        strategic_context = temp_h_hidden.mean(dim=1) if len(temp_h_hidden.shape) == 3 else temp_h_hidden
                        
                        # L-module update
                        temp_l_hidden, _ = self.low_level_module(l_input, strategic_context, z_L)
                        z_L = temp_l_hidden.mean(dim=1) if len(temp_l_hidden.shape) == 3 else temp_l_hidden
                
                # H-module update (except for the last cycle)
                if h_cycle != self.H_cycles - 1:
                    temp_h_hidden, _ = self.high_level_module(h_input, None, instrument_id, timeframe_id)
                    z_H = temp_h_hidden.mean(dim=1) if len(temp_h_hidden.shape) == 3 else temp_h_hidden
        
        # Final step with gradients (one-step approximation)
        h_hidden, strategic_outputs = self.high_level_module(h_input, None, instrument_id, timeframe_id)
        z_H = h_hidden.mean(dim=1) if len(h_hidden.shape) == 3 else h_hidden
        
        strategic_context = z_H  # Use H-module hidden state as strategic context
        l_hidden, tactical_outputs = self.low_level_module(l_input, strategic_context, z_L)
        z_L = l_hidden.mean(dim=1) if len(l_hidden.shape) == 3 else l_hidden
        
        # Performance metrics (placeholder - would be updated with actual performance)
        performance_metrics = torch.zeros(batch_size, 4, device=self.device)
        if carry.performance_history:
            recent_performance = carry.performance_history[-4:]
            perf_tensor = torch.tensor(recent_performance + [0.0] * (4 - len(recent_performance)))
            performance_metrics = perf_tensor.unsqueeze(0).expand(batch_size, -1).to(self.device)
        
        # Adaptive computation time decision
        act_outputs = {}
        new_halted = carry.halted.clone()
        
        if training and self.halt_max_steps > 1:
            act_outputs = self.act_module(
                strategic_outputs['strategic_hidden'],
                tactical_outputs['tactical_hidden'],
                strategic_outputs['regime_probabilities'],
                performance_metrics,
                carry.inner_state.step_count,
                self.halt_max_steps
            )
            
            # Update halting logic
            carry.inner_state.step_count += 1
            is_max_steps = carry.inner_state.step_count >= self.halt_max_steps
            
            should_halt = self.act_module.should_halt(
                act_outputs, 
                carry.inner_state.step_count, 
                self.halt_max_steps, 
                self.halt_exploration_prob
            )
            
            new_halted = torch.tensor([is_max_steps or should_halt] * batch_size, 
                                    dtype=torch.bool, device=self.device)
        else:
            # During evaluation, always use max steps for consistency
            carry.inner_state.step_count += 1
            new_halted = torch.tensor([carry.inner_state.step_count >= self.halt_max_steps] * batch_size,
                                    dtype=torch.bool, device=self.device)
        
        # Deep supervision (if in training mode)
        supervision_outputs = {}
        if training:
            supervision_outputs = self.deep_supervision(
                strategic_outputs['strategic_hidden'],
                tactical_outputs['tactical_hidden'],
                carry.inner_state.segment_count % 4
            )
            carry.inner_state.segment_count += 1
        
        # Update carry state
        new_carry = HRMCarry(
            inner_state=HRMTradingState(
                z_H=z_H.detach(),
                z_L=z_L.detach(),
                step_count=carry.inner_state.step_count,
                segment_count=carry.inner_state.segment_count
            ),
            halted=new_halted,
            performance_history=carry.performance_history
        )
        
        # Combine all outputs
        outputs = {
            **strategic_outputs,
            **tactical_outputs,
            **act_outputs,
            **supervision_outputs,
            'final_z_H': z_H,
            'final_z_L': z_L
        }
        
        return new_carry, outputs
    
    def extract_trading_decision(self, 
                               outputs: Dict[str, torch.Tensor],
                               current_position: float = 0.0,
                               available_capital: float = 100000.0) -> Dict[str, any]:
        """Extract final trading decision from all outputs"""
        
        # Use L-module's trading decision logic
        decision = self.low_level_module.extract_trading_decision(
            outputs, outputs, current_position, available_capital
        )
        
        # Add strategic context
        if 'regime_probabilities' in outputs:
            regime_analysis = self.high_level_module.interpret_regime(outputs['regime_probabilities'])
            decision['market_regime'] = regime_analysis
        
        # Add ACT information
        if 'market_complexity' in outputs:
            decision['market_complexity'] = outputs['market_complexity'].item()
        if 'model_confidence' in outputs:
            decision['model_confidence'] = outputs['model_confidence'].item()
        
        return decision
    
    def should_act(self, 
                   outputs: Dict[str, torch.Tensor],
                   current_position: float,
                   min_confidence: float = 0.6) -> bool:
        """Determine if the model is confident enough to take action"""
        
        return self.low_level_module.should_act(outputs, current_position, min_confidence)
    
    def update_performance_metrics(self, 
                                 carry: HRMCarry, 
                                 reward: float, 
                                 total_return: float,
                                 sharpe_ratio: float, 
                                 max_drawdown: float):
        """Update performance metrics for ACT decision making"""
        
        carry.performance_history.append(reward)
        
        # Keep recent history only
        if len(carry.performance_history) > 100:
            carry.performance_history.pop(0)
    
    def get_model_summary(self) -> Dict[str, any]:
        """Get comprehensive model summary"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        h_params = sum(p.numel() for p in self.high_level_module.parameters())
        l_params = sum(p.numel() for p in self.low_level_module.parameters())
        act_params = sum(p.numel() for p in self.act_module.parameters())
        ds_params = sum(p.numel() for p in self.deep_supervision.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'h_module_parameters': h_params,
            'l_module_parameters': l_params,
            'act_module_parameters': act_params,
            'deep_supervision_parameters': ds_params,
            'h_lookback_window': self.h_lookback,
            'l_lookback_window': self.l_lookback,
            'hidden_dimension': self.hidden_dim,
            'H_cycles': self.H_cycles,
            'L_cycles': self.L_cycles,
            'max_computation_steps': self.halt_max_steps,
            'device': str(self.device)
        }