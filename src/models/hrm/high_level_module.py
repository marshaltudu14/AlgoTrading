"""
High-Level Module (H-Module) for Strategic Reasoning
Focuses on long-term market analysis and strategic decision making
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
from enum import Enum

from .base_components import HierarchicalReasoningModule


class MarketRegime(Enum):
    """Market regime classification - emerges naturally from data"""
    REGIME_0 = 0  # Will learn what this represents (e.g., trending up)
    REGIME_1 = 1  # Will learn what this represents (e.g., trending down)
    REGIME_2 = 2  # Will learn what this represents (e.g., ranging)
    REGIME_3 = 3  # Will learn what this represents (e.g., high volatility)
    REGIME_4 = 4  # Will learn what this represents (e.g., low volatility)


class HighLevelModule(HierarchicalReasoningModule):
    """
    Strategic reasoning module for long-term market analysis
    
    Key characteristics:
    - Sees 100 candles for broader market context
    - Updates less frequently (every H_cycles)
    - Focuses on strategic decisions, risk management, regime detection
    - Provides context and constraints to L-module
    """
    
    def __init__(self, 
                 feature_dim: int,  # Number of features per candle
                 lookback_window: int,  # Number of candles to look back (100)
                 hidden_dim: int, 
                 num_layers: int, 
                 num_heads: int,
                 instruments_loader=None,  # InstrumentsLoader instance for dynamic embeddings
                 dropout: float = 0.1):
        
        # Input dimension is feature_dim per timestep (not flattened)
        # Base module expects [batch, seq, features] format
        input_dim = feature_dim
        
        super().__init__(input_dim, hidden_dim, num_layers, num_heads, dropout)
        
        self.feature_dim = feature_dim
        self.lookback_window = lookback_window
        
        # Initialize embeddings for instruments and timeframes
        self._init_embeddings(instruments_loader)
        
        # Strategic output heads
        self._init_output_heads(hidden_dim)
        
    def _init_embeddings(self, instruments_loader):
        """Initialize dynamic embeddings for instruments and timeframes"""
        if instruments_loader is not None:
            num_instruments, num_timeframes = instruments_loader.get_embedding_dimensions()
            
            # Create embeddings
            embedding_dim = min(64, self.hidden_dim // 4)  # Reasonable embedding size
            self.instrument_embedding = nn.Embedding(num_instruments, embedding_dim)
            self.timeframe_embedding = nn.Embedding(num_timeframes, embedding_dim)
            
            # Projection to integrate embeddings with hidden state
            self.market_context_projector = nn.Sequential(
                nn.Linear(2 * embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.use_embeddings = True
        else:
            self.use_embeddings = False
    
    def _init_output_heads(self, hidden_dim: int):
        """Initialize strategic output heads"""
        
        # Market regime classification (5 regimes - let model learn what they represent)
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, len(MarketRegime))
        )
        
        # Strategic risk parameters
        self.risk_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4),  # position_size_factor, risk_multiplier, max_exposure, volatility_adjustment
            nn.Sigmoid()  # Bound to [0, 1]
        )
        
        # Overall strategic signal strength and direction
        self.strategic_signal = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # strength, direction
        )
        self.signal_strength_activation = nn.Sigmoid()  # Strength [0, 1]
        self.signal_direction_activation = nn.Tanh()    # Direction [-1, 1]
        
        # Long-term trend analysis
        self.trend_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # trend_strength, trend_consistency, trend_momentum
        )
        
    def forward(self, market_data: torch.Tensor, context: Optional[torch.Tensor] = None, 
                instrument_id: Optional[torch.Tensor] = None, timeframe_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for strategic reasoning
        
        Args:
            market_data: [batch_size, lookback_window, feature_dim] - sequence market data
            context: Optional context from previous H-module state
            
        Returns:
            hidden_state: Updated hidden state for this module
            outputs: Dictionary of strategic outputs
        """
        
        # Market data is already in sequence format
        sequence_data = market_data
        
        # Base forward pass through transformer layers
        hidden = super().forward(sequence_data, context)
        
        # Extract strategic representation (use mean pooling over sequence)
        if len(hidden.shape) == 3:  # [batch, seq_len, hidden]
            strategic_repr = hidden.mean(dim=1)  # Pool over sequence dimension
        else:
            strategic_repr = hidden
            
        # Integrate instrument and timeframe embeddings if available
        if self.use_embeddings and instrument_id is not None and timeframe_id is not None:
            batch_size = strategic_repr.size(0)
            
            # Get embeddings
            instrument_emb = self.instrument_embedding(instrument_id)  # [batch_size, embed_dim]
            timeframe_emb = self.timeframe_embedding(timeframe_id)     # [batch_size, embed_dim]
            
            # Combine embeddings
            market_context = torch.cat([instrument_emb, timeframe_emb], dim=-1)  # [batch_size, 2*embed_dim]
            market_context = self.market_context_projector(market_context)       # [batch_size, hidden_dim]
            
            # Add market context to strategic representation
            strategic_repr = strategic_repr + market_context
            
        # Generate strategic outputs
        regime_logits = self.regime_classifier(strategic_repr)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        risk_params = self.risk_estimator(strategic_repr)
        
        strategic_signals = self.strategic_signal(strategic_repr)
        signal_strength = self.signal_strength_activation(strategic_signals[:, 0:1])
        signal_direction = self.signal_direction_activation(strategic_signals[:, 1:2])
        
        trend_analysis = torch.sigmoid(self.trend_analyzer(strategic_repr))
        
        # Package outputs
        outputs = {
            'regime_logits': regime_logits,
            'regime_probabilities': regime_probs,
            'risk_parameters': risk_params,
            'signal_strength': signal_strength,
            'signal_direction': signal_direction,
            'trend_strength': trend_analysis[:, 0:1],
            'trend_consistency': trend_analysis[:, 1:2], 
            'trend_momentum': trend_analysis[:, 2:3],
            'strategic_hidden': strategic_repr
        }
        
        return hidden, outputs
    
    def get_strategic_context(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Create strategic context vector to pass to L-module
        This summarizes the strategic state for tactical use
        """
        context_components = [
            outputs['strategic_hidden'],
            outputs['regime_probabilities'],
            outputs['risk_parameters'],
            outputs['signal_strength'],
            outputs['signal_direction'],
            outputs['trend_strength'],
            outputs['trend_consistency'],
            outputs['trend_momentum']
        ]
        
        # Concatenate all strategic information
        strategic_context = torch.cat(context_components, dim=-1)
        
        # Project to same dimension as hidden for easy integration
        if not hasattr(self, 'context_projector'):
            self.context_projector = nn.Linear(
                strategic_context.size(-1), 
                self.hidden_dim
            ).to(strategic_context.device)
            
        return self.context_projector(strategic_context)
    
    def interpret_regime(self, regime_probs: torch.Tensor) -> Dict[str, any]:
        """
        Interpret regime probabilities (for logging/analysis)
        Note: Actual regime meanings will emerge from training
        """
        dominant_regime_idx = torch.argmax(regime_probs, dim=-1)
        confidence = torch.max(regime_probs, dim=-1)[0]
        
        return {
            'dominant_regime_idx': dominant_regime_idx.item(),
            'regime_confidence': confidence.item(),
            # GPU-optimized: only transfer to CPU when necessary
            'regime_distribution': regime_probs.detach().cpu().numpy().tolist() if regime_probs.device.type == 'cuda' else regime_probs.detach().numpy().tolist(),
            'regime_enum': MarketRegime(dominant_regime_idx.item())
        }