"""
Hierarchical Reasoning Model (HRM) for Trading Agents

Brain-inspired dual-module architecture with hierarchical convergence mechanism.
Based on cutting-edge research achieving 27M parameter efficiency with unlimited
computational depth for complex reasoning tasks.

Architecture:
- H-module (High-Level): Strategic reasoning, abstract planning
- L-module (Low-Level): Tactical execution, detailed computations
- Hierarchical Convergence: N cycles × T timesteps for deep reasoning
- Unified Output Heads: Policy, Quantity, Value, Q-learning compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Dict, Tuple, Optional, Any, Union
from src.utils.config_loader import ConfigLoader
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * x.size(-1) ** -0.5
        return self.scale * x / (norm + self.eps)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Encoding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=x.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embedding"""
    # Ensure cos and sin match the input dimensions
    if cos.size(-1) != x.size(-1) // 2:
        # Truncate or pad cos/sin to match input dimensions
        target_dim = x.size(-1) // 2
        if cos.size(-1) > target_dim:
            cos = cos[..., :target_dim]
            sin = sin[..., :target_dim]
        else:
            # Pad with zeros if needed
            pad_size = target_dim - cos.size(-1)
            cos = torch.cat([cos, torch.zeros(*cos.shape[:-1], pad_size, device=cos.device)], dim=-1)
            sin = torch.cat([sin, torch.zeros(*sin.shape[:-1], pad_size, device=sin.device)], dim=-1)
    
    x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.sigmoid(self.gate(x)) * self.linear(x)


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with modern improvements"""
    def __init__(self, dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Multi-head attention
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed-forward network with GLU
        self.ff_glu = GLU(dim, ff_dim)
        self.ff_out = nn.Linear(ff_dim, dim, bias=False)
        self.ff_dropout = nn.Dropout(dropout)
        
        # Layer normalization (Post-Norm architecture)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Rotary positional embedding
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention with RoPE
        residual = x
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Apply rotary positional embedding
        cos, sin = self.rope(x, seq_len)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.attn_out(out)
        x = self.norm1(residual + out)
        
        # Feed-forward with GLU
        residual = x
        ff = self.ff_glu(x)
        ff = self.ff_out(ff)
        ff = self.ff_dropout(ff)
        x = self.norm2(residual + ff)
        
        return x


class InputEmbeddingNetwork(nn.Module):
    """Input embedding network for market data preprocessing"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = config.get('input_dim', 256)
        self.embedding_dim = config.get('embedding_dim', 512)
        
        self.linear_projection = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
        self.norm = RMSNorm(self.embedding_dim)
        self.dropout = nn.Dropout(config.get('dropout', 0.1))

    def forward(self, x):
        """Process market features into embedded representation"""
        x = self.linear_projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class HighLevelModule(nn.Module):
    """H-module: Strategic reasoning with recurrent Transformer architecture"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_layers = config.get('num_layers', 4)
        self.n_heads = config.get('n_heads', 8)
        self.ff_dim = config.get('ff_dim', 2048)
        self.dropout = config.get('dropout', 0.1)
        
        # Transformer blocks for strategic reasoning
        self.layers = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Recurrent update mechanism
        self.recurrent_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, z_h_prev, z_l_converged):
        """Strategic update based on converged L-module state"""
        # Combine previous H-state with converged L-state
        combined = torch.cat([z_h_prev, z_l_converged], dim=-1)
        
        # Gated update mechanism
        update_gate = torch.sigmoid(self.recurrent_gate(combined))
        candidate = z_h_prev  # Maintain strategic stability
        
        # Apply Transformer layers for strategic reasoning
        x = candidate.unsqueeze(1)  # Add sequence dimension
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Gated integration
        z_h_new = update_gate * x + (1 - update_gate) * z_h_prev
        z_h_new = self.norm(z_h_new)
        
        return z_h_new


class LowLevelModule(nn.Module):
    """L-module: Tactical execution with recurrent Transformer architecture"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.n_heads = config.get('n_heads', 8)
        self.ff_dim = config.get('ff_dim', 1024)
        self.dropout = config.get('dropout', 0.1)
        
        # Store config for proper serialization
        self.config = config
        
        # Transformer blocks for tactical execution
        self.layers = nn.ModuleList([
            TransformerBlock(self.hidden_dim, self.n_heads, self.ff_dim, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Integration networks
        self.h_integration = nn.Linear(self.hidden_dim, self.hidden_dim)  # H-module guidance
        self.x_integration = nn.Linear(self.hidden_dim, self.hidden_dim)  # Input context
        self.recurrent_update = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = RMSNorm(self.hidden_dim)
        
        # Pre-initialize projection layers to avoid dynamic creation
        # These will be properly sized if needed
        self.h_projection = None
        self.x_projection = None

    def _ensure_projection_layers(self, h_dim, x_dim):
        """Ensure projection layers exist with correct dimensions"""
        if h_dim != self.hidden_dim and self.h_projection is None:
            self.h_projection = nn.Linear(h_dim, self.hidden_dim)
            
        if x_dim != self.hidden_dim and self.x_projection is None:
            self.x_projection = nn.Linear(x_dim, self.hidden_dim)

    def forward(self, z_l_prev, z_h, x_embedded):
        """Tactical update guided by H-module and input features"""
        # Ensure projection layers exist before forward pass
        self._ensure_projection_layers(z_h.size(-1), x_embedded.size(-1))
        
        # Project H-module state to L-module dimension if needed
        if z_h.size(-1) != self.hidden_dim and self.h_projection is not None:
            h_projected = self.h_projection(z_h)
        else:
            h_projected = z_h
        
        # Project input embedding to L-module dimension if needed
        if x_embedded.size(-1) != self.hidden_dim and self.x_projection is not None:
            x_projected = self.x_projection(x_embedded)
        else:
            x_projected = x_embedded
        
        # Integrate guidance from H-module and input features
        h_guidance = self.h_integration(h_projected)
        x_context = self.x_integration(x_projected)
        
        # Combine with previous L-state
        combined = z_l_prev + h_guidance + x_context
        
        # Apply Transformer layers for tactical reasoning
        x = combined.unsqueeze(1)  # Add sequence dimension
        for layer in self.layers:
            x = layer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        
        # Recurrent update
        z_l_new = self.recurrent_update(x)
        z_l_new = self.norm(z_l_new)
        
        return z_l_new


class PolicyHead(nn.Module):
    """Policy head for discrete action prediction"""
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, action_dim, bias=False)

    def forward(self, z_h):
        return self.fc(z_h)


class QuantityHead(nn.Module):
    """Quantity head for continuous position sizing"""
    def __init__(self, input_dim: int, quantity_min: float = 1.0, quantity_max: float = 100000.0):
        super().__init__()
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
        self.fc = nn.Linear(input_dim, 1, bias=False)

    def forward(self, z_h):
        return self.fc(z_h).squeeze(-1)


class QHead(nn.Module):
    """Q-head for ACT halting mechanism (future Epic 5)"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2, bias=False)  # halt, continue

    def forward(self, z_h):
        return torch.sigmoid(self.fc(z_h))


class InstrumentEmbedding(nn.Module):
    """Embedding layer for trading instruments"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        
    def forward(self, instrument_ids):
        return self.embedding(instrument_ids)


class TimeframeEmbedding(nn.Module):
    """Embedding layer for trading timeframes"""
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        
    def forward(self, timeframe_ids):
        return self.embedding(timeframe_ids)


class HierarchicalReasoningModel(nn.Module, BaseAgent):
    """
    Brain-inspired HRM with dual-module architecture for algorithmic trading.
    
    Implements hierarchical convergence mechanism with unlimited computational depth
    through N cycles of T timesteps each. Achieves 27M parameter efficiency with
    revolutionary performance on complex reasoning tasks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Load configuration
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.get_config()
        
        self.config = config
        self._setup_architecture_config()
        self._initialize_components()
        self._setup_convergence_parameters()
        
        logger.info(f"HRM initialized with {self.count_parameters()} parameters")

    def _setup_architecture_config(self):
        """Setup architecture configuration from settings.yaml"""
        # Extract HRM-specific config or create defaults
        hrm_config = self.config.get('hierarchical_reasoning_model', {})
        
        # H-module configuration
        self.h_config = hrm_config.get('h_module', {
            'hidden_dim': 512,
            'num_layers': 4,
            'n_heads': 8,
            'ff_dim': 2048,
            'dropout': 0.1
        })
        
        # L-module configuration
        self.l_config = hrm_config.get('l_module', {
            'hidden_dim': 256,
            'num_layers': 3,
            'n_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1
        })
        
        # Input embedding configuration
        self.input_config = hrm_config.get('input_embedding', {
            'input_dim': self.config.get('model', {}).get('observation_dim', 256),
            'embedding_dim': self.h_config['hidden_dim'],
            'dropout': 0.1
        })
        
        # Embedding configurations
        embedding_config = hrm_config.get('embeddings', {})
        self.instrument_embedding_config = {
            'vocab_size': embedding_config.get('max_instruments', 1000),
            'embedding_dim': embedding_config.get('instrument_dim', 64)
        }
        self.timeframe_embedding_config = {
            'vocab_size': embedding_config.get('max_timeframes', 10),
            'embedding_dim': embedding_config.get('timeframe_dim', 32)
        }
        
        # Output head configuration
        output_config = hrm_config.get('output_heads', {})
        model_config = self.config.get('model', {})
        self.output_config = {
            'action_dim': output_config.get('action_dim', model_config.get('action_dim_discrete', 5)),
            'quantity_min': output_config.get('quantity_min', 1.0),
            'quantity_max': output_config.get('quantity_max', 100000.0),
            'value_estimation': output_config.get('value_estimation', True),
            'q_learning_prep': output_config.get('q_learning_prep', True)
        }

    def _initialize_components(self):
        """Initialize all HRM components"""
        # Input embedding network
        self.input_network = InputEmbeddingNetwork(self.input_config)
        
        # Instrument and timeframe embeddings
        self.instrument_embedding = InstrumentEmbedding(
            self.instrument_embedding_config['vocab_size'],
            self.instrument_embedding_config['embedding_dim']
        )
        self.timeframe_embedding = TimeframeEmbedding(
            self.timeframe_embedding_config['vocab_size'],
            self.timeframe_embedding_config['embedding_dim']
        )
        
        # Embedding projection to match H-module dimension
        embedding_total_dim = (self.instrument_embedding_config['embedding_dim'] + 
                             self.timeframe_embedding_config['embedding_dim'])
        self.embedding_projection = nn.Linear(
            self.input_config['embedding_dim'] + embedding_total_dim,
            self.h_config['hidden_dim'],
            bias=False
        )
        
        # Dual recurrent modules
        self.h_module = HighLevelModule(self.h_config)
        self.l_module = LowLevelModule(self.l_config)
        
        # L-module dimension projection to match H-module
        if self.l_config['hidden_dim'] != self.h_config['hidden_dim']:
            self.l_to_h_projection = nn.Linear(
                self.l_config['hidden_dim'], 
                self.h_config['hidden_dim'], 
                bias=False
            )
        else:
            self.l_to_h_projection = nn.Identity()
        
        # Output heads
        self.policy_head = PolicyHead(self.h_config['hidden_dim'], self.output_config['action_dim'])
        self.quantity_head = QuantityHead(
            self.h_config['hidden_dim'], 
            self.output_config['quantity_min'], 
            self.output_config['quantity_max']
        )
        
        if self.output_config['value_estimation']:
            self.value_head = ValueHead(self.h_config['hidden_dim'])
        
        if self.output_config['q_learning_prep']:
            self.q_head = QHead(self.h_config['hidden_dim'])

    def _setup_convergence_parameters(self):
        """Setup hierarchical convergence parameters"""
        hierarchical_config = self.config.get('hierarchical_reasoning_model', {}).get('hierarchical', {})
        
        self.N = hierarchical_config.get('N_cycles', 3)  # High-level cycles
        self.T = hierarchical_config.get('T_timesteps', 5)  # Low-level timesteps per cycle
        self.convergence_threshold = hierarchical_config.get('convergence_threshold', 1e-6)
        self.max_convergence_steps = hierarchical_config.get('max_convergence_steps', 100)

    def initialize_states(self, batch_size: int, device: torch.device, z_init: Optional[Tuple] = None):
        """Initialize hidden states for H-module and L-module"""
        if z_init is not None:
            z_h_init, z_l_init = z_init
            return z_h_init.to(device), z_l_init.to(device)
        
        # Truncated normal initialization as per paper
        z_h = torch.randn(batch_size, self.h_config['hidden_dim'], device=device)
        z_h = torch.clamp(z_h, -2.0, 2.0)  # Truncation at 2 standard deviations
        
        z_l = torch.randn(batch_size, self.l_config['hidden_dim'], device=device)
        z_l = torch.clamp(z_l, -2.0, 2.0)
        
        return z_h, z_l

    def reset_l_module(self, z_l: torch.Tensor) -> torch.Tensor:
        """Reset L-module for fresh convergence in next cycle"""
        # Partial reset: maintain some information while encouraging fresh convergence
        reset_factor = 0.3  # Configurable reset strength
        noise = torch.randn_like(z_l) * 0.1
        return reset_factor * z_l + (1 - reset_factor) * noise

    def forward(self, x: torch.Tensor, instrument_ids: Optional[torch.Tensor] = None, 
                timeframe_ids: Optional[torch.Tensor] = None, z_init: Optional[Tuple] = None):
        """
        Execute N cycles of T timesteps each for hierarchical reasoning
        
        Args:
            x: Market features [batch_size, feature_dim]
            instrument_ids: Instrument identifiers [batch_size]
            timeframe_ids: Timeframe identifiers [batch_size]
            z_init: Initial hidden states (z_H, z_L)
            
        Returns:
            outputs: Dictionary with action_type, quantity, value, q_values
            final_states: (z_H, z_L) for potential continuation
        """
        try:
            # Input validation
            if x is None or x.numel() == 0:
                raise ValueError("Input tensor x cannot be None or empty")
            
            if x.dim() != 2:
                raise ValueError(f"Input tensor x must be 2D [batch_size, feature_dim], got {x.dim()}D")
            
            batch_size = x.size(0)
            device = x.device
            
            # Validate input dimension
            expected_input_dim = self.input_config['input_dim']
            if x.size(1) != expected_input_dim:
                logger.warning(f"Input dimension mismatch: expected {expected_input_dim}, got {x.size(1)}. "
                             f"Attempting to adapt...")
                # Adaptive input handling
                if x.size(1) < expected_input_dim:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, expected_input_dim - x.size(1), device=device)
                    x = torch.cat([x, padding], dim=1)
                else:
                    # Truncate
                    x = x[:, :expected_input_dim]
                logger.info(f"Input adapted to dimension {x.size(1)}")
            
            # Process input features with error handling
            try:
                x_embedded = self.input_network(x)
            except Exception as e:
                logger.error(f"Input embedding failed: {e}")
                # Fallback: simple linear projection
                if not hasattr(self, '_fallback_input_proj'):
                    self._fallback_input_proj = nn.Linear(
                        x.size(1), self.h_config['hidden_dim'], device=device
                    )
                x_embedded = self._fallback_input_proj(x)
                logger.warning("Using fallback input projection")
            
            # Add instrument and timeframe embeddings if provided
            if instrument_ids is not None and timeframe_ids is not None:
                try:
                    # Validate embedding IDs
                    if instrument_ids.max() >= self.instrument_embedding_config['vocab_size']:
                        logger.warning(f"Instrument ID {instrument_ids.max()} exceeds vocab size "
                                     f"{self.instrument_embedding_config['vocab_size']}. Clamping.")
                        instrument_ids = torch.clamp(instrument_ids, 0, 
                                                   self.instrument_embedding_config['vocab_size'] - 1)
                    
                    if timeframe_ids.max() >= self.timeframe_embedding_config['vocab_size']:
                        logger.warning(f"Timeframe ID {timeframe_ids.max()} exceeds vocab size "
                                     f"{self.timeframe_embedding_config['vocab_size']}. Clamping.")
                        timeframe_ids = torch.clamp(timeframe_ids, 0, 
                                                  self.timeframe_embedding_config['vocab_size'] - 1)
                    
                    instrument_emb = self.instrument_embedding(instrument_ids)
                    timeframe_emb = self.timeframe_embedding(timeframe_ids)
                    
                    # Concatenate all embeddings
                    combined_embedded = torch.cat([x_embedded, instrument_emb, timeframe_emb], dim=-1)
                    x_embedded = self.embedding_projection(combined_embedded)
                    
                    logger.debug(f"Applied instrument/timeframe embeddings: "
                               f"instruments={instrument_ids.tolist()}, timeframes={timeframe_ids.tolist()}")
                
                except Exception as e:
                    logger.error(f"Embedding processing failed: {e}. Continuing without embeddings.")
                    # Continue with x_embedded as is
            
            # Initialize hidden states with error handling
            try:
                z_H, z_L = self.initialize_states(batch_size, device, z_init)
            except Exception as e:
                logger.error(f"State initialization failed: {e}")
                # Fallback initialization
                z_H = torch.zeros(batch_size, self.h_config['hidden_dim'], device=device)
                z_L = torch.zeros(batch_size, self.l_config['hidden_dim'], device=device)
                logger.warning("Using fallback zero state initialization")
            
            # Hierarchical convergence over N cycles with monitoring
            convergence_errors = 0
            max_convergence_errors = 3  # Allow some failures before fallback
            
            for cycle in range(self.N):
                try:
                    # L-module converges within cycle (T timesteps)
                    for t in range(self.T):
                        try:
                            z_L_new = self.l_module(z_L, z_H, x_embedded)
                            
                            # Check for NaN/Inf values
                            if torch.isnan(z_L_new).any() or torch.isinf(z_L_new).any():
                                logger.warning(f"NaN/Inf detected in L-module at cycle {cycle}, timestep {t}")
                                # Use previous state or reset
                                if t > 0:
                                    continue  # Keep previous z_L
                                else:
                                    z_L_new = self.initialize_states(batch_size, device)[1]
                            
                            z_L = z_L_new
                            
                        except Exception as e:
                            logger.error(f"L-module forward failed at cycle {cycle}, timestep {t}: {e}")
                            convergence_errors += 1
                            if convergence_errors > max_convergence_errors:
                                logger.error("Too many convergence errors, using fallback")
                                break
                            continue
                    
                    # Project L-module state to H-module dimension for update
                    try:
                        z_L_projected = self.l_to_h_projection(z_L)
                    except Exception as e:
                        logger.error(f"L-to-H projection failed: {e}")
                        z_L_projected = z_L  # Fallback if dimensions match
                    
                    # H-module updates once per cycle using converged L-state
                    try:
                        z_H_new = self.h_module(z_H, z_L_projected)
                        
                        # Check for NaN/Inf values
                        if torch.isnan(z_H_new).any() or torch.isinf(z_H_new).any():
                            logger.warning(f"NaN/Inf detected in H-module at cycle {cycle}")
                            # Keep previous H-state
                        else:
                            z_H = z_H_new
                            
                    except Exception as e:
                        logger.error(f"H-module forward failed at cycle {cycle}: {e}")
                        convergence_errors += 1
                        if convergence_errors > max_convergence_errors:
                            break
                    
                    # Reset L-module for next cycle's fresh convergence
                    if cycle < self.N - 1:
                        try:
                            z_L = self.reset_l_module(z_L)
                        except Exception as e:
                            logger.error(f"L-module reset failed: {e}")
                            # Fallback: partial noise injection
                            z_L = z_L + torch.randn_like(z_L) * 0.01
                
                except Exception as e:
                    logger.error(f"Hierarchical convergence failed at cycle {cycle}: {e}")
                    convergence_errors += 1
                    if convergence_errors > max_convergence_errors:
                        logger.error("Maximum convergence errors reached, using current states")
                        break
            
            # Generate outputs from final H-module state with error handling
            outputs = {}
            
            # Policy head (required)
            try:
                outputs['action_type'] = self.policy_head(z_H)
            except Exception as e:
                logger.error(f"Policy head failed: {e}")
                # Fallback: uniform random policy
                outputs['action_type'] = torch.zeros(batch_size, self.output_config['action_dim'], device=device)
                logger.warning("Using fallback uniform policy")
            
            # Quantity head (required)
            try:
                outputs['quantity'] = self.quantity_head(z_H)
            except Exception as e:
                logger.error(f"Quantity head failed: {e}")
                # Fallback: minimum quantity
                outputs['quantity'] = torch.full((batch_size,), self.output_config['quantity_min'], device=device)
                logger.warning("Using fallback minimum quantity")
            
            # Optional heads
            if hasattr(self, 'value_head'):
                try:
                    outputs['value'] = self.value_head(z_H)
                except Exception as e:
                    logger.error(f"Value head failed: {e}")
                    outputs['value'] = torch.zeros(batch_size, device=device)
            
            if hasattr(self, 'q_head'):
                try:
                    outputs['q_values'] = self.q_head(z_H)
                except Exception as e:
                    logger.error(f"Q-head failed: {e}")
                    outputs['q_values'] = torch.zeros(batch_size, 2, device=device)  # halt, continue
            
            # Log convergence statistics
            if convergence_errors > 0:
                logger.warning(f"Forward pass completed with {convergence_errors} convergence errors")
            
            return outputs, (z_H, z_L)
        
        except Exception as e:
            logger.error(f"Critical error in HRM forward pass: {e}")
            # Emergency fallback: return safe default values
            batch_size = x.size(0) if x is not None else 1
            device = x.device if x is not None else torch.device('cpu')
            
            fallback_outputs = {
                'action_type': torch.zeros(batch_size, self.output_config['action_dim'], device=device),
                'quantity': torch.full((batch_size,), self.output_config['quantity_min'], device=device)
            }
            
            if self.output_config['value_estimation']:
                fallback_outputs['value'] = torch.zeros(batch_size, device=device)
            
            if self.output_config['q_learning_prep']:
                fallback_outputs['q_values'] = torch.zeros(batch_size, 2, device=device)
            
            # Fallback states
            fallback_z_H = torch.zeros(batch_size, self.h_config['hidden_dim'], device=device)
            fallback_z_L = torch.zeros(batch_size, self.l_config['hidden_dim'], device=device)
            
            logger.error("Using emergency fallback outputs")
            return fallback_outputs, (fallback_z_H, fallback_z_L)

    def act(self, observation: torch.Tensor, instrument_id: Optional[int] = None, 
            timeframe_id: Optional[int] = None) -> Tuple[int, float]:
        """
        Generate trading action from observation with comprehensive error handling
        
        Args:
            observation: Market features
            instrument_id: Trading instrument identifier
            timeframe_id: Trading timeframe identifier
            
        Returns:
            action_type: Discrete action (0-4)
            quantity: Continuous quantity
        """
        try:
            # Input validation
            if observation is None:
                logger.error("Observation is None, using default HOLD action")
                return 2, self.output_config['quantity_min']  # HOLD action, minimum quantity
            
            if not isinstance(observation, torch.Tensor):
                logger.error(f"Observation must be torch.Tensor, got {type(observation)}")
                return 2, self.output_config['quantity_min']
            
            if observation.numel() == 0:
                logger.error("Observation tensor is empty")
                return 2, self.output_config['quantity_min']
            
            self.eval()
            with torch.no_grad():
                # Ensure proper tensor dimensions
                if observation.dim() == 1:
                    observation = observation.unsqueeze(0)
                elif observation.dim() > 2:
                    logger.warning(f"Observation has {observation.dim()} dimensions, reshaping to 2D")
                    observation = observation.view(1, -1)
                
                # Validate instrument and timeframe IDs
                instrument_ids = None
                timeframe_ids = None
                
                if instrument_id is not None:
                    if instrument_id < 0 or instrument_id >= self.instrument_embedding_config['vocab_size']:
                        logger.warning(f"Invalid instrument_id {instrument_id}, clamping to valid range")
                        instrument_id = max(0, min(instrument_id, self.instrument_embedding_config['vocab_size'] - 1))
                    instrument_ids = torch.tensor([instrument_id], device=observation.device)
                
                if timeframe_id is not None:
                    if timeframe_id < 0 or timeframe_id >= self.timeframe_embedding_config['vocab_size']:
                        logger.warning(f"Invalid timeframe_id {timeframe_id}, clamping to valid range")
                        timeframe_id = max(0, min(timeframe_id, self.timeframe_embedding_config['vocab_size'] - 1))
                    timeframe_ids = torch.tensor([timeframe_id], device=observation.device)
                
                # Forward pass with error handling
                try:
                    outputs, _ = self.forward(observation, instrument_ids, timeframe_ids)
                except Exception as e:
                    logger.error(f"Forward pass failed in act(): {e}")
                    return 2, self.output_config['quantity_min']  # Safe fallback
                
                # Action sampling with error handling
                try:
                    action_logits = outputs['action_type']
                    
                    # Check for valid logits
                    if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                        logger.warning("Invalid action logits detected, using uniform distribution")
                        action_probs = torch.ones_like(action_logits) / action_logits.size(-1)
                    else:
                        action_probs = F.softmax(action_logits, dim=-1)
                    
                    # Ensure probabilities are valid
                    if torch.isnan(action_probs).any() or (action_probs < 0).any():
                        logger.warning("Invalid action probabilities, using HOLD action")
                        action_type = 4  # HOLD (Fixed: was incorrectly set to 2=CLOSE_LONG)
                    else:
                        # Safe sampling
                        try:
                            action_type = torch.multinomial(action_probs, 1).item()
                        except Exception as e:
                            logger.warning(f"Sampling failed: {e}, using argmax")
                            action_type = torch.argmax(action_probs, dim=-1).item()
                    
                    # Validate action type
                    if action_type < 0 or action_type >= self.output_config['action_dim']:
                        logger.warning(f"Invalid action_type {action_type}, using HOLD")
                        action_type = 4  # HOLD (Fixed: was incorrectly set to 2=CLOSE_LONG)
                
                except Exception as e:
                    logger.error(f"Action sampling failed: {e}")
                    action_type = 4  # Safe fallback to HOLD (Fixed: was incorrectly set to 2=CLOSE_LONG)
                
                # Quantity extraction with error handling
                try:
                    quantity = outputs['quantity'].item()
                    
                    # Validate quantity
                    if math.isnan(quantity) or math.isinf(quantity):
                        logger.warning("Invalid quantity detected, using minimum")
                        quantity = self.output_config['quantity_min']
                    elif quantity < self.output_config['quantity_min']:
                        logger.debug(f"Quantity {quantity} below minimum, clamping to {self.output_config['quantity_min']}")
                        quantity = self.output_config['quantity_min']
                    elif quantity > self.output_config['quantity_max']:
                        logger.debug(f"Quantity {quantity} above maximum, clamping to {self.output_config['quantity_max']}")
                        quantity = self.output_config['quantity_max']
                
                except Exception as e:
                    logger.error(f"Quantity extraction failed: {e}")
                    quantity = self.output_config['quantity_min']
                
                logger.debug(f"HRM action: type={action_type}, quantity={quantity:.2f}")
                return action_type, quantity
                
        except Exception as e:
            logger.error(f"Critical error in act(): {e}")
            # Emergency fallback
            return 2, self.output_config['quantity_min']  # HOLD action, minimum quantity

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_convergence_diagnostics(self, x: torch.Tensor, instrument_ids: Optional[torch.Tensor] = None, 
                                   timeframe_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Get detailed convergence diagnostics for debugging hierarchical reasoning
        
        Args:
            x: Market features [batch_size, feature_dim]
            instrument_ids: Instrument identifiers [batch_size]
            timeframe_ids: Timeframe identifiers [batch_size]
            
        Returns:
            Dictionary with convergence statistics and state trajectories
        """
        self.eval()
        diagnostics = {
            'cycles': [],
            'h_module_states': [],
            'l_module_states': [],
            'convergence_metrics': {},
            'output_statistics': {},
            'parameter_statistics': {}
        }
        
        with torch.no_grad():
            try:
                batch_size = x.size(0)
                device = x.device
                
                # Process input features
                x_embedded = self.input_network(x)
                
                # Add instrument and timeframe embeddings if provided
                if instrument_ids is not None and timeframe_ids is not None:
                    instrument_emb = self.instrument_embedding(instrument_ids)
                    timeframe_emb = self.timeframe_embedding(timeframe_ids)
                    combined_embedded = torch.cat([x_embedded, instrument_emb, timeframe_emb], dim=-1)
                    x_embedded = self.embedding_projection(combined_embedded)
                
                # Initialize hidden states
                z_H, z_L = self.initialize_states(batch_size, device)
                
                # Store initial states
                diagnostics['h_module_states'].append({
                    'cycle': -1,
                    'timestep': -1,
                    'state_norm': torch.norm(z_H, dim=-1).mean().item(),
                    'state_mean': z_H.mean().item(),
                    'state_std': z_H.std().item(),
                    'state_min': z_H.min().item(),
                    'state_max': z_H.max().item()
                })
                
                diagnostics['l_module_states'].append({
                    'cycle': -1,
                    'timestep': -1,
                    'state_norm': torch.norm(z_L, dim=-1).mean().item(),
                    'state_mean': z_L.mean().item(),
                    'state_std': z_L.std().item(),
                    'state_min': z_L.min().item(),
                    'state_max': z_L.max().item()
                })
                
                # Hierarchical convergence with detailed tracking
                for cycle in range(self.N):
                    cycle_diagnostics = {
                        'cycle': cycle,
                        'l_timesteps': [],
                        'h_update': {},
                        'convergence_achieved': False,
                        'convergence_residual': float('inf')
                    }
                    
                    # L-module convergence within cycle
                    l_convergence_residuals = []
                    for t in range(self.T):
                        z_L_prev = z_L.clone()
                        z_L = self.l_module(z_L, z_H, x_embedded)
                        
                        # Calculate convergence residual
                        residual = torch.norm(z_L - z_L_prev, dim=-1).mean().item()
                        l_convergence_residuals.append(residual)
                        
                        # Store L-module state
                        l_state_stats = {
                            'cycle': cycle,
                            'timestep': t,
                            'state_norm': torch.norm(z_L, dim=-1).mean().item(),
                            'state_mean': z_L.mean().item(),
                            'state_std': z_L.std().item(),
                            'state_min': z_L.min().item(),
                            'state_max': z_L.max().item(),
                            'convergence_residual': residual
                        }
                        diagnostics['l_module_states'].append(l_state_stats)
                        cycle_diagnostics['l_timesteps'].append(l_state_stats)
                    
                    # Check L-module convergence
                    final_residual = l_convergence_residuals[-1] if l_convergence_residuals else float('inf')
                    cycle_diagnostics['convergence_residual'] = final_residual
                    cycle_diagnostics['convergence_achieved'] = final_residual < self.convergence_threshold
                    cycle_diagnostics['residual_trend'] = l_convergence_residuals
                    
                    # H-module update
                    z_H_prev = z_H.clone()
                    z_L_projected = self.l_to_h_projection(z_L)
                    z_H = self.h_module(z_H, z_L_projected)
                    
                    # H-module statistics
                    h_residual = torch.norm(z_H - z_H_prev, dim=-1).mean().item()
                    h_state_stats = {
                        'cycle': cycle,
                        'timestep': -1,  # H-module updates once per cycle
                        'state_norm': torch.norm(z_H, dim=-1).mean().item(),
                        'state_mean': z_H.mean().item(),
                        'state_std': z_H.std().item(),
                        'state_min': z_H.min().item(),
                        'state_max': z_H.max().item(),
                        'update_residual': h_residual
                    }
                    diagnostics['h_module_states'].append(h_state_stats)
                    cycle_diagnostics['h_update'] = h_state_stats
                    
                    # Reset L-module for next cycle
                    if cycle < self.N - 1:
                        z_L_before_reset = z_L.clone()
                        z_L = self.reset_l_module(z_L)
                        reset_magnitude = torch.norm(z_L - z_L_before_reset, dim=-1).mean().item()
                        cycle_diagnostics['reset_magnitude'] = reset_magnitude
                    
                    diagnostics['cycles'].append(cycle_diagnostics)
                
                # Generate final outputs and analyze
                outputs = {
                    'action_type': self.policy_head(z_H),
                    'quantity': self.quantity_head(z_H)
                }
                
                if hasattr(self, 'value_head'):
                    outputs['value'] = self.value_head(z_H)
                
                if hasattr(self, 'q_head'):
                    outputs['q_values'] = self.q_head(z_H)
                
                # Output statistics
                diagnostics['output_statistics'] = {
                    'action_logits_mean': outputs['action_type'].mean().item(),
                    'action_logits_std': outputs['action_type'].std().item(),
                    'action_entropy': -torch.sum(F.softmax(outputs['action_type'], dim=-1) * 
                                                F.log_softmax(outputs['action_type'], dim=-1), dim=-1).mean().item(),
                    'quantity_mean': outputs['quantity'].mean().item(),
                    'quantity_std': outputs['quantity'].std().item(),
                    'quantity_min': outputs['quantity'].min().item(),
                    'quantity_max': outputs['quantity'].max().item()
                }
                
                if 'value' in outputs:
                    diagnostics['output_statistics']['value_mean'] = outputs['value'].mean().item()
                    diagnostics['output_statistics']['value_std'] = outputs['value'].std().item()
                
                # Convergence metrics
                h_norms = [state['state_norm'] for state in diagnostics['h_module_states']]
                l_norms = [state['state_norm'] for state in diagnostics['l_module_states']]
                
                diagnostics['convergence_metrics'] = {
                    'h_module_norm_trend': h_norms,
                    'l_module_norm_trend': l_norms,
                    'h_module_stability': max(h_norms) - min(h_norms) if h_norms else 0,
                    'l_module_stability': max(l_norms) - min(l_norms) if l_norms else 0,
                    'cycles_converged': sum(1 for cycle in diagnostics['cycles'] if cycle['convergence_achieved']),
                    'total_cycles': len(diagnostics['cycles']),
                    'final_h_norm': h_norms[-1] if h_norms else 0,
                    'final_l_norm': l_norms[-1] if l_norms else 0
                }
                
                # Parameter statistics
                diagnostics['parameter_statistics'] = {
                    'total_parameters': self.count_parameters(),
                    'h_module_parameters': sum(p.numel() for p in self.h_module.parameters()),
                    'l_module_parameters': sum(p.numel() for p in self.l_module.parameters()),
                    'embedding_parameters': (sum(p.numel() for p in self.instrument_embedding.parameters()) +
                                           sum(p.numel() for p in self.timeframe_embedding.parameters())),
                    'output_head_parameters': (sum(p.numel() for p in self.policy_head.parameters()) +
                                             sum(p.numel() for p in self.quantity_head.parameters()))
                }
                
                logger.info(f"HRM Convergence Diagnostics Summary:")
                logger.info(f"  Cycles converged: {diagnostics['convergence_metrics']['cycles_converged']}/{diagnostics['convergence_metrics']['total_cycles']}")
                logger.info(f"  Final H-module norm: {diagnostics['convergence_metrics']['final_h_norm']:.6f}")
                logger.info(f"  Final L-module norm: {diagnostics['convergence_metrics']['final_l_norm']:.6f}")
                logger.info(f"  Action entropy: {diagnostics['output_statistics']['action_entropy']:.6f}")
                
            except Exception as e:
                logger.error(f"Error in convergence diagnostics: {e}")
                diagnostics['error'] = str(e)
        
        return diagnostics

    def log_hierarchical_reasoning_step(self, cycle: int, timestep: int, z_h: torch.Tensor, 
                                       z_l: torch.Tensor, module_type: str, additional_info: Dict = None):
        """
        Log detailed information about hierarchical reasoning steps
        
        Args:
            cycle: Current cycle number
            timestep: Current timestep number  
            z_h: H-module state
            z_l: L-module state
            module_type: 'H' or 'L' to indicate which module
            additional_info: Additional logging information
        """
        if additional_info is None:
            additional_info = {}
        
        # Calculate state statistics
        h_norm = torch.norm(z_h, dim=-1).mean().item()
        l_norm = torch.norm(z_l, dim=-1).mean().item()
        
        log_msg = (f"HRM {module_type}-module | Cycle: {cycle}, Timestep: {timestep} | "
                  f"H-norm: {h_norm:.6f}, L-norm: {l_norm:.6f}")
        
        if additional_info:
            for key, value in additional_info.items():
                log_msg += f" | {key}: {value}"
        
        logger.debug(log_msg)

    def analyze_reasoning_patterns(self, market_data_batch: torch.Tensor, 
                                 num_samples: int = 10) -> Dict[str, Any]:
        """
        Analyze reasoning patterns across multiple market scenarios
        
        Args:
            market_data_batch: Batch of market data scenarios
            num_samples: Number of scenarios to analyze
            
        Returns:
            Analysis of reasoning patterns and decision consistency
        """
        analysis = {
            'decision_consistency': {},
            'reasoning_depth_usage': {},
            'convergence_patterns': {},
            'output_diversity': {}
        }
        
        self.eval()
        with torch.no_grad():
            sample_indices = torch.randint(0, market_data_batch.size(0), (num_samples,))
            sample_data = market_data_batch[sample_indices]
            
            decisions = []
            convergence_stats = []
            
            for i, data in enumerate(sample_data):
                # Get diagnostics for this sample
                diagnostics = self.get_convergence_diagnostics(data.unsqueeze(0))
                
                # Extract decision information
                outputs, _ = self.forward(data.unsqueeze(0))
                action_probs = F.softmax(outputs['action_type'], dim=-1)
                decision = {
                    'sample_idx': i,
                    'action_probs': action_probs.squeeze().cpu().numpy(),
                    'quantity': outputs['quantity'].item(),
                    'convergence_achieved': diagnostics['convergence_metrics']['cycles_converged'],
                    'reasoning_depth': diagnostics['convergence_metrics']['total_cycles']
                }
                decisions.append(decision)
                convergence_stats.append(diagnostics['convergence_metrics'])
            
            # Analyze decision consistency
            action_entropies = [decision['action_probs'].max() for decision in decisions]
            analysis['decision_consistency'] = {
                'mean_confidence': np.mean(action_entropies),
                'confidence_std': np.std(action_entropies),
                'high_confidence_ratio': sum(1 for conf in action_entropies if conf > 0.7) / len(action_entropies)
            }
            
            # Analyze reasoning depth usage
            convergence_rates = [stats['cycles_converged'] / stats['total_cycles'] for stats in convergence_stats]
            analysis['reasoning_depth_usage'] = {
                'mean_convergence_rate': np.mean(convergence_rates),
                'convergence_rate_std': np.std(convergence_rates),
                'full_convergence_ratio': sum(1 for rate in convergence_rates if rate == 1.0) / len(convergence_rates)
            }
            
            logger.info(f"Reasoning Pattern Analysis:")
            logger.info(f"  Mean decision confidence: {analysis['decision_consistency']['mean_confidence']:.3f}")
            logger.info(f"  Mean convergence rate: {analysis['reasoning_depth_usage']['mean_convergence_rate']:.3f}")
        
        return analysis

    def load_model(self, path: str):
        """Load model weights from checkpoint"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Handle dynamic layers that might not exist in the current model
            current_state = self.state_dict()
            
            # Filter out keys that don't exist in current model (e.g., dynamic projection layers)
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if key in current_state:
                    filtered_state_dict[key] = value
                else:
                    logger.warning(f"Skipping unexpected key in checkpoint: {key}")
            
            # Load the filtered state dict
            self.load_state_dict(filtered_state_dict, strict=False)
            
            logger.info(f"HRM model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load HRM model from {path}: {e}")
            raise

    def save_model(self, path: str):
        """Save model weights to checkpoint"""
        try:
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'parameter_count': self.count_parameters(),
                'architecture': 'HierarchicalReasoningModel'
            }
            torch.save(checkpoint, path)
            logger.info(f"HRM model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save HRM model to {path}: {e}")
            raise

    # BaseAgent interface methods
    def select_action(self, observation: Union[torch.Tensor, np.ndarray],
                      available_capital: float,
                      current_position_quantity: float,
                      current_price: float,
                      instrument: Any, # Will import Instrument later
                      return_probabilities: bool = False
                     ) -> Union[Tuple[int, float], Tuple[int, float, np.ndarray]]:
        """
        Select action using HRM hierarchical reasoning with action masking.
        
        Args:
            observation: Market observation tensor
            available_capital: Current available capital
            current_position_quantity: Current quantity of held position (positive for long, negative for short, 0 for none)
            current_price: Current market price
            instrument: The Instrument object for capital calculations
            
        Returns:
            Tuple of (action_type, quantity)
        """
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).unsqueeze(0)
        elif observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        # Extract instrument and timeframe IDs from instrument object
        instrument_id = None
        timeframe_id = None
        
        if instrument and hasattr(instrument, 'symbol'):
            # Extract instrument and timeframe from symbol (e.g., 'Bankex_180')
            symbol_parts = instrument.symbol.split('_')
            
            # Map instrument name to ID (simple hash for now)
            instrument_name = symbol_parts[0] if len(symbol_parts) > 0 else 'Unknown'
            instrument_id = abs(hash(instrument_name)) % self.instrument_embedding_config['vocab_size']
            
            # Map timeframe to ID if available
            if len(symbol_parts) > 1:
                try:
                    timeframe = int(symbol_parts[1])
                    # Map common timeframes to IDs: 1->0, 10->1, 15->2, 120->3, 180->4
                    timeframe_map = {1: 0, 10: 1, 15: 2, 120: 3, 180: 4}
                    timeframe_id = timeframe_map.get(timeframe, abs(hash(str(timeframe))) % self.timeframe_embedding_config['vocab_size'])
                except ValueError:
                    timeframe_id = 0  # Default timeframe
            else:
                timeframe_id = 0  # Default timeframe
        
        # Generate action through hierarchical reasoning
        with torch.no_grad():
            # Pass instrument and timeframe IDs to the model
            instrument_ids = torch.tensor([instrument_id], device=observation.device) if instrument_id is not None else None
            timeframe_ids = torch.tensor([timeframe_id], device=observation.device) if timeframe_id is not None else None
            
            outputs_dict, _ = self.forward(observation, instrument_ids, timeframe_ids)
            action_logits = outputs_dict['action_type']

            # --- Action Masking Logic ---
            # Initialize mask with all ones (no actions masked)
            action_mask = torch.ones_like(action_logits, dtype=torch.bool)

            # Action types: 0=BUY_LONG, 1=SELL_SHORT, 2=CLOSE_LONG, 3=CLOSE_SHORT, 4=HOLD

            # Import CapitalAwareQuantitySelector here to avoid circular imports
            from src.utils.capital_aware_quantity import CapitalAwareQuantitySelector
            selector = CapitalAwareQuantitySelector()
            
            # Calculate max_affordable_quantity for opening new positions
            max_affordable_quantity_buy_sell = selector.get_max_affordable_quantity(
                available_capital=available_capital,
                current_price=current_price,
                instrument=instrument
            )

            if max_affordable_quantity_buy_sell <= 0:
                # If no capital, mask out BUY_LONG (0) and SELL_SHORT (1)
                action_mask[0, 0] = False # BUY_LONG
                action_mask[0, 1] = False # SELL_SHORT

            # Masking based on current position (for CLOSE_LONG, CLOSE_SHORT)
            if current_position_quantity <= 0:
                # If not long (or short), mask out CLOSE_LONG (2)
                action_mask[0, 2] = False
            if current_position_quantity >= 0:
                # If not short (or long), mask out CLOSE_SHORT (3)
                action_mask[0, 3] = False
            
            # If agent is already in a position, it should not open a new one
            if current_position_quantity != 0:
                action_mask[0, 0] = False # Mask BUY_LONG
                action_mask[0, 1] = False # Mask SELL_SHORT

            # Apply mask: set logits of masked actions to a very small number
            # This effectively gives them a probability of zero after softmax
            action_logits = torch.where(action_mask, action_logits, torch.tensor(-1e9, device=action_logits.device))

            action_probs = F.softmax(action_logits, dim=-1)
            
            
            # If all actions are masked (should not happen with HOLD always available), default to HOLD
            if torch.all(~action_mask): # If all actions are False in the mask
                action_type = 4 # Default to HOLD
            else:
                # Use sampling instead of deterministic argmax for exploration
                action_type = torch.multinomial(action_probs, 1).item()
            
            # Get continuous quantity - let model predict freely within configured range
            # Use quantity_min and quantity_max from model config (set during initialization)
            quantity_min = getattr(self, 'quantity_min', 1.0)
            quantity_max = getattr(self, 'quantity_max', 100000.0)
            raw_quantity = torch.clamp(outputs_dict['quantity'], quantity_min, quantity_max).item()
            
            # Convert to actual quantity based on capital and action type
            if action_type in [0, 1]:  # BUY_LONG or SELL_SHORT (opening positions)
                if max_affordable_quantity_buy_sell > 0:
                    # Use capital-aware quantity for opening positions
                    quantity = selector.adjust_quantity_for_capital(
                        predicted_quantity=raw_quantity,
                        available_capital=available_capital,
                        current_price=current_price,
                        instrument=instrument
                    )
                else:
                    quantity = 0  # No capital available
            elif action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT (closing positions)
                # For closing, quantity should be the current position quantity
                quantity = abs(current_position_quantity)
            else:  # HOLD (action_type == 4)
                quantity = 0  # HOLD doesn't need quantity
            
        if return_probabilities:
            prob_values = action_probs.squeeze().cpu().numpy()
            return action_type, quantity, prob_values
        else:
            return action_type, quantity


    def act(self, observation: Union[torch.Tensor, np.ndarray]) -> Tuple[int, float]:
        """
        Alternative interface for live trading system.
        """
        return self.select_action(observation)

    def update(self) -> None:
        """
        Update HRM internal states. Used by some trainers.
        """
        # HRM updates happen implicitly during hierarchical convergence
        pass
