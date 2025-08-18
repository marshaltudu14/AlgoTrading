"""
HRM Core Components - Basic Building Blocks

Implements the fundamental neural components used throughout the HRM architecture:
- RMSNorm (Root Mean Square Layer Normalization)
- Rotary Positional Embedding (RoPE)
- Gated Linear Unit (GLU)
- Enhanced Transformer Block with modern improvements

Based on the HRM research paper's architectural specifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = None):
        super().__init__()
        from src.utils.config_loader import ConfigLoader
        if eps is None:
            config = ConfigLoader().get_config()
            eps = config.get('hierarchical_reasoning_model', {}).get('architecture', {}).get('rms_norm_eps', 1e-8)
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * x.size(-1) ** -0.5
        # Ensure self.eps is a scalar float, not a string
        eps = self.eps if isinstance(self.eps, (int, float)) else 1e-8
        return self.scale * x / (norm + eps)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Encoding (RoPE)"""
    def __init__(self, dim: int, max_seq_len: int = None, base_freq: float = None):
        super().__init__()
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader().get_config()
        arch_config = config.get('hierarchical_reasoning_model', {}).get('architecture', {})
        
        if max_seq_len is None:
            max_seq_len = arch_config.get('rope_max_seq_len', 2048)
        if base_freq is None:
            base_freq = arch_config.get('rope_base_freq', 10000)
            
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, dim, 2).float() / dim))
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
    # x has shape (batch, seq_len, heads, head_dim)
    # cos, sin have shape (seq_len, head_dim)
    
    # Split x into two halves along the last dimension
    x1, x2 = x[..., :x.size(-1)//2], x[..., x.size(-1)//2:]
    
    # Split cos and sin to match the dimensions of x1 and x2
    cos1, cos2 = cos[..., :x.size(-1)//2], cos[..., x.size(-1)//2:]
    sin1, sin2 = sin[..., :x.size(-1)//2], sin[..., x.size(-1)//2:]
    
    # Expand cos and sin to match x dimensions
    # cos and sin are (seq_len, head_dim//2)
    # We need to expand to (1, seq_len, 1, head_dim//2) to broadcast with (batch, seq_len, heads, head_dim//2)
    if cos1.dim() == 2:  # (seq_len, head_dim//2)
        cos1 = cos1.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        sin1 = sin1.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        cos2 = cos2.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
        sin2 = sin2.unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim//2)
    
    # Apply rotary embedding
    rotated = torch.cat([x1 * cos1 - x2 * sin1, x1 * sin2 + x2 * cos2], dim=-1)
    return rotated


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
            from src.utils.config_loader import ConfigLoader
            config = ConfigLoader().get_config()
            mask_value = config.get('hierarchical_reasoning_model', {}).get('architecture', {}).get('transformer_mask_value', -1e9)
            attn = attn.masked_fill(mask == 0, mask_value)
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


class ConvergenceTracker:
    """Track convergence metrics for hierarchical reasoning"""
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.residuals = []
        self.norms = []
        
    def update(self, z_prev: torch.Tensor, z_curr: torch.Tensor):
        """Update convergence metrics"""
        residual = torch.norm(z_curr - z_prev, dim=-1).mean().item()
        norm = torch.norm(z_curr, dim=-1).mean().item()
        
        self.residuals.append(residual)
        self.norms.append(norm)
        
        return residual < self.threshold
        
    def reset(self):
        """Reset tracking"""
        self.residuals.clear()
        self.norms.clear()
        
    def get_stats(self):
        """Get convergence statistics"""
        return {
            'final_residual': self.residuals[-1] if self.residuals else float('inf'),
            'mean_norm': sum(self.norms) / len(self.norms) if self.norms else 0,
            'converged': self.residuals[-1] < self.threshold if self.residuals else False,
            'convergence_steps': len(self.residuals)
        }