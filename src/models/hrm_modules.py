"""
HRM Dual Modules - H-Module and L-Module

Implements the core hierarchical reasoning modules:
- HighLevelModule (H): Strategic reasoning with slow updates
- LowLevelModule (L): Tactical execution with fast updates

Based on the HRM research paper's hierarchical convergence mechanism.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .hrm_components import TransformerBlock, RMSNorm


class HighLevelModule(nn.Module):
    """H-module: Strategic reasoning with recurrent Transformer architecture"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_layers = config.get('num_layers', 4)
        self.n_heads = config.get('n_heads', 8)
        self.ff_dim = config.get('ff_dim', 2048)
        self.dropout = config.get('dropout', 0.1)
        
        # Store config for serialization
        self.config = config
        
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
        
        # Store config for serialization
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
        
        # Dynamic projection layers (created as needed)
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


class HierarchicalConvergenceEngine:
    """
    Orchestrates the hierarchical convergence process between H and L modules.
    
    Implements the core algorithm from the paper:
    - N cycles of T timesteps each
    - L-module converges within each cycle
    - H-module updates once per cycle using converged L-state
    - L-module reset between cycles for fresh convergence
    """
    
    def __init__(self, N: int = 3, T: int = 5, convergence_threshold: float = 1e-6):
        self.N = N  # High-level cycles
        self.T = T  # Low-level timesteps per cycle
        self.convergence_threshold = convergence_threshold
        
    def reset_l_module(self, z_l: torch.Tensor, reset_factor: float = 0.3) -> torch.Tensor:
        """Reset L-module for fresh convergence in next cycle"""
        noise = torch.randn_like(z_l) * 0.1
        return reset_factor * z_l + (1 - reset_factor) * noise
    
    def execute_hierarchical_reasoning(
        self, 
        h_module: HighLevelModule,
        l_module: LowLevelModule, 
        z_h_init: torch.Tensor,
        z_l_init: torch.Tensor,
        x_embedded: torch.Tensor,
        l_to_h_projection: nn.Module
    ):
        """
        Execute the complete hierarchical convergence process.
        
        Returns:
            z_h_final: Final H-module state
            z_l_final: Final L-module state
            convergence_info: Detailed convergence statistics
        """
        z_h, z_l = z_h_init, z_l_init
        convergence_info = {
            'cycles': [],
            'total_l_steps': 0,
            'h_updates': 0
        }
        
        for cycle in range(self.N):
            cycle_info = {
                'cycle': cycle,
                'l_convergence_residuals': [],
                'converged': False
            }
            
            # L-module converges within cycle (T timesteps)
            for t in range(self.T):
                z_l_prev = z_l.clone()
                z_l = l_module(z_l, z_h, x_embedded)
                
                # Track convergence
                residual = torch.norm(z_l - z_l_prev, dim=-1).mean().item()
                cycle_info['l_convergence_residuals'].append(residual)
                convergence_info['total_l_steps'] += 1
                
                # Check for premature convergence
                if residual < self.convergence_threshold:
                    cycle_info['converged'] = True
                    break
            
            # Project L-module state to H-module dimension for update
            z_l_projected = l_to_h_projection(z_l)
            
            # H-module updates once per cycle using converged L-state
            z_h = h_module(z_h, z_l_projected)
            convergence_info['h_updates'] += 1
            
            # Store cycle information
            convergence_info['cycles'].append(cycle_info)
            
            # Reset L-module for next cycle's fresh convergence
            if cycle < self.N - 1:
                z_l = self.reset_l_module(z_l)
        
        return z_h, z_l, convergence_info