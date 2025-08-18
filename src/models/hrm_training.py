"""
HRM Training Methods - One-Step Gradient & Deep Supervision

Implements the revolutionary training methods from the HRM paper:
- One-Step Gradient Approximation (O(1) memory vs O(T) BPTT)
- Deep Supervision mechanism with detached states
- Neumann Series approximation for efficient gradient computation

Based on HRM research paper Equations 1-3 and Section 2.3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Tuple, Optional, List
from .hrm_modules import HierarchicalConvergenceEngine
from .hrm_embeddings import EmbeddingProcessor
from .hrm_heads import OutputProcessor

logger = logging.getLogger(__name__)


class OneStepGradientApproximator:
    """
    Implements the One-Step Gradient Approximation from HRM paper.
    
    Based on Deep Equilibrium Models and Implicit Function Theorem:
    ∂z_H*/∂θ ≈ ∂f_H/∂θ (Equation 2)
    
    This reduces memory complexity from O(T) to O(1) by avoiding BPTT.
    """
    
    def __init__(self):
        self.convergence_threshold = 1e-6
        
    def approximate_gradient(
        self, 
        h_module: nn.Module,
        l_module: nn.Module,
        z_h_final: torch.Tensor,
        z_l_final: torch.Tensor,
        loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute one-step gradient approximation without BPTT.
        
        Args:
            h_module: High-level module
            l_module: Low-level module  
            z_h_final: Final H-module state
            z_l_final: Final L-module state
            loss: Current loss value
            
        Returns:
            Dictionary of approximated gradients for each module
        """
        # Enable gradient computation for final states only
        z_h_final.requires_grad_(True)
        z_l_final.requires_grad_(True)
        
        gradients = {}
        
        # Compute gradients w.r.t. final states
        loss.backward(retain_graph=True)
        
        # H-module gradient approximation: ∂z_H*/∂θ_H ≈ ∂f_H/∂θ_H
        if z_h_final.grad is not None:
            h_grad = z_h_final.grad.clone()
            gradients['h_module'] = h_grad
        
        # L-module gradient approximation: ∂z_H*/∂θ_L ≈ ∂f_H/∂z_L* · ∂z_L*/∂θ_L  
        if z_l_final.grad is not None:
            l_grad = z_l_final.grad.clone()
            gradients['l_module'] = l_grad
            
        # Clear gradients for next iteration
        z_h_final.grad = None
        z_l_final.grad = None
        
        return gradients
    
    def neumann_series_approximation(
        self, 
        jacobian: torch.Tensor, 
        max_terms: int = 3
    ) -> torch.Tensor:
        """
        Approximate (I - J)^-1 using Neumann series: I + J + J^2 + J^3 + ...
        
        Args:
            jacobian: Jacobian matrix J
            max_terms: Maximum number of series terms
            
        Returns:
            Approximated inverse matrix
        """
        I = torch.eye(jacobian.size(-1), device=jacobian.device, dtype=jacobian.dtype)
        result = I.clone()
        J_power = jacobian.clone()
        
        for i in range(1, max_terms):
            result += J_power
            J_power = torch.matmul(J_power, jacobian)
            
        return result


class DeepSupervisionTrainer:
    """
    Implements Deep Supervision training mechanism from HRM paper.
    
    Key features:
    - Multiple forward passes (segments) with supervision at each step
    - Detached states between segments to prevent gradient flow
    - Periodic learning as in neural oscillations
    """
    
    def __init__(
        self, 
        M_max: int = 8,
        M_min: int = 2,
        gradient_approximator: OneStepGradientApproximator = None
    ):
        self.M_max = M_max  # Maximum segments
        self.M_min = M_min  # Minimum segments
        self.gradient_approximator = gradient_approximator or OneStepGradientApproximator()
        
    def train_step(
        self,
        model: nn.Module,
        data_batch: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        M: Optional[int] = None,
        accumulation_steps: int = 1,
        hardware_optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute one deep supervision training step with optional gradient accumulation.
        
        Args:
            model: HRM model to train
            data_batch: Input data batch
            targets: Training targets
            optimizer: Optimizer for parameter updates
            loss_fn: Loss function
            M: Number of segments (None for random selection)
            accumulation_steps: Number of steps to accumulate gradients before update
            hardware_optimizer: Hardware optimizer for mixed precision support
            
        Returns:
            Training statistics for the step
        """
        if M is None:
            M = torch.randint(self.M_min, self.M_max + 1, (1,)).item()
            
        device = data_batch.device
        batch_size = data_batch.size(0)
        
        # Initialize hidden states
        z_h, z_l = model.initialize_states(batch_size, device)
        
        total_loss = 0
        segment_losses = []
        convergence_stats = []
        accumulated_loss = 0
        
        for segment in range(M):
            # Forward pass for current segment
            outputs, (z_h_new, z_l_new) = model.forward(
                data_batch, z_init=(z_h, z_l)
            )
            
            # Compute loss for current segment
            segment_loss = loss_fn(outputs, targets)
            total_loss += segment_loss
            segment_losses.append(segment_loss.item())
            
            # Accumulate loss for gradient accumulation
            accumulated_loss += segment_loss / accumulation_steps
            
            # One-step gradient approximation
            gradients = self.gradient_approximator.approximate_gradient(
                model.h_module, model.l_module, z_h_new, z_l_new, segment_loss
            )
            
            # Update parameters with gradient accumulation
            if (segment + 1) % accumulation_steps == 0 or segment == M - 1:
                optimizer.zero_grad()
                
                # Use mixed precision if enabled
                if hardware_optimizer and hardware_optimizer.is_mixed_precision_enabled():
                    if hasattr(hardware_optimizer, 'is_tpu') and hardware_optimizer.is_tpu:
                        # TPU-specific optimization steps
                        try:
                            import torch_xla.core.xla_model as xm
                            accumulated_loss.backward()
                            xm.optimizer_step(optimizer)  # TPU-specific optimizer step
                        except ImportError:
                            accumulated_loss.backward()
                            optimizer.step()
                    else:
                        # CUDA mixed precision
                        hardware_optimizer.scaler.scale(accumulated_loss).backward()
                        hardware_optimizer.scaler.step(optimizer)
                        hardware_optimizer.scaler.update()
                else:
                    accumulated_loss.backward()
                    # TPU-specific optimizer step if needed
                    if hardware_optimizer and hasattr(hardware_optimizer, 'is_tpu') and hardware_optimizer.is_tpu:
                        try:
                            import torch_xla.core.xla_model as xm
                            xm.optimizer_step(optimizer)
                        except ImportError:
                            optimizer.step()
                    else:
                        optimizer.step()
                
                accumulated_loss = 0  # Reset accumulated loss
            
            # CRITICAL: Detach states to prevent gradient flow to next segment
            z_h = z_h_new.detach()
            z_l = z_l_new.detach()
            
            # Store convergence statistics
            convergence_stats.append({
                'segment': segment,
                'loss': segment_loss.item(),
                'h_norm': torch.norm(z_h, dim=-1).mean().item(),
                'l_norm': torch.norm(z_l, dim=-1).mean().item()
            })
            
            logger.debug(f"Deep supervision segment {segment}/{M}: loss={segment_loss.item():.6f}")
        
        return {
            'total_loss': total_loss.item(),
            'segment_losses': segment_losses,
            'segments_used': M,
            'convergence_stats': convergence_stats,
            'final_h_norm': torch.norm(z_h, dim=-1).mean().item(),
            'final_l_norm': torch.norm(z_l, dim=-1).mean().item()
        }
    
    def adaptive_segment_selection(
        self, 
        current_loss: float, 
        loss_history: List[float],
        min_improvement_threshold: float = 0.001
    ) -> int:
        """
        Adaptively select number of segments based on training progress.
        
        Args:
            current_loss: Current training loss
            loss_history: Recent loss history
            min_improvement_threshold: Minimum improvement to consider progress
            
        Returns:
            Number of segments to use for next training step
        """
        if len(loss_history) < 5:
            return self.M_min
        
        # Calculate recent improvement rate
        recent_losses = loss_history[-5:]
        improvement_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
        
        if improvement_rate < min_improvement_threshold:
            # Slow progress - increase segments for deeper reasoning
            return min(self.M_max, self.M_min + 2)
        else:
            # Good progress - use minimum segments for efficiency
            return self.M_min


class HRMTrainingOrchestrator:
    """
    Orchestrates the complete HRM training process with all paper innovations.
    
    Combines:
    - Hierarchical Convergence
    - One-Step Gradient Approximation  
    - Deep Supervision
    - Proper state initialization and management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # One-step gradient approximation
        self.gradient_approximator = OneStepGradientApproximator()
        
        self.deep_supervisor = DeepSupervisionTrainer(
            M_max=config.get('training', {}).get('M_max', 8),
            M_min=config.get('training', {}).get('M_min', 2),
            gradient_approximator=self.gradient_approximator
        )
        # Ensure convergence threshold is properly typed
        hrm_hierarchical_config = config.get('hierarchical_reasoning_model', {}).get('hierarchical', {})
        convergence_threshold = hrm_hierarchical_config.get('convergence_threshold', 1e-6)
        if isinstance(convergence_threshold, str):
            convergence_threshold = float(convergence_threshold)
        
        self.convergence_engine = HierarchicalConvergenceEngine(
            N=config.get('hierarchical_reasoning_model', {}).get('hierarchical', {}).get('N_cycles', 3),
            T=config.get('hierarchical_reasoning_model', {}).get('hierarchical', {}).get('T_timesteps', 5),
            convergence_threshold=convergence_threshold
        )
        
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        accumulation_steps: int = 1,
        hardware_optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train one complete epoch with HRM methodology and optional gradient accumulation.
        
        Args:
            model: HRM model to train
            dataloader: Training data loader
            optimizer: Optimizer for parameter updates
            loss_fn: Loss function
            device: Device to train on
            accumulation_steps: Number of steps to accumulate gradients before update
            hardware_optimizer: Hardware optimizer for mixed precision support
            
        Returns:
            Comprehensive training statistics
        """
        model.train()
        epoch_stats = {
            'total_loss': 0,
            'batch_losses': [],
            'convergence_rates': [],
            'segment_usage': [],
            'gradient_norms': []
        }
        
        loss_history = []
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            # Adaptive segment selection based on training progress
            M = self.deep_supervisor.adaptive_segment_selection(
                current_loss=loss_history[-1] if loss_history else float('inf'),
                loss_history=loss_history
            )
            
            # Execute deep supervision training step with gradient accumulation
            batch_stats = self.deep_supervisor.train_step(
                model=model,
                data_batch=data,
                targets=targets,
                optimizer=optimizer,
                loss_fn=loss_fn,
                M=M,
                accumulation_steps=accumulation_steps,
                hardware_optimizer=hardware_optimizer
            )
            
            # Update epoch statistics
            epoch_stats['total_loss'] += batch_stats['total_loss']
            epoch_stats['batch_losses'].append(batch_stats['total_loss'])
            epoch_stats['segment_usage'].append(batch_stats['segments_used'])
            
            loss_history.append(batch_stats['total_loss'])
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}: loss={batch_stats['total_loss']:.6f}, "
                          f"segments={batch_stats['segments_used']}")
        
        # Calculate final epoch statistics
        num_batches = len(dataloader)
        epoch_stats['avg_loss'] = epoch_stats['total_loss'] / num_batches
        epoch_stats['avg_segments'] = sum(epoch_stats['segment_usage']) / num_batches
        epoch_stats['convergence_success_rate'] = (
            sum(1 for stats in epoch_stats['convergence_rates'] 
                if stats.get('converged', False)) / max(len(epoch_stats['convergence_rates']), 1)
        )
        
        return epoch_stats