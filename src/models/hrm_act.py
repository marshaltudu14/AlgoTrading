"""
HRM Adaptive Computation Time (ACT) - "Thinking, Fast and Slow"

Implements the ACT mechanism from the HRM paper allowing dynamic computational allocation:
- Q-learning based halt/continue decisions
- "System 1" (automatic) vs "System 2" (deliberate) thinking
- Inference-time scaling for improved performance
- MDP formulation for optimal stopping

Based on HRM research paper Section 2.5 and brain-inspired dual-process theory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class ACTController:
    """
    Adaptive Computation Time controller implementing Q-learning for halt/continue decisions.
    
    MDP Formulation:
    - State: Current H-module state z_H^m
    - Actions: {halt, continue}  
    - Rewards: Binary correctness for halt (1 if correct, 0 otherwise), 0 for continue
    - Q-targets: G_halt = 1{y_hat = y}, G_continue = max(Q_halt^{m+1}, Q_continue^{m+1})
    """
    
    def __init__(
        self,
        M_max: int = 16,
        M_min_range: Tuple[int, int] = (1, 8),
        epsilon: float = 0.3,
        gamma: float = 0.99
    ):
        self.M_max = M_max
        self.M_min_range = M_min_range
        self.epsilon = epsilon  # Exploration probability for M_min sampling
        self.gamma = gamma  # Discount factor
        
        # Q-learning parameters
        self.q_learning_lr = 0.001
        self.target_update_freq = 100
        self.update_counter = 0
        
        # Statistics tracking
        self.segment_usage_history = deque(maxlen=1000)
        self.accuracy_history = deque(maxlen=1000)
        
    def should_halt(
        self, 
        q_values: torch.Tensor, 
        segment_count: int,
        M_min: int
    ) -> bool:
        """
        Determine whether to halt based on Q-values and constraints.
        
        Args:
            q_values: Q-values [batch_size, 2] for [halt, continue]
            segment_count: Current segment count
            M_min: Minimum segments for this episode
            
        Returns:
            Boolean decision to halt
        """
        # Force continue if below minimum segments
        if segment_count < M_min:
            return False
            
        # Force halt if at maximum segments
        if segment_count >= self.M_max:
            return True
            
        # Q-learning based decision
        q_halt, q_continue = q_values[0, 0], q_values[0, 1]
        return q_halt > q_continue
    
    def sample_M_min(self) -> int:
        """
        Sample minimum segments using epsilon-greedy strategy.
        
        With probability epsilon: sample uniformly from {2, ..., M_max}
        With probability 1-epsilon: use M_min = 1
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.M_min_range[0], self.M_min_range[1] + 1)
        else:
            return 1
    
    def compute_q_targets(
        self,
        segment_predictions: List[torch.Tensor],
        target: torch.Tensor,
        q_values_history: List[torch.Tensor],
        halt_segment: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Q-learning targets for all segments.
        
        Args:
            segment_predictions: Predictions from each segment
            target: Ground truth target
            q_values_history: Q-values from each segment
            halt_segment: Segment where halting occurred
            
        Returns:
            List of (q_targets_halt, q_targets_continue) for each segment
        """
        q_targets = []
        
        for m in range(len(segment_predictions)):
            # Halt target: binary correctness
            prediction_correct = self._check_prediction_correctness(
                segment_predictions[m], target
            )
            g_halt = torch.tensor(
                1.0 if prediction_correct else 0.0, 
                device=target.device, 
                dtype=torch.float32
            )
            
            # Continue target
            if m >= self.M_max - 1:
                # At maximum, must halt
                g_continue = g_halt
            elif m < len(q_values_history) - 1:
                # Use next segment's Q-values
                next_q = q_values_history[m + 1]
                g_continue = torch.max(next_q[0, 0], next_q[0, 1])
            else:
                # Final segment
                g_continue = g_halt
            
            q_targets.append((g_halt.unsqueeze(0), g_continue.unsqueeze(0)))
        
        return q_targets
    
    def _check_prediction_correctness(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> bool:
        """Check if prediction matches target (task-specific)"""
        if prediction.dim() > 1 and prediction.size(-1) > 1:
            # Classification: check if argmax matches
            pred_class = torch.argmax(prediction, dim=-1)
            if target.dim() > 0:
                target_class = torch.argmax(target, dim=-1) if target.size(-1) > 1 else target
            else:
                target_class = target
            return (pred_class == target_class).all().item()
        else:
            # Regression: check if close enough (within 5% tolerance)
            relative_error = torch.abs(prediction - target) / (torch.abs(target) + 1e-8)
            return (relative_error < 0.05).all().item()
    
    def update_q_network(
        self,
        q_head: nn.Module,
        q_values_history: List[torch.Tensor],
        q_targets: List[Tuple[torch.Tensor, torch.Tensor]],
        z_h_history: List[torch.Tensor],
        optimizer: torch.optim.Optimizer
    ):
        """
        Update Q-network using computed targets.
        
        Args:
            q_head: Q-head network
            q_values_history: Q-values from each segment
            q_targets: Target Q-values for each segment
            z_h_history: H-module states from each segment
            optimizer: Optimizer for Q-head
        """
        total_q_loss = 0
        
        for m, (q_values, (g_halt, g_continue), z_h) in enumerate(
            zip(q_values_history, q_targets, z_h_history)
        ):
            # Compute Q-loss for this segment
            q_pred = q_head(z_h.detach())  # Detach to prevent gradient flow
            
            target_q = torch.stack([g_halt, g_continue], dim=-1)
            
            q_loss = F.mse_loss(q_pred, target_q)
            total_q_loss += q_loss
        
        # Update Q-head
        optimizer.zero_grad()
        total_q_loss.backward()
        optimizer.step()
        
        self.update_counter += 1
        
        return total_q_loss.item()
    
    def get_inference_time_scaling_factor(
        self, 
        task_difficulty: float = None,
        available_compute_budget: float = None
    ) -> float:
        """
        Determine scaling factor for inference-time computation.
        
        Args:
            task_difficulty: Estimated difficulty (0-1, higher = more difficult)
            available_compute_budget: Available computation budget
            
        Returns:
            Scaling factor for M_max during inference
        """
        base_factor = 1.0
        
        # Scale based on task difficulty
        if task_difficulty is not None:
            difficulty_factor = 1.0 + task_difficulty * 2.0  # Up to 3x for very difficult tasks
            base_factor *= difficulty_factor
        
        # Scale based on compute budget
        if available_compute_budget is not None:
            budget_factor = min(available_compute_budget, 4.0)  # Cap at 4x
            base_factor *= budget_factor
        
        return min(base_factor, 4.0)  # Maximum 4x scaling


class ACTIntegratedTrainer:
    """
    Training orchestrator that integrates ACT with HRM training process.
    
    Combines deep supervision with adaptive computation time for optimal
    performance vs efficiency trade-offs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        act_config = config.get('act', {})
        
        self.act_controller = ACTController(
            M_max=act_config.get('M_max', 16),
            M_min_range=act_config.get('M_min_range', (1, 8)),
            epsilon=act_config.get('epsilon', 0.3),
            gamma=act_config.get('gamma', 0.99)
        )
        
        # Separate optimizer for Q-head
        self.q_optimizer = None
        
    def train_step_with_act(
        self,
        model: nn.Module,
        data_batch: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module
    ) -> Dict[str, Any]:
        """
        Execute training step with ACT mechanism.
        
        Args:
            model: HRM model with Q-head
            data_batch: Input batch
            targets: Target batch
            optimizer: Main model optimizer
            loss_fn: Primary loss function
            
        Returns:
            Training statistics including ACT metrics
        """
        device = data_batch.device
        batch_size = data_batch.size(0)
        
        # Initialize Q-head optimizer if not exists
        if self.q_optimizer is None and hasattr(model, 'q_head'):
            self.q_optimizer = torch.optim.Adam(
                model.q_head.parameters(), 
                lr=self.act_controller.q_learning_lr
            )
        
        # Sample minimum segments for this episode
        M_min = self.act_controller.sample_M_min()
        
        # Initialize states and tracking
        z_h, z_l = model.initialize_states(batch_size, device)
        
        segment_predictions = []
        q_values_history = []
        z_h_history = []
        segment_losses = []
        
        segment_count = 0
        halted = False
        
        # Adaptive computation loop
        while segment_count < self.act_controller.M_max and not halted:
            # Forward pass for current segment
            outputs, (z_h_new, z_l_new) = model.forward(
                data_batch, z_init=(z_h, z_l)
            )
            
            # Store segment information
            segment_predictions.append(outputs['action_type'].clone())
            z_h_history.append(z_h_new.clone())
            
            # Get Q-values for halt/continue decision
            if hasattr(model, 'q_head'):
                q_values = model.q_head(z_h_new.detach())
                q_values_history.append(q_values.clone())
                
                # Decide whether to halt
                should_halt = self.act_controller.should_halt(
                    q_values, segment_count + 1, M_min
                )
                
                if should_halt:
                    halted = True
            
            # Compute segment loss
            segment_loss = loss_fn(outputs, targets)
            segment_losses.append(segment_loss.item())
            
            # Update main model parameters
            optimizer.zero_grad()
            segment_loss.backward()
            optimizer.step()
            
            # Detach states for next segment
            z_h = z_h_new.detach()
            z_l = z_l_new.detach()
            
            segment_count += 1
        
        # Update Q-network if we have Q-head
        q_loss = 0
        if hasattr(model, 'q_head') and self.q_optimizer and len(q_values_history) > 0:
            # Compute Q-targets
            q_targets = self.act_controller.compute_q_targets(
                segment_predictions, targets, q_values_history, segment_count
            )
            
            # Update Q-network
            q_loss = self.act_controller.update_q_network(
                model.q_head, q_values_history, q_targets, z_h_history, self.q_optimizer
            )
        
        # Update statistics
        final_prediction = segment_predictions[-1]
        accuracy = self.act_controller._check_prediction_correctness(
            final_prediction, targets
        )
        
        self.act_controller.segment_usage_history.append(segment_count)
        self.act_controller.accuracy_history.append(accuracy)
        
        return {
            'total_loss': sum(segment_losses),
            'segment_losses': segment_losses,
            'segments_used': segment_count,
            'halted_early': halted,
            'M_min_used': M_min,
            'q_loss': q_loss,
            'accuracy': accuracy,
            'avg_segments_recent': np.mean(list(self.act_controller.segment_usage_history)),
            'avg_accuracy_recent': np.mean(list(self.act_controller.accuracy_history))
        }
    
    def inference_with_scaling(
        self,
        model: nn.Module,
        data_batch: torch.Tensor,
        scaling_factor: float = 1.0,
        task_difficulty: float = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Run inference with computational scaling.
        
        Args:
            model: HRM model
            data_batch: Input data
            scaling_factor: Computational scaling factor
            task_difficulty: Estimated task difficulty
            
        Returns:
            Tuple of (outputs, inference_stats)
        """
        device = data_batch.device
        batch_size = data_batch.size(0)
        
        # Scale maximum segments based on factor
        scaled_M_max = int(self.act_controller.M_max * scaling_factor)
        
        # Initialize states
        z_h, z_l = model.initialize_states(batch_size, device)
        
        segment_count = 0
        halted = False
        confidence_history = []
        
        model.eval()
        with torch.no_grad():
            while segment_count < scaled_M_max and not halted:
                # Forward pass
                outputs, (z_h_new, z_l_new) = model.forward(
                    data_batch, z_init=(z_h, z_l)
                )
                
                # Check Q-values for halting decision
                if hasattr(model, 'q_head'):
                    q_values = model.q_head(z_h_new)
                    should_halt = self.act_controller.should_halt(
                        q_values, segment_count + 1, M_min=1
                    )
                    
                    confidence = torch.max(q_values, dim=-1)[0].item()
                    confidence_history.append(confidence)
                    
                    if should_halt:
                        halted = True
                
                # Update states
                z_h, z_l = z_h_new, z_l_new
                segment_count += 1
        
        inference_stats = {
            'segments_used': segment_count,
            'scaling_factor': scaling_factor,
            'scaled_M_max': scaled_M_max,
            'halted_early': halted,
            'confidence_trajectory': confidence_history,
            'final_confidence': confidence_history[-1] if confidence_history else 0
        }
        
        return outputs, inference_stats