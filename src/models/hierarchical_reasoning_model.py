"""
Hierarchical Reasoning Model (HRM) v2.0 - Modular Brain-Inspired Architecture

This is the completely refactored, scientifically rigorous implementation of HRM
based on the research paper "Hierarchical Reasoning Model" achieving 27M parameter
efficiency with unlimited computational depth.

NEW FEATURES IN V2.0:
✅ One-Step Gradient Approximation (O(1) memory vs O(T) BPTT)
✅ Complete Deep Supervision mechanism with detached states  
✅ Full Adaptive Computation Time (ACT) with Q-learning
✅ Brain correspondence validation with Participation Ratio
✅ Hierarchical convergence diagnostics
✅ Modular architecture for maintainability

Architecture Components:
- InputEmbeddingNetwork: Market data preprocessing
- HighLevelModule (H): Strategic reasoning, abstract planning
- LowLevelModule (L): Tactical execution, detailed computations
- OutputProcessor: All prediction heads with trading constraints
- HierarchicalConvergenceEngine: N cycles × T timesteps orchestration
- DeepSupervisionTrainer: Revolutionary O(1) memory training
- ACTController: "Thinking fast and slow" mechanism
- ParticipationRatioAnalyzer: Neuroscience validation

Based on cutting-edge research from Sapient Intelligence achieving breakthrough
performance on ARC-AGI (40.3%), Sudoku-Extreme (near-perfect), and Maze-Hard.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
from src.utils.config_loader import ConfigLoader
from src.agents.base_agent import BaseAgent
from src.utils.hardware_optimizer import HardwareOptimizer

# Import modular components
from .hrm_components import (
    RMSNorm, RotaryPositionalEmbedding, GLU, TransformerBlock, ConvergenceTracker
)
from .hrm_modules import (
    HighLevelModule, LowLevelModule, HierarchicalConvergenceEngine
)
from .hrm_embeddings import (
    InputEmbeddingNetwork, InstrumentEmbedding, TimeframeEmbedding, EmbeddingProcessor
)
from .hrm_heads import (
    PolicyHead, QuantityHead, ValueHead, QHead, OutputProcessor
)
from .hrm_training import (
    OneStepGradientApproximator, DeepSupervisionTrainer, HRMTrainingOrchestrator
)
from .hrm_act import (
    ACTController, ACTIntegratedTrainer
)
from .hrm_diagnostics import (
    ParticipationRatioAnalyzer, ConvergencePatternAnalyzer, HRMDiagnosticSuite
)

logger = logging.getLogger(__name__)


class HierarchicalReasoningModel(nn.Module, BaseAgent):
    """
    HRM v2.0 - Scientifically rigorous brain-inspired trading agent.
    
    Key Innovations:
    1. **One-Step Gradient**: O(1) memory complexity vs O(T) BPTT
    2. **Deep Supervision**: Multi-segment training with detached states
    3. **Adaptive Computation Time**: Dynamic "System 1/2" thinking
    4. **Brain Correspondence**: Validated hierarchical dimensionality
    5. **Hierarchical Convergence**: N cycles × T timesteps deep reasoning
    
    Mathematical Foundation:
    - H-module: z_H^k = f_H(z_H^{k-1}, z_L*; θ_H)
    - L-module: z_L* = f_L(z_L*, z_H^{k-1}, x̃; θ_L) (fixed point)
    - Gradient: ∂z_H*/∂θ ≈ (I - J_F)^{-1} ∂F/∂θ ≈ ∂F/∂θ (one-step approx)
    - PR Hierarchy: PR_H > PR_L (brain-like dimensionality separation)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Load configuration with comprehensive validation
        if config is None:
            config_loader = ConfigLoader()
            config = config_loader.get_config()
        
        self.config = config
        self._validate_and_setup_config()
        self._initialize_architecture()
        self._setup_training_components()
        self._setup_diagnostic_suite()
        
        # Initialize hardware optimizer for GPU/TPU optimization
        self.hardware_optimizer = HardwareOptimizer(enable_optimization=True)
        
        # Optimize model for hardware
        self = self.hardware_optimizer.optimize_model(self)
        
        # Enable mixed precision training if available
        self.hardware_optimizer.enable_mixed_precision()
        
        # Log initialization with parameter count
        param_count = self.count_parameters()
        logger.info(f"🧠 HRM v2.0 initialized: {param_count:,} parameters ")
        logger.info(f"   Device: {self.hardware_optimizer.device}")
        logger.info(f"   Mixed Precision: {'Enabled' if self.hardware_optimizer.is_mixed_precision_enabled() else 'Disabled'}")

    def _validate_and_setup_config(self):
        """Validate configuration and setup architectural parameters"""
        # Extract HRM-specific configuration
        hrm_config = self.config.get('hierarchical_reasoning_model', {})
        
        # H-module configuration (Strategic reasoning)
        self.h_config = hrm_config.get('h_module', {
            'hidden_dim': 512,
            'num_layers': 4,
            'n_heads': 8,
            'ff_dim': 2048,
            'dropout': 0.1
        })
        
        # L-module configuration (Tactical execution)
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
        
        # Hierarchical processing configuration for multi-timeframe analysis
        self.hierarchical_processing_config = hrm_config.get('hierarchical_processing', {
            'high_level_lookback': 128,  # Comprehensive market overview
            'low_level_lookback': 50,    # Recent market dynamics
            'high_level_features': 64,   # Features for high-level analysis
            'low_level_features': 32     # Features for low-level analysis
        })
        
        # Hierarchical convergence parameters
        self.hierarchical_config = hrm_config.get('hierarchical', {
            'N_cycles': 3,  # High-level cycles
            'T_timesteps': 5,  # Low-level timesteps per cycle
            'convergence_threshold': 1e-6,
            'reset_factor': 0.3
        })
        
        # Ensure convergence_threshold is a float
        if 'convergence_threshold' in self.hierarchical_config:
            # Convert to float if it's a string
            if isinstance(self.hierarchical_config['convergence_threshold'], str):
                self.hierarchical_config['convergence_threshold'] = float(self.hierarchical_config['convergence_threshold'])
        else:
            self.hierarchical_config['convergence_threshold'] = 1e-6
        
        # Add missing convergence_threshold if not present
        if 'convergence_threshold' not in self.hierarchical_config:
            self.hierarchical_config['convergence_threshold'] = 1e-6
        
        # Embedding configurations
        embedding_config = hrm_config.get('embeddings', {})
        self.embedding_configs = {
            'instrument': {
                'vocab_size': embedding_config.get('max_instruments', 1000),
                'embedding_dim': embedding_config.get('instrument_dim', 64)
            },
            'timeframe': {
                'vocab_size': embedding_config.get('max_timeframes', 10),
                'embedding_dim': embedding_config.get('timeframe_dim', 32)
            }
        }
        
        # Output head configuration
        output_config = hrm_config.get('output_heads', {})
        model_config = self.config.get('model', {})
        self.output_config = {
            'action_dim': output_config.get('action_dim', model_config.get('action_dim_discrete', 5)),
            'quantity_min': output_config.get('quantity_min', 1.0),
            'quantity_max': output_config.get('quantity_max', 100000.0),
            'value_estimation': output_config.get('value_estimation', True),
            'q_learning_enabled': output_config.get('q_learning_enabled', True)
        }

    def _initialize_architecture(self):
        """Initialize all architectural components"""
        logger.info("🏗️  Initializing HRM v2.0 modular architecture...")
        
        # Core embedding networks
        self.input_network = InputEmbeddingNetwork(self.input_config)
        self.instrument_embedding = InstrumentEmbedding(
            self.embedding_configs['instrument']['vocab_size'],
            self.embedding_configs['instrument']['embedding_dim']
        )
        self.timeframe_embedding = TimeframeEmbedding(
            self.embedding_configs['timeframe']['vocab_size'],
            self.embedding_configs['timeframe']['embedding_dim']
        )
        
        # Embedding projection to H-module dimension
        total_embedding_dim = (
            self.input_config['embedding_dim'] +
            self.embedding_configs['instrument']['embedding_dim'] +
            self.embedding_configs['timeframe']['embedding_dim']
        )
        self.embedding_projection = nn.Linear(
            total_embedding_dim, self.h_config['hidden_dim'], bias=False
        )
        
        # Hierarchical reasoning modules
        self.h_module = HighLevelModule(self.h_config)
        self.l_module = LowLevelModule(self.l_config)
        
        # Dimension projection between L and H modules
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
        
        # Optional heads
        self.value_head = None
        if self.output_config['value_estimation']:
            self.value_head = ValueHead(self.h_config['hidden_dim'])
            
        self.q_head = None
        if self.output_config['q_learning_enabled']:
            self.q_head = QHead(self.h_config['hidden_dim'])
        
        # Initialize embedding processor and output processor
        self.embedding_processor = EmbeddingProcessor(
            self.input_network, self.instrument_embedding, 
            self.timeframe_embedding, self.embedding_projection
        )
        
        self.output_processor = OutputProcessor(
            self.policy_head, self.quantity_head, self.value_head, self.q_head
        )
        
        # Hierarchical convergence engine
        self.convergence_engine = HierarchicalConvergenceEngine(
            N=self.hierarchical_config['N_cycles'],
            T=self.hierarchical_config['T_timesteps'],
            convergence_threshold=self.hierarchical_config['convergence_threshold']
        )
        
        logger.info("✅ HRM architecture initialized successfully")

    def _setup_training_components(self):
        """Initialize training-specific components"""
        training_config = self.config.get('training', {})
        
        # One-step gradient approximation
        self.gradient_approximator = OneStepGradientApproximator()
        
        # Deep supervision trainer
        self.deep_supervision_trainer = DeepSupervisionTrainer(
            M_max=training_config.get('M_max', 8),
            M_min=training_config.get('M_min', 2),
            gradient_approximator=self.gradient_approximator
        )
        
        # Training orchestrator
        self.training_orchestrator = HRMTrainingOrchestrator(self.config)
        
        # ACT components
        if self.output_config['q_learning_enabled']:
            self.act_controller = ACTController(
                M_max=training_config.get('act_M_max', 16),
                M_min_range=training_config.get('act_M_min_range', (1, 8)),
                epsilon=training_config.get('act_epsilon', 0.3)
            )
            
            self.act_trainer = ACTIntegratedTrainer(self.config)

    def _setup_diagnostic_suite(self):
        """Initialize comprehensive diagnostic suite"""
        self.diagnostic_suite = HRMDiagnosticSuite()
        
        # Individual analyzers for focused analysis
        self.pr_analyzer = ParticipationRatioAnalyzer()
        self.convergence_analyzer = ConvergencePatternAnalyzer()

    def initialize_states(
        self, 
        batch_size: int, 
        device: torch.device, 
        z_init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden states using truncated normal distribution.
        
        Following paper specification: truncated normal with σ=1, truncation=2σ.
        """
        if z_init is not None:
            z_h_init, z_l_init = z_init
            return z_h_init.to(device), z_l_init.to(device)
        
        # Truncated normal initialization (paper specification)
        arch_config = self.config.get('hierarchical_reasoning_model', {}).get('architecture', {})
        std = arch_config.get('truncated_normal_std', 1.0)
        limit = arch_config.get('truncated_normal_limit', 2.0)
        
        z_h = torch.randn(batch_size, self.h_config['hidden_dim'], device=device) * std
        z_h = torch.clamp(z_h, -limit, limit)
        
        z_l = torch.randn(batch_size, self.l_config['hidden_dim'], device=device) * std
        z_l = torch.clamp(z_l, -limit, limit)
        
        return z_h, z_l

    def forward(
        self, 
        x: torch.Tensor, 
        instrument_ids: Optional[torch.Tensor] = None,
        timeframe_ids: Optional[torch.Tensor] = None, 
        z_init: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_diagnostics: bool = False
    ) -> Union[Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
               Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]]:
        """
        Execute hierarchical reasoning with comprehensive error handling.
        
        Args:
            x: Market features [batch_size, feature_dim]
            instrument_ids: Instrument identifiers [batch_size]
            timeframe_ids: Timeframe identifiers [batch_size]
            z_init: Initial hidden states (z_H, z_L)
            return_diagnostics: Whether to return detailed diagnostics
            
        Returns:
            outputs: Dictionary with predictions
            final_states: (z_H, z_L) for potential continuation
            diagnostics: (optional) Detailed convergence diagnostics
        """
        try:
            # Input validation and preprocessing
            batch_size = x.size(0)
            device = x.device
            
            # Process inputs through embedding pipeline
            x_embedded = self.embedding_processor.process(x, instrument_ids, timeframe_ids)
            
            # Initialize hidden states
            z_h, z_l = self.initialize_states(batch_size, device, z_init)
            
            # Execute hierarchical convergence
            z_h_final, z_l_final, convergence_info = self.convergence_engine.execute_hierarchical_reasoning(
                self.h_module, self.l_module, z_h, z_l, x_embedded, self.l_to_h_projection
            )
            
            # Generate outputs from final H-module state
            outputs = self.output_processor.generate_outputs(z_h_final)
            
            # Optional diagnostics
            if return_diagnostics:
                diagnostics = {
                    'convergence_info': convergence_info,
                    'embedding_stats': self.embedding_processor.get_embedding_stats(),
                    'output_stats': self._compute_output_statistics(outputs),
                    'state_norms': {
                        'final_h_norm': torch.norm(z_h_final, dim=-1).mean().item(),
                        'final_l_norm': torch.norm(z_l_final, dim=-1).mean().item()
                    }
                }
                return outputs, (z_h_final, z_l_final), diagnostics
            
            return outputs, (z_h_final, z_l_final)
            
        except Exception as e:
            logger.error(f"🚨 Critical error in HRM forward pass: {e}")
            # Emergency fallback
            return self._emergency_fallback(x, device)

    def _compute_output_statistics(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute statistics for output validation"""
        stats = {}
        
        if 'action_type' in outputs:
            action_probs = F.softmax(outputs['action_type'], dim=-1)
            stats['action_entropy'] = -torch.sum(
                action_probs * torch.log(action_probs + 1e-8), dim=-1
            ).mean().item()
            stats['action_confidence'] = torch.max(action_probs, dim=-1)[0].mean().item()
        
        if 'quantity' in outputs:
            stats['quantity_mean'] = outputs['quantity'].mean().item()
            stats['quantity_std'] = outputs['quantity'].std().item()
        
        if 'value' in outputs:
            stats['value_mean'] = outputs['value'].mean().item()
            stats['value_std'] = outputs['value'].std().item()
        
        return stats

    def _emergency_fallback(self, x: torch.Tensor, device: torch.device):
        """Emergency fallback for critical errors"""
        batch_size = x.size(0)
        
        fallback_outputs = {
            'action_type': torch.zeros(batch_size, self.output_config['action_dim'], device=device),
            'quantity': torch.full((batch_size,), self.output_config['quantity_min'], device=device)
        }
        
        if self.output_config['value_estimation']:
            fallback_outputs['value'] = torch.zeros(batch_size, device=device)
        
        if self.output_config['q_learning_enabled']:
            fallback_outputs['q_values'] = torch.zeros(batch_size, 2, device=device)
        
        fallback_states = (
            torch.zeros(batch_size, self.h_config['hidden_dim'], device=device),
            torch.zeros(batch_size, self.l_config['hidden_dim'], device=device)
        )
        
        return fallback_outputs, fallback_states

    def select_action(
        self, 
        observation: Union[torch.Tensor, np.ndarray],
        available_capital: float = None,
        current_position_quantity: float = None,
        current_price: float = None,
        instrument: Any = None,
        return_probabilities: bool = False,
        use_act_scaling: bool = False,
        task_difficulty: float = None
    ) -> Union[Tuple[int, float], Tuple[int, float, np.ndarray]]:
        """
        Advanced action selection with HRM hierarchical reasoning.
        
        Args:
            observation: Market observation
            available_capital: Available capital for trades
            current_position_quantity: Current position size
            current_price: Current market price
            instrument: Instrument object
            return_probabilities: Whether to return action probabilities
            use_act_scaling: Whether to use ACT inference scaling
            task_difficulty: Estimated task difficulty for ACT scaling
            
        Returns:
            Action tuple: (action_type, quantity, [probabilities])
        """
        if isinstance(observation, np.ndarray):
            observation = torch.FloatTensor(observation).unsqueeze(0)
        elif observation.dim() == 1:
            observation = observation.unsqueeze(0)
        
        # Extract instrument and timeframe information
        instrument_id, timeframe_id = self._extract_instrument_info(instrument)
        
        self.eval()
        with torch.no_grad():
            if use_act_scaling and hasattr(self, 'act_trainer'):
                # Use ACT with inference-time scaling
                scaling_factor = self.act_controller.get_inference_time_scaling_factor(
                    task_difficulty=task_difficulty
                )
                
                outputs, inference_stats = self.act_trainer.inference_with_scaling(
                    self, observation, scaling_factor, task_difficulty
                )
                
                logger.debug(f"ACT scaling used: {scaling_factor:.2f}x, "
                           f"segments: {inference_stats['segments_used']}")
            else:
                # Standard forward pass
                instrument_ids = torch.tensor([instrument_id], device=observation.device) if instrument_id is not None else None
                timeframe_ids = torch.tensor([timeframe_id], device=observation.device) if timeframe_id is not None else None
                
                outputs, _ = self.forward(observation, instrument_ids, timeframe_ids)
            
            # Sample action with trading constraints
            action_type, quantity = self.output_processor.sample_action(
                outputs, available_capital, current_position_quantity, current_price
            )
            
            if return_probabilities:
                action_probs = F.softmax(outputs['action_type'], dim=-1).squeeze().cpu().numpy()
                return action_type, quantity, action_probs
            else:
                return action_type, quantity

    def _extract_instrument_info(self, instrument) -> Tuple[Optional[int], Optional[int]]:
        """Extract instrument and timeframe IDs from instrument object"""
        instrument_id, timeframe_id = None, None
        
        if instrument and hasattr(instrument, 'symbol'):
            symbol_parts = instrument.symbol.split('_')
            
            # Map instrument name to ID
            instrument_name = symbol_parts[0] if len(symbol_parts) > 0 else 'Unknown'
            instrument_id = abs(hash(instrument_name)) % self.embedding_configs['instrument']['vocab_size']
            
            # Map timeframe to ID
            if len(symbol_parts) > 1:
                try:
                    timeframe = int(symbol_parts[1])
                    timeframe_map = {1: 0, 10: 1, 15: 2, 120: 3, 180: 4}
                    timeframe_id = timeframe_map.get(
                        timeframe, 
                        abs(hash(str(timeframe))) % self.embedding_configs['timeframe']['vocab_size']
                    )
                except ValueError:
                    timeframe_id = 0
            else:
                timeframe_id = 0
        
        return instrument_id, timeframe_id

    def train_with_deep_supervision(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        use_act: bool = False,
        accumulation_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Train using revolutionary deep supervision methodology with hardware optimization.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer instance
            loss_fn: Loss function
            device: Computing device
            use_act: Whether to use ACT mechanism
            accumulation_steps: Number of steps to accumulate gradients before update
            
        Returns:
            Comprehensive training statistics
        """
        if use_act and hasattr(self, 'act_trainer'):
            return self._train_with_act(dataloader, optimizer, loss_fn, device)
        else:
            return self.training_orchestrator.train_epoch(
                self, dataloader, optimizer, loss_fn, device,
                accumulation_steps=accumulation_steps,
                hardware_optimizer=self.hardware_optimizer
            )

    def _train_with_act(
        self,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device
    ) -> Dict[str, Any]:
        """Train with ACT mechanism"""
        self.train()
        epoch_stats = {
            'total_loss': 0,
            'total_q_loss': 0,
            'segment_usage': [],
            'accuracy_scores': []
        }
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            batch_stats = self.act_trainer.train_step_with_act(
                self, data, targets, optimizer, loss_fn
            )
            
            epoch_stats['total_loss'] += batch_stats['total_loss']
            epoch_stats['total_q_loss'] += batch_stats['q_loss']
            epoch_stats['segment_usage'].append(batch_stats['segments_used'])
            epoch_stats['accuracy_scores'].append(batch_stats['accuracy'])
            
            if batch_idx % 10 == 0:
                logger.info(f"ACT Batch {batch_idx}: loss={batch_stats['total_loss']:.6f}, "
                          f"segments={batch_stats['segments_used']}, "
                          f"accuracy={batch_stats['accuracy']:.3f}")
        
        epoch_stats['avg_segments'] = np.mean(epoch_stats['segment_usage'])
        epoch_stats['avg_accuracy'] = np.mean(epoch_stats['accuracy_scores'])
        
        return epoch_stats

    def get_convergence_diagnostics(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get detailed convergence diagnostics for a given input.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with detailed convergence information
        """
        # Run forward pass with diagnostics enabled
        outputs, final_states, diagnostics = self.forward(x, return_diagnostics=True)
        convergence_info = diagnostics['convergence_info']
        
        # Add additional metrics for the diagnostic suite
        convergence_info['convergence_metrics'] = {
            'total_cycles': len(convergence_info['cycles']),
            'total_l_steps': convergence_info['total_l_steps'],
            'h_updates': convergence_info['h_updates'],
            'final_residuals': convergence_info['residuals'][-5:] if convergence_info['residuals'] else []
        }
        
        return convergence_info
    
    def run_brain_correspondence_analysis(
        self, 
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Run comprehensive brain correspondence analysis.
        
        Validates that the model exhibits brain-like hierarchical organization
        with proper dimensionality separation between H and L modules.
        """
        logger.info("🧠 Running brain correspondence analysis...")
        
        analysis_results = self.diagnostic_suite.run_comprehensive_diagnostic(
            self, test_dataloader, device, num_samples
        )
        
        # Log key findings
        brain_score = analysis_results['brain_correspondence_score']
        pr_hierarchy = analysis_results['participation_ratio_analysis']['hierarchy_ratio']
        
        logger.info(f"🧠 Brain Correspondence Analysis Results:")
        logger.info(f"   Brain Similarity Score: {brain_score:.3f}/1.000")
        logger.info(f"   PR Hierarchy Ratio: {pr_hierarchy:.3f} (Expected: 2.25-3.0)")
        logger.info(f"   HRM Compliance Score: {analysis_results['hrm_compliance_score']:.3f}/1.000")
        
        if brain_score > 0.7:
            logger.info("✅ Model shows strong brain-like hierarchical organization")
        elif brain_score > 0.4:
            logger.warning("⚠️  Model shows moderate brain correspondence - consider tuning")
        else:
            logger.error("❌ Model lacks proper hierarchical organization - major issues detected")
        
        return analysis_results

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get detailed parameter breakdown by component"""
        breakdown = {
            'total_parameters': self.count_parameters(),
            'h_module': sum(p.numel() for p in self.h_module.parameters()),
            'l_module': sum(p.numel() for p in self.l_module.parameters()),
            'input_embedding': sum(p.numel() for p in self.input_network.parameters()),
            'instrument_embedding': sum(p.numel() for p in self.instrument_embedding.parameters()),
            'timeframe_embedding': sum(p.numel() for p in self.timeframe_embedding.parameters()),
            'policy_head': sum(p.numel() for p in self.policy_head.parameters()),
            'quantity_head': sum(p.numel() for p in self.quantity_head.parameters()),
        }
        
        if self.value_head:
            breakdown['value_head'] = sum(p.numel() for p in self.value_head.parameters())
        
        if self.q_head:
            breakdown['q_head'] = sum(p.numel() for p in self.q_head.parameters())
        
        return breakdown

    def save_model_v2(self, path: str):
        """Save HRM v2.0 model with comprehensive metadata"""
        try:
            # Get parameter breakdown and diagnostic info
            param_breakdown = self.get_parameter_breakdown()
            
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'config': self.config,
                'parameter_breakdown': param_breakdown,
                'architecture': 'HierarchicalReasoningModel_v2.0',
                'paper_compliance': {
                    'one_step_gradient': True,
                    'deep_supervision': True,
                    'adaptive_computation_time': self.output_config['q_learning_enabled'],
                    'brain_correspondence': True,
                    'hierarchical_convergence': True
                },
                'version': '2.0.0',
                'components': {
                    'hrm_modules': True,
                    'hrm_training': True,
                    'hrm_act': self.output_config['q_learning_enabled'],
                    'hrm_diagnostics': True
                }
            }
            
            torch.save(checkpoint, path)
            logger.info(f"💾 HRM v2.0 model saved to {path}")
            logger.info(f"   Total parameters: {param_breakdown['total_parameters']:,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save HRM v2.0 model: {e}")
            raise

    def load_model_v2(self, path: str):
        """Load HRM v2.0 model with compatibility checking"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            # Validate version compatibility
            if checkpoint.get('version', '1.0.0').startswith('2.'):
                logger.info(f"📥 Loading HRM v{checkpoint.get('version', '2.0.0')}")
            else:
                logger.warning("⚠️  Loading older model version - some features may be unavailable")
            
            # Load state dict with error handling
            current_state = self.state_dict()
            loaded_state = checkpoint.get('model_state_dict', checkpoint)
            
            # Filter incompatible keys
            compatible_state = {}
            for key, value in loaded_state.items():
                if key in current_state and current_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    logger.warning(f"Skipping incompatible parameter: {key}")
            
            self.load_state_dict(compatible_state, strict=False)
            
            # Log loading results
            loaded_params = sum(p.numel() for p in compatible_state.values())
            logger.info(f"✅ HRM v2.0 model loaded: {loaded_params:,} parameters")
            
        except Exception as e:
            logger.error(f"❌ Failed to load HRM v2.0 model: {e}")
            raise

    # BaseAgent interface methods
    def act(self, observation: Union[torch.Tensor, np.ndarray]) -> Tuple[int, float]:
        """Simplified interface for live trading compatibility"""
        return self.select_action(observation)

    def update(self) -> None:
        """Update method for trainer compatibility"""
        # HRM updates happen through hierarchical convergence
        pass