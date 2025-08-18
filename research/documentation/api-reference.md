# HRM API Reference Documentation

## Overview

This document provides comprehensive API documentation for the Hierarchical Reasoning Model (HRM) implementation for algorithmic trading. The API is designed for seamless integration with existing trading systems while providing advanced hierarchical reasoning capabilities.

## Table of Contents

1. [Core Model API](#core-model-api)
2. [Configuration Management](#configuration-management)
3. [Training Interface](#training-interface)
4. [Inference API](#inference-api)
5. [Diagnostic Tools](#diagnostic-tools)
6. [Integration Helpers](#integration-helpers)
7. [Error Handling](#error-handling)
8. [Examples](#examples)

## Core Model API

### HierarchicalReasoningModel

The main model class implementing the brain-inspired dual-module architecture.

```python
class HierarchicalReasoningModel(nn.Module):
    """
    Brain-inspired HRM with dual-module architecture for algorithmic trading.
    
    Implements hierarchical convergence mechanism with unlimited computational depth
    through N cycles of T timesteps each.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize HRM model.
        
        Args:
            config: Configuration dictionary. If None, loads from settings.yaml
            
        Example:
            >>> config = {'model': {'observation_dim': 256}}
            >>> model = HierarchicalReasoningModel(config)
        """
```

#### Key Methods

##### forward()
```python
def forward(self, 
           x: torch.Tensor, 
           instrument_ids: Optional[torch.Tensor] = None,
           timeframe_ids: Optional[torch.Tensor] = None, 
           z_init: Optional[Tuple] = None) -> Tuple[Dict[str, torch.Tensor], Tuple]:
    """
    Execute hierarchical reasoning over N cycles of T timesteps.
    
    Args:
        x: Market features [batch_size, feature_dim]
        instrument_ids: Instrument identifiers [batch_size] (optional)
        timeframe_ids: Timeframe identifiers [batch_size] (optional)
        z_init: Initial hidden states (z_H, z_L) (optional)
        
    Returns:
        outputs: Dictionary with keys:
            - 'action_type': Discrete action logits [batch_size, action_dim]
            - 'quantity': Continuous position size [batch_size]
            - 'value': State value estimation [batch_size] (if enabled)
            - 'q_values': Q-learning values [batch_size, 2] (if enabled)
        final_states: Tuple (z_H, z_L) for potential continuation
        
    Example:
        >>> x = torch.randn(4, 256)  # Batch of market features
        >>> outputs, states = model.forward(x)
        >>> action_probs = F.softmax(outputs['action_type'], dim=-1)
    """
```

##### act()
```python
def act(self, 
        observation: torch.Tensor, 
        instrument_id: Optional[int] = None,
        timeframe_id: Optional[int] = None) -> Tuple[int, float]:
    """
    Generate trading action from observation.
    
    Args:
        observation: Market features [feature_dim] or [1, feature_dim]
        instrument_id: Trading instrument identifier (0 to max_instruments-1)
        timeframe_id: Trading timeframe identifier (0 to max_timeframes-1)
        
    Returns:
        action_type: Discrete action (0=BUY, 1=SELL, 2=HOLD, 3=CLOSE_LONG, 4=CLOSE_SHORT)
        quantity: Continuous position size (quantity_min to quantity_max)
        
    Raises:
        ValueError: If observation is invalid
        
    Example:
        >>> observation = torch.randn(256)
        >>> action, quantity = model.act(observation, instrument_id=5, timeframe_id=2)
        >>> print(f"Action: {action}, Quantity: {quantity:.2f}")
    """
```

##### save_model() / load_model()
```python
def save_model(self, path: str):
    """
    Save model weights and configuration to checkpoint.
    
    Args:
        path: File path for saving (.pth extension recommended)
        
    Saves:
        - model_state_dict: Model parameters
        - config: Model configuration
        - parameter_count: Total parameters
        - architecture: Model architecture identifier
        
    Example:
        >>> model.save_model("models/hrm_checkpoint_epoch_100.pth")
    """

def load_model(self, path: str):
    """
    Load model weights from checkpoint.
    
    Args:
        path: File path to checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint is incompatible
        
    Example:
        >>> model.load_model("models/hrm_checkpoint_epoch_100.pth")
    """
```

## Configuration Management

### ConfigLoader

Manages configuration loading and validation.

```python
class ConfigLoader:
    """Generic configuration loader for settings.yaml"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """Initialize configuration loader."""
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get specific configuration section."""
```

### Configuration Schema

```yaml
# Core model configuration
model:
  observation_dim: 256          # Input feature dimension
  action_dim_discrete: 5        # Number of discrete actions
  model_type: "hrm"            # Model architecture type

# HRM-specific configuration
hierarchical_reasoning_model:
  # H-module (Strategic reasoning)
  h_module:
    hidden_dim: 512             # Strategic reasoning dimension
    num_layers: 4               # Strategic reasoning depth
    n_heads: 8                  # Multi-head attention heads
    ff_dim: 2048                # Feed-forward dimension
    dropout: 0.1                # Dropout rate
  
  # L-module (Tactical execution)
  l_module:
    hidden_dim: 256             # Tactical execution dimension
    num_layers: 3               # Tactical execution depth
    n_heads: 8                  # Multi-head attention heads
    ff_dim: 1024                # Feed-forward dimension
    dropout: 0.1                # Dropout rate
  
  # Input processing
  input_embedding:
    input_dim: 256              # Market feature input dimension
    embedding_dim: 512          # Embedded representation dimension
    dropout: 0.1                # Input dropout rate
  
  # Hierarchical convergence
  hierarchical:
    N_cycles: 3                 # High-level reasoning cycles
    T_timesteps: 5              # Low-level timesteps per cycle
    convergence_threshold: 1e-6  # Convergence detection threshold
  
  # Embeddings for multi-instrument learning
  embeddings:
    instrument_dim: 64          # Instrument embedding dimension
    timeframe_dim: 32           # Timeframe embedding dimension
    max_instruments: 1000       # Maximum tradeable instruments
    max_timeframes: 10          # Maximum timeframes
  
  # Output heads
  output_heads:
    action_dim: 5               # Discrete actions
    quantity_min: 1.0           # Minimum position size
    quantity_max: 100000.0      # Maximum position size
    value_estimation: true      # Enable value head
    q_learning_prep: true       # Enable Q-head for ACT
```

## Training Interface

### Model Training

```python
def train_hrm_model(model, train_dataloader, val_dataloader, config):
    """
    Train HRM model with hierarchical reasoning objectives.
    
    Args:
        model: HierarchicalReasoningModel instance
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
        
    Returns:
        training_history: Dictionary with training metrics
        
    Example:
        >>> history = train_hrm_model(model, train_loader, val_loader, train_config)
        >>> print(f"Best validation Sharpe: {history['best_val_sharpe']:.3f}")
    """
```

### Loss Functions

```python
class HRMLoss(nn.Module):
    """
    Composite loss function for HRM training.
    
    Combines:
    - Policy loss (action prediction)
    - Value loss (state value estimation)
    - Quantity loss (position sizing)
    - Convergence regularization
    """
    
    def forward(self, outputs, targets, convergence_stats):
        """
        Compute composite HRM loss.
        
        Args:
            outputs: Model outputs from forward()
            targets: Target dictionary with action, quantity, value
            convergence_stats: Convergence diagnostics
            
        Returns:
            total_loss: Weighted sum of component losses
            loss_components: Dictionary of individual loss components
        """
```

## Inference API

### Real-time Trading Interface

```python
class HRMTradingInterface:
    """High-level interface for real-time trading with HRM."""
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize trading interface.
        
        Args:
            model_path: Path to trained HRM model
            config_path: Path to configuration file
        """
    
    def generate_signal(self, 
                       market_data: Dict[str, float],
                       instrument: str,
                       timeframe: str) -> Dict[str, Any]:
        """
        Generate trading signal from market data.
        
        Args:
            market_data: Dictionary with OHLCV and technical indicators
            instrument: Trading instrument identifier
            timeframe: Timeframe identifier
            
        Returns:
            signal: Dictionary with:
                - action: Trading action string
                - quantity: Position size
                - confidence: Model confidence (0-1)
                - reasoning: Hierarchical reasoning breakdown
                
        Example:
            >>> interface = HRMTradingInterface("models/hrm_final.pth")
            >>> data = {"close": 2750.50, "volume": 100000, "rsi": 65.2, ...}
            >>> signal = interface.generate_signal(data, "NIFTY", "5m")
            >>> print(f"Action: {signal['action']}, Qty: {signal['quantity']}")
        """
```

### Batch Processing

```python
def batch_inference(model, 
                   data_batch: torch.Tensor,
                   batch_size: int = 32) -> List[Tuple[int, float]]:
    """
    Process batch of observations efficiently.
    
    Args:
        model: Trained HRM model
        data_batch: Batch of market observations
        batch_size: Processing batch size
        
    Returns:
        predictions: List of (action, quantity) tuples
        
    Example:
        >>> predictions = batch_inference(model, large_dataset, batch_size=64)
        >>> actions, quantities = zip(*predictions)
    """
```

## Diagnostic Tools

### Convergence Analysis

```python
def get_convergence_diagnostics(model, 
                              x: torch.Tensor,
                              instrument_ids: Optional[torch.Tensor] = None,
                              timeframe_ids: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Get detailed convergence diagnostics for debugging.
    
    Args:
        model: HRM model
        x: Market features
        instrument_ids: Instrument identifiers (optional)
        timeframe_ids: Timeframe identifiers (optional)
        
    Returns:
        diagnostics: Dictionary with:
            - cycles: Per-cycle convergence information
            - h_module_states: H-module state trajectory
            - l_module_states: L-module state trajectory
            - convergence_metrics: Summary statistics
            - output_statistics: Final output analysis
            - parameter_statistics: Model parameter breakdown
            
    Example:
        >>> x = torch.randn(1, 256)
        >>> diagnostics = model.get_convergence_diagnostics(x)
        >>> print(f"Convergence rate: {diagnostics['convergence_metrics']['cycles_converged']}/3")
    """
```

### Reasoning Pattern Analysis

```python
def analyze_reasoning_patterns(model,
                             market_data_batch: torch.Tensor,
                             num_samples: int = 10) -> Dict[str, Any]:
    """
    Analyze reasoning patterns across multiple market scenarios.
    
    Args:
        model: HRM model
        market_data_batch: Batch of market scenarios
        num_samples: Number of scenarios to analyze
        
    Returns:
        analysis: Dictionary with:
            - decision_consistency: Confidence and consistency metrics
            - reasoning_depth_usage: Convergence utilization patterns
            - convergence_patterns: Convergence behavior analysis
            
    Example:
        >>> batch = torch.randn(100, 256)
        >>> analysis = model.analyze_reasoning_patterns(batch, num_samples=20)
        >>> print(f"Mean confidence: {analysis['decision_consistency']['mean_confidence']:.3f}")
    """
```

## Integration Helpers

### Data Preprocessing

```python
class MarketDataProcessor:
    """Preprocess market data for HRM input."""
    
    def __init__(self, feature_config: Dict[str, Any]):
        """Initialize with feature configuration."""
    
    def process(self, raw_data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert raw market data to HRM input format.
        
        Args:
            raw_data: Dictionary with OHLCV and indicators
            
        Returns:
            processed_tensor: Normalized feature tensor
        """
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
```

### Risk Management Integration

```python
class HRMRiskManager:
    """Risk management wrapper for HRM trading signals."""
    
    def __init__(self, risk_config: Dict[str, Any]):
        """Initialize with risk parameters."""
    
    def validate_signal(self, 
                       action: int, 
                       quantity: float,
                       market_context: Dict[str, Any]) -> Tuple[int, float, bool]:
        """
        Validate and adjust trading signal based on risk rules.
        
        Args:
            action: Proposed action
            quantity: Proposed quantity
            market_context: Current market conditions
            
        Returns:
            validated_action: Risk-adjusted action
            validated_quantity: Risk-adjusted quantity
            approved: Whether signal was approved
        """
```

## Error Handling

### Exception Classes

```python
class HRMError(Exception):
    """Base exception for HRM-related errors."""
    pass

class ConvergenceError(HRMError):
    """Raised when hierarchical convergence fails."""
    pass

class ConfigurationError(HRMError):
    """Raised when configuration is invalid."""
    pass

class ModelLoadError(HRMError):
    """Raised when model loading fails."""
    pass

class InferenceError(HRMError):
    """Raised when inference fails."""
    pass
```

### Error Recovery

```python
def safe_inference(model, observation, max_retries=3):
    """
    Perform inference with automatic error recovery.
    
    Args:
        model: HRM model
        observation: Market observation
        max_retries: Maximum retry attempts
        
    Returns:
        result: (action, quantity) or fallback values
        
    Handles:
        - NaN/Inf in inputs
        - Convergence failures
        - Memory errors
        - Device mismatches
    """
```

## Examples

### Basic Usage

```python
import torch
from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
from src.utils.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader("config/settings.yaml")
config = config_loader.get_config()

# Initialize model
model = HierarchicalReasoningModel(config)

# Load trained weights
model.load_model("models/hrm_production.pth")

# Generate trading signal
market_features = torch.randn(256)  # Your market features
action, quantity = model.act(market_features, instrument_id=5, timeframe_id=2)

print(f"Recommended action: {action}")
print(f"Position size: {quantity:.2f}")
```

### Advanced Usage with Diagnostics

```python
# Enable diagnostic mode
model.eval()

# Process market data with full diagnostics
market_data = torch.randn(1, 256)
outputs, states = model.forward(market_data)

# Get convergence diagnostics
diagnostics = model.get_convergence_diagnostics(market_data)

# Analyze reasoning patterns
print(f"Cycles converged: {diagnostics['convergence_metrics']['cycles_converged']}/3")
print(f"Final H-module norm: {diagnostics['convergence_metrics']['final_h_norm']:.6f}")
print(f"Action entropy: {diagnostics['output_statistics']['action_entropy']:.6f}")

# Log hierarchical reasoning steps (for debugging)
for cycle in diagnostics['cycles']:
    print(f"Cycle {cycle['cycle']}: Convergence = {cycle['convergence_achieved']}")
```

### Production Integration

```python
from src.trading.live_trading_service import LiveTradingService

# Initialize trading service with HRM
trading_service = LiveTradingService(
    model_path="models/hrm_production.pth",
    config_path="config/trading_config.yaml"
)

# Set up real-time data feed
async def handle_market_update(market_data):
    """Process real-time market updates."""
    try:
        # Generate trading signal
        signal = trading_service.generate_signal(market_data)
        
        # Execute trade if signal is valid
        if signal['confidence'] > 0.7:
            await trading_service.execute_trade(signal)
            
    except Exception as e:
        logger.error(f"Trading error: {e}")
        # Fallback to safe state
        await trading_service.emergency_stop()
```

## Performance Considerations

### Memory Optimization

- Use `torch.no_grad()` during inference
- Process in batches for large datasets
- Clear intermediate states when not needed
- Consider mixed precision for large models

### Computational Efficiency

- Cache embedding lookups for repeated instruments
- Use compiled models for production inference
- Monitor convergence patterns to optimize N and T
- Implement early stopping for faster inference

### Monitoring and Maintenance

- Track convergence rates over time
- Monitor inference latency
- Log reasoning patterns for analysis
- Set up alerts for performance degradation

---

*For additional support and examples, see the research documentation and test cases in the repository.*