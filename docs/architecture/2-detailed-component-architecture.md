# 2. Detailed Component Architecture

## 2.1 HRM Core Model (`src/models/hierarchical_reasoning_model.py`)

### 2.1.1 Model Structure

```python
class HierarchicalReasoningModel(nn.Module):
    """
    Brain-inspired HRM with dual-module architecture
    """
    def __init__(self, config):
        # Input embedding network
        self.input_network = InputEmbeddingNetwork(config)
        
        # Dual recurrent modules
        self.h_module = HighLevelModule(config)  # Strategic reasoning
        self.l_module = LowLevelModule(config)   # Tactical execution
        
        # Output heads
        self.policy_head = PolicyHead(config)    # Action type (5 discrete)
        self.quantity_head = QuantityHead(config) # Continuous quantity
        self.value_head = ValueHead(config)      # State value estimation
        self.q_head = QHead(config)              # ACT halting mechanism
        
        # Hierarchical convergence parameters
        self.N = config.hierarchical.N_cycles    # High-level cycles
        self.T = config.hierarchical.T_timesteps # Low-level timesteps per cycle
```

### 2.1.2 Hierarchical Convergence Mechanism

```python
def forward(self, x, z_init=None):
    """
    Execute N cycles of T timesteps each for hierarchical reasoning
    """
    x_embedded = self.input_network(x)
    
    # Initialize hidden states
    z_H, z_L = self.initialize_states(z_init)
    
    # Hierarchical convergence over N cycles
    for cycle in range(self.N):
        # L-module converges within cycle (T timesteps)
        for t in range(self.T):
            z_L = self.l_module(z_L, z_H, x_embedded)
        
        # H-module updates once per cycle using converged L-state
        z_H = self.h_module(z_H, z_L)
        
        # Reset L-module for next cycle's fresh convergence
        if cycle < self.N - 1:
            z_L = self.reset_l_module(z_L)
    
    # Generate outputs from final H-module state
    outputs = {
        'action_type': self.policy_head(z_H),
        'quantity': self.quantity_head(z_H),
        'value': self.value_head(z_H),
        'q_values': self.q_head(z_H)
    }
    
    return outputs, (z_H, z_L)
```

## 2.2 Adaptive Computation Time (ACT) Implementation

### 2.2.1 ACT Decision Framework

```python
class AdaptiveComputationTime:
    """
    Q-learning based halting mechanism for dynamic reasoning depth
    """
    def __init__(self, config):
        self.M_max = config.act.max_segments     # Maximum computation segments
        self.M_min = config.act.min_segments     # Minimum segments (dynamic)
        self.epsilon = config.act.exploration    # Exploration probability
    
    def should_halt(self, q_values, segment_count, market_complexity):
        """
        Determine whether to halt computation based on Q-values and constraints
        """
        q_halt, q_continue = q_values
        
        # Dynamic minimum based on market complexity
        dynamic_min = self.compute_dynamic_min(market_complexity)
        
        # Halt conditions
        if segment_count >= self.M_max:
            return True, "max_segments_reached"
        
        if segment_count >= dynamic_min and q_halt > q_continue:
            return True, "q_learning_decision"
        
        return False, "continue_reasoning"
    
    def compute_dynamic_min(self, market_complexity):
        """
        Adapt minimum segments based on market volatility and complexity
        """
        base_min = 1
        if market_complexity > 0.7:  # High volatility
            return base_min + 2
        elif market_complexity > 0.4:  # Medium volatility
            return base_min + 1
        return base_min
```

### 2.2.2 Market Complexity Detection

```python
class MarketComplexityDetector:
    """
    Detect market complexity for adaptive reasoning depth
    """
    def __init__(self):
        self.volatility_window = 20
        self.trend_strength_window = 14
    
    def compute_complexity(self, market_data):
        """
        Compute normalized market complexity score (0-1)
        """
        volatility_score = self.compute_volatility_score(market_data)
        trend_strength_score = self.compute_trend_strength(market_data)
        momentum_score = self.compute_momentum_score(market_data)
        
        # Weighted complexity score
        complexity = (
            0.4 * volatility_score +
            0.3 * trend_strength_score +
            0.3 * momentum_score
        )
        
        return np.clip(complexity, 0.0, 1.0)
```

## 2.3 Deep Supervision Training Pipeline

### 2.3.1 One-Step Gradient Approximation

```python
class DeepSupervisionTrainer:
    """
    O(1) memory training with segment-based supervision
    """
    def __init__(self, model, config):
        self.model = model
        self.max_segments = config.training.max_supervision_segments
        self.optimizer = torch.optim.AdamW(model.parameters())
    
    def train_step(self, batch):
        """
        Deep supervision training with detached segment progression
        """
        total_loss = 0.0
        
        for x, y_true in batch:
            z = self.initialize_hidden_state()
            
            # Multi-segment supervision
            for segment in range(self.max_segments):
                # Forward pass through HRM
                with torch.no_grad() if segment > 0 else nullcontext():
                    if segment < self.max_segments - 1:
                        # Detached forward pass (no gradients)
                        outputs, z = self.model(x, z)
                        z = z.detach()  # Break gradient flow
                    else:
                        # Final segment with gradients
                        outputs, z = self.model(x, z)
                
                # Compute segment loss
                segment_loss = self.compute_loss(outputs, y_true)
                
                if segment == self.max_segments - 1:
                    # Only final segment contributes to gradients
                    total_loss += segment_loss
                
                # ACT halting decision
                if self.should_halt_training(outputs, segment):
                    break
        
        # Single backward pass (O(1) memory)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()
```

## 2.4 HRM Agent Implementation

### 2.4.1 Agent Interface Compatibility

```python
class HRMAgent(BaseAgent):
    """
    HRM-based trading agent maintaining PPO interface compatibility
    """
    def __init__(self, observation_dim, action_dim_discrete, action_dim_continuous, config):
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim_discrete = action_dim_discrete
        self.action_dim_continuous = action_dim_continuous
        
        # HRM model with dual-module architecture
        self.hrm_model = HierarchicalReasoningModel(config)
        self.act_mechanism = AdaptiveComputationTime(config)
        self.complexity_detector = MarketComplexityDetector()
        
        # Training components
        self.trainer = DeepSupervisionTrainer(self.hrm_model, config)
        
        # Inference optimization
        self.inference_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hrm_model.to(self.device)
    
    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Select trading action using HRM with adaptive computation
        Maintains PPO interface: returns (action_type, quantity)
        """
        # Input validation and preprocessing
        if not np.isfinite(observation).all():
            logger.warning("Invalid observation detected, using safe default")
            return 4, 1.0  # HOLD action
        
        # Convert to tensor
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Market complexity analysis
        market_complexity = self.complexity_detector.compute_complexity(observation)
        
        # Initialize hidden state
        z_init = self.initialize_hidden_state()
        
        # Adaptive computation with ACT
        segment_count = 0
        z = z_init
        
        while segment_count < self.act_mechanism.M_max:
            # HRM forward pass
            outputs, z = self.hrm_model(state, z)
            segment_count += 1
            
            # ACT halting decision
            should_halt, reason = self.act_mechanism.should_halt(
                outputs['q_values'], segment_count, market_complexity
            )
            
            if should_halt:
                logger.debug(f"Halted after {segment_count} segments: {reason}")
                break
        
        # Extract actions from final outputs
        action_probs = torch.softmax(outputs['action_type'], dim=-1)
        action_type = torch.multinomial(action_probs, 1).item()
        
        # Continuous quantity with constraints
        quantity = torch.clamp(outputs['quantity'], min=1.0, max=100000.0).item()
        
        return action_type, quantity
    
    def learn(self, experiences: List[Tuple]) -> None:
        """
        Learn from experiences using deep supervision training
        Maintains PPO interface for service compatibility
        """
        if not experiences:
            return
        
        # Convert experiences to HRM training format
        training_batch = self.prepare_training_batch(experiences)
        
        # Deep supervision training
        loss = self.trainer.train_step(training_batch)
        
        logger.debug(f"HRM training loss: {loss:.6f}")
    
    def save_model(self, path: str) -> None:
        """Save HRM model state"""
        torch.save({
            'hrm_state_dict': self.hrm_model.state_dict(),
            'config': self.hrm_model.config,
            'training_step': self.trainer.step_count
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load HRM model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.hrm_model.load_state_dict(checkpoint['hrm_state_dict'])
        logger.info(f"Loaded HRM model from {path}")
```

---
