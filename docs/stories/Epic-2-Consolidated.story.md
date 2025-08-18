---
Status: Ready for Development
Epic: 2
Duration: 2-3 weeks
Dependencies: Epic 1 (clean codebase)
---

# Epic 2: HRM Core Architecture - Consolidated Implementation Story

## Epic Overview

**As a** developer,
**I want to** implement the complete foundational HRM dual-module architecture with hierarchical convergence,
**so that** we have a brain-inspired trading system capable of unlimited computational depth and superior performance with only 27M parameters.

## Epic Goal

Implement the foundational HRM model with hierarchical dual-module architecture that replaces the PPO system with a revolutionary brain-inspired approach featuring H-module (strategic) and L-module (tactical) reasoning.

## Key Requirements from PRD

### Functional Requirements (FR2: HRM Core Architecture)
- **FR2.1**: Implement HRM dual-module architecture with H-module (strategic) and L-module (tactical)
- **FR2.2**: Create hierarchical convergence mechanism for unlimited computational depth
- **FR2.3**: Implement instrument and timeframe embeddings for multi-instrument learning
- **FR2.4**: Design unified output heads compatible with existing 5-action trading space

### Performance Requirements
- **Inference latency**: Must remain under 50ms for live trading (improved from 100ms)
- **Model size**: Must remain under 100MB for deployment (27M parameters vs billions)
- **Memory usage**: Should not exceed 4GB during training (O(1) complexity advantage)

## Consolidated Acceptance Criteria

### Core HRM Model Implementation
1. **Complete rewrite** of `src/models/hierarchical_reasoning_model.py` following cutting-edge HRM architecture
2. **Implement H-module** (High-Level/Strategic) with recurrent architecture for strategic reasoning
3. **Implement L-module** (Low-Level/Tactical) with recurrent architecture for tactical execution
4. **Add hierarchical convergence mechanism** supporting unlimited computational depth
5. **Support configurable N cycles and T timesteps** through settings.yaml integration
6. **Create input embedding network** for market data preprocessing
7. **Implement dual-module communication** where L-module receives guidance from H-module

### Instrument and Timeframe Embeddings
8. **Create instrument embedding layer** (64 dimensions) for multi-instrument learning
9. **Create timeframe embedding layer** (32 dimensions) for different trading timeframes
10. **Implement embedding lookup and concatenation** with input features
11. **Support dynamic vocabulary expansion** for new instruments and timeframes
12. **Integrate embeddings** with the dual-module architecture

### Hierarchical Convergence Mechanism
13. **Implement L-module convergence** within each cycle (T timesteps)
14. **Create H-module updates** based on L-module final states after each cycle
15. **Add reset mechanism** for L-module between cycles for fresh convergence
16. **Configure convergence parameters** through settings.yaml (N cycles, T timesteps)
17. **Implement state initialization** and hidden state management
18. **Add convergence monitoring** and early stopping mechanisms

### Unified Output Heads
19. **Create policy head** for action type (5 discrete actions: BUY, SELL, HOLD, etc.)
20. **Implement quantity prediction head** (continuous) for position sizing
21. **Add value estimation head** for state values (strategic planning)
22. **Implement Q-head** for ACT halting mechanism preparation
23. **Ensure output compatibility** with existing action space and trading services
24. **Add output validation** and constraint enforcement

### Integration and Compatibility
25. **Maintain existing API interfaces** for live trading service compatibility
26. **Preserve existing action space** (5 discrete actions + continuous quantity)
27. **Maintain compatibility** with current data pipeline and feature engineering
28. **Implement error handling** and graceful degradation mechanisms
29. **Add comprehensive logging** for debugging and monitoring
30. **Create model serialization** methods compatible with existing infrastructure

## Tasks / Subtasks

### Task 1: Study HRM Research Paper and Architecture Design
- [ ] **Subtask 1.1**: Read `C:\AlgoTrading\hrm-research-paper.txt` thoroughly to understand cutting-edge HRM architecture
- [ ] **Subtask 1.2**: Study brain-inspired dual-module design principles from research paper
- [ ] **Subtask 1.3**: Understand hierarchical convergence mechanism and unlimited computational depth concepts
- [ ] **Subtask 1.4**: Review 27M parameter efficiency claims and architectural innovations
- [ ] **Subtask 1.5**: Plan implementation strategy based on research paper insights

### Task 2: Complete HRM Core Model Implementation
- [ ] **Subtask 2.1**: Create `HierarchicalReasoningModel` class with proper PyTorch inheritance
- [ ] **Subtask 2.2**: Implement `InputEmbeddingNetwork` for market data preprocessing
- [ ] **Subtask 2.3**: Implement `HighLevelModule` (H-module) with recurrent architecture for strategic reasoning
- [ ] **Subtask 2.4**: Implement `LowLevelModule` (L-module) with recurrent architecture for tactical execution
- [ ] **Subtask 2.5**: Add hierarchical convergence logic in forward() method with N cycles and T timesteps
- [ ] **Subtask 2.6**: Implement state initialization and reset mechanisms
- [ ] **Subtask 2.7**: Add configuration loading from settings.yaml for all HRM parameters

### Task 3: Implement Embedding Systems
- [ ] **Subtask 3.1**: Create `InstrumentEmbedding` class (64 dimensions) with vocabulary management
- [ ] **Subtask 3.2**: Create `TimeframeEmbedding` class (32 dimensions) for different trading timeframes
- [ ] **Subtask 3.3**: Implement embedding lookup tables with dynamic expansion capability
- [ ] **Subtask 3.4**: Add embedding concatenation logic with input features
- [ ] **Subtask 3.5**: Create embedding initialization strategies (Xavier/He initialization)
- [ ] **Subtask 3.6**: Add embedding validation and error handling

### Task 4: Implement Hierarchical Convergence Mechanism
- [ ] **Subtask 4.1**: Design L-module convergence loop within each cycle (T timesteps)
- [ ] **Subtask 4.2**: Implement H-module update mechanism based on converged L-module states
- [ ] **Subtask 4.3**: Add L-module reset functionality between cycles for fresh convergence
- [ ] **Subtask 4.4**: Create convergence monitoring with metrics and early stopping
- [ ] **Subtask 4.5**: Implement hidden state management and memory optimization
- [ ] **Subtask 4.6**: Add cycle and timestep configuration validation

### Task 5: Create Unified Output Heads
- [ ] **Subtask 5.1**: Implement `PolicyHead` for discrete action prediction (5 actions)
- [ ] **Subtask 5.2**: Implement `QuantityHead` for continuous quantity prediction with constraints
- [ ] **Subtask 5.3**: Implement `ValueHead` for state value estimation
- [ ] **Subtask 5.4**: Implement `QHead` for ACT halting mechanism (future Epic 5 preparation)
- [ ] **Subtask 5.5**: Add output validation, constraint enforcement, and numerical stability
- [ ] **Subtask 5.6**: Ensure compatibility with existing action space expectations

### Task 6: Integration and Compatibility
- [ ] **Subtask 6.1**: Update model loading/saving methods compatible with existing infrastructure
- [ ] **Subtask 6.2**: Add comprehensive error handling and fallback mechanisms
- [ ] **Subtask 6.3**: Implement detailed logging for debugging hierarchical reasoning
- [ ] **Subtask 6.4**: Create unit tests for all HRM components and convergence mechanisms
- [ ] **Subtask 6.5**: Validate integration with existing data pipeline and feature formats
- [ ] **Subtask 6.6**: Document API interfaces and usage patterns for future integration

## Implementation Architecture Reference

### From Technical Architecture Document

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

### Key Design Principles from Research Paper

1. **Brain-inspired Architecture**: H-module (strategic/cortical) and L-module (tactical/subcortical) mimic brain's hierarchical reasoning
2. **Unlimited Computational Depth**: Hierarchical convergence allows arbitrarily deep reasoning without premature convergence
3. **Revolutionary Efficiency**: 27M parameters achieving performance of billion-parameter models through architectural innovation
4. **Adaptive Computation**: Dynamic reasoning depth based on problem complexity (foundation for Epic 5)
5. **O(1) Memory Complexity**: One-step gradient approximation for efficient training (Epic 4)

## Configuration Requirements

### Expected settings.yaml Extension

```yaml
hierarchical_reasoning_model:
  # Dual-module architecture
  h_module:
    hidden_dim: 512
    num_layers: 4
    dropout: 0.1
  
  l_module:
    hidden_dim: 256
    num_layers: 3
    dropout: 0.1
  
  # Hierarchical convergence
  hierarchical:
    N_cycles: 3          # High-level reasoning cycles
    T_timesteps: 5       # Low-level timesteps per cycle
    convergence_threshold: 1e-6
    max_convergence_steps: 100
  
  # Embeddings
  embeddings:
    instrument_dim: 64
    timeframe_dim: 32
    max_instruments: 1000
    max_timeframes: 10
  
  # Output heads
  output_heads:
    action_dim: 5        # BUY, SELL, HOLD, etc.
    quantity_min: 1.0
    quantity_max: 100000.0
    value_estimation: true
    q_learning_prep: true  # For future ACT implementation
```

## Dev Notes

### Critical Implementation Guidelines

1. **Study Research Paper First**: The developer MUST read `hrm-research-paper.txt` to understand the cutting-edge architecture before implementation
2. **Brain-inspired Design**: Follow the biological inspiration - H-module for strategic planning, L-module for tactical execution
3. **Parameter Efficiency**: Target 27M parameters total - use efficient architectures, not brute force scaling
4. **Hierarchical Convergence**: Implement true unlimited computational depth through the N cycles × T timesteps mechanism
5. **Future-proofing**: Design interfaces for Epic 5 (ACT) and Epic 6 (Agent Integration)

### Performance Optimization

From `C:\AlgoTrading\docs\architecture\4-performance-optimization.md`:
- Target <50ms inference latency (50% improvement over PPO)
- Implement memory-efficient hidden state management
- Use gradient checkpointing if needed for large models
- Profile critical paths and optimize bottlenecks

### Testing Strategy

From `C:\AlgoTrading\docs\architecture\8-testing-strategy.md`:
- Unit tests for each module (H-module, L-module, embeddings, output heads)
- Integration tests for hierarchical convergence mechanism
- Performance benchmarks against PPO baseline
- Memory usage validation tests

### API Compatibility

From `C:\AlgoTrading\docs\architecture\7-api-compatibility.md`:
- Maintain existing model.forward() interface patterns
- Preserve action space compatibility (5 discrete + continuous)
- Ensure model.save_model() and model.load_model() compatibility
- Keep inference API consistent for LiveTradingService integration

## Success Criteria

### Technical Success
- [ ] **HRM model passes all unit tests** with >95% code coverage
- [ ] **Hierarchical convergence works correctly** with configurable N and T parameters
- [ ] **Memory usage stays under 4GB** during training and inference
- [ ] **Inference latency under 50ms** on target hardware
- [ ] **Model size under 100MB** (27M parameters achieved)

### Functional Success
- [ ] **All output heads produce valid results** within expected ranges
- [ ] **Embeddings work correctly** for multiple instruments and timeframes
- [ ] **Configuration loading works** from settings.yaml without errors
- [ ] **Error handling prevents crashes** under edge conditions
- [ ] **Logging provides sufficient detail** for debugging and monitoring

### Integration Success
- [ ] **Compatible with existing data pipeline** and feature engineering
- [ ] **Model serialization works** with current infrastructure
- [ ] **API interfaces maintained** for service compatibility
- [ ] **No breaking changes** to downstream consumers
- [ ] **Ready for Epic 5 (ACT) integration** with Q-head implementation

## Epic Dependencies

### Input Dependencies (Epic 1)
- PPO architecture completely removed
- Clean codebase without PPO references
- Base agent interfaces updated/verified

### Output Dependencies (Future Epics)
- **Epic 3**: Configuration system will extend settings.yaml with HRM parameters
- **Epic 4**: Deep supervision training will use this HRM architecture
- **Epic 5**: ACT mechanism will build on Q-head and hierarchical convergence
- **Epic 6**: HRM Agent will wrap this model for service integration

## Change Log

- **Date**: 2025-08-18, **Version**: 1.0, **Description**: Consolidated Epic 2 story from PRD and Architecture docs, **Author**: AI Story Creator

## Research Paper Requirement

🔬 **CRITICAL**: Developer must read `C:\AlgoTrading\hrm-research-paper.txt` thoroughly before beginning implementation. This paper contains the cutting-edge AI architecture details essential for proper HRM implementation that will work effectively with the algorithmic trading bot.

The HRM architecture represents a breakthrough in neural network design with brain-inspired dual-module processing, unlimited computational depth, and revolutionary parameter efficiency. Understanding these concepts from the research paper is essential for successful implementation.

---

*This consolidated story represents the complete Epic 2 implementation requirements, combining all individual stories (2.1-2.4) into a single comprehensive development guide. The developer should implement all components as an integrated system following the brain-inspired HRM architecture principles from the research paper.*