# Requirements

## Functional Requirements (FR)

**FR1: PPO Architecture Removal**

- FR1.1: Delete all PPO-related files and components
- FR1.2: Remove PPO agent implementation and dependencies
- FR1.3: Clean up PPO-specific configuration parameters
- FR1.4: Archive existing PPO models for reference

**FR2: HRM Core Architecture**

- FR2.1: Implement HRM dual-module architecture with H-module (strategic) and L-module (tactical)
- FR2.2: Create hierarchical convergence mechanism for unlimited computational depth
- FR2.3: Implement instrument and timeframe embeddings for multi-instrument learning
- FR2.4: Design unified output heads compatible with existing 5-action trading space

**FR3: Adaptive Computation Time (ACT)**

- FR3.1: Implement Q-learning based halting mechanism for dynamic computation
- FR3.2: Create market complexity detection for adaptive reasoning depth
- FR3.3: Develop deep supervision training with segment-based learning
- FR3.4: Configure ACT parameters through settings.yaml (max segments, min segments)

**FR4: HRM Training Pipeline**

- FR4.1: Implement one-step gradient approximation replacing BPTT
- FR4.2: Create deep supervision training with O(1) memory complexity
- FR4.3: Support multi-segment training on diverse trading scenarios
- FR4.4: Maintain model checkpointing with hierarchical state preservation

**FR5: Configuration Management**

- FR5.1: Extend settings.yaml with HRM dual-module parameters
- FR5.2: Configure hierarchical convergence settings (N cycles, T timesteps)
- FR5.3: Support ACT mechanism configuration (M_max, M_min, epsilon)
- FR5.4: Remove PPO-specific configuration sections and add HRM parameters

**FR6: Integration & Compatibility**

- FR6.1: Maintain existing API interfaces for live trading service
- FR6.2: Preserve existing action space (5 discrete actions + continuous quantity)
- FR6.3: Maintain compatibility with current data pipeline
- FR6.4: Update all training and inference scripts

## Non-Functional Requirements (NFR)

**NFR1: Performance**

- NFR1.1: Inference latency must remain under 50ms for live trading (improved from 100ms)
- NFR1.2: Support real-time hierarchical reasoning with adaptive computation depth
- NFR1.3: Memory usage should not exceed 4GB during training (O(1) complexity advantage)
- NFR1.4: Model size should remain under 100MB for deployment (27M parameters vs billions)

**NFR2: Reliability**

- NFR2.1: Training must be deterministic and reproducible
- NFR2.2: Model must gracefully handle missing or invalid market data
- NFR2.3: System must maintain 99.9% uptime during live trading
- NFR2.4: Implement comprehensive error handling and recovery mechanisms

**NFR3: Maintainability**

- NFR3.1: Code must follow existing project architecture patterns
- NFR3.2: All components must have comprehensive unit test coverage (>90%)
- NFR3.3: Configuration changes must not require code modifications
- NFR3.4: Documentation must be updated for all new components

**NFR4: Scalability**

- NFR4.1: Support training on multiple instruments with shared hierarchical representations
- NFR4.2: Architecture must support future expansion to more asset classes and timeframes
- NFR4.3: Model must handle variable computation depth through ACT mechanism
- NFR4.4: Support efficient training on large datasets with one-step gradient approximation
