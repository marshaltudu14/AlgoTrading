# Product Requirements Document: PPO to Hierarchical Reasoning Model (HRM) Migration

## Goals and Background Context

### Project Name

PPO to Hierarchical Reasoning Model (HRM) Migration for Algorithmic Trading System

### Goals

1. **Replace PPO Architecture**: Transition from online PPO reinforcement learning to brain-inspired Hierarchical Reasoning Model (HRM)
2. **Hierarchical Multi-Timescale Processing**: Implement H-module (strategic) and L-module (tactical) for natural trading hierarchy
3. **Revolutionary Efficiency**: Achieve superior performance with only 27M parameters vs traditional billion-parameter models
4. **Adaptive Computation Time**: Dynamic reasoning depth based on market complexity and volatility
5. **Internal Reasoning**: Enable latent reasoning without Chain-of-Thought overhead for real-time trading decisions

### Background

The current algorithmic trading system uses PPO (Proximal Policy Optimization) with transformer-based Actor-Critic models. While functional, this approach has limitations:

- Limited computational depth due to fixed architecture
- Massive parameter requirements for reasonable performance
- Online learning instability and sample inefficiency
- Lack of hierarchical reasoning across trading timescales
- Missing adaptive computation for varying market complexity

Hierarchical Reasoning Model (HRM) represents a breakthrough in neural architecture, offering brain-inspired hierarchical processing, unlimited computational depth through hierarchical convergence, and revolutionary efficiency (27M parameters achieving performance of billion-parameter models). HRM's two-module design naturally maps to trading's strategic/tactical decision hierarchy.

### Change Log

- **v1.0**: Initial PPO implementation with transformer models
- **v2.0** (Proposed): Hierarchical Reasoning Model (HRM) with brain-inspired dual-module architecture

## Requirements

### Functional Requirements (FR)

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

### Non-Functional Requirements (NFR)

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

## User Interface Design Goals

### UX Vision

The HRM migration is a **backend-only architectural change** with **no frontend UI modifications**. All interactions remain through terminal-based commands and configuration files. The focus is on seamless CLI experience and enhanced terminal output showcasing HRM's adaptive reasoning capabilities.

### Interaction Paradigms

**Primary Paradigm: CLI-First Configuration**

- Settings.yaml-driven model configuration
- Terminal-based training and monitoring commands
- Command-line model management and validation

**Secondary Paradigm: Enhanced Terminal Output**

- Rich terminal feedback during training
- Detailed logging and progress indicators
- Improved error messages and debugging information

### Core Terminal Interfaces

**Training Commands:**

```bash
python run_training.py              # Now uses HRM with deep supervision
python run_unified_backtest.py      # Enhanced with HRM hierarchical reasoning
python run_live_bot.py              # HRM-powered adaptive live trading
```

**Configuration Interface:**

- `config/settings.yaml` - Primary configuration file
- Enhanced parameter validation and error reporting
- Clear documentation for HRM dual-module and ACT settings

**Monitoring Interface:**

- Enhanced terminal logging during training
- Progress bars for trajectory processing
- Detailed performance metrics output
- Model checkpointing status updates

### Terminal Output Enhancements

- Clear HRM training progress with hierarchical convergence metrics
- Adaptive computation time status indicators
- Memory usage and performance metrics (O(1) efficiency)
- Enhanced error messages with HRM-specific diagnostics

### No UI/UX Changes Required

- Next.js frontend remains unchanged
- All HRM functionality accessible via existing CLI
- Backend API interfaces preserved for frontend compatibility
- Focus on terminal user experience improvements only

## Technical Assumptions

### Repository Structure

- **Preserved Architecture**: Current service-oriented backend structure remains intact
- **New Components**: HRM models added to `src/models/` directory
- **Agent Replacement**: `src/agents/ppo_agent.py` replaced with `src/agents/hrm_agent.py`
- **Configuration Extension**: `config/settings.yaml` extended with HRM dual-module and ACT parameters
- **Model Storage**: New model files stored in `models/` directory alongside existing structure

### Service Architecture

- **API Compatibility**: FastAPI endpoints remain unchanged for frontend compatibility
- **Training Services**: `src/training/` modules updated to support deep supervision training
- **Live Trading**: `src/trading/live_trading_service.py` maintains same interface with HRM backend
- **Backtesting**: `src/trading/backtest_service.py` updated for HRM hierarchical evaluation
- **Data Pipeline**: `src/data_processing/` enhanced to support multi-segment training data

### Technology Stack

- **Python 3.11+**: Current Python version maintained
- **PyTorch**: Continue using PyTorch for model implementation
- **FastAPI**: Backend API framework unchanged
- **Dependencies**: New requirements for HRM (specialized recurrent modules, ACT mechanism, one-step gradient)
- **Testing Framework**: Pytest remains primary testing framework

### Data Pipeline Assumptions

- **Historical Data**: Existing OHLC data format compatible with multi-segment training
- **Hierarchical Processing**: Dual-module reasoning feasible with current data diversity
- **Storage Requirements**: Reduced storage needs due to O(1) memory complexity
- **Memory Constraints**: 8GB RAM sufficient for training and inference

### Model Training Assumptions

- **Deep Supervision Training**: Historical data sufficient for segment-based training
- **Compute Resources**: Current hardware more than adequate for efficient HRM training (27M params)
- **Training Time**: Acceptable training duration (hours vs days for convergence)
- **Data Quality**: Existing market data quality sufficient for sequence modeling

### Integration Assumptions

- **Backward Compatibility**: Existing model loading/saving interfaces adaptable
- **API Contracts**: Current trading API contracts preserved
- **Configuration Migration**: Settings.yaml structure extensible without breaking changes
- **Testing Coverage**: Existing test suite adaptable to Decision Transformer architecture

### Performance Assumptions

- **Inference Speed**: Hierarchical reasoning processable within 50ms latency requirement
- **Memory Usage**: Dramatically reduced model size and memory footprint (27M vs billions of parameters)
- **Scalability**: Architecture supports future multi-instrument training
- **Hardware Compatibility**: Current deployment hardware sufficient

### Risk Mitigation Assumptions

- **Rollback Capability**: Current PPO models preserved as fallback option during transition
- **Validation Framework**: Existing backtesting framework adequate for DT validation
- **Monitoring Systems**: Current logging and monitoring sufficient for DT operations
- **Error Handling**: Existing error handling patterns adaptable to new architecture

## Epic List

### Epic 1: PPO Architecture Removal and Cleanup

**Objective**: Complete removal of PPO-related code and preparation for HRM implementation
**Duration**: 1 week
**Dependencies**: None
**Key Deliverables**:

- Delete all PPO agent files and references
- Remove PPO-specific configuration parameters
- Archive existing PPO models for reference
- Clean codebase preparation for HRM implementation

### Epic 2: HRM Core Architecture

**Objective**: Implement the foundational HRM dual-module architecture with hierarchical convergence
**Duration**: 2-3 weeks
**Dependencies**: Epic 1 (clean codebase)
**Key Deliverables**:

- Core HRM model with H-module and L-module
- Hierarchical convergence mechanism implementation
- Instrument and timeframe embedding layers
- Unified output heads for trading actions

### Epic 3: Configuration System Enhancement

**Objective**: Extend settings.yaml with HRM dual-module and ACT parameters and validation
**Duration**: 1 week
**Dependencies**: Epic 2 (architecture understanding)
**Key Deliverables**:

- Enhanced settings.yaml with HRM configuration sections
- Configuration validation and error handling
- Parameter documentation and examples
- Remove legacy PPO configuration sections

### Epic 4: Deep Supervision Training Pipeline

**Objective**: Implement HRM deep supervision training with one-step gradient approximation
**Duration**: 2 weeks
**Dependencies**: Epic 3 (configuration system)
**Key Deliverables**:

- Deep supervision training implementation
- One-step gradient approximation (O(1) memory)
- Multi-segment training data processing
- Training data validation for HRM format

### Epic 5: Adaptive Computation Time (ACT) Implementation

**Objective**: Implement ACT mechanism for dynamic reasoning depth based on market complexity
**Duration**: 2-3 weeks
**Dependencies**: Epic 2 (core architecture), Epic 4 (training pipeline)
**Key Deliverables**:

- Q-learning based halting mechanism
- Market complexity detection algorithms
- Dynamic computation depth adjustment
- ACT training and inference optimization

### Epic 6: HRM Agent Implementation and Integration

**Objective**: Implement HRMAgent and integrate with existing services
**Duration**: 2 weeks
**Dependencies**: Epic 2 (core architecture), Epic 5 (ACT implementation)
**Key Deliverables**:

- HRMAgent class implementation with dual-module reasoning
- Integration with LiveTradingService and BacktestService
- Hierarchical action selection and inference optimization
- Error handling and fallback mechanisms

### Epic 7: Testing and Validation Framework

**Objective**: Comprehensive testing suite for HRM implementation
**Duration**: 1-2 weeks
**Dependencies**: Epic 6 (agent implementation)
**Key Deliverables**:

- Unit tests for all HRM components
- Integration tests for training and inference
- Performance benchmarking suite
- Backtesting validation framework

### Epic 8: Documentation and Migration Guide

**Objective**: Complete documentation for HRM system and migration process
**Duration**: 1 week
**Dependencies**: Epic 7 (testing completion)
**Key Deliverables**:

- Technical documentation for HRM architecture
- Configuration guide and examples
- Migration procedures and best practices
- Troubleshooting guide and FAQ

### Epic 9: Production Deployment and Monitoring

**Objective**: Deploy HRM to production and establish monitoring
**Duration**: 1 week
**Dependencies**: Epic 8 (documentation complete)
**Key Deliverables**:

- Production deployment procedures
- Enhanced monitoring and alerting
- Performance tracking and optimization
- Rollback procedures and contingency plans

**Total Estimated Duration**: 11-15 weeks
**Critical Path**: Epic 1 → Epic 2 → Epic 4 → Epic 5 → Epic 6 → Epic 7 → Epic 9

## Epic Details

### Epic 1: PPO Architecture Removal and Cleanup

**Epic Goal**: Complete removal of PPO-related code and preparation for HRM implementation

**User Stories**:

**Story 1.1**: Remove PPO Agent Implementation

- **As a** developer
- **I want to** delete the PPO agent and related files
- **So that** the codebase is clean for HRM implementation
- **Acceptance Criteria**:
  - [ ] Delete `src/agents/ppo_agent.py`
  - [ ] Remove PPO imports from all files
  - [ ] Update base agent interfaces if needed
  - [ ] Verify no broken references remain

**Story 1.2**: Clean PPO Configuration

- **As a** developer
- **I want to** remove PPO-specific configuration parameters
- **So that** settings.yaml is prepared for HRM parameters
- **Acceptance Criteria**:
  - [ ] Remove PPO training parameters from settings.yaml
  - [ ] Archive PPO configuration for reference
  - [ ] Update configuration loading code
  - [ ] Validate configuration parsing still works

**Story 1.3**: Archive PPO Models

- **As a** developer
- **I want to** safely archive existing PPO models
- **So that** we can reference or rollback if needed
- **Acceptance Criteria**:
  - [ ] Move existing PPO models to archive directory
  - [ ] Document model versions and performance metrics
  - [ ] Update model loading paths
  - [ ] Create rollback procedure documentation

**Story 1.4**: Update Training Scripts

- **As a** developer
- **I want to** remove PPO references from training scripts
- **So that** scripts are ready for HRM integration
- **Acceptance Criteria**:
  - [ ] Update `run_training.py` to remove PPO initialization
  - [ ] Clean `src/training/` modules of PPO references
  - [ ] Remove PPO-specific training loops
  - [ ] Ensure scripts can run without errors (with placeholder logic)

### Epic 2: HRM Core Architecture

**Epic Goal**: Implement the foundational HRM model with hierarchical dual-module architecture

**User Stories**:

**Story 2.1**: Core HRM Model

- **As a** developer
- **I want to** implement the base HRM dual-module architecture
- **So that** we have the foundation for hierarchical trading decisions
- **Acceptance Criteria**:
  - [ ] Create `src/models/hierarchical_reasoning_model.py`
  - [ ] Implement H-module and L-module with recurrent architecture
  - [ ] Add hierarchical convergence mechanism
  - [ ] Support configurable N cycles and T timesteps

**Story 2.2**: Instrument and Timeframe Embeddings

- **As a** developer
- **I want to** implement embedding layers for instruments and timeframes
- **So that** the model can learn shared patterns across different trading contexts
- **Acceptance Criteria**:
  - [ ] Create instrument embedding layer (64 dimensions)
  - [ ] Create timeframe embedding layer (32 dimensions)
  - [ ] Implement embedding lookup and concatenation
  - [ ] Support dynamic vocabulary expansion

**Story 2.3**: Hierarchical Convergence Mechanism

- **As a** developer
- **I want to** implement hierarchical convergence for unlimited computational depth
- **So that** the model can perform deep reasoning without premature convergence
- **Acceptance Criteria**:
  - [ ] Implement L-module convergence within each cycle
  - [ ] Create H-module updates based on L-module final states
  - [ ] Add reset mechanism for L-module between cycles
  - [ ] Configure convergence parameters through settings

**Story 2.4**: Policy and Value Heads

- **As a** developer
- **I want to** implement unified output heads for actions and values
- **So that** the model can generate trading decisions and value estimates
- **Acceptance Criteria**:
  - [ ] Create policy head for action type (5 discrete actions)
  - [ ] Implement quantity prediction head (continuous)
  - [ ] Add value estimation head for state values
  - [ ] Ensure output compatibility with existing action space

### Epic 3: Configuration System Enhancement

**Epic Goal**: Extend settings.yaml with HRM dual-module and ACT parameters and validation

**User Stories**:

**Story 3.1**: HRM Configuration Schema

- **As a** developer
- **I want to** add comprehensive HRM configuration to settings.yaml
- **So that** all model parameters are configurable without code changes
- **Acceptance Criteria**:
  - [ ] Add hierarchical_reasoning_model section to settings.yaml
  - [ ] Include dual-module architecture parameters (N, T, dimensions)
  - [ ] Add ACT mechanism configuration (M_max, M_min, epsilon)
  - [ ] Include deep supervision training hyperparameters

**Story 3.2**: Configuration Validation

- **As a** developer
- **I want to** implement robust configuration validation
- **So that** invalid parameters are caught early with helpful error messages
- **Acceptance Criteria**:
  - [ ] Validate parameter ranges and types
  - [ ] Check parameter compatibility and dependencies
  - [ ] Provide clear error messages for invalid configurations
  - [ ] Support configuration schema documentation

**Story 3.3**: Parameter Documentation

- **As a** developer
- **I want to** document all HRM parameters
- **So that** users understand configuration options and their effects
- **Acceptance Criteria**:
  - [ ] Add inline comments for all HRM parameters
  - [ ] Create configuration examples for different trading scenarios
  - [ ] Document hierarchical convergence and ACT trade-offs
  - [ ] Include efficiency optimization guidelines

## Checklist Results Report

### PM Checklist Validation

**Requirements Clarity**: ✅ PASS

- All functional and non-functional requirements clearly defined
- Requirements mapped to specific epics and stories
- Acceptance criteria specified for each user story

**Technical Feasibility**: ✅ PASS

- Architecture builds on existing transformer foundation
- No fundamental technology stack changes required
- Performance requirements within reasonable bounds

**Scope Management**: ✅ PASS

- Clear boundaries established (backend-only, no UI changes)
- PPO removal scope clearly defined
- Integration points identified and preserved

**Risk Assessment**: ✅ PASS

- Technical risks identified and mitigation strategies planned
- Rollback capabilities preserved through model archival
- Performance assumptions documented for validation

**Timeline Realism**: ⚠️ CAUTION

- 11-15 week estimate may be optimistic for single developer
- Dependencies clearly mapped but critical path is lengthy
- Consider parallel development opportunities

**Success Metrics**: ✅ PASS

- Performance benchmarks defined (latency, memory, accuracy)
- Testing framework includes comprehensive validation
- Monitoring and observability addressed

## Next Steps

### Immediate Actions (Week 1)

1. **Epic 1 Execution**: Begin PPO architecture removal
2. **Environment Setup**: Prepare development environment for Decision Transformer dependencies
3. **Baseline Establishment**: Document current system performance metrics
4. **Team Alignment**: Review PRD with all stakeholders and confirm scope

### UX Expert Prompt

_Not applicable - backend-only migration with no UI changes_

### Architect Prompt

"Review the proposed HRM architecture for the following considerations:

- Validate hierarchical convergence performance assumptions and O(1) memory efficiency
- Assess computational requirements for dual-module reasoning and ACT mechanism
- Confirm integration strategy with existing FastAPI services and trading interfaces
- Evaluate deep supervision training data requirements and storage implications
- Review configuration system extensibility for future HRM enhancements"

### Success Criteria

- **Performance**: Achieve <50ms inference latency with hierarchical reasoning (50% improvement)
- **Efficiency**: Demonstrate 27M parameter model outperforming traditional billion-parameter approaches
- **Compatibility**: Zero changes required to existing frontend or API contracts
- **Reliability**: Achieve superior trading performance compared to PPO baseline through adaptive reasoning
- **Maintainability**: Complete test coverage and documentation for all HRM components
- **Scalability**: Support for multiple instruments with shared hierarchical representations

---

_This PRD serves as the definitive guide for migrating from PPO to Hierarchical Reasoning Model (HRM) in the algorithmic trading system. All development work should align with the requirements, assumptions, and success criteria outlined in this document, leveraging HRM's revolutionary efficiency and brain-inspired architecture for superior trading performance._
