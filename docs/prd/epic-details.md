# Epic Details

## Epic 1: PPO Architecture Removal and Cleanup

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

## Epic 2: HRM Core Architecture

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

## Epic 3: Configuration System Enhancement

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
