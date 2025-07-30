---
Status: Approved
Story:
  Sequence: 3.4
  Title: Implement the Self-Modification Logic
  Description: As a developer, I want to implement the logic for self-modification so that the agent can react to its own performance and trigger adaptive changes.
Acceptance Criteria:
  1: A new file `src/reasoning/self_modification.py` is created.
  2: The module contains a function `check_performance_and_adapt(agent, performance_metrics)`.
  3: This function can trigger specific adaptive actions based on configurable metrics and thresholds, such as:
    - Calling the NAS controller to search for a new architecture if the profit factor is too low.
    - Temporarily reducing the agent's risk-taking if the drawdown is too high.
  4: This function will be called at the end of each evaluation phase in the main training loop (to be built in Epic 4).
Tasks:
  - Task: Create `src/reasoning/self_modification.py`
    Subtasks:
      - Subtask: Define `SelfModificationManager` class or function
  - Task: Implement `check_performance_and_adapt` function
    Subtasks:
      - Subtask: Define `check_performance_and_adapt(agent, performance_metrics, config)`
      - Subtask: Implement logic to evaluate `performance_metrics` against configurable thresholds (e.g., from `config/autonomous_config.yaml`)
      - Subtask: Implement conditional calls to `NASController` (from Story 3.1) to trigger architectural changes
      - Subtask: Implement conditional logic to adjust agent's internal risk parameters (e.g., `agent.risk_aversion_factor`)
      - Subtask: Implement conditional logic for synaptic pruning or neurogenesis (e.g., by modifying `CoreTransformer` parameters and re-initializing part of the model)
  - Task: Create unit tests for `SelfModificationManager`
    Subtasks:
      - Subtask: Create `tests/test_self_modification.py`
      - Subtask: Write test cases to verify that `check_performance_and_adapt` triggers correct actions based on dummy performance metrics
      - Subtask: Test different threshold scenarios
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: