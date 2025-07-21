---
Status: Approved
Story:
  Sequence: 1.4
  Title: Refactor MAML Agent to Use Transformer
  Description: As a developer, I want to replace the LSTM layer in the MAML agent with the new `CoreTransformer` module so that the meta-learning process can leverage a more powerful underlying architecture.
Acceptance Criteria:
  1: The MAML agent's model definition in `src/models/maml_model.py` (or equivalent file) is modified to replace the LSTM layer with the `CoreTransformer` module.
  2: The `run_training.py` script can successfully complete the `stage_3_maml` training, including the inner and outer loop updates, with the refactored model.
  3: The backtesting script can successfully load and run the new Transformer-based MAML model.
  4: Performance metrics (e.g., adaptation speed, cross-symbol performance) of the refactored MAML agent are comparable to or improved over the LSTM-based version on historical data.
Tasks:
  - Task: Identify and modify MAML agent model definition
    Subtasks:
      - Subtask: Locate the relevant file defining MAML's actor/critic networks (likely `src/agents/moe_agent.py` if MAML uses MoE internally, or a dedicated MAML model file)
      - Subtask: Remove or comment out existing LSTM layer instantiation and forward pass logic
  - Task: Integrate CoreTransformer into MAML agent
    Subtasks:
      - Subtask: Import `CoreTransformer`
      - Subtask: Instantiate `CoreTransformer` for both actor and critic networks, ensuring correct input/output dimensions
      - Subtask: Adjust forward pass of actor and critic to use the `CoreTransformer`
  - Task: Verify MAML training and backtesting with new model
    Subtasks:
      - Subtask: Run `run_training.py` for `stage_3_maml` and confirm successful completion without errors
      - Subtask: Verify that inner and outer loop updates function correctly with the new architecture
      - Subtask: Run `run_backtest.py` with a model trained from the refactored MAML agent and confirm successful loading and execution
      - Subtask: Collect and compare performance metrics (adaptation speed, cross-symbol performance) against a baseline of the LSTM-based MAML agent
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: