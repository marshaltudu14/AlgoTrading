---
Status: Approved
Story:
  Sequence: 1.2
  Title: Refactor PPO Agent to Use Transformer
  Description: As a developer, I want to replace the LSTM layer in the PPO agent with the new `CoreTransformer` module so that the baseline agent's performance is improved by the new architecture.
Acceptance Criteria:
  1: The PPO agent's model definition in `src/models/ppo_model.py` (or equivalent file) is modified to remove the LSTM layer.
  2: The `CoreTransformer` module is imported and instantiated within the PPO model.
  3: The `run_training.py` script can successfully complete the `stage_1_ppo` training using the refactored model without errors.
  4: The backtesting script can successfully load and run the new Transformer-based PPO model.
  5: Performance metrics (e.g., win rate, profit factor) of the refactored PPO agent are comparable to or improved over the LSTM-based version on historical data.
Tasks:
  - Task: Identify and modify PPO agent model definition
    Subtasks:
      - Subtask: Locate `src/models/ppo_model.py` (or relevant file defining PPO's actor/critic networks)
      - Subtask: Remove or comment out existing LSTM layer instantiation and forward pass logic
  - Task: Integrate CoreTransformer into PPO agent
    Subtasks:
      - Subtask: Import `CoreTransformer` from `src/models/core_transformer.py`
      - Subtask: Instantiate `CoreTransformer` for both actor and critic networks, ensuring correct input/output dimensions
      - Subtask: Adjust forward pass of actor and critic to use the `CoreTransformer`
  - Task: Verify PPO training and backtesting with new model
    Subtasks:
      - Subtask: Run `run_training.py` for `stage_1_ppo` and confirm successful completion without errors
      - Subtask: Run `run_backtest.py` with a model trained from the refactored PPO agent and confirm successful loading and execution
      - Subtask: Collect and compare performance metrics (win rate, profit factor) against a baseline of the LSTM-based PPO agent
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: