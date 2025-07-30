---
Status: Approved
Story:
  Sequence: 1.3
  Title: Refactor MoE Agent to Use Transformer
  Description: As a developer, I want to replace the LSTM layers in the Mixture of Experts (MoE) agent with the new `CoreTransformer` module so that the specialized experts benefit from enhanced memory.
Acceptance Criteria:
  1: The MoE agent's model definition in `src/models/moe_model.py` (or equivalent file) is modified to replace LSTM layers with the `CoreTransformer` module in each expert network.
  2: The `CoreTransformer` module is imported and instantiated within the MoE model.
  3: The gating network correctly handles the outputs from the new Transformer-based experts.
  4: The `run_training.py` script can successfully complete the `stage_2_moe` training using the refactored model.
  5: The backtesting script can successfully load and run the new Transformer-based MoE model.
  6: Performance metrics (e.g., win rate, profit factor) of the refactored MoE agent are comparable to or improved over the LSTM-based version on historical data.
Tasks:
  - Task: Identify and modify MoE agent model definition
    Subtasks:
      - Subtask: Locate `src/agents/moe_agent.py` and relevant expert model files (e.g., `src/models/lstm_model.py` if shared)
      - Subtask: For each expert (Trend, Mean Reversion, Volatility, Consolidation), remove or comment out existing LSTM layer instantiation and forward pass logic in their actor/critic networks
  - Task: Integrate CoreTransformer into MoE agent experts
    Subtasks:
      - Subtask: Import `CoreTransformer` into relevant expert model files
      - Subtask: Instantiate `CoreTransformer` for both actor and critic networks of each expert, ensuring correct input/output dimensions
      - Subtask: Adjust forward pass of expert actor and critic networks to use the `CoreTransformer`
  - Task: Verify Gating Network compatibility
    Subtasks:
      - Subtask: Ensure the Gating Network in `MoEAgent` correctly processes outputs from the new Transformer-based experts
      - Subtask: Confirm that the `select_action` method in `MoEAgent` functions as expected with the new expert outputs
  - Task: Verify MoE training and backtesting with new model
    Subtasks:
      - Subtask: Run `run_training.py` for `stage_2_moe` and confirm successful completion without errors
      - Subtask: Run `run_backtest.py` with a model trained from the refactored MoE agent and confirm successful loading and execution
      - Subtask: Collect and compare performance metrics (win rate, profit factor) against a baseline of the LSTM-based MoE agent
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: