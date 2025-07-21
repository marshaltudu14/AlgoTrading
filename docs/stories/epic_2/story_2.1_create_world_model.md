---
Status: Approved
Story:
  Sequence: 2.1
  Title: Create the Transformer-Based World Model
  Description: As a developer, I want to create a new `TransformerWorldModel` that can predict future market states so that the agent has a foundation for its reasoning engine.
Acceptance Criteria:
  1: A new file is created at `src/models/world_model.py`.
  2: The file contains a PyTorch `nn.Module` class named `TransformerWorldModel`.
  3: The model's architecture is based on the `CoreTransformer` from Epic 1.
  4: The model has two distinct output "heads":
    - A "Prediction Head" that outputs a predicted future market state (e.g., next N candles' OHLCV, or higher-level market regime shifts over a defined horizon) to assess potential profitability.
    - A "Policy Head" that outputs a distribution over trading actions.
  5: A unit test is created to verify that the model can process a sequence of data and produce outputs of the correct shape for both heads, and that the prediction head's output is in a format interpretable for "what-if" scenarios.
Tasks:
  - Task: Create `src/models/world_model.py`
    Subtasks:
      - Subtask: Define `TransformerWorldModel` class inheriting from `nn.Module`
      - Subtask: Import and instantiate `CoreTransformer` within `TransformerWorldModel`
      - Subtask: Implement the "Prediction Head" (e.g., linear layer to predict OHLCV or market regime)
      - Subtask: Implement the "Policy Head" (e.g., linear layer with softmax for action distribution)
  - Task: Implement `forward` pass for World Model
    Subtasks:
      - Subtask: Define `forward` method to take market state input
      - Subtask: Pass input through `CoreTransformer`
      - Subtask: Route `CoreTransformer` output to both Prediction and Policy Heads
      - Subtask: Return outputs from both heads
  - Task: Create unit tests for `TransformerWorldModel`
    Subtasks:
      - Subtask: Create `tests/test_world_model.py`
      - Subtask: Write test case to instantiate `TransformerWorldModel`
      - Subtask: Write test case to verify correct output shapes for both heads
      - Subtask: Write test case to verify interpretability of prediction head output (e.g., check value ranges, basic structure)
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: