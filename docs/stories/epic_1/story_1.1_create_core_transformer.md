---
Status: Approved
Story:
  Sequence: 1.1
  Title: Create a Core Transformer Module
  Description: As a developer, I want to create a generic, reusable Transformer module with configurable parameters so that it can be easily integrated into all existing and future trading agents.
Acceptance Criteria:
  1: A new file is created at `src/models/core_transformer.py`.
  2: The file contains a PyTorch `nn.Module` class named `CoreTransformer`.
  3: The module accepts parameters for input dimension, number of attention heads, number of encoder layers, feed-forward dimensions, and output dimension, ensuring sufficient flexibility for future NAS.
  4: The module correctly processes a sequence of input data and returns an output of the specified dimension.
  5: The module is accompanied by a simple unit test to verify its input/output functionality.
Tasks:
  - Task: Define CoreTransformer class structure
    Subtasks:
      - Subtask: Create `src/models/core_transformer.py`
      - Subtask: Implement `__init__` method with configurable parameters (input_dim, num_heads, num_layers, ff_dim, output_dim)
      - Subtask: Implement Transformer Encoder layers using `nn.TransformerEncoder` and `nn.TransformerEncoderLayer`
      - Subtask: Add a final linear layer to map to output_dim
  - Task: Implement forward pass logic
    Subtasks:
      - Subtask: Define `forward` method to process input sequence
      - Subtask: Ensure correct tensor shapes and data types through the Transformer layers
  - Task: Create unit test for CoreTransformer
    Subtasks:
      - Subtask: Create `tests/test_core_transformer.py`
      - Subtask: Write a test case to instantiate `CoreTransformer` with various parameters
      - Subtask: Write a test case to verify input/output dimensions and data types
      - Subtask: Write a test case to ensure the module runs without errors on dummy data
Dev Notes:
  Testing:
    - Test file location: `tests/test_core_transformer.py`
    - Test standards: Unit tests
    - Testing frameworks and patterns to use: PyTorch's `unittest` or `pytest` with `torch` assertions.
    - Any specific testing requirements for this story: Verify input/output shapes and data types; ensure module runs without errors on dummy data.
Dev Agent Record:
QA Results:
  - [ ] Task:
  - [ ] Subtask:
Change Log: