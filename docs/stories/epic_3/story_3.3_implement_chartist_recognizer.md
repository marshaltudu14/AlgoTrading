---
Status: Approved
Story:
  Sequence: 3.3
  Title: Implement the "Chartist" Pattern Recognizer
  Description: As a developer, I want to create a pattern recognition module so that the agent can identify classic technical analysis patterns from price data.
Acceptance Criteria:
  1: A new file `src/reasoning/pattern_recognizer.py` is created.
  2: The module contains a function or class that uses a vision-based approach (e.g., a 1D CNN) to analyze a sequence of price data.
  3: The module outputs an identified pattern if one is detected (e.g., "Head and Shoulders," "Double Top").
  4: The `AutonomousAgent` is updated to take this pattern information as an additional input to its `act` method.
Tasks:
  - Task: Create `src/reasoning/pattern_recognizer.py`
    Subtasks:
      - Subtask: Define `PatternRecognizer` class or function
  - Task: Implement pattern recognition logic
    Subtasks:
      - Subtask: Choose initial set of patterns to detect (e.g., Doji, Hammer, Engulfing from `feature_generator.py`)
      - Subtask: Implement `recognize_pattern(price_data_sequence)` method using 1D CNN or rule-based logic
      - Subtask: Output identified pattern (e.g., one-hot encoding or string label)
  - Task: Update `AutonomousAgent` to use `PatternRecognizer`
    Subtasks:
      - Subtask: Import `PatternRecognizer` into `AutonomousAgent`
      - Subtask: Instantiate `PatternRecognizer` in `AutonomousAgent`'s `__init__`
      - Subtask: Call `recognize_pattern` in `AutonomousAgent`'s `act` method and pass the result to the `TransformerWorldModel` input
  - Task: Create unit tests for `PatternRecognizer`
    Subtasks:
      - Subtask: Create `tests/test_pattern_recognizer.py`
      - Subtask: Write test cases for different price data sequences to verify correct pattern detection
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: