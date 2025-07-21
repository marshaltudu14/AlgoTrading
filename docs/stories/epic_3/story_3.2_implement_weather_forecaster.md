---
Status: Approved
Story:
  Sequence: 3.2
  Title: Implement the Market "Weather Forecaster"
  Description: As a developer, I want to create a market classification module so that the agent can understand the current market regime (e.g., Trending, Ranging, Volatile).
Acceptance Criteria:
  1: A new directory `src/reasoning/` is created.
  2: A new file `src/reasoning/market_classifier.py` is created within it.
  3: The module contains a function or class that takes market data as input.
  4: The module outputs a classification of the current market state (e.g., using a simple moving average crossover system or a volatility breakout system initially).
  5: The `AutonomousAgent` from Epic 2 is updated to take this classification as an additional input to its `act` method.
Tasks:
  - Task: Create `src/reasoning/` directory and `market_classifier.py`
    Subtasks:
      - Subtask: Create the directory structure
      - Subtask: Define `MarketClassifier` class or function
  - Task: Implement market classification logic
    Subtasks:
      - Subtask: Choose initial classification criteria (e.g., based on ATR, ADX, or moving average slopes)
      - Subtask: Implement `classify_market(market_data)` method to return a categorical classification (e.g., "Trending", "Ranging", "Volatile")
  - Task: Update `AutonomousAgent` to use `MarketClassifier`
    Subtasks:
      - Subtask: Import `MarketClassifier` into `AutonomousAgent`
      - Subtask: Instantiate `MarketClassifier` in `AutonomousAgent`'s `__init__`
      - Subtask: Call `classify_market` in `AutonomousAgent`'s `act` method and pass the result to the `TransformerWorldModel` input
  - Task: Create unit tests for `MarketClassifier`
    Subtasks:
      - Subtask: Create `tests/test_market_classifier.py`
      - Subtask: Write test cases for different market data scenarios to verify correct classification outputs
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: