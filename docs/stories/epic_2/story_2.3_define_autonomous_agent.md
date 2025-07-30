---
Status: Approved
Story:
  Sequence: 2.3
  Title: Define the Core Autonomous Agent
  Description: As a developer, I want to create a new `AutonomousAgent` class that integrates the World Model and the External Memory so that the core components of the new agent are brought together.
Acceptance Criteria:
  1: A new file is created at `src/agents/autonomous_agent.py`.
  2: The file contains an `AutonomousAgent` class.
  3: In its `__init__` method, the agent initializes an instance of the `TransformerWorldModel` and the `ExternalMemory`.
  4: The agent has an `act(market_state)` method that performs the "Think" loop:
    - It embeds the current `market_state`.
    - It retrieves relevant memories from its `ExternalMemory`.
    - It feeds the state and memories into its `TransformerWorldModel`.
    - It returns a trading action from the model's policy head.
  5: The agent can be successfully instantiated without errors.
Tasks:
  - Task: Create `src/agents/autonomous_agent.py`
    Subtasks:
      - Subtask: Define `AutonomousAgent` class
  - Task: Implement `__init__` method
    Subtasks:
      - Subtask: Import `TransformerWorldModel` and `ExternalMemory`
      - Subtask: Instantiate `TransformerWorldModel` and `ExternalMemory` within `__init__`
      - Subtask: Pass necessary configuration parameters to instantiated components
  - Task: Implement `act` method (the "Think" loop)
    Subtasks:
      - Subtask: Define `act(market_state)` method
      - Subtask: Implement logic to embed `market_state` (e.g., simple linear layer or direct pass if already embedded)
      - Subtask: Call `ExternalMemory.retrieve` with the embedded state
      - Subtask: Concatenate or combine embedded state and retrieved memories for input to `TransformerWorldModel`
      - Subtask: Call `TransformerWorldModel`'s `forward` method
      - Subtask: Extract and return the trading action from the policy head output
  - Task: Create unit tests for `AutonomousAgent`
    Subtasks:
      - Subtask: Create `tests/test_autonomous_agent.py`
      - Subtask: Write test case to instantiate `AutonomousAgent` without errors
      - Subtask: Write test case to call `act` method with dummy market state and verify output format
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: