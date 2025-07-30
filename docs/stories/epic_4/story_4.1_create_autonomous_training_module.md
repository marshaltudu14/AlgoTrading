---
Status: Approved
Story:
  Sequence: 4.1
  Title: Create the Autonomous Training Module
  Description: As a developer, I want to create a new training module that orchestrates the generational training loop for the autonomous agents.
Acceptance Criteria:
  1: A new file is created at `src/training/autonomous_trainer.py`.
  2: The file contains a main function `run_autonomous_stage(config)`.
  3: This function initializes a population of `AutonomousAgent` instances based on the configuration.
  4: It contains the main loop that iterates for the specified number of `generations`.
  5: Inside the loop, it calls the backtesting engine to evaluate the fitness of each agent in the current population.
  6: It calls the NAS controller (from Epic 3) to select the best agents and create a new, evolved population for the next generation.
Tasks:
  - Task: Create `src/training/autonomous_trainer.py`
    Subtasks:
      - Subtask: Define `run_autonomous_stage(config)` function
  - Task: Implement population initialization
    Subtasks:
      - Subtask: Instantiate `NASController` (from Story 3.1)
      - Subtask: Use `NASController` to generate initial architectures for a population of `AutonomousAgent` instances (from Story 2.3)
  - Task: Implement generational training loop
    Subtasks:
      - Subtask: Loop for `config['generations']`
      - Subtask: For each agent in the population, run a backtest using `src/backtesting/environment.py` and `src/backtesting/engine.py`
      - Subtask: Calculate fitness score (e.g., Sharpe Ratio, Profit Factor) for each agent based on backtest results
      - Subtask: Call `NASController.evolve_population` to generate the next generation of agents based on fitness
Dev Notes:
Testing:
Dev Agent Record:
  - [ ] Task:
  - [ ] Subtask:
Change Log: