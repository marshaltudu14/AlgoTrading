# Autonomous Trading Agent Product Requirements Document (PRD)

## Goals and Background Context

**Goals**

*   Develop a fully autonomous trading agent that can learn, adapt, and evolve through direct market exposure.
*   Achieve a state of perpetual self-improvement through the use of a self-evolving neural architecture.
*   Implement self-hyperparameter tuning to automatically optimize its own learning process.
*   Implement a "World Model" based reasoning engine, allowing the agent to simulate and predict future market states.
*   Create a robust system for self-learning and self-modification, enabling the agent to create its own trading strategies and rules.
*   Ensure long-term viability and profitability through advanced overfitting mitigation techniques.
*   Establish a long-term memory system, allowing the agent to learn from specific, critical past events.

**Background Context**

This project aims to evolve the existing RL-based trading system into a truly autonomous entity, codenamed the "Super Trader Baby." The current system, while sophisticated, relies on a predefined training pipeline and fixed architectures. This PRD outlines the requirements for the next evolutionary step: creating an agent that can transcend its initial design. The core problem this solves is the inherent limitation of human-designed models, which cannot adapt at the speed and complexity of modern financial markets. By creating an agent that designs and refines itself, we aim to unlock a new level of performance and autonomy in algorithmic trading.

**Change Log**

| Date       | Version | Description                | Author |
| :--------- | :------ | :------------------------- | :----- |
| 2025-07-21 | 1.0     | Initial draft of the PRD. | John   |

---

## Requirements

**Functional**

1.  **FR1: The system must support a new "Autonomous" training stage**, selectable via the `training_sequence.yaml` file, which follows the existing PPO, MoE, and MAML stages.
2.  **FR2: The Autonomous stage must manage a population of agents** and run a generational training loop, evaluating each agent's fitness per generation.
3.  **FR3: The system must implement a Transformer-based World Model** capable of predicting future market states (e.g., next N candles' OHLCV, or higher-level market regime shifts over a defined horizon) to assess potential profitability.
4.  **FR4: The system must include an External Memory module** that allows an agent to store and retrieve significant past events (e.g., flash crashes, large wins/losses, specific market regime shifts, or rare actions/outcomes) based on predefined criteria.
5.  **FR5: The agent must be able to perform Neural Architecture Search (NAS)** to modify and evolve its own architecture during the training loop.
6.  **FR6: The agent must be able to perform self-hyperparameter tuning** as part of the generational training loop, adjusting parameters like learning rate and batch size based on performance.
7.  **FR7: The system must include a market classification module (the "Weather Forecaster")** to identify the current market regime (e.g., Trending, Ranging, Volatile).
8.  **FR8: The system must include a pattern recognition module (the "Chartist")** to identify technical chart patterns from price data.
9.  **FR9: The system must implement robust overfitting mitigation**, including the use of adversarial data generation and advanced regularization techniques.
10. **FR10: The final, best-performing autonomous agent model must be saved** in a format that can be loaded by both the `run_backtest.py` and `run_live_bot.py` scripts.
11. **FR11: The existing RL agents (PPO, MoE, MAML) must be refactored to use a Transformer-based architecture**, replacing the current LSTM components to improve memory and sequence processing capabilities across the entire system.

**Non-Functional**

1.  **NFR1: All new architectural components (World Model, NAS, Memory, etc.) must be modular** and contained within their own subdirectories in the `src/` folder.
2.  **NFR2: The new Autonomous training stage must integrate with the existing logging and monitoring system**, providing clear output for each generation's progress and fitness scores.
3.  **NFR3: The system's configuration must be centralized**, with all new parameters for the autonomous stage managed within the `training_sequence.yaml` file.
4.  **NFR4: The implementation should not break the existing PPO, MoE, and MAML training stages**, which must remain fully functional (albeit with their new Transformer architecture).
5.  **NFR5: The Neural Architecture Search (NAS) implementation must be extensible**, allowing for easy swapping or addition of different search algorithms (e.g., DARTS, Evolutionary Algorithms) in the future.
6.  **NFR6: The market classification ("Weather Forecaster") and pattern recognition ("Chartist") modules must be designed to be pluggable**, allowing for easy swapping or addition of new market understanding modules without major refactoring of the `AutonomousAgent`.

---

## Technical Assumptions

**Repository Structure: Monorepo**

*   The entire project, including all source code, configuration, and documentation, will continue to be managed within the existing single Git repository. This approach simplifies dependency management and ensures consistency across the codebase.

**Service Architecture: Monolithic Application (with future extensibility in mind)**

*   The system will be developed as a single, cohesive application. The new components (World Model, NAS, Memory, etc.) will be integrated as modules within the existing `src/` directory structure, not as separate microservices. While monolithic for now, the design should consider *future extensibility* towards a more distributed architecture without requiring a complete rewrite.

**Testing Requirements: Full Testing Pyramid**

*   A comprehensive testing strategy is mandatory. This includes:
    *   **Unit Tests:** For all new, individual modules (e.g., NAS controller, Memory module) to ensure they function correctly in isolation. The existing `tests/` directory should be expanded to accommodate these.
    *   **Integration Tests:** To verify that the new modules interact correctly with each other and with existing components like the backtesting engine.
    *   **End-to-End (E2E) Tests:** The full `run_training.py` execution of the "Autonomous" stage will serve as the primary E2E test, validating that the entire evolutionary process runs successfully. The backtesting environment (`src/backtesting/environment.py` and `engine.py`) will be the primary "sandbox" for evaluating the evolving agents.

**Additional Technical Assumptions and Requests**

*   The core technology stack will remain **Python**, with **PyTorch** for neural network development and **Ray RLlib** for reinforcement learning orchestration where applicable.
*   All new components must be designed to be compatible with the existing data processing pipeline and the backtesting environment in `src/backtesting/`.
*   The implementation of the Transformer architecture (FR11) should be done in a way that it becomes a reusable, core component for all agents (PPO, MoE, MAML, and Autonomous).

---

## Epic List

1.  **Epic 1: Foundational Upgrade - Transformer Architecture**
    *   **Goal:** To refactor the existing PPO, MoE, and MAML agents to use a more powerful Transformer-based architecture, replacing the current LSTM components.

2.  **Epic 2: Core Autonomous Agent Components**
    *   **Goal:** To build the fundamental components of the new autonomous agent: the Transformer-based "World Model" and the "External Memory" module.

3.  **Epic 3: Self-Evolution and Reasoning Engine**
    *   **Goal:** To implement the advanced capabilities of the agent, including the Neural Architecture Search (NAS) for self-evolution and the reasoning modules ("Weather Forecaster" and "Chartist") for market understanding.

4.  **Epic 4: The Autonomous Training Loop & Final Integration**
    *   **Goal:** To develop the full generational training loop, integrate all components into the main `run_training.py` pipeline, and enable the agent's self-hyperparameter tuning and final model saving.

---

## Epic 1: Foundational Upgrade - Transformer Architecture

**Goal:** To refactor the existing PPO, MoE, and MAML agents to use a more powerful and reusable Transformer-based architecture, replacing the current LSTM components. This will enhance the sequence processing and memory capabilities of the entire existing pipeline, setting a strong foundation for the future autonomous agent.

---

#### **Story 1.1: Create a Core Transformer Module**
*As a developer, I want to create a generic, reusable Transformer module with configurable parameters so that it can be easily integrated into all existing and future trading agents.*

**Acceptance Criteria:**
1.  A new file is created at `src/models/core_transformer.py`.
2.  The file contains a PyTorch `nn.Module` class named `CoreTransformer`.
3.  The module accepts parameters for input dimension, number of attention heads, number of encoder layers, feed-forward dimensions, and output dimension, ensuring sufficient flexibility for future NAS.
4.  The module correctly processes a sequence of input data and returns an output of the specified dimension.
5.  The module is accompanied by a simple unit test to verify its input/output functionality.

---

#### **Story 1.2: Refactor PPO Agent to Use Transformer**
*As a developer, I want to replace the LSTM layer in the PPO agent with the new `CoreTransformer` module so that the baseline agent's performance is improved by the new architecture.*

**Acceptance Criteria:**
1.  The PPO agent's model definition in `src/models/ppo_model.py` (or equivalent file) is modified to remove the LSTM layer.
2.  The `CoreTransformer` module is imported and instantiated within the PPO model.
3.  The `run_training.py` script can successfully complete the `stage_1_ppo` training using the refactored model without errors.
4.  The backtesting script can successfully load and run the new Transformer-based PPO model.

---

#### **Story 1.3: Refactor MoE Agent to Use Transformer**
*As a developer, I want to replace the LSTM layers in the Mixture of Experts (MoE) agent with the new `CoreTransformer` module so that the specialized experts benefit from enhanced memory.*

**Acceptance Criteria:**
1.  The MoE agent's model definition in `src/models/moe_model.py` (or equivalent file) is modified to replace LSTM layers with the `CoreTransformer` module in each expert network.
2.  The `run_training.py` script can successfully complete the `stage_2_moe` training using the refactored model.
3.  The gating network correctly handles the outputs from the new Transformer-based experts.
4.  The backtesting script can successfully load and run the new Transformer-based MoE model.

---

#### **Story 1.4: Refactor MAML Agent to Use Transformer**
*As a developer, I want to replace the LSTM layer in the MAML agent with the new `CoreTransformer` module so that the meta-learning process can leverage a more powerful underlying architecture.*

**Acceptance Criteria:**
1.  The MAML agent's model definition in `src/models/maml_model.py` (or equivalent file) is modified to replace the LSTM layer with the `CoreTransformer` module.
2.  The `run_training.py` script can successfully complete the `stage_3_maml` training, including the inner and outer loop updates, with the refactored model.
3.  The backtesting script can successfully load and run the new Transformer-based MAML model.
4.  Performance metrics (e.g., adaptation speed, cross-symbol performance) of the refactored MAML agent are comparable to or improved over the LSTM-based version on historical data.

---

## Epic 2: Core Autonomous Agent Components

**Goal:** To build the fundamental, standalone components of the new autonomous agent: the Transformer-based "World Model" which will serve as its brain, and the "External Memory" module which will serve as its notebook.

---

#### **Story 2.1: Create the Transformer-Based World Model**
*As a developer, I want to create a new `TransformerWorldModel` that can predict future market states so that the agent has a foundation for its reasoning engine.*

**Acceptance Criteria:**
1.  A new file is created at `src/models/world_model.py`.
2.  The file contains a PyTorch `nn.Module` class named `TransformerWorldModel`.
3.  The model's architecture is based on the `CoreTransformer` from Epic 1.
4.  The model has two distinct output "heads":
    *   A "Prediction Head" that outputs a predicted future market state.
    *   A "Policy Head" that outputs a distribution over trading actions.
5.  A unit test is created to verify that the model can process a sequence of data and produce outputs of the correct shape for both heads, and that the prediction head's output is in a format interpretable for "what-if" scenarios.

---

#### **Story 2.2: Implement the External Memory Module**
*As a developer, I want to create an `ExternalMemory` module where the agent can store and retrieve significant past events so that it can learn from specific historical experiences.*

**Acceptance Criteria:**
1.  A new directory `src/memory/` is created.
2.  A new file `src/memory/episodic_memory.py` is created within it.
3.  The file contains an `ExternalMemory` class.
4.  The class has a `store(event_embedding, outcome)` method to save a vector representation of an event.
5.  The class has a `retrieve(current_state_embedding)` method that returns the most similar past events from memory.
6.  The initial implementation can use a simple vector similarity search library (e.g., Faiss) or even NumPy for the backend, ensuring efficient querying by the `AutonomousAgent`.
7.  A unit test is created to verify that memories can be stored and retrieved correctly.

---

#### **Story 2.3: Define the Core Autonomous Agent**
*As a developer, I want to create a new `AutonomousAgent` class that integrates the World Model and the External Memory so that the core components of the new agent are brought together.*

**Acceptance Criteria:**
1.  A new file is created at `src/agents/autonomous_agent.py`.
2.  The file contains an `AutonomousAgent` class.
3.  In its `__init__` method, the agent initializes an instance of the `TransformerWorldModel` and the `ExternalMemory`.
4.  The agent has an `act(market_state)` method that performs the "Think" loop:
    1.  It embeds the current `market_state`.
    2.  It retrieves relevant memories from its `ExternalMemory`.
    3.  It feeds the state and memories into its `TransformerWorldModel`.
    4.  It returns a trading action from the model's policy head.
5.  The agent can be successfully instantiated without errors.

---

## Epic 3: Self-Evolution and Reasoning Engine

**Goal:** To implement the advanced capabilities of the agent, including the Neural Architecture Search (NAS) for self-evolution, the reasoning modules ("Weather Forecaster" and "Chartist") for market understanding, and the logic for self-modification.

---

#### **Story 3.1: Implement the Neural Architecture Search (NAS) Framework**
*As a developer, I want to create a framework for Neural Architecture Search so that the agent can discover and evolve its own model architectures.*

**Acceptance Criteria:**
1.  A new directory `src/nas/` is created.
2.  A file `src/nas/search_space.py` is created to define the "Lego box" of possible layers and operations.
3.  A file `src/nas/search_controller.py` is created to house the search algorithm (e.g., DARTS or an Evolutionary Algorithm).
4.  The `search_controller` can take a population of agent architectures and generate a new, evolved population based on fitness scores, interacting with the `CoreTransformer` module to define and modify architectures.
5.  A unit test is created to verify that the NAS controller can generate valid new architectures from the defined search space.

---

#### **Story 3.2: Implement the Market "Weather Forecaster"**
*As a developer, I want to create a market classification module so that the agent can understand the current market regime (e.g., Trending, Ranging, Volatile).*

**Acceptance Criteria:**
1.  A new directory `src/reasoning/` is created.
2.  A new file `src/reasoning/market_classifier.py` is created within it.
3.  The module contains a function or class that takes market data as input.
4.  The module outputs a classification of the current market state (e.g., using a simple moving average crossover system or a volatility breakout system initially).
5.  The `AutonomousAgent` from Epic 2 is updated to take this classification as an additional input to its `act` method.

---

#### **Story 3.3: Implement the "Chartist" Pattern Recognizer**
*As a developer, I want to create a pattern recognition module so that the agent can identify classic technical analysis patterns from price data.*

**Acceptance Criteria:**
1.  A new file `src/reasoning/pattern_recognizer.py` is created.
2.  The module contains a function or class that uses a vision-based approach (e.g., a 1D CNN) to analyze a sequence of price data.
3.  The module outputs an identified pattern if one is detected (e.g., "Head and Shoulders," "Double Top").
4.  The `AutonomousAgent` is updated to take this pattern information as an additional input to its `act` method.

---

#### **Story 3.4: Implement the Self-Modification Logic**
*As a developer, I want to implement the logic for self-modification so that the agent can react to its own performance and trigger adaptive changes.*

**Acceptance Criteria:**
1.  A new file `src/reasoning/self_modification.py` is created.
2.  The module contains a function `check_performance_and_adapt(agent, performance_metrics)`.
3.  This function can trigger specific adaptive actions based on configurable metrics and thresholds, such as:
    *   Calling the NAS controller to search for a new architecture if the profit factor is too low.
    *   Temporarily reducing the agent's risk-taking if the drawdown is too high.
4.  This function will be called at the end of each evaluation phase in the main training loop (to be built in Epic 4).

---

## Epic 4: The Autonomous Training Loop & Final Integration

**Goal:** To develop the full generational training loop, integrate all previously built components into the main `run_training.py` pipeline, and enable the agent's self-hyperparameter tuning and final model saving.

---

#### **Story 4.1: Create the Autonomous Training Module**
*As a developer, I want to create a new training module that orchestrates the generational training loop for the autonomous agents.*

**Acceptance Criteria:**
1.  A new file is created at `src/training/autonomous_trainer.py`.
2.  The file contains a main function `run_autonomous_stage(config)`.
3.  This function initializes a population of `AutonomousAgent` instances based on the configuration.
4.  It contains the main loop that iterates for the specified number of `generations`.
5.  Inside the loop, it calls the backtesting engine to evaluate the fitness of each agent in the current population.
6.  It calls the NAS controller (from Epic 3) to select the best agents and create a new, evolved population for the next generation.

---

#### **Story 4.2: Integrate the Autonomous Stage into the Main Pipeline**
*As a developer, I want to modify the main `run_training.py` script to recognize and execute the new "Autonomous" stage so that it becomes the final step in the training pipeline.*

**Acceptance Criteria:**
1.  The `run_training.py` script is modified to include a condition for `algorithm: "Autonomous"`.
2.  When this condition is met, it calls the `run_autonomous_stage` function from the new training module.
3.  The script can successfully run the full PPO -> MoE -> MAML -> Autonomous pipeline in sequence.
4.  The `config/training_sequence.yaml` is updated with a complete `stage_4_autonomous` section, including all necessary settings.

---

#### **Story 4.3: Implement Self-Hyperparameter Tuning**
*As a developer, I want to enable the agent to automatically tune its own hyperparameters during the evolutionary process so that it can continuously optimize its own learning.*

**Acceptance Criteria:**
1.  The `AutonomousAgent` class is updated to include its own set of hyperparameters (e.g., learning rate, batch size), with a configurable search space for these parameters.
2.  During the evolution phase in the `autonomous_trainer`, the "mutation" process not only changes the architecture (via NAS) but also slightly perturbs the hyperparameters of the new "child" agents.
3.  The system demonstrates that agents with more effective hyperparameters achieve higher fitness scores and are more likely to be selected for the next generation.

---

#### **Story 4.4: Save the Champion Agent for Backtesting and Live Trading**
*As a developer, I want to ensure the final, best-performing agent from the autonomous stage is saved correctly so that it can be used as the definitive model for backtesting and live trading.*

**Acceptance Criteria:**
1.  At the end of the `run_autonomous_stage`, the single best-performing agent from the final generation is identified.
2.  Its complete state (model weights, architecture definition if dynamically generated by NAS, and best tuned hyperparameters) is saved to a file named `{symbol}_autonomous_final.pth`.
3.  The `run_backtest.py` script is modified to **exclusively** load and use the `{symbol}_autonomous_final.pth` model when running a backtest on a symbol that has completed the autonomous stage.
4.  The `run_live_bot.py` script is also modified to **exclusively** load and use the `{symbol}_autonomous_final.pth` model for live trading.
