# Autonomous Trading Bot Product Requirements Document (PRD)

## Goals and Background Context

### Goals
*   Enhance the `BacktestingEngine` to support different instrument types and resolve the "insufficient capital" error for options trading.
*   Improve the `TradingEnv` for more realistic and effective RL agent training, including dynamic action spaces, sophisticated reward functions, and robust observation space normalization.
*   Implement an `Instrument` class and `instruments.yaml` for scalable instrument management.
*   Develop a proxy model to simulate options trading, enabling the agent to learn options-like behavior without direct options data.
*   Complete the implementation of the Meta-Learning (MAML) and Mixture of Experts (MoE) architecture for adaptive agent behavior.
*   Establish a robust and scalable training architecture using data streaming and parallel/distributed reinforcement learning (e.g., Ray RLlib).

### Background Context
This Product Requirements Document (PRD) builds upon the foundational work of the Autonomous Trading Bot project, which aims to develop a fully autonomous algorithmic trading bot leveraging a multi-agent reinforcement learning system with a Mixture of Experts (MoE) architecture. Initial analysis of the existing codebase reveals a solid foundation with components like `TradingEnv`, `BacktestingEngine`, `DataLoader`, and a baseline `PPOAgent`. This PRD outlines the necessary enhancements and new implementations to evolve the system from its current state towards a more robust, scalable, and sophisticated trading solution capable of handling diverse instrument types, particularly options, and advancing towards the full MoE and meta-learning vision. The focus is on addressing identified limitations and laying the groundwork for advanced RL capabilities and scalable training.

### Change Log
| Date       | Version | Description                               | Author |
|------------|---------|-------------------------------------------|--------|
| 2025-07-20 | 1.0     | Initial draft based on analysis_and_planning.md | John   |

## Requirements

### Functional
*   FR1: The `BacktestingEngine` shall handle different instrument types.
*   FR2: The `BacktestingEngine` shall separate `quantity` (agent's trade decision) and `lot_size` (instrument-specific property).
*   FR3: The `BacktestingEngine` shall calculate trade cost based on `proxy_premium * quantity * lot_size`.
*   FR4: The `BacktestingEngine` shall track a `_trailing_stop_price` for any open position based on the peak price.
*   FR5: The `TradingEnv` shall calculate and pass a `proxy_premium` to the `BacktestingEngine`.
*   FR6: The `TradingEnv` shall incorporate trading `quantity` as part of the action space.
*   FR7: The `TradingEnv` shall support sophisticated reward functions beyond simple P&L.
*   FR8: The `TradingEnv` shall use reward shaping to guide agent behavior (e.g., penalty for idleness, bonus for realizing profits, penalty for over-trading).
*   FR9: The `TradingEnv` shall use z-score normalization for the observation space.
*   FR10: The `TradingEnv` shall terminate an episode if the account's drawdown exceeds a predefined limit.
*   FR11: The `TradingEnv` shall terminate an episode if the remaining capital is insufficient to place a new trade.
*   FR12: The `TradingEnv` shall add normalized `distance_to_trail` to the agent's observation space.
*   FR13: The `TradingEnv` shall provide reward shaping for trailing stops (bonus for holding profitable position as trailing stop improves, penalty for premature closing).
*   FR14: A new Python class named `Instrument` shall be created to hold instrument-specific information (`symbol`, `type`, `lot_size`, `tick_size`).
*   FR15: The `Instrument` object shall be passed to the `TradingEnv` and `BacktestingEngine`.
*   FR16: An `instruments.yaml` file shall act as a registry for all tradable instruments.
*   FR17: A utility shall load `instruments.yaml` at startup and create a dictionary of `Instrument` objects.
*   FR18: The agent shall continue to make decisions based on the underlying's price chart for options simulation.
*   FR19: The cost model for options simulation shall be `cost = proxy_premium * quantity * lot_size`.
*   FR20: The P&L model for options simulation shall be based on the intrinsic value of the option, with maximum loss capped at the premium paid.
*   FR21: The `learn` method in specialized agents shall be implemented with core PPO update logic.
*   FR22: The `adapt` method in specialized agents shall implement the inner loop adaptation with a differentiable copy of parameters.
*   FR23: `MoEAgent.select_action` shall use a weighted average of expert action probabilities.
*   FR24: The `learn` method in `MoEAgent` shall orchestrate learning for both `GatingNetwork` and experts.
*   FR25: The `adapt` method in `MoEAgent` shall orchestrate adaptation of `GatingNetwork` and individual experts during MAML inner loop.
*   FR26: The outer loop meta-update in `Trainer.meta_train` shall be fully implemented.
*   FR27: The `DataLoader` shall provide data in smaller chunks or as an iterator.
*   FR28: `TradingEnv.reset()` shall load only the necessary data segment for the current episode.
*   FR29: The system shall adopt an Actor-Learner architecture for parallel training (e.g., using Ray RLlib).
*   FR30: The system shall utilize GPU for PyTorch models and data, with optimized batch sizes and `torch.backends.cudnn.benchmark = True`.
*   FR31: The system shall ensure graceful fallback to CPU if GPU/TPU is unavailable.

### Non Functional
*   NFR1: The training architecture shall be robust and scalable to handle millions of rows across hundreds of instruments and timeframes.
*   NFR2: The training process shall be optimized for faster training times.
*   NFR3: Efficient data storage formats like Parquet or HDF5 should be considered for data streaming and on-demand loading.
*   NFR4: The system should abstract device management for distributed training (e.g., via RLlib).

## User Interface Design Goals
This section is not applicable as the core trading bot does not have a user-facing interface.

## Technical Assumptions

*   **Repository Structure:** Not explicitly defined, but a multi-module or monorepo structure could be considered to manage different components (data pipeline, models, trading, backtesting).
*   **Service Architecture:** The bot will primarily operate as a monolithic, locally run Python application for execution.
*   **Testing Requirements:** Comprehensive testing, including unit tests, integration tests, and extensive backtesting, will be crucial to ensure the reliability and performance of the trading bot.
*   **Additional Technical Assumptions and Requests:**
    *   The primary development language will be Python. C++ for performance-critical components is not currently planned, but could be considered if profiling indicates a bottleneck.
    *   PyTorch will be the primary deep learning framework.
    *   For training, Ray RLlib will be utilized, especially for future cloud-based or distributed training to achieve faster results. Current execution of the bot will be local.
    *   Efficient data storage formats like Parquet or HDF5 will be considered for managing large datasets locally.
    *   GPU acceleration will be utilized for training if available locally or in the cloud, with graceful fallback to CPU.
    *   The Fyers API will be the primary external trading API for live execution and data.
    *   No database is currently in use for the core bot's operations. For future commercialization, Supabase (self-hosted or cloud) is a potential consideration for data storage.
    *   For future commercialization, a subscription-based frontend built with Next.js is a possibility.

## Epic List

*   **Epic 1: Backtesting & Environment Enhancements for Options Simulation**
    *   **Goal:** Establish a robust backtesting environment capable of simulating options-like trading behavior, handling different instrument types, and providing a more realistic training ground for RL agents.

*   **Epic 2: Meta-Learning (MAML) and Mixture of Experts (MoE) Implementation**
    *   **Goal:** Implement the core Meta-Learning and Mixture of Experts architecture, enabling the RL agents to adapt and specialize across diverse market conditions.

*   **Epic 3: Scalable Training Architecture**
    *   **Goal:** Develop a robust and scalable training architecture to efficiently handle large datasets and enable faster training of complex RL models, including the use of Ray RLlib for distributed capabilities.

## Epic 1 Backtesting & Environment Enhancements for Options Simulation
**Expanded Goal:** This epic aims to establish a robust backtesting environment capable of simulating options-like trading behavior, handling different instrument types, and providing a more realistic training ground for RL agents. This involves modifying the core `BacktestingEngine` and `TradingEnv` to support instrument-specific properties, options proxy cost/P&L models, and advanced reward/observation mechanisms, ultimately resolving the "insufficient capital" error and enabling more sophisticated strategy development.

### Story 1.1 Instrument Class and Configuration
As a developer,
I want to define and manage instrument-specific properties (symbol, type, lot_size, tick_size) in a structured way,
so that the backtesting engine and trading environment can correctly handle different asset types, including options.

#### Acceptance Criteria
1.  **1.1.1:** A new Python class named `Instrument` shall be created in `src/config/` (or a suitable new `src/instruments/` directory if preferred) with attributes for `symbol` (str), `type` (str, e.g., "STOCK", "OPTION"), `lot_size` (int), and `tick_size` (float).
2.  **1.1.2:** An `instruments.yaml` file shall be created in the `config/` directory to serve as a centralized registry for all tradable instruments, defining their properties (symbol, type, lot_size, tick_size).
3.  **1.1.3:** A utility function or class method shall be implemented to load the `instruments.yaml` file at startup and parse its contents into a dictionary of `Instrument` objects, accessible by their symbol.
4.  **1.1.4:** The `Instrument` object (or relevant properties from it) shall be passed to the `TradingEnv` and `BacktestingEngine` during their initialization or relevant method calls, allowing them to access instrument-specific data.

### Story 1.2 BacktestingEngine Enhancements
As a developer,
I want the `BacktestingEngine` to accurately simulate options trading and handle instrument-specific properties,
so that the RL agent can be trained in a more realistic environment and the "insufficient capital" error is resolved.

#### Acceptance Criteria
1.  **1.2.1:** The `BacktestingEngine` shall be updated to separate `quantity` (agent's trade decision) and `lot_size` (instrument-specific property).
2.  **1.2.2:** The `BacktestingEngine` shall calculate trade cost based on `proxy_premium * quantity * lot_size` for options simulation.
3.  **1.2.3:** The `BacktestingEngine` shall track a `_trailing_stop_price` for any open position based on the peak price reached during the trade.
4.  **1.2.4:** The P&L model for options simulation shall be based on the intrinsic value of the option, with maximum loss capped at the premium paid.

### Story 1.3 TradingEnv Enhancements
As a developer,
I want the `TradingEnv` to provide a more realistic and effective training environment for RL agents,
so that the agents can learn more sophisticated trading strategies and risk management.

#### Acceptance Criteria
1.  **1.3.1:** The `TradingEnv` shall calculate and pass a `proxy_premium` for each trade to the `BacktestingEngine`.
2.  **1.3.2:** The `TradingEnv` shall incorporate trading `quantity` as part of the action space, allowing the agent to learn position sizing.
3.  **1.3.3:** The `TradingEnv` shall support sophisticated reward functions beyond simple P&L (e.g., Sharpe Ratio, Sortino Ratio, Profit Factor).
4.  **1.3.4:** The `TradingEnv` shall use reward shaping to guide agent behavior (e.g., penalty for idleness, bonus for realizing profits, penalty for over-trading).
5.  **1.3.5:** The `TradingEnv` shall use z-score normalization for the observation space.
6.  **1.3.6:** The `TradingEnv` shall terminate an episode prematurely if the account's drawdown exceeds a predefined limit (e.g., 20%).
7.  **1.3.7:** The `TradingEnv` shall terminate an episode prematurely if the remaining capital is insufficient to place a new trade.
8.  **1.3.8:** The `TradingEnv` shall add the normalized `distance_to_trail` to the agent's observation space.
9.  **1.3.9:** The `TradingEnv` shall provide reward shaping for trailing stops (bonus for holding a profitable position as the trailing stop improves, penalty for closing a profitable position prematurely when the trend is still strong).

## Epic 2 Meta-Learning (MAML) and Mixture of Experts (MoE) Implementation
**Expanded Goal:** This epic focuses on implementing the core Meta-Learning and Mixture of Experts architecture, enabling the RL agents to adapt and specialize across diverse market conditions. This involves completing the placeholder implementations for `learn` and `adapt` methods in specialized agents and the `MoEAgent`, refining the `GatingNetwork`, and fully implementing the outer loop meta-update in the `Trainer`.

### Story 2.1 Specialized Agent Implementation
As a developer,
I want the specialized RL agents to have fully functional learning and adaptation capabilities,
so that they can be effectively trained and integrated into the MoE framework.

#### Acceptance Criteria
1.  **2.1.1:** The `learn` method in each specialized agent (`TrendAgent`, `MeanReversionAgent`, `VolatilityAgent`, `ConsolidationAgent`) shall be implemented with the core PPO update logic (or chosen RL algorithm).
2.  **2.1.2:** The `adapt` method in each specialized agent shall implement the inner loop adaptation, including creating a differentiable copy of the agent's parameters and performing gradient steps on them.

### Story 2.2 MoEAgent and GatingNetwork Refinement
As a developer,
I want the `MoEAgent` to effectively orchestrate specialized agents and the `GatingNetwork` to dynamically select experts,
so that the overall system can leverage the "wisdom of the crowd" and adapt to market regimes.

#### Acceptance Criteria
1.  **2.2.1:** The `MoEAgent.select_action` method shall be changed from hard routing (`argmax`) to a weighted average of expert action probabilities.
2.  **2.2.2:** The `learn` method in `MoEAgent` shall orchestrate the learning for both the `GatingNetwork` and the individual experts.
3.  **2.2.3:** The `adapt` method in `MoEAgent` shall orchestrate the adaptation of both the `GatingNetwork` and the individual experts during the MAML inner loop.

### Story 2.3 Trainer Meta-Update Completion
As a developer,
I want the `Trainer` to fully implement the meta-optimization logic,
so that the `MoEAgent`'s meta-parameters can be updated based on meta-loss accumulated across tasks.

#### Acceptance Criteria
1.  **2.3.1:** The critical "outer loop meta-update" in `Trainer.meta_train` shall be fully implemented, enabling the meta-optimization of the `MoEAgent`.

## Epic 3 Scalable Training Architecture
**Expanded Goal:** This epic aims to develop a robust and scalable training architecture to efficiently handle large datasets and enable faster training of complex RL models. This includes optimizing data loading, implementing parallel training using an Actor-Learner architecture (e.g., Ray RLlib), and ensuring hardware optimization.

### Story 3.1 Data Streaming and On-Demand Loading
As a developer,
I want the data loading process to be efficient and scalable for large datasets,
so that the training process can handle millions of rows across hundreds of instruments and timeframes.

#### Acceptance Criteria
1.  **3.1.1:** The `DataLoader` shall be modified to provide data in smaller, manageable chunks or as an iterator.
2.  **3.1.2:** `TradingEnv.reset()` shall be updated to load only the necessary data segment for the current episode.
3.  **3.1.3:** Efficient data storage formats like Parquet or HDF5 shall be considered for data streaming and on-demand loading.

### Story 3.2 Parallel Training Implementation
As a developer,
I want to implement parallel training for RL agents,
so that training times can be significantly reduced and the system can handle complex models.

#### Acceptance Criteria
1.  **3.2.1:** The system shall adopt an Actor-Learner architecture for parallel training, utilizing a framework like Ray RLlib.
2.  **3.2.2:** Multiple processes (Actors) shall interact with `TradingEnv` instances in parallel, collecting experiences.
3.  **3.2.3:** A central process (Learner) shall update the agent's model based on experiences from actors.
4.  **3.2.4:** A shared, distributed experience replay buffer shall be implemented.
5.  **3.2.5:** The Learner shall periodically send updated weights to actors for model synchronization.

### Story 3.3 Hardware Optimization
As a developer,
I want to ensure optimal utilization of available hardware for training,
so that training is as fast and efficient as possible.

#### Acceptance Criteria
1.  **3.3.1:** PyTorch models and data shall be configured to utilize GPU (`.to('cuda')`) when available.
2.  **3.3.2:** Batch sizes shall be optimized for GPU utilization.
3.  **3.3.3:** `torch.backends.cudnn.benchmark = True` shall be enabled for performance optimization on GPUs.
4.  **3.3.4:** The system shall ensure graceful fallback to CPU if GPU/TPU is unavailable.
5.  **3.3.5:** Ray RLlib's capabilities for abstracting device management for distributed training shall be leveraged.

## Architect's Review and Recommendations

Overall, the PRD provides a clear and comprehensive outline of the project's goals, requirements, and a well-structured plan for implementation through epics and stories. The detailed functional and non-functional requirements, especially for the `BacktestingEngine` and `TradingEnv` enhancements, are well-articulated.

Here are my architectural observations and recommendations:

1.  **Clarity on "Monolithic, Locally Run Python Application" vs. Ray RLlib:**
    *   The "Service Architecture" in Technical Assumptions states "The bot will primarily operate as a monolithic, locally run Python application for execution."
    *   However, "Additional Technical Assumptions" and "Epic 3: Scalable Training Architecture" clearly state the intent to use Ray RLlib for distributed training, potentially in the cloud.
    *   **Architectural Recommendation:** While the execution might be local initially, the PRD should explicitly clarify the architectural distinction between the *execution environment* (local, monolithic) and the *training environment* (potentially distributed, cloud-based with Ray RLlib). This distinction is crucial for future infrastructure planning and resource allocation. It might be beneficial to add a section or a clear statement about the intended deployment model for training vs. live execution.

2.  **`Instrument` Class Location:**
    *   Story 1.1.1 suggests creating the `Instrument` class in `src/config/` or `src/instruments/`.
    *   **Architectural Recommendation:** The `instruments.yaml` is at `C:/AlgoTrading/config/instruments.yaml`. To keep the `Instrument` class logically grouped with its configuration, it should be placed within `src/config/`. Any loading utility should be in `src/utils/`.

3.  **Data Storage Formats (Parquet/HDF5):**
    *   NFR3 and Story 3.1.3 mention considering Parquet or HDF5.
    *   **Architectural Recommendation:** Given the emphasis on scalable training and handling millions of rows, **Parquet** is the recommended format for large-scale historical data storage and retrieval for processed data. Raw data will remain in CSV.

4.  **GPU/CPU Fallback and RLlib's Role:**
    *   Story 3.3.4 and 3.3.5 discuss graceful fallback to CPU and RLlib's abstraction of device management.
    *   **Architectural Recommendation:** This is a sound approach. Ensure that the implementation details for this abstraction are well-defined, especially how models and data are moved between devices and how RLlib's capabilities are leveraged to minimize manual device management.

5.  **Options Simulation Proxy Model:**
    *   The PRD details a proxy model for options simulation.
    *   **Architectural Recommendation:** The options proxy model as described in `docs\analysis_and_planning.md` will be used. It's important to ensure that the proxy model's assumptions and limitations are well-documented and understood, as they will directly impact the RL agent's learned behavior and its transferability to real options trading.

6.  **Meta-Learning and MoE Implementation:**
    *   The detailed steps for implementing `learn` and `adapt` methods in specialized agents and `MoEAgent` are clear.
    *   **Architectural Recommendation:** The success of the MoE will heavily depend on the `GatingNetwork`'s ability to accurately identify market regimes and weight experts. Consider adding a non-functional requirement or a specific task to evaluate the `GatingNetwork`'s performance and its impact on overall ensemble decision-making.

## Checklist Results Report
(This section will be populated after running the pm-checklist)

## Next Steps

### UX Expert Prompt
This section is not applicable as the core trading bot does not have a user-facing interface.

### Architect Prompt
This section will contain the prompt for the Architect, keep it short and to the point to initiate create architecture mode using this document as input.
