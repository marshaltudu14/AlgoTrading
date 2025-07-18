# Autonomous Trading Bot Product Requirements Document (PRD)

## Goals and Background Context

### Goals
- Maximize risk-adjusted profits through autonomous trading.
- Achieve true trading autonomy for users.
- Develop an instrument-agnostic Reinforcement Learning (RL) model capable of generating universal trading signals (BUY, SELL, HOLD, CLOSE).
- Enable rapid adaptation to changing market conditions and instrument types using Meta-Learning (MAML).
- Leverage a Mixture of Experts (MoE) framework to enhance trading performance and adaptability.

### Background Context
This PRD outlines the requirements for an autonomous algorithmic trading bot designed to provide traders with a fully automated solution for market operations. The core of this system is a sophisticated Reinforcement Learning (RL) model that employs multi-layer decision-making and adaptive learning to achieve true trading autonomy. The initial MVP focuses on delivering this fully autonomous RL agent, with the primary objective of maximizing profits for the end-user. The system is being developed with an emphasis on instrument agnosticism and rapid adaptation to diverse market conditions.

### Change Log
| Date       | Version | Description          | Author |
|------------|---------|----------------------|--------|
| 2025-07-18 | 0.1     | Initial Draft        | John   |

## Requirements

### Functional
- FR1: The system shall ingest historical OHLCV data for training solely from `data/final/`.
- FR2: The system shall generate technical indicators and price action features from raw market data for backtesting purposes.
- FR3: The RL model shall output discrete trading actions: `BUY_LONG`, `SELL_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`, `HOLD`.
- FR4: The system shall provide a backtesting environment for simulating trading strategies using `data/raw/**` initially, with the capability to integrate different datasets in the future.
- FR5: The system shall support meta-learning (MAML) for rapid adaptation of RL agents to new market regimes and instruments.
- FR6: The system shall incorporate a Mixture of Experts (MoE) framework to dynamically select or weight specialized agents based on market conditions.

### Non Functional
- NFR1: The RL model shall achieve a Sharpe Ratio > 1.5 during backtesting.
- NFR2: The RL model shall maintain a Maximum Drawdown < 15% during backtesting.
- NFR3: The RL model's decision-making process shall contribute to robust risk management by incorporating risk factors into its reward function and policy optimization.
- NFR4: The system shall provide monitoring and observability for RL model behavior and training performance.
- NFR5: The system shall be developed primarily using Python, with C++ for low-latency components (if applicable to the RL model itself).
- NFR6: The system shall utilize PyTorch for deep learning and Ray RLlib for distributed RL training.
- NFR7: The system shall comply with SEBI guidelines for algorithmic trading regulations, specifically concerning the auditable and explainable aspects of the RL model's decisions.

## User Interface Design Goals

### Overall UX Vision
The primary UX vision for this phase is to provide a clear, efficient, and informative experience for developers and data scientists interacting with the RL model. This includes easy initiation of training and backtesting runs, and comprehensive display of model performance and behavior directly within the terminal.

### Key Interaction Paradigms
Interaction will be exclusively through:
-   **Command-Line Interface (CLI):** For triggering training, backtesting, and data processing tasks. All outputs, including metrics and results, will be presented directly in the terminal in a clear and concise manner, avoiding overwhelming the user.

### Core Screens and Views
From a product perspective, the most critical "views" will be:
-   Backtesting Performance Summaries (e.g., key P&L figures, drawdown, trade counts) displayed in terminal.
-   Training Progress Updates (e.g., current episode, reward, loss) displayed in terminal.
-   Model Performance Metrics (e.g., Sharpe Ratio, Profit Factor, Win Rate) displayed in terminal.

### Accessibility: None
(As this is a developer-focused tool with a text-based interface, formal accessibility requirements are not applicable at this stage.)

### Branding
(No specific branding elements or style guides are required for this internal development tool.)

### Target Device and Platforms: Desktop Only (Developer Workstations)
(The primary platform will be developer workstations for running scripts and viewing terminal output.)

## Technical Assumptions

### Repository Structure: Polyrepo
(Assumption: Given the current focus on a single, specialized RL model component, a polyrepo structure where this component resides in its own repository is assumed. This can be revisited if the project expands to a larger monorepo strategy.)

### Service Architecture: Monolith (for the RL Model Component)
(Assumption: The RL model itself, including its agents, training, and backtesting components, will initially be developed as a cohesive, single application or service. This allows for focused development and easier management of dependencies within the RL system.)

### Testing Requirements: Full Testing Pyramid
(The project will implement a comprehensive testing strategy including unit tests for individual functions and modules, integration tests for component interactions (e.g., data loading, feature generation, model inference), and end-to-end tests for the full backtesting pipeline. This ensures the robustness and reliability of the RL model.)

### Additional Technical Assumptions and Requests
-   **Programming Languages**: Primary development will be in Python. C++ may be used for specific low-latency components within the RL model if identified as a performance bottleneck.
-   **Machine Learning Frameworks**: PyTorch will be used for deep learning model development. Ray RLlib will be utilized for distributed reinforcement learning training.
-   **Data Processing Libraries**: Pandas, NumPy, and Scikit-learn will be used for data manipulation, feature engineering, and general machine learning utilities.
-   **Data Sources**:
    -   **Training Data**: The RL model will exclusively use pre-processed data located in `data/final/` for training.
    -   **Backtesting Data**: The backtesting environment will initially use raw data from `data/raw/**`. The architecture will be designed to allow for easy integration of different backtesting datasets in the future.

## Epic List

1.  **Epic 1: Foundation & Core Backtesting Environment:** Establish the foundational project structure, integrate data loaders for `data/final` (training) and `data/raw` (backtesting), and implement the basic backtesting engine and environment to simulate trades and calculate rewards.
    #### Story 1.1 Project Setup & Basic Structure
    As a developer,
    I want the project to have a well-defined and organized directory structure,
    so that I can easily navigate the codebase and understand its components.

    *   **Acceptance Criteria**
        1.  The project root contains `src/`, `data/`, `docs/`, `tests/`, `config/` directories.
        2.  `src/` contains subdirectories for `agents`, `backtesting`, `config`, `data_processing`, `models`, `trading`, `training`, and `utils`.
        3.  A `requirements.txt` file lists all initial Python dependencies (e.g., pandas, numpy, pytorch, ray[rllib], fyers-apiv3, pyotp, requests, transformers, scikit-learn, joblib, pytz).
        4.  A basic `src/config/config.py` file is present, defining initial configuration parameters.

    #### Story 1.2 Data Loader for `data/final` (Training Data)
    As an RL engineer,
    I want a data loader that can efficiently load processed data from `data/final/`,
    so that I can feed it to the RL model for training.

    *   **Acceptance Criteria**
        1.  `src/utils/data_loader.py` contains a class `DataLoader` with a method `load_all_data()`.
        2.  `load_all_data()` successfully reads all CSV files from the `data/final/` directory.
        3.  `load_all_data()` concatenates the data into a single pandas DataFrame.
        4.  The loaded DataFrame retains all columns from the processed data.
        5.  Error handling is implemented for cases where `data/final/` is empty or contains corrupted files, logging warnings/errors without crashing.

    #### Story 1.3 Data Loader for `data/raw` (Backtesting Data)
    As an RL engineer,
    I want a data loader that can efficiently load raw OHLCV data from `data/raw/`,
    so that I can use it for backtesting the RL model.

    *   **Acceptance Criteria**
        1.  `src/utils/data_loader.py` contains a method `load_data_for_symbol(symbol: str)`.
        2.  `load_data_for_symbol()` successfully reads the specified CSV file (e.g., `Nifty_2.csv`) from `data/raw/` into a pandas DataFrame.
        3.  The loaded DataFrame contains `datetime`, `open`, `high`, `low`, `close` columns.
        4.  Basic data validation ensures `high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close` for each row.
        5.  Error handling is implemented for missing files or invalid data, logging warnings/errors without crashing.

    #### Story 1.4 Basic Backtesting Engine (`src/backtesting/engine.py`)
    As an RL engineer,
    I want a core backtesting engine that simulates trading mechanics,
    so that I can accurately evaluate the RL agent's performance.

    *   **Acceptance Criteria**
        1.  `src/backtesting/engine.py` contains a class `BacktestingEngine`.
        2.  `BacktestingEngine` can be initialized with a starting capital.
        3.  It supports `execute_trade(action, price, quantity)` method that updates internal capital and position.
        4.  It accurately tracks current capital, open positions (symbol, quantity, entry price), and unrealized/realized P&L.
        5.  Fixed brokerage costs (25 INR entry, 35 INR exit) are applied to each trade.
        6.  Negligible slippage is simulated (e.g., by executing at the provided `price` without modification).
        7.  It provides methods to retrieve current capital, position, and P&L.

    #### Story 1.5 Backtesting Environment (`src/backtesting/environment.py`)
    As an RL engineer,
    I want an RL environment that interfaces with the backtesting engine,
    so that I can train and evaluate RL agents.

    *   **Acceptance Criteria**
        1.  `src/backtesting/environment.py` contains a class `TradingEnv` that inherits from a suitable RL environment base class (e.g., OpenAI Gym's `gym.Env` if used, or a custom base).
        2.  `TradingEnv` can be initialized with a dataset (loaded via `DataLoader` from `data/raw`).
        3.  The `step(action)` method:
            *   Translates an RL agent's discrete action (`BUY_LONG`, `SELL_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`, `HOLD`) into `BacktestingEngine` calls.
            *   Calculates and returns a reward signal based on the change in portfolio value (P&L) per step, after accounting for transaction costs.
            *   Applies penalties for invalid actions (e.g., trying to sell when no position, or buying with insufficient capital).
            *   Returns the next observation, reward, `done` flag, and `info` dictionary.
        4.  The `reset()` method resets the environment to an initial state, returning the initial observation.
        5.  The observation space includes: OHLCV data for the current step, current capital, current position, and unrealized P&L.
        6.  All components of the observation space are normalized (e.g., using Z-score scaling).

2.  **Epic 2: Baseline RL Agent & Training Loop:** Implement a single, baseline Reinforcement Learning agent (e.g., PPO) with its neural network architecture (e.g., LSTM) and a functional training loop, enabling the agent to learn a basic trading policy within the established backtesting environment.
    #### Story 2.1 Base Agent Interface
    As an RL engineer,
    I want a common interface for all RL agents,
    so that I can ensure consistency and interchangeability between different agent implementations.

    *   **Acceptance Criteria**
        1.  `src/agents/base_agent.py` defines an abstract base class (e.g., `BaseAgent`) with abstract methods for `select_action(observation)`, `learn(experience)`, and `save_model(path)`.
        2.  The `select_action` method takes an observation (normalized state) and returns a discrete action.
        3.  The `learn` method takes an experience tuple (e.g., `(state, action, reward, next_state, done)`) and updates the agent's policy.
        4.  The `save_model` method saves the agent's learned policy to a specified path.

    #### Story 2.2 LSTM Neural Network Model
    As an RL engineer,
    I want a generic LSTM-based neural network model,
    so that it can process sequential market data and learn robust state representations for the RL agents.

    *   **Acceptance Criteria**
        1.  `src/models/lstm_model.py` contains a PyTorch `nn.Module` class (e.g., `LSTMModel`).
        2.  `LSTMModel` takes `input_dim`, `hidden_dim`, and `output_dim` as initialization parameters.
        3.  The model includes at least one LSTM layer followed by a linear layer to produce outputs suitable for action selection.
        4.  The `forward` method correctly processes sequential input data.
        5.  The model can be instantiated and its forward pass executed without errors.

    #### Story 2.3 PPO Agent Implementation
    As an RL engineer,
    I want a baseline Proximal Policy Optimization (PPO) agent,
    so that I can establish a performance benchmark and begin training the RL model.

    *   **Acceptance Criteria**
        1.  `src/agents/ppo_agent.py` contains a class `PPOAgent` that inherits from `BaseAgent`.
        2.  `PPOAgent` initializes with an `LSTMModel` for its policy and value networks.
        3.  It implements the `select_action` method to choose actions based on the current policy.
        4.  It implements the `learn` method to update the policy and value networks using the PPO algorithm.
        5.  The agent can be initialized and interact with the `TradingEnv` (from Epic 1) by selecting actions and receiving experiences.

    #### Story 2.4 RL Training Loop
    As an RL engineer,
    I want a training script that orchestrates the learning process,
    so that I can train the PPO agent within the backtesting environment.

    *   **Acceptance Criteria**
        1.  `src/training/trainer.py` contains a `Trainer` class or function.
        2.  The `Trainer` can be initialized with a `TradingEnv` instance and a `PPOAgent` instance.
        3.  It implements a training loop that:
            *   Resets the environment for each episode.
            *   Interacts with the environment by calling `agent.select_action()` and `env.step()`.
            *   Collects experiences (`(state, action, reward, next_state, done)`).
            *   Calls `agent.learn()` periodically to update the agent's policy.
            *   Logs training progress (e.g., episode rewards, losses) to the terminal.
        4.  The training loop can run for a specified number of episodes.
        5.  The `Trainer` can save the trained agent's model at the end of training.

3.  **Epic 3: Mixture of Experts (MoE) Framework:** Develop and integrate specialized trading agents (e.g., Trend, Mean Reversion) and a Gating Network to form a Mixture of Experts (MoE) architecture, allowing the system to dynamically select or combine agent decisions based on market regimes.
    #### Story 3.1 Specialized Trading Agents
    As an RL engineer,
    I want to implement specialized trading agents (e.g., Trend Following, Mean Reversion, Volatility, Consolidation),
    so that the MoE framework can leverage diverse strategies tailored to specific market conditions.

    *   **Acceptance Criteria**
        1.  `src/agents/trend_agent.py`, `src/agents/mean_reversion_agent.py`, `src/agents/volatility_agent.py`, and `src/agents/consolidation_agent.py` are created.
        2.  Each specialized agent class inherits from `BaseAgent` (from Story 2.1).
        3.  Each specialized agent utilizes an `LSTMModel` (or other appropriate `src/models` architecture) for its policy and value networks.
        4.  Each agent's `learn` method is designed to be trainable on specific market conditions (though the actual specialized training data filtering will be handled by the `Trainer` in a later story).
        5.  Each agent's `select_action` method outputs the same discrete trading actions (`BUY_LONG`, `SELL_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`, `HOLD`).

    #### Story 3.2 Gating Network Implementation
    As an RL engineer,
    I want a Gating Network that dynamically assesses the current market regime,
    so that it can assign appropriate weights to the specialized agents within the MoE framework.

    *   **Acceptance Criteria**
        1.  `src/agents/moe_agent.py` is updated to include a Gating Network component (e.g., a separate PyTorch `nn.Module`).
        2.  The Gating Network takes market state features (e.g., volatility measures, trend strength from observation space) as input.
        3.  It outputs probability weights for each specialized agent.
        4.  The Gating Network can be trained to optimize its weighting decisions.

    #### Story 3.3 MoE Agent & Consensus System
    As an RL engineer,
    I want a Mixture of Experts (MoE) agent that combines recommendations from specialized agents,
    so that the system can make a unified trading decision based on the perceived market regime.

    *   **Acceptance Criteria**
        1.  `src/agents/moe_agent.py` contains a class `MoEAgent` that inherits from `BaseAgent`.
        2.  `MoEAgent` initializes with instances of the specialized agents (from Story 3.1) and the Gating Network (from Story 3.2).
        3.  The `select_action` method in `MoEAgent` uses the Gating Network to get weights, then combines the actions/recommendations of the specialized agents based on these weights to produce a single, final trading action.
        4.  The `learn` method in `MoEAgent` orchestrates the learning for both the specialized agents and the Gating Network, potentially using a joint optimization strategy.
        5.  The `MoEAgent` can be instantiated and interact with the `TradingEnv`.

    #### Story 3.4 MoE Training Integration
    As an RL engineer,
    I want the training loop to support the Mixture of Experts architecture,
    so that I can effectively train the Gating Network and specialized agents for optimal ensemble performance.

    *   **Acceptance Criteria**
        1.  `src/training/trainer.py` is updated to support training the `MoEAgent`.
        2.  The training loop can handle the joint optimization of the Gating Network and the specialized agents.
        3.  The `Trainer` can log performance metrics relevant to the MoE (e.g., which agents are activated under which conditions, individual agent performance).
        4.  The `Trainer` can save the complete `MoEAgent` model, including all specialized agents and the Gating Network.

4.  **Epic 4: Meta-Learning (MAML) for Adaptability:** Implement the Meta-Learning (MAML) algorithm to enable the RL agents to rapidly adapt to new market conditions, instruments, and timeframes with minimal retraining, enhancing the model's generalization capabilities.
    #### Story 4.1 Task Sampling for Meta-Training
    As an RL engineer,
    I want a mechanism to sample diverse tasks (instrument/timeframe combinations) from the `data/final` dataset,
    so that I can effectively meta-train the RL agents for rapid adaptation.

    *   **Acceptance Criteria**
        1.  `src/utils/data_loader.py` (or a new utility in `src/training/`) contains a function or method to identify unique `(instrument_type, timeframe)` combinations within the `data/final` dataset.
        2.  This function can sample a batch of these unique combinations, representing distinct meta-training tasks.
        3.  Each sampled task provides the necessary data subset for a training episode.
        4.  The sampling mechanism ensures diversity across tasks to promote generalization.

    #### Story 4.2 Inner Loop (Adaptation) Implementation
    As an RL engineer,
    I want to implement the inner loop of the MAML algorithm,
    so that the RL agent can quickly adapt its policy to a specific sampled task.

    *   **Acceptance Criteria**
        1.  The `BaseAgent` (or a new MAML-specific base class) is updated to support temporary, task-specific parameter updates without affecting the global meta-parameters.
        2.  The `learn` method of the RL agents (PPO and MoE) can perform a specified number of gradient steps on a given task's data, effectively adapting the agent's policy to that task.
        3.  This adaptation process is isolated for each task within the inner loop.

    #### Story 4.3 Outer Loop (Meta-Update) Implementation
    As an RL engineer,
    I want to implement the outer loop of the MAML algorithm,
    so that the meta-learner can update the agent's initial parameters based on performance across adapted tasks.

    *   **Acceptance Criteria**
        1.  `src/training/trainer.py` is updated to manage the MAML outer loop.
        2.  After performing inner-loop adaptations for a batch of tasks, the `Trainer` calculates a meta-loss based on the performance of the adapted policies on new data from those tasks.
        3.  The `Trainer` then updates the agent's global (meta) parameters using this meta-loss, aiming for an initialization that facilitates rapid learning on new, unseen tasks.
        4.  The meta-update process correctly propagates gradients through the inner-loop computations.

    #### Story 4.4 MAML Integration with Trainer
    As an RL engineer,
    I want the `Trainer` to orchestrate the full MAML meta-training process,
    so that I can effectively train the RL agents for rapid adaptability.

    *   **Acceptance Criteria**
        1.  `src/training/trainer.py` fully integrates the task sampling (Story 4.1), inner loop (Story 4.2), and outer loop (Story 4.3) into a cohesive meta-training workflow.
        2.  The `Trainer` can run MAML for a specified number of meta-iterations.
        3.  The `Trainer` can save the meta-trained agent's initial parameters, which are optimized for rapid adaptation.
        4.  Terminal output clearly indicates the progress of meta-training, including meta-losses and adaptation performance.

5.  **Epic 5: Comprehensive Evaluation & Reporting:** Enhance the backtesting and training evaluation framework with advanced performance metrics, robust reporting mechanisms, and clear, concise terminal-based output for monitoring model behavior and performance.
    #### Story 5.1 Standard Trading Performance Metrics
    As an RL engineer,
    I want to calculate standard trading performance metrics,
    so that I can rigorously evaluate the profitability and risk of the RL agent's strategies.

    *   **Acceptance Criteria**
        1.  `src/utils/metrics.py` contains functions to calculate:
            *   Sharpe Ratio
            *   Total P&L (absolute profit generated)
            *   Profit Factor (ratio of gross profit to gross loss)
            *   Maximum Drawdown
            *   Win Rate (percentage of profitable trades)
            *   Average P&L per Trade
            *   Number of Trades
        2.  These functions take a trade history (e.g., a list of trade objects or a DataFrame from the `BacktestingEngine`) as input.
        3.  All metrics are calculated correctly and handle edge cases (e.g., no trades, all losing trades).

    #### Story 5.2 Comprehensive Backtesting Reports
    As an RL engineer,
    I want to generate comprehensive backtesting reports,
    so that I can analyze the RL agent's performance in detail and identify areas for improvement.

    *   **Acceptance Criteria**
        1.  The `Trainer` (or a new reporting module) can generate a summary report of backtesting results.
        2.  The report includes all metrics from Story 5.1.
        3.  The report is displayed in the terminal in a clear, well-formatted, and concise manner.
        4.  The report highlights key performance indicators (e.g., Sharpe Ratio, Max Drawdown, Total P&L).
        5.  The report can be generated for both baseline PPO and MoE agents.

    #### Story 5.3 Real-time Training Progress Updates
    As an RL engineer,
    I want to see real-time updates on the training progress,
    so that I can monitor the learning process and identify potential issues early.

    *   **Acceptance Criteria**
        1.  The `Trainer` provides concise, terminal-based updates during the training loop.
        2.  Updates include: current episode number, current reward, average reward over a window, and potentially loss values.
        3.  Updates are non-intrusive and do not flood the terminal, perhaps updating on the same line or at a fixed interval.
        4.  The updates are clear and easy to understand at a glance.

    #### Story 5.4 Clear and Concise Terminal Output
    As an RL engineer,
    I want all terminal output to be clear, concise, and easy to understand,
    so that I can quickly grasp the results and status of the RL model.

    *   **Acceptance Criteria**
        1.  All output from training, backtesting, and reporting modules is formatted for readability in the terminal.
        2.  Key information is highlighted or easily discernible.
        3.  Unnecessary verbose logging is suppressed by default.
        4.  Error messages are clear and actionable.
        5.  The overall terminal experience is not overwhelming, adhering to the "without being too overwhelming or hard to understand" principle.

## Checklist Results Report
(Skipped as per user request.)

## Next Steps

### UX Expert Prompt
Given the CLI-only nature of the UI for this phase, the UX Expert's role will be minimal. However, they could provide valuable input on ensuring the terminal output is as clear, concise, and user-friendly as possible for developers and data scientists. They could also advise on the structure and presentation of metrics to maximize readability and comprehension within a text-based interface.

### Architect Prompt
This PRD outlines the functional and non-functional requirements, as well as the technical assumptions for the Reinforcement Learning model component of the Autonomous Trading Bot. The Architect should now proceed with designing the detailed technical architecture, including class diagrams, module interactions, and specific technology choices for implementing the RL agents, backtesting environment, training loops, and evaluation metrics as described in this document. Special attention should be paid to the Meta-Learning (MAML) and Mixture of Experts (MoE) frameworks, ensuring scalability and maintainability.
