# Detailed Reinforcement Learning Model Architecture

## 1. Executive Summary

This document provides a detailed architectural design for the Reinforcement Learning (RL) component of the Autonomous Trading Bot, building upon the high-level vision outlined in `mainIdea.md` and `docs/architecture/rl-model-architecture.md`, and fulfilling the requirements specified in `docs/prd-live-trading-mvp.md`. The core principle is to develop an **instrument-agnostic** RL model capable of generating universal trading signals across diverse market data, emphasizing **Meta-Learning (MAML)** for rapid adaptation and a **Mixture of Experts (MoE)** framework for robust decision-making. The focus is on the RL model's development, training, and backtesting, with a CLI-only interface for developers.

## 2. Core Architectural Principles

*   **Instrument Agnosticism:** Input features will be normalized and standardized to allow the RL model to learn patterns independent of specific instrument types or timeframes.
*   **Simplified Action Space:** The model's output will be limited to four fundamental trading actions: `BUY_LONG`, `SELL_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`, `HOLD`.
*   **Meta-Learning (MAML):** The system will incorporate MAML to enable RL agents to quickly adapt to new market regimes, instruments, or timeframes with minimal additional training.
*   **Mixture of Experts (MoE):** Specialized agents will focus on different market conditions, with a gating network dynamically selecting or weighting their outputs based on the perceived market regime.
*   **Realistic Simulation:** The backtesting environment will accurately simulate Indian market conditions, including fixed brokerage costs (25 INR entry, 35 INR exit) and negligible slippage.
*   **CLI-Only Interface:** All interactions, metrics, and results will be displayed directly in the terminal, ensuring clarity and conciseness for developers.

## 3. Architectural Components and Detailed Design (Epic by Epic)

### Epic 1: Foundation & Core Backtesting Environment

**Objective:** To establish the fundamental components for data handling, simulated trading, and the Reinforcement Learning environment, providing a stable base for agent development.

#### 1. `src/utils/data_loader.py` - Data Loading Module

*   **Purpose:** Centralized and efficient loading of historical market data for both training and backtesting. It abstracts away the file system details and provides validated dataframes.
*   **Class:** `DataLoader`
    *   **Responsibilities:**
        *   Locate and read CSV files from specified directories (`data/final/` for training, `data/raw/` for backtesting).
        *   Concatenate multiple data files into a single DataFrame for training.
        *   Load individual data files for specific symbols for backtesting.
        *   Perform basic data validation (e.g., required columns, OHLC relationships) and cleaning (e.g., timezone conversion, duplicate removal).
    *   **Key Methods:**
        *   `__init__(self, final_data_dir: str = "data/final", raw_data_dir: str = "data/raw")`: Initializes the loader with paths to processed and raw data directories.
        *   `load_all_processed_data(self) -> pd.DataFrame`: Loads and concatenates all CSV files from `data/final/`. Returns a single DataFrame.
        *   `load_raw_data_for_symbol(self, symbol: str) -> pd.DataFrame`: Loads a specific CSV file (e.g., `Nifty_2.csv`) from `data/raw/`. Performs OHLC validation and returns a DataFrame.
    *   **Interactions:**
        *   Used by `src/backtesting/environment.py` (`TradingEnv`) to load data for backtesting.
        *   Will be used by `src/training/trainer.py` to load training data.

#### 2. `src/backtesting/engine.py` - Backtesting Engine

*   **Purpose:** To simulate the core mechanics of a trading account, including capital management, position tracking, and the application of transaction costs. It acts as the financial ledger for the simulated environment.
*   **Class:** `BacktestingEngine`
    *   **Responsibilities:**
        *   Manage the simulated trading capital.
        *   Track open positions (symbol, quantity, entry price, side).
        *   Calculate and update realized and unrealized Profit & Loss (P&L).
        *   Apply fixed brokerage costs (25 INR entry, 35 INR exit) to each trade.
        *   Simulate negligible slippage (executing trades at the provided price).
        *   Provide the current state of the trading account.
    *   **Key Methods:**
        *   `__init__(self, initial_capital: float)`: Sets the starting capital and initializes all account variables.
        *   `execute_trade(self, action: str, price: float, quantity: float) -> Tuple[float, float]`: Processes a trade (`BUY_LONG`, `SELL_SHORT`, `CLOSE_LONG`, `CLOSE_SHORT`). Updates capital, position, and P&L. Returns `(realized_pnl_from_this_trade, current_unrealized_pnl)`.
        *   `get_account_state(self) -> Dict[str, float]`: Returns a dictionary with `capital`, `current_position_quantity`, `current_position_entry_price`, `realized_pnl`, `unrealized_pnl`.
        *   `reset(self)`: Resets the engine to its initial state (initial capital, no positions).
    *   **Internal State:** `_capital`, `_current_position` (dict), `_realized_pnl`, `_unrealized_pnl`.
    *   **Interactions:**
        *   Called by `src/backtesting/environment.py` (`TradingEnv`) to perform simulated trade executions.

#### 3. `src/backtesting/environment.py` - Trading Environment

*   **Purpose:** To serve as the Reinforcement Learning environment, providing the "world" for RL agents to interact with. It translates agent actions into simulated trades, calculates rewards, and constructs observations. This will adhere to the OpenAI Gym API for compatibility with RL frameworks.
*   **Class:** `TradingEnv` (inherits from `gym.Env` for standard RL integration)
    *   **Responsibilities:**
        *   Load and manage the historical data for a specific backtesting run.
        *   Interface with the `BacktestingEngine` to execute trades.
        *   Construct the observation space for the RL agent, including market data, engineered features (implicitly from `data/final` or `data/raw` after feature generation), and account state.
        *   Normalize all components of the observation space (e.g., using Z-score scaling).
        *   Calculate and provide reward signals to the agent based on P&L changes and penalties for invalid actions.
        *   Manage the episode lifecycle (start, end).
    *   **Key Methods:**
        *   `__init__(self, data_loader: DataLoader, symbol: str, initial_capital: float, lookback_window: int = 50)`: Initializes the environment. Sets up `observation_space` and `action_space`.
        *   `reset(self) -> np.ndarray`: Resets the environment to an initial state, loads data for the specified symbol, resets the `BacktestingEngine`, and returns the initial observation.
        *   `step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]`: Takes a discrete action from the agent.
            *   Translates the action into `BacktestingEngine` calls.
            *   Advances the simulation by one time step.
            *   Calculates the reward.
            *   Constructs the next observation.
            *   Determines if the episode is `done`.
            *   Returns `(observation, reward, done, info)`.
        *   `_get_observation(self) -> np.ndarray`: Internal method to construct the current observation by combining market data, features, and normalized account state.
        *   `_calculate_reward(self, prev_pnl: float, current_pnl: float, is_invalid_action: bool) -> float`: Internal method to compute the reward signal.
    *   **Observation Space:** A `gym.spaces.Box` representing the normalized concatenation of:
        *   Lookback window of OHLCV data (e.g., 50 periods).
        *   Engineered features for the current period (assumed to be part of the loaded data).
        *   Normalized current capital, current position (quantity, entry price), and normalized unrealized P&L.
    *   **Action Space:** A `gym.spaces.Discrete(5)` representing: `0: BUY_LONG`, `1: SELL_SHORT`, `2: CLOSE_LONG`, `3: CLOSE_SHORT`, `4: HOLD`.
    *   **Interactions:**
        *   Uses `src/utils/data_loader.py` (`DataLoader`) to load historical data.
        *   Uses `src/backtesting/engine.py` (`BacktestingEngine`) to simulate trades and manage the account.
        *   Provides the interface for `src/training/trainer.py` to interact with the RL environment.

### Epic 2: Baseline RL Agent & Training Loop

**Objective:** To implement the fundamental components for a Reinforcement Learning agent, including its abstract interface, neural network architecture, a concrete PPO agent, and the training orchestration, enabling the agent to learn within the simulated backtesting environment.

#### 1. `src/agents/base_agent.py` - Base Agent Interface

*   **Purpose:** To define a common, abstract interface that all Reinforcement Learning agents in the system must adhere to. This ensures consistency and allows for easy interchangeability of different agent implementations.
*   **Class:** `BaseAgent` (Abstract Base Class)
    *   **Responsibilities:**
        *   Enforce a standard set of methods for all RL agents.
    *   **Key Abstract Methods:**
        *   `select_action(self, observation: np.ndarray) -> int`: Takes a normalized observation from the environment and returns a discrete action (integer representing `BUY_LONG`, `SELL_SHORT`, etc.).
        *   `learn(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None`: Processes a single or batch of experiences (`(state, action, reward, next_state, done)`) to update the agent's internal policy.
        *   `save_model(self, path: str) -> None`: Saves the agent's learned policy (e.g., neural network weights) to a specified file path.
        *   `load_model(self, path: str) -> None`: Loads a previously saved policy from a specified file path.
    *   **Interactions:**
        *   All concrete agent implementations (e.g., `PPOAgent`, `MoEAgent`) will inherit from this class.
        *   `src/training/trainer.py` will interact with agents through this interface.

#### 2. `src/models/lstm_model.py` - LSTM Neural Network Model

*   **Purpose:** To provide a generic, reusable LSTM-based neural network architecture capable of processing sequential market data. This model will serve as the policy and/or value network for various RL agents.
*   **Class:** `LSTMModel` (inherits from `torch.nn.Module`)
    *   **Responsibilities:**
        *   Process sequential input data (e.g., historical OHLCV and features).
        *   Learn robust state representations from the time-series data.
        *   Output values suitable for action probabilities (policy) or state-value estimation (value function).
    *   **Key Methods:**
        *   `__init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1)`: Initializes the LSTM layers and linear layers.
        *   `forward(self, x: torch.Tensor) -> torch.Tensor`: Defines the forward pass of the network.
            *   Input `x` is expected to be `(batch_size, sequence_length, input_dim)`.
            *   Passes `x` through LSTM layers.
            *   Passes the output of the LSTM (e.g., last hidden state) through a linear layer to produce the final output.
    *   **Internal Components:** `torch.nn.LSTM`, `torch.nn.Linear`.
    *   **Interactions:**
        *   Instantiated and used by `src/agents/ppo_agent.py` (and later `moe_agent.py`) to build their policy and value networks.

#### 3. `src/agents/ppo_agent.py` - PPO Agent Implementation

*   **Purpose:** To provide a concrete implementation of the Proximal Policy Optimization (PPO) algorithm, serving as the baseline RL agent for initial training and performance benchmarking.
*   **Class:** `PPOAgent` (inherits from `BaseAgent`)
    *   **Responsibilities:**
        *   Maintain and update a policy network and a value network (both using `LSTMModel`).
        *   Select actions based on the current policy's probability distribution.
        *   Compute advantages and value targets for policy updates.
        *   Implement the PPO update rule using collected experiences.
    *   **Key Methods:**
        *   `__init__(self, observation_dim: int, action_dim: int, hidden_dim: int, lr_actor: float, lr_critic: float, gamma: float, epsilon_clip: float, k_epochs: int)`: Initializes actor (policy) and critic (value) networks, optimizers, and PPO-specific hyperparameters.
        *   `select_action(self, observation: np.ndarray) -> int`:
            *   Converts `observation` to `torch.Tensor`.
            *   Passes through the actor network to get action probabilities.
            *   Samples an action from the distribution.
            *   Returns the selected action (integer).
        *   `learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None`:
            *   Processes a batch of `experiences`.
            *   Calculates advantages using Generalized Advantage Estimation (GAE) or similar.
            *   Performs `k_epochs` of policy and value network updates using the PPO clipping objective.
            *   Clears the experience buffer after updates.
    *   **Internal Components:** Instances of `LSTMModel` (for actor and critic), `torch.optim.Adam`, experience buffer.
    *   **Interactions:**
        *   Interacts with `src/backtesting/environment.py` (`TradingEnv`) by receiving observations and returning actions.
        *   Receives experiences from `TradingEnv` via `src/training/trainer.py`.
        *   Uses `LSTMModel` for its neural network architectures.

#### 4. `src/training/trainer.py` - RL Training Loop

*   **Purpose:** To orchestrate the entire Reinforcement Learning training process. It manages the interaction between the RL agent and the environment, handles experience collection, triggers learning updates, and logs training progress.
*   **Class:** `Trainer`
    *   **Responsibilities:**
        *   Initialize the `TradingEnv` and the chosen `BaseAgent` (e.g., `PPOAgent`).
        *   Run the main training loop for a specified number of episodes.
        *   Collect experiences from environment interactions.
        *   Periodically call the agent's `learn` method.
        *   Log training progress (episode rewards, losses, etc.) to the terminal.
        *   Save the trained agent's model.
    *   **Key Methods:**
        *   `__init__(self, env: TradingEnv, agent: BaseAgent, num_episodes: int, log_interval: int = 10)`: Initializes the trainer with the environment, agent, and training parameters.
        *   `train(self) -> None`:
            *   Main training loop:
                *   For each episode:
                    *   `observation = env.reset()`
                    *   Loop until `done`:
                        *   `action = agent.select_action(observation)`
                        *   `next_observation, reward, done, info = env.step(action)`
                        *   Collect `(observation, action, reward, next_observation, done)` into an experience buffer.
                        *   `observation = next_observation`
                    *   After episode, call `agent.learn(experience_buffer)`.
                    *   Log episode results to terminal (e.g., total reward, episode length).
                    *   Periodically save the agent's model.
        *   `_log_progress(self, episode: int, total_reward: float, loss: float = None) -> None`: Internal method for formatted terminal output of training progress.
    *   **Interactions:**
        *   Uses `src/backtesting/environment.py` (`TradingEnv`) to simulate market interactions.
        *   Uses `src/agents/ppo_agent.py` (or any `BaseAgent` implementation) to select actions and learn.
        *   Will use `src/utils/metrics.py` (from Epic 5) for more detailed evaluation during training.

### Epic 3: Mixture of Experts (MoE) Framework

**Objective:** To enhance the RL model's adaptability and performance by implementing a Mixture of Experts (MoE) architecture. This involves creating specialized trading agents, each focusing on different market conditions, and developing a Gating Network to dynamically select or weight their outputs. This will allow the overall system to leverage diverse strategies and make more robust decisions across varying market regimes.

#### 1. `src/agents/` - Specialized Trading Agents

*   **Purpose:** To develop individual RL agents, each designed to excel in specific market conditions (e.g., trending, ranging, volatile). These agents will form the "experts" within the MoE framework.
*   **Classes:**
    *   `TrendAgent` (inherits from `BaseAgent`)
    *   `MeanReversionAgent` (inherits from `BaseAgent`)
    *   `VolatilityAgent` (inherits from `BaseAgent`)
    *   `ConsolidationAgent` (inherits from `BaseAgent`)
*   **Responsibilities (for each specialized agent):**
    *   Implement the `select_action` and `learn` methods as defined by `BaseAgent`.
    *   Utilize an `LSTMModel` (or potentially other `src/models` architectures if deemed more suitable for their specialization, e.g., CNN for pattern recognition) for their policy and value networks.
    *   Their `learn` method will be designed to optimize for performance within their specific market regime.
*   **Key Methods (for each specialized agent):**
    *   `__init__(self, observation_dim: int, action_dim: int, hidden_dim: int, ...)`: Standard agent initialization.
    *   `select_action(self, observation: np.ndarray) -> int`: Selects an action based on its specialized policy.
    *   `learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None`: Updates its policy based on collected experiences.
*   **Interactions:**
    *   These agents will be instantiated and managed by the `MoEAgent`.
    *   They will receive observations and provide actions/recommendations to the `MoEAgent`.

#### 2. `src/agents/moe_agent.py` - Gating Network Implementation

*   **Purpose:** To dynamically assess the current market regime and determine which specialized agent(s) are most relevant or trustworthy for the given conditions. It assigns probability weights to each expert.
*   **Class:** `GatingNetwork` (inherits from `torch.nn.Module`)
    *   **Responsibilities:**
        *   Take relevant market state features as input.
        *   Output a probability distribution (weights) over the specialized agents.
    *   **Key Methods:**
        *   `__init__(self, input_dim: int, num_experts: int, hidden_dim: int)`: Initializes the network layers.
        *   `forward(self, market_features: torch.Tensor) -> torch.Tensor`:
            *   Input `market_features` will be a subset of the observation space relevant for regime detection (e.g., volatility measures, trend strength, specific indicators).
            *   Outputs a tensor of shape `(batch_size, num_experts)` representing the weights (e.g., using `softmax` for probabilities).
*   **Interactions:**
    *   Instantiated and used by the `MoEAgent`.
    *   Receives market features from the `MoEAgent` (derived from the environment's observation).

#### 3. `src/agents/moe_agent.py` - MoE Agent & Consensus System

*   **Purpose:** To act as the central coordinator for the Mixture of Experts. It combines the recommendations of the specialized agents, weighted by the Gating Network, to produce a single, unified trading decision.
*   **Class:** `MoEAgent` (inherits from `BaseAgent`)
    *   **Responsibilities:**
        *   Manage and coordinate multiple `BaseAgent` instances (the specialized experts).
        *   Utilize the `GatingNetwork` to determine expert weights.
        *   Combine expert recommendations into a single action.
        *   Orchestrate the learning process for both the Gating Network and the individual experts.
    *   **Key Methods:**
        *   `__init__(self, observation_dim: int, action_dim: int, hidden_dim: int, expert_configs: Dict)`: Initializes the `GatingNetwork` and instances of all specialized agents.
        *   `select_action(self, observation: np.ndarray) -> int`:
            *   Extracts market features relevant for the `GatingNetwork` from the `observation`.
            *   Passes these features to the `GatingNetwork` to get expert weights.
            *   For each expert, calls `expert.select_action(observation)` to get their individual recommendation.
            *   Combines these recommendations using the weights (e.g., weighted average of action probabilities, or selecting the action of the highest-weighted expert).
            *   Returns the final discrete action.
        *   `learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None`:
            *   This method will be more complex, potentially involving:
                *   Distributing experiences to relevant experts for their individual learning.
                *   Updating the `GatingNetwork` based on the performance of the experts it selected.
                *   Implementing a joint optimization strategy for the entire MoE system.
        *   `save_model(self, path: str) -> None`: Saves the state of the `MoEAgent`, including the `GatingNetwork` and all specialized experts.
        *   `load_model(self, path: str) -> None`: Loads the state of the `MoEAgent`.
*   **Interactions:**
    *   Interacts with `src/backtesting/environment.py` (`TradingEnv`) to receive observations and return actions.
    *   Manages and calls methods on the specialized agents and the `GatingNetwork`.

#### 4. `src/training/trainer.py` - MoE Training Integration

*   **Purpose:** To adapt the existing `Trainer` to effectively train the complex `MoEAgent`, handling the joint optimization of multiple experts and the gating mechanism.
*   **Class:** `Trainer` (updates to existing class)
    *   **Responsibilities:**
        *   Initialize the `TradingEnv` and an `MoEAgent` instance.
        *   Manage the training loop, ensuring experiences are collected and passed to the `MoEAgent`'s `learn` method.
        *   Log performance metrics relevant to the MoE, such as which experts are activated under different market conditions, and potentially individual expert performance.
    *   **Key Updates to `train(self)` method:**
        *   The core loop remains similar, but `agent.learn()` will now call the `MoEAgent`'s learning logic, which in turn orchestrates the learning of its sub-components.
        *   Logging will be enhanced to provide insights into the MoE's behavior.
*   **Interactions:**
    *   Interacts with `src/backtesting/environment.py` (`TradingEnv`).
    *   Interacts with `src/agents/moe_agent.py` (`MoEAgent`).

### Epic 4: Meta-Learning (MAML) for Adaptability

**Objective:** To implement the Meta-Learning (MAML) algorithm, enabling the RL agents (both baseline PPO and MoE) to quickly adapt to new market conditions, instruments, or timeframes with minimal additional training. This will significantly improve the model's generalization capabilities.

#### 1. `src/utils/data_loader.py` (or new utility) - Task Sampling for Meta-Training

*   **Purpose:** To provide a robust mechanism for identifying and sampling diverse "tasks" for meta-training. In this context, a task is defined as a unique `(instrument_type, timeframe)` combination from the `data/final` dataset.
*   **Class/Function:** `DataLoader` (enhancement) or a new `MetaTaskSampler` utility.
    *   **Responsibilities:**
        *   Scan `data/final/` to identify all available instrument/timeframe combinations.
        *   Provide methods to sample a batch of these combinations for a meta-training iteration.
        *   For each sampled task, provide access to its corresponding data subset.
    *   **Key Methods (additions to `DataLoader` or new `MetaTaskSampler`):**
        *   `get_available_tasks(self) -> List[Tuple[str, str]]`: Scans `data/final/` and returns a list of `(symbol, timeframe)` tuples.
        *   `sample_tasks(self, num_tasks: int) -> List[Tuple[str, str]]`: Randomly samples `num_tasks` from the available tasks.
        *   `get_task_data(self, symbol: str, timeframe: str) -> pd.DataFrame`: Loads the specific data for a given task.
    *   **Interactions:**
        *   Used by `src/training/trainer.py` to select tasks for each meta-training iteration.

#### 2. `src/agents/base_agent.py` - Inner Loop (Adaptation) Implementation

*   **Purpose:** To enable RL agents to perform rapid, task-specific adaptations without permanently altering their global meta-parameters. This involves creating a temporary, differentiable copy of the agent's parameters.
*   **Class:** `BaseAgent` (enhancement)
    *   **Responsibilities:**
        *   Provide a mechanism to create a "fast-adapted" version of the agent for the inner loop.
        *   Allow for gradient computations through this adaptation process.
    *   **Key Methods (additions to `BaseAgent`):**
        *   `adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> BaseAgent`:
            *   Creates a temporary copy of the agent's policy and value network parameters.
            *   Performs `num_gradient_steps` of a standard RL update (e.g., PPO update) using the provided experience on these temporary parameters.
            *   Returns a new `BaseAgent` instance (or a representation of the adapted parameters) that is differentiable with respect to the original meta-parameters.
    *   **Implementation Detail:** This will likely involve using PyTorch's `clone()` and `detach()` methods carefully, or a library like `higher` for automatic differentiation through optimization steps. The `learn` method of `PPOAgent` and `MoEAgent` will be modified to support this inner-loop adaptation.
    *   **Interactions:**
        *   Called by `src/training/trainer.py` during the inner loop of MAML.

#### 3. `src/training/trainer.py` - Outer Loop (Meta-Update) Implementation

*   **Purpose:** To manage the meta-learning process, where the agent's initial parameters are updated based on how well they adapt to a batch of diverse tasks. This aims to find a good initialization that allows for rapid learning on *new*, unseen tasks.
*   **Class:** `Trainer` (enhancement)
    *   **Responsibilities:**
        *   Orchestrate the sampling of tasks.
        *   Execute the inner-loop adaptation for each task.
        *   Compute the meta-loss based on the performance of the adapted agents on new data from their respective tasks.
        *   Perform a meta-update step on the agent's global (meta) parameters.
    *   **Key Methods (additions/modifications to `Trainer`):**
        *   `meta_train(self, num_meta_iterations: int, num_inner_loop_steps: int, num_evaluation_steps: int, meta_batch_size: int) -> None`:
            *   Main MAML training loop:
                *   For `num_meta_iterations`:
                    *   `tasks = self.data_loader.sample_tasks(meta_batch_size)` (using `MetaTaskSampler` or `DataLoader` enhancement).
                    *   Initialize `meta_loss = 0`.
                    *   For each `task` in `tasks`:
                        *   Load data for `task`.
                        *   Run `num_inner_loop_steps` of adaptation using `agent.adapt()`.
                        *   Evaluate the adapted agent on a separate set of data from the same `task` for `num_evaluation_steps`.
                        *   Accumulate the loss from this evaluation into `meta_loss`.
                    *   Perform a gradient step on `meta_loss` to update the agent's global parameters.
                    *   Log meta-training progress.
    *   **Interactions:**
        *   Uses `src/utils/data_loader.py` (or `MetaTaskSampler`) for task sampling.
        *   Calls the `adapt` method of the `BaseAgent` (or its concrete implementations).
        *   Manages the overall MAML optimization process.

#### 4. `src/training/trainer.py` - MAML Integration with Trainer

*   **Purpose:** To ensure the `Trainer` seamlessly orchestrates the entire MAML meta-training process, providing clear terminal output and saving the meta-trained model.
*   **Class:** `Trainer` (enhancement)
    *   **Responsibilities:**
        *   Provide a unified interface for initiating MAML training.
        *   Handle the saving and loading of meta-trained models.
        *   Present clear and concise terminal output for MAML progress.
    *   **Key Methods (modifications to `Trainer`):**
        *   The `train` method will be extended or a new `meta_train` method will be the primary entry point for MAML.
        *   `save_meta_model(self, path: str) -> None`: Saves the agent's meta-parameters.
        *   `_log_meta_progress(self, iteration: int, meta_loss: float, avg_task_reward: float) -> None`: Formatted terminal output for MAML progress.
    *   **Interactions:**
        *   Coordinates all components of the MAML framework.

### Epic 5: Comprehensive Evaluation & Reporting

**Objective:** To provide robust and clear insights into the RL model's performance during both training and backtesting. This involves implementing a suite of standard trading performance metrics, generating comprehensive backtesting reports, and ensuring that all evaluation results and training progress are presented clearly and concisely in the terminal.

#### 1. `src/utils/metrics.py` - Standard Trading Performance Metrics

*   **Purpose:** To provide a collection of functions for calculating key financial and trading performance metrics from a series of trades or portfolio values.
*   **Class/Module:** `metrics.py` (collection of static functions)
    *   **Responsibilities:**
        *   Calculate various profitability and risk-adjusted return metrics.
        *   Handle edge cases (e.g., division by zero, empty trade lists).
    *   **Key Functions:**
        *   `calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float`: Computes the Sharpe Ratio.
        *   `calculate_total_pnl(trade_history: List[Dict]) -> float`: Sums up the P&L from all trades.
        *   `calculate_profit_factor(trade_history: List[Dict]) -> float`: Calculates the ratio of gross profit to gross loss.
        *   `calculate_max_drawdown(equity_curve: pd.Series) -> float`: Determines the maximum peak-to-trough decline in equity.
        *   `calculate_win_rate(trade_history: List[Dict]) -> float`: Computes the percentage of profitable trades.
        *   `calculate_avg_pnl_per_trade(trade_history: List[Dict]) -> float`: Calculates the average profit or loss per trade.
        *   `calculate_num_trades(trade_history: List[Dict]) -> int`: Counts the total number of executed trades.
    *   **Inputs:** Functions will typically take `pd.Series` for equity curves/returns or a `List[Dict]` representing trade history (e.g., from `BacktestingEngine`).
    *   **Interactions:**
        *   Used by `src/training/trainer.py` for evaluating backtesting results and potentially during training.
        *   Used by any dedicated reporting module.

#### 2. `src/training/trainer.py` (Enhancement) - Comprehensive Backtesting Reports

*   **Purpose:** To generate and display a summary of the RL agent's performance after a backtesting run, presenting key metrics in a clear, terminal-friendly format.
*   **Class:** `Trainer` (enhancement)
    *   **Responsibilities:**
        *   Collect the necessary data (e.g., trade history, equity curve) from the `TradingEnv` and `BacktestingEngine` after a backtesting episode or run.
        *   Utilize functions from `src/utils/metrics.py` to compute all required performance indicators.
        *   Format and print the results to the terminal.
    *   **Key Methods (additions/modifications to `Trainer`):**
        *   `run_backtest_and_report(self, agent: BaseAgent, env: TradingEnv) -> Dict`:
            *   Runs a full backtest episode/simulation using the provided `agent` and `env`.
            *   Collects trade history and equity curve.
            *   Calls `metrics.py` functions to calculate all performance metrics.
            *   Calls `_display_backtest_report()` to print results.
            *   Returns a dictionary of all calculated metrics.
        *   `_display_backtest_report(self, metrics: Dict) -> None`:
            *   Internal method to format and print the backtesting report to the terminal.
            *   Ensures clear headings, aligned values, and highlights for key metrics (e.g., Sharpe Ratio, Max Drawdown, Total P&L).
    *   **Interactions:**
        *   Uses `src/utils/metrics.py`.
        *   Interacts with `src/backtesting/environment.py` and `src/backtesting/engine.py` to get simulation data.

#### 3. `src/training/trainer.py` (Enhancement) - Real-time Training Progress Updates

*   **Purpose:** To provide concise, non-intrusive updates on the RL agent's learning progress directly in the terminal during training.
*   **Class:** `Trainer` (enhancement)
    *   **Responsibilities:**
        *   Periodically output training statistics (e.g., episode number, current reward, average reward, loss).
        *   Ensure updates are formatted to avoid overwhelming the terminal.
    *   **Key Methods (modifications to `Trainer`):**
        *   `train(self)`: The main training loop will be modified.
        *   `_log_training_progress(self, episode: int, total_reward: float, avg_reward: float, loss: float = None) -> None`:
            *   Internal method called at regular intervals (e.g., every 10 episodes).
            *   Uses `` (carriage return) to overwrite the previous line, creating a dynamic, single-line progress bar effect.
            *   Displays episode number, current/average reward, and relevant loss values.
    *   **Interactions:**
        *   Called within the main training loop.

#### 4. General Terminal Output Guidelines (Across all modules)

*   **Purpose:** To ensure a consistent, clear, and user-friendly experience for all terminal-based interactions and outputs.
*   **Guidelines:**
    *   **Conciseness:** Avoid verbose logging unless explicitly debugging. Focus on essential information.
    *   **Readability:** Use clear headings, consistent indentation, and appropriate spacing.
    *   **Highlighting:** Use simple text formatting (e.g., `===`, `---`) to draw attention to important sections or results.
    *   **Actionable Errors:** Error messages should be clear, indicating what went wrong and, if possible, suggesting a solution or where to find more details (e.g., log file).
    *   **Progress Indicators:** For long-running operations, provide clear progress indicators (e.g., "Processing file X of Y...", "Episode Z/Total Episodes...").
    *   **No Overwhelm:** Adhere strictly to the requirement of not overwhelming the user. If a lot of data needs to be presented, consider summarizing it or offering options to view more detail.
*   **Implementation:** This is a cross-cutting concern that will be applied during the implementation of all modules that produce terminal output.
