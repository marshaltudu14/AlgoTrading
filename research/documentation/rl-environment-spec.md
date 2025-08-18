# Reinforcement Learning Environment: A Technical and Theoretical Specification

## 1. Overview and Design Philosophy

### 1.1. Conceptual Framework

This document specifies the Reinforcement Learning (RL) environment engineered to serve as a sophisticated and realistic virtual laboratory for training advanced algorithmic trading agents. The core design philosophy is to bridge the gap between abstract reasoning models, like the Hierarchical Reasoning Model (HRM), and the complexities of live financial markets. The environment is built to be a challenging and informative simulation that fosters the development of agents that are not just profitable, but also robust, adaptive, and risk-aware.

### 1.2. Technical Implementation

The environment is implemented as a `TradingEnv` class that inherits from `gymnasium.Env`, ensuring compatibility with standard RL libraries and frameworks. It operates in three distinct modalities to support a full agent development lifecycle:

-   **`TradingMode.TRAINING`**: A mode for agent learning, which utilizes an intelligent data feeding strategy. This exposes the agent to a curriculum of diverse market scenarios (e.g., high volatility, trending, sideways) to promote the development of a generalized trading intelligence.
-   **`TradingMode.BACKTESTING`**: A mode for rigorous validation of a trained agent. It runs the agent over a complete, sequential historical dataset to generate unbiased performance metrics.
-   **`TradingMode.LIVE`**: A placeholder mode for future integration with a live data feed, allowing the trained agent to operate in real-time markets.

## 2. Observation Space: The Agent's Perception

### 2.1. Design Philosophy

The observation space is engineered to provide the agent with a rich, multi-dimensional perception of the market and its own state. A key design choice is the **deliberate exclusion of raw price data (OHLC)**. This forces the agent to learn from the abstract relationships and patterns within derived market features, preventing it from overfitting to specific price levels and fostering a more universal, generalizable model.

### 2.2. Technical Specification

The observation is a `gym.spaces.Box` with a flattened shape, composed of three distinct parts:

1.  **Market Context Features**:
    *   **Lookback Window**: `50` historical timesteps.
    *   **Features per Timestep**: `256` derived technical indicators and market features (e.g., RSI, MACD, Bollinger Bands, volatility, momentum).
    *   **Temporal Feature**: The `datetime_epoch` is included to allow the agent to learn time-based patterns.

2.  **Account State Features** (5 features):
    *   `capital`: The agent's current available trading capital.
    *   `position_quantity`: The number of lots currently held (positive for long, negative for short).
    *   `position_entry_price`: The average price at which the current position was entered.
    *   `unrealized_pnl`: The current floating profit or loss of the open position.
    *   `is_position_open`: A binary flag (`1.0` or `0.0`) indicating an open position.

3.  **Implicit Risk Feature** (1 feature):
    *   `distance_to_trail`: The normalized distance of the current price to the trailing stop-loss. This gives the agent a continuous signal of the trade's current risk-reward status.

**Normalization**: To ensure training stability, all observation features, **except for `datetime_epoch`**, undergo Z-score normalization based on the statistics of the last 100 observations.

## 3. Action Space: The Agent's Capabilities

### 3.1. Design Philosophy

The action space is designed as a hybrid, mirroring the dual nature of a human trading decision, which combines a strategic choice with a tactical execution parameter.

-   **The Strategic Choice**: The agent decides on its core intent (e.g., to enter a long position or to hold).
-   **The Tactical Parameter**: The agent determines the magnitude of its action (i.e., the position size).

### 3.2. Technical Specification

The action space is a `gym.spaces.Box` of shape `(2,)`, representing `[action_type, quantity]`.

1.  **`action_type`** (Discrete, mapped from the first continuous value):
    *   `0`: **BUY_LONG**
    *   `1`: **SELL_SHORT**
    *   `2`: **CLOSE_LONG**
    *   `3`: **CLOSE_SHORT**
    *   `4`: **HOLD**

2.  **`quantity`** (Continuous):
    *   This value represents the desired position size (in lots).
    *   The environment enforces realism by clamping this value to the **maximum affordable quantity**, calculated based on the agent's current capital and the instrument's price.

## 4. Reward Architecture: The Learning Signal

### 4.1. Design Philosophy

The reward architecture is engineered to be a nuanced learning signal that guides the agent toward behaviors that are not just profitable, but also robust, consistent, and risk-aware. It moves beyond simple P&L to reward the *quality* of the agent's performance, as a professional trader would be evaluated.

### 4.2. Technical Specification

The environment uses the `enhanced_trading_focused` reward function, which is a composite of several weighted components:

-   **Base Reward**: The fundamental signal derived from the step-by-step change in the agent's total equity (realized capital + unrealized P&L).
-   **Profit Factor Bonus**: A heavily weighted bonus for achieving a high Profit Factor (Gross Profit / Gross Loss). The reward scales aggressively for exceptional ratios (>2.0).
-   **Win Rate Bonus**: An incentive for maintaining a high percentage of winning trades, with a target of 60-70% and above.
-   **Drawdown Penalty**: A negative reward that increases in magnitude as the agent's equity drops from its peak, discouraging risky behavior.
-   **Risk-Reward Ratio Bonus**: A bonus for trades where the average profit is significantly larger than the average loss.
-   **Behavioral Shaping**: Smaller, heuristic-based rewards and penalties to fine-tune behavior, including penalties for **idleness** and **over-trading**, and a bonus for effective **trend-following** (holding a position as its trailing stop improves).

## 5. Environment Dynamics and Configuration

### 5.1. Conceptual Dynamics

The simulation incorporates several mechanisms to ensure realism and to focus the agent's learning on meaningful strategic decisions.

-   **Meaningful Episode Termination**: Episodes conclude based on significant events, such as a catastrophic drawdown (a risk violation) or the end of a curated data segment, rather than an arbitrary time limit. This frames the learning process around goal-oriented scenarios.
-   **Logical Action Filtering**: The environment enforces logical consistency. For example, an agent's attempt to buy while already in a long position is automatically interpreted as a `HOLD`. This prevents the agent from learning noisy or redundant policies.

### 5.2. Key Configuration Parameters

The environment's dynamics are controlled by parameters in the `config/settings.yaml` file.

| Parameter (`environment` section) | Type | Description | Default Value |
| :--- | :--- | :--- | :--- |
| `initial_capital` | `float` | The starting capital for each episode. | `100000.0` |
| `lookback_window` | `int` | The number of historical timesteps in each observation. | `50` |
| `episode_length` | `int` | The number of steps in a standard training episode. | `1000` |
| `reward_function` | `str` | The reward calculation strategy to be used. | `enhanced_trading_focused` |
| `trailing_stop_percentage`| `float` | The percentage below the peak price to set the trailing stop. | `0.015` |
| `smart_action_filtering`| `bool` | If true, prevents redundant actions. | `true` |
| `max_drawdown_pct` | `float` | The maximum allowed drawdown in training before an episode is terminated. | `0.20` (20%) |
