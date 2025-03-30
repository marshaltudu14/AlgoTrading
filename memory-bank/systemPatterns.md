# System Patterns: RL Algorithmic Trading Agent

## Core Architecture

The system follows a typical pipeline for developing and evaluating an RL-based trading agent:

```mermaid
graph TD
    A[Raw Data Acquisition] --> B(Data Preprocessing & Normalization);
    B --> C{RL Training Environment};
    C --> D[RL Agent Training];
    D --> E{Trained Agent Model};
    E --> F[Agent Evaluation];
    F --> G[Performance Metrics & Logs];

    subgraph Configuration
        H[dynamic_config.json] --> B;
        H --> C;
        H --> D;
        H --> F;
    end

    subgraph Fyers Integration (Potential)
        I[Fyers Auth] --> J{Live Data Feed};
        J --> E;
        E --> K[Trade Signal Generation];
        K --> L[Order Execution via API];
        L --> M[API Logs];
        I --> L;
    end

    style Configuration fill:#f9f,stroke:#333,stroke-width:2px
    style Fyers Integration (Potential) fill:#ccf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
```

## Key Components & Patterns

1.  **Data Handling (`data/`, `src/data_handler.py`, `preprocess_norm_stats.py`, `run_data_setup.py`):**

    - Raw historical data is stored (`data/historical_raw/`).
    - A preprocessing step generates features and saves processed data (`data/historical_processed/`). This likely involves calculating technical indicators (using `pandas-ta`?) and handling time series data (`pandas`).
    - Normalization statistics are calculated and saved (`normalization_stats.json`, `preprocess_norm_stats.py`) to scale data for the RL agent, likely using `scikit-learn`.

2.  **RL Environment (`src/rl_environment.py`):**

    - Implements a custom environment following the `Gymnasium` API standard.
    - Simulates the trading process: takes actions (buy, sell, hold), updates portfolio state, calculates rewards based on profit/loss or other metrics, and provides observations (market data, portfolio status) to the agent.
    - Uses processed historical data for simulation.

3.  **Agent Training (`train_agent.py`, `models/rl_models/`, `logs/rl_logs/`):**

    - Uses `Stable Baselines3` (specifically PPO, based on logs/models) to train the agent within the custom environment.
    - Loads configuration (`dynamic_config.json`).
    - Saves trained model checkpoints (`models/rl_models/`).
    - Logs training progress (rewards, episode lengths, etc.) using TensorBoard format (`logs/rl_logs/`). `tqdm` is likely used for progress bars.

4.  **Agent Evaluation (`evaluate_agent.py`):**

    - Loads a trained model.
    - Runs the agent on a separate (test) dataset within the environment.
    - Calculates performance metrics (e.g., total return, Sharpe ratio, max drawdown). May use libraries like `matplotlib`, `mplfinance` for plotting results.

5.  **Fyers API Interaction (`src/fyers_auth.py`, `fyersApi.log`, `fyersRequests.log`, `fyersDataSocket.log`):**

    - Handles authentication with the Fyers API (`fyers-apiv3`, `pyotp`).
    - Likely includes functions to fetch historical data, place/modify/cancel orders, and potentially stream live data via WebSockets (suggested by `fyersDataSocket.log`).
    - Logging is implemented for API requests and responses.

6.  **Configuration (`dynamic_config.json`, `src/config.py`):**

    - A central JSON file likely stores dynamic parameters (instrument, timeframe, model paths, hyperparameters).
    - `src/config.py` might load and provide access to these settings.

7.  **Signals (`src/signals.py`):**

    - This module's exact role is unclear without reading the code, but it might be involved in translating agent actions into specific trade signals or integrating technical indicators separate from the RL state.

8.  **Utilities (`sounds/`):**
    - Audio files suggest potential use of sound alerts (`pygame`?) for specific events during operation (e.g., successful trade, error).
