# Product Context: RL Algorithmic Trading Agent

## Problem Solved

Automating the complex and time-consuming process of making trading decisions based on market data analysis. This aims to:

- Remove emotional biases from trading.
- Potentially identify and exploit market patterns that are difficult for humans to discern consistently.
- Allow for systematic backtesting and evaluation of trading strategies.
- Enable continuous operation without constant human monitoring (during market hours).

## How It Should Work

1.  **Configuration:** The user configures the agent through a settings file (`dynamic_config.json` likely), specifying parameters like the trading instrument, timeframe, risk parameters, and potentially RL model details.
2.  **Data Preparation:** Historical data is processed and normalized (`preprocess_norm_stats.py`, `run_data_setup.py`).
3.  **Training:** The user initiates the training process (`train_agent.py`), which uses the prepared data and the RL environment (`src/rl_environment.py`) to train an agent model (saved in `models/rl_models/`). Training progress is logged (`logs/rl_logs/`).
4.  **Evaluation:** The user evaluates a trained model (`evaluate_agent.py`) on unseen historical data to assess its performance based on metrics like profitability, drawdown, Sharpe ratio, etc.
5.  **(Potential) Live Operation:** If deployed, the agent would connect to the Fyers API (`src/fyers_auth.py`), receive market data (potentially via `fyersDataSocket.log`), use the trained model to generate trading signals (`src/signals.py` might be related), and execute orders. Activity would be logged (`fyersApi.log`, `fyersRequests.log`).

## User Experience Goals

- **Clarity:** Clear logs and outputs for training, evaluation, and potential live trading.
- **Control:** Easy configuration of key parameters.
- **Reliability:** Stable operation during training and evaluation. Robust error handling, especially for API interactions.
- **Reproducibility:** Ability to reproduce training runs and evaluation results.
- **Performance Insight:** Meaningful metrics and visualizations (potentially via `matplotlib`, `mplfinance`) to understand agent behavior and performance.
