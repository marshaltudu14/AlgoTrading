# Project Brief: AlgoTrading System

**Objective:** Develop an algorithmic trading system capable of fetching historical data, processing it, backtesting strategies, and potentially deploying them.

**Core Functionality:**
1.  **Authentication:** Securely authenticate with the Fyers API (`src/fyers_auth.py`).
2.  **Data Fetching:** Retrieve historical candle data for configured indices and timeframes (`src/data_handler.py`, `run_data_setup.py`).
3.  **Raw Data Storage:** Save fetched raw data into CSV files (`data/historical_raw/`).
4.  **Data Processing:** Process raw data, calculate technical indicators, and save to Parquet format (`src/data_handler.py`, `run_data_processing.py`, `data/historical_processed/`).
5.  **Backtesting:** Implement and run trading strategies using a custom backtesting engine (`src/custom_backtester.py`, `run_custom_backtest.py`).
6.  **Configuration:** Manage Fyers credentials, API settings, instrument mappings, and parameters via `src/config.py`.
7.  **Strategy Documentation:** Maintain documentation for implemented strategies (`memory-bank/strategies.md`).

**Current Status:**
*   Project structure established.
*   Fyers authentication implemented.
*   Raw data fetching implemented.
*   Data processing pipeline implemented (calculates indicators like ATR).
*   Custom backtesting engine implemented for Inside Candle strategy.
*   Strategy documentation initiated.
*   Removed previous `backtesting.py` library integration.
*   Removed previous RL components.
