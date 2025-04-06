# Product Context: AlgoTrading System

**Goal:** Develop an end-to-end algorithmic trading system foundation, including data acquisition, processing, strategy backtesting, and potential future deployment.

**Key Components:**
*   **Fyers API Integration:** Connects to Fyers V3 API for authentication and historical data retrieval (`src/fyers_auth.py`).
*   **Data Fetcher:** Script (`run_data_setup.py`) that uses the Fyers API wrapper to download candle data.
*   **Raw Data Storage:** Stores downloaded data in CSV format (`data/historical_raw/`).
*   **Data Processor:** Script (`run_data_processing.py`) using `src/data_handler.py` to clean raw data, calculate technical indicators (e.g., ATR), and save processed data in Parquet format (`data/historical_processed/`).
*   **Custom Backtester:** A manual backtesting engine (`src/custom_backtester.py`) and execution script (`run_custom_backtest.py`) to simulate strategy performance on processed data.
*   **Configuration:** Centralized settings for credentials, instruments, timeframes, file paths, and strategy parameters (`src/config.py`).
*   **Strategy Documentation:** A central place to document strategy logic and performance (`memory-bank/strategies.md`).

**User:** The primary user is the developer (Marshal Tudu) building and testing algorithmic trading strategies.

**Use Case:** The system allows the user to fetch historical data, process it with relevant indicators, backtest custom trading strategies against this data to evaluate their performance, and iterate on strategy development.
