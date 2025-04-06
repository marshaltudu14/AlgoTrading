# Product Context: AlgoTrading System

**Goal:** Develop an end-to-end algorithmic trading system, including data acquisition, processing, strategy backtesting, and real-time strategy execution via the Fyers API. The system aims to automate trading decisions and provide real-time market analysis.

**Key Components:**
*   **Fyers API Integration:** Connects to Fyers V3 API for authentication, REST calls (historical data, order placement, positions), and WebSocket connections (market data, order updates) (`src/fyers_auth.py`, `src/order_manager.py`, WebSocket handlers).
*   **Data Fetcher:** Script (`run_data_setup.py`) using the Fyers API wrapper to download historical candle data.
*   **Raw Data Storage:** Stores downloaded data in CSV format (`data/historical_raw/`).
*   **Data Processor:** Script (`run_data_processing.py`) using `src/data_handler.py` to clean raw data, calculate technical indicators, and save processed data in Parquet format.
*   **Custom Backtester:** Engine (`src/custom_backtester.py`) and script (`run_custom_backtest.py`) to simulate strategy performance.
*   **Real-Time Data Handlers:**
    *   `src/realtime_data_handler.py`: Manages WebSocket for real-time market data (underlying index, options).
    *   `src/order_update_handler.py`: Manages WebSocket for real-time order, trade, and position updates.
*   **Real-Time Strategy Executor:** (`src/realtime_strategy_executor.py`): Orchestrates live trading by processing real-time data, generating signals (`src/signals.py`), selecting options (`src/options_utils.py`), placing orders (`src/order_manager.py`), managing positions (`src/position_manager.py`), and handling exits based on index SL/TP.
*   **Order Manager:** (`src/order_manager.py`): Handles placing/modifying/cancelling orders via the Fyers REST API.
*   **Position Manager:** (`src/position_manager.py`): Tracks current positions based on trade updates (currently in-memory).
*   **Configuration:** Centralized settings (`src/config.py`) including credentials, paths, strategy parameters, real-time settings (`TRADING_MODE`, etc.).
*   **Execution Scripts:** `run_data_setup.py`, `run_data_processing.py`, `run_custom_backtest.py`, `run_realtime_bot.py`.
*   **Memory Banks:** Documentation files (`memory-bank/`) tracking progress, context, etc.

**User:** The primary user is the developer (Marshal Tudu) building, testing, and deploying algorithmic options trading strategies.

**Use Case:** The system allows the user to fetch historical data, process it, backtest strategies (currently Inside Candle), and deploy validated strategies for automated real-time options trading (Buy or Sell mode) based on underlying index signals and index-based SL/TP levels. It provides real-time logging of position status.
