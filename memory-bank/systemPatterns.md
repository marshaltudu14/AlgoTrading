# System Patterns: AlgoTrading System

*   **Configuration Management:** Centralized configuration (`src/config.py`) for API keys, paths, instrument lists, fetching parameters, and backtesting settings. Sensitive credentials should ideally be moved to environment variables or a more secure store.
*   **Modular Design:** Code is separated into modules for authentication (`fyers_auth.py`), data handling/processing (`data_handler.py`), backtesting logic (`custom_backtester.py`), and configuration (`config.py`).
*   **Scripted Execution:** Separate scripts orchestrate distinct workflows: data setup (`run_data_setup.py`), data processing (`run_data_processing.py`), backtesting (`run_custom_backtest.py`), and real-time trading (`run_realtime_bot.py`).
*   **Idempotency (Partial):** Data fetching (`run_data_setup.py`) includes checks to avoid re-downloading existing raw data.
*   **Error Handling:** Basic error handling (try/except blocks) is present, printing messages to the console. Real-time components use the `logging` module. More robust logging and specific error handling (e.g., API rate limits, connection drops) could be added.
*   **Data Persistence:** Raw data stored as CSV (`data/historical_raw/`), processed data with indicators stored as Parquet (`data/historical_processed/`). Backtest results (metrics) are printed. Real-time logs are generated. Position state is currently in-memory (`PositionManager`).
*   **Strategy Encapsulation:**
    *   Backtesting: Strategy logic is within `src/custom_backtester.py`.
    *   Real-Time: Signal generation logic is encapsulated in classes within `src/signals.py` (e.g., `InsideCandleRealtimeSignalGenerator`). The `RealtimeStrategyExecutor` orchestrates the interaction between signals, data, and order management based on the selected strategy configuration.
*   **Real-Time Architecture:**
    *   **Event-Driven:** Uses WebSocket callbacks (`on_message`, `on_orders`, etc.) to react to market data and order updates.
    *   **Concurrency:** Employs threading (`threading.Thread`) to run WebSocket connections in the background without blocking the main execution flow.
    *   **State Management:** Uses `PositionManager` (in-memory) and internal dictionaries (`active_orders`, `position_meta` in `RealtimeStrategyExecutor`) to track live orders and positions. Thread safety is managed using `threading.Lock`.
    *   **Separation of Concerns:** WebSocket communication is separated into dedicated handlers for market data (`RealtimeMarketDataHandler`) and order updates (`RealtimeOrderUpdateHandler`). Options logic is further separated into `options_utils.py`.
    *   **Dynamic Configuration:** Real-time bot (`run_realtime_bot.py`) loads strategy parameters dynamically from `config.py`.
