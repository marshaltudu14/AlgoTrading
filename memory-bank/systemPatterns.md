# System Patterns: AlgoTrading System

*   **Configuration Management:** Centralized configuration (`src/config.py`) for API keys, paths, instrument lists, fetching parameters, and backtesting settings. Sensitive credentials should ideally be moved to environment variables or a more secure store.
*   **Modular Design:** Code is separated into modules for authentication (`fyers_auth.py`), data handling/processing (`data_handler.py`), backtesting logic (`custom_backtester.py`), and configuration (`config.py`).
*   **Scripted Execution:** Separate scripts orchestrate distinct workflows: data setup (`run_data_setup.py`), data processing (`run_data_processing.py`), and backtesting (`run_custom_backtest.py`).
*   **Idempotency (Partial):** Data fetching (`run_data_setup.py`) and processing (`run_data_processing.py`) scripts include checks to avoid re-downloading/re-processing existing data.
*   **Error Handling:** Basic error handling (try/except blocks) is present, printing messages to the console. More robust logging could be added.
*   **Data Persistence:** Raw data stored as CSV (`data/historical_raw/`), processed data with indicators stored as Parquet (`data/historical_processed/`). Backtest results (metrics) are printed; trade logs can be optionally saved.
*   **Strategy Encapsulation:** Strategy logic (signal generation) is currently within the custom backtester module (`src/custom_backtester.py`). Could be further modularized if more strategies are added.
