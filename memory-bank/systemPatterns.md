# System Patterns: AlgoTrading - Fyers Data Fetcher

*   **Configuration Management:** Centralized configuration (`src/config.py`) for API keys, paths, instrument lists, and fetching parameters. Sensitive credentials should ideally be moved to environment variables or a more secure store.
*   **Modular Design:** Code is separated into modules for authentication (`fyers_auth.py`), data handling/fetching (`data_handler.py`), and configuration (`config.py`).
*   **Scripted Execution:** A main script (`run_data_setup.py`) orchestrates the workflow (authentication -> data fetching -> saving).
*   **Idempotency (Partial):** The `run_data_setup.py` script checks if raw data files already exist and skips fetching, making the raw data download step partially idempotent.
*   **Error Handling:** Basic error handling (try/except blocks) is present in authentication and data fetching, printing messages to the console. More robust logging could be added.
*   **Data Persistence:** Raw historical data is persisted as CSV files on the local filesystem.
