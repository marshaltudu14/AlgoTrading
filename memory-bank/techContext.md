# Technical Context: AlgoTrading System

**Language:** Python 3

**Key Libraries:**
*   **fyers-apiv3:** Official Fyers API V3 library for authentication and data interaction.
*   **requests:** Used internally by `fyers_auth.py` for the multi-step authentication flow.
*   **pyotp:** Used for generating Time-based One-Time Passwords (TOTP) required for Fyers login.
*   **pandas:** Core library for data manipulation (DataFrames), used in data handling, processing, and backtesting.
*   **numpy:** Used for numerical operations, particularly within pandas and the custom backtester.
*   **pytz:** Used for timezone handling during data fetching.
*   **pandas-ta:** Used in `src/data_handler.py` for calculating technical indicators (e.g., ATR) during data processing.
*   **pyarrow / fastparquet:** Required by pandas for reading/writing Parquet files (used for processed data). Ensure one is installed.
*   **Removed:** `backtesting` library is no longer used.

**Environment:**
*   Assumes Python 3 environment with libraries listed in `requirements.txt` installed.
*   Requires Fyers API credentials to be correctly set in `src/config.py` (or environment variables).

**Execution:**
*   Data Setup: `run_data_setup.py`
*   Data Processing: `run_data_processing.py`
*   Backtesting: `run_custom_backtest.py`
