# Project Brief: AlgoTrading - Fyers Data Fetcher

**Objective:** Establish a robust system to fetch historical market data for multiple Indian indices (Nifty, Bank Nifty, Finnifty, Sensex, Bankex) across various timeframes from the Fyers API.

**Core Functionality:**
1.  **Authentication:** Securely authenticate with the Fyers API using App ID, Secret Key, User ID, PIN, and TOTP.
2.  **Data Fetching:** Retrieve historical candle data (OHLCV) for configured indices and timeframes (2m, 3m, 5m, 10m, 15m, 20m, 30m, 45m, 60m, 120m, 180m, 240m).
3.  **Data Storage:** Save the fetched raw data into CSV files organized by instrument and timeframe within the `data/historical_raw/` directory.
4.  **Configuration:** Manage Fyers credentials, API settings, instrument mappings, and data fetching parameters via `src/config.py`.

**Current Status:**
*   Project structure established.
*   Fyers authentication mechanism implemented (`src/fyers_auth.py`).
*   Raw data fetching logic implemented (`src/data_handler.py`).
*   Configuration file set up (`src/config.py`).
*   Main setup script (`run_data_setup.py`) orchestrates authentication and raw data fetching.
*   Reinforcement learning components and processed data have been removed for a fresh start.
