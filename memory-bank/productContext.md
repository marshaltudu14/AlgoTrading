# Product Context: AlgoTrading - Fyers Data Fetcher

**Goal:** Create a foundational data pipeline for an algorithmic trading system by reliably fetching and storing historical market data from Fyers.

**Key Components:**
*   **Fyers API Integration:** Connects to Fyers V3 API for authentication and historical data retrieval.
*   **Data Fetcher:** Script (`run_data_setup.py`) that uses the Fyers API wrapper to download candle data for specified indices and timeframes.
*   **Raw Data Storage:** Stores downloaded data in CSV format in `data/historical_raw/`.
*   **Configuration:** Centralized settings for credentials, instruments, timeframes, and file paths (`src/config.py`).

**User:** The primary user is the developer (Marshal Tudu) building the algorithmic trading system.

**Use Case:** The system needs accurate and comprehensive historical data as the basis for backtesting trading strategies and potentially training machine learning models in the future. This module focuses solely on acquiring and storing this raw data.
