# Technical Context: AlgoTrading - Fyers Data Fetcher

**Language:** Python 3

**Key Libraries:**
*   **fyers-apiv3:** Official Fyers API V3 library for authentication and data interaction.
*   **requests:** Used internally by `fyers_auth.py` for the multi-step authentication flow.
*   **pyotp:** Used for generating Time-based One-Time Passwords (TOTP) required for Fyers login.
*   **pandas:** Used for handling and manipulating the fetched candle data (creating DataFrames, saving to CSV).
*   **numpy:** Used by pandas and potentially for numerical operations (though direct usage is minimal after cleanup).
*   **pytz:** Used for timezone handling (converting UTC timestamps from Fyers to IST).
*   **numba:** Used in the (now removed) `signals.py` for JIT compilation. Kept in requirements for now as `data_handler.py` still uses `pandas-ta` which might have optional numba dependency.
*   **pandas-ta:** Used in `data_handler.py` for calculating technical indicators (ATR, RSI, MACD, etc.) within the `FullFeaturePipeline`.

**Environment:**
*   Assumes Python 3 environment with libraries listed in `requirements.txt` installed.
*   Requires Fyers API credentials (APP_ID, SECRET_KEY, FYERS_USER, FYERS_PIN, FYERS_TOTP_KEY) to be correctly set in `src/config.py` (or environment variables).

**Execution:**
*   The primary entry point for data setup is running the `run_data_setup.py` script.
