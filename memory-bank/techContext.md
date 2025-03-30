# Technical Context: RL Algorithmic Trading Agent

## Core Technologies

- **Language:** Python 3.x
- **Reinforcement Learning:**
  - `stable-baselines3[extra]`: Core library for RL algorithms (PPO confirmed from logs/models). The `[extra]` likely includes TensorBoard support.
  - `gymnasium`: Standard API for RL environments. The custom environment (`src/rl_environment.py`) adheres to this.
- **Data Handling & Analysis:**
  - `pandas`: Primary library for data manipulation and time series analysis.
  - `numpy`: Foundational library for numerical operations.
  - `scikit-learn`: Used for data preprocessing, specifically normalization/scaling.
  - `pandas-ta`: Likely used for calculating technical indicators as features.
  - `statsmodels`, `scipy`: Available for more advanced statistical analysis if needed.
  - `numba`: Potentially used for performance optimization of numerical code.
- **API Integration (Fyers):**
  - `fyers-apiv3`: Official Fyers API v3 library for Python.
  - `requests`: Underlying HTTP library, likely used by `fyers-apiv3` or for other potential API calls.
  - `pyotp`: Used for Time-based One-Time Password (TOTP) generation, required for Fyers API authentication.
- **Visualization:**
  - `matplotlib`: Standard plotting library.
  - `mplfinance`: Specialized library for financial data visualization (candlestick charts, etc.).
- **Utilities:**
  - `pytz`: Handling timezones, crucial for financial data.
  - `tqdm`: Progress bars for long-running processes like training.
  - `pygame`: Used for playing sound alerts found in the `sounds/` directory.

## Development Setup

- **Dependencies:** Managed via `requirements.txt`. Install using `pip install -r requirements.txt`.
- **Configuration:** Key parameters are managed through `dynamic_config.json`.
- **Environment:** Assumed to be a standard Python environment (virtual environment recommended).
- **Operating System:** Developed on Windows 11 (based on system info), but likely cross-platform compatible due to Python usage.

## Technical Constraints & Considerations

- **API Limits:** Fyers API will have rate limits and usage constraints that need to be handled gracefully.
- **Data Quality:** Performance is highly dependent on the quality and relevance of the historical data used for training.
- **Computational Resources:** Training RL models can be computationally intensive, requiring significant CPU time and potentially RAM.
- **Real-time Performance:** If deployed live, the system needs to process data and make decisions within acceptable latency limits.
- **Backtesting vs. Reality:** Simulation results may not perfectly translate to live trading due to factors like slippage, latency, and changing market dynamics.
