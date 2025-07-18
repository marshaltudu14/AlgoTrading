
# System Architecture: Autonomous Trading Bot - Live Trading MVP

**Author:** Winston, Architect
**Status:** Draft
**Version:** 1.0
**Date:** 2025-07-18

---

## 1. Introduction

### 1.1. Purpose
This document provides the technical architecture for the Live Trading MVP. It translates the functional requirements defined in the PRD into a concrete technical design, detailing the system components, their interactions, data flows, and operational considerations. This architecture is designed to be robust, scalable, and directly integrate with the project's existing data processing and model training pipelines.

### 1.2. Guiding Principles
- **Modularity:** Each component has a single, well-defined responsibility (e.g., authentication, data fetching, execution) to ensure separation of concerns and ease of maintenance.
- **Reliability:** The system must be resilient to common failures, such as API connection errors or invalid data, through robust error handling and logging.
- **Extensibility:** While building for the MVP, the design will allow for future expansion, such as adding new trading models or instruments.

---

## 2. System Overview & Components

The live trading system is designed as a single, long-running Python application orchestrated by a central `LiveTrader` class. This application will run in a continuous loop, driven by a schedule based on the selected trading timeframe.

### 2.1. Component Diagram

```
+-------------------------+
|   run_live_bot.py       | (Main execution script)
+-------------------------+
           |
           V
+-------------------------+
|  LiveTrader (Orchestrator) |
+-------------------------+
| - __init__()            |
| - run()                 |
| - get_trade_decision()  |
| - manage_active_trade() |
+-------------------------+
      |         |         |
      V         V         V
+---------+ +---------+ +---------+
| Fyers   | |Inference| | Logger  |
| Client  | | Engine  | | Service |
+---------+ +---------+ +---------+

```

### 2.2. Component Responsibilities

- **`run_live_bot.py` (Entry Point):** The main script that initializes the `LiveTrader` and starts its main `run()` loop.

- **`LiveTrader` (Orchestrator):** The core component responsible for managing the entire trading lifecycle. It coordinates all other components.

- **`FyersClient` (API Wrapper):** A dedicated class encapsulating all interactions with the Fyers API. This includes authentication, fetching historical data, subscribing to WebSockets, placing/modifying/canceling orders, and getting account details.

- **`InferenceEngine` (Model Wrapper):** A class responsible for loading the trained supervised model, scaler, and encoders. It exposes a simple method to get a prediction from new data.

- **`Logger` (Logging Service):** A centralized logging utility to record all system activities, decisions, trades, and errors in a structured format.

---

## 3. Detailed Component Design

### 3.1. `src/trading/live_trader.py`

This file will contain the `LiveTrader` class.

```python
class LiveTrader:
    def __init__(self, config):
        # - Initializes FyersClient and authenticates.
        # - Initializes InferenceEngine, loading the model.
        # - Initializes the Logger.
        # - Sets trading parameters (instrument, timeframe) from config.

    def run(self):
        # - The main, scheduled loop of the bot.
        # - Checks if a position is active.
        # - If no active position, calls get_trade_decision().
        # - If a signal is generated, calls manage_active_trade().

    def get_trade_decision(self):
        # - Fetches latest historical data via FyersClient.
        # - Runs data through the existing processing pipeline (feature/reasoning).
        # - Gets a prediction from the InferenceEngine.
        # - Returns the signal (0, 1, or 2).

    def manage_active_trade(self, entry_signal):
        # - Calculates SL/TP levels based on ATR.
        # - Places the entry order via FyersClient.
        # - Subscribes to WebSocket via FyersClient.
        # - Monitors ticks for SL/TP hit.
        # - Places exit order when SL/TP is hit.
        # - Unsubscribes from WebSocket.
```

### 3.2. `src/trading/fyers_client.py`

This file will abstract all Fyers-specific logic.

```python
class FyersClient:
    def __init__(self, config):
        # - Handles the entire authentication flow from src/auth/fyers_auth.py.

    def get_historical_data(self, symbol, timeframe, num_candles):
        # - Fetches historical candle data.

    def place_order(self, symbol, side, qty):
        # - Places a market order.

    def exit_position(self, symbol):
        # - Places a market order to close the current position.

    def get_open_positions(self):
        # - Returns current open positions.

    def connect_websocket(self, symbol, on_tick_callback):
        # - Connects to the Fyers data socket and streams ticks.
        # - Calls the provided callback function for each new tick.
```

### 3.3. `src/trading/inference_engine.py`

This file will handle all model-related tasks.

```python
class InferenceEngine:
    def __init__(self, model_path, scaler_path, encoder_path):
        # - Loads the .joblib files for the model, scaler, and label encoder.

    def predict(self, processed_data):
        # - Takes a single row of processed data.
        # - Applies scaling and returns the predicted signal.
```

---

## 4. Data & Execution Flow

### 4.1. Scheduled Loop (No Active Position)
1.  **Scheduler** (e.g., `apscheduler`) triggers `LiveTrader.run()` at the start of each new candle.
2.  `LiveTrader` confirms there are no open positions via `FyersClient`.
3.  `LiveTrader` calls `get_trade_decision()`.
4.  `get_trade_decision()` fetches data, processes it, and gets a prediction from `InferenceEngine`.
5.  If the signal is `1` or `2`, `LiveTrader` calls `manage_active_trade()`.

### 4.2. In-Trade Loop (Active Position)
1.  `manage_active_trade()` calculates SL/TP and places an entry order via `FyersClient`.
2.  It then calls `FyersClient.connect_websocket()`, passing a callback method (`self.on_tick`).
3.  `on_tick()` receives each tick and compares it to the SL/TP prices.
4.  If a price is hit, `on_tick()` calls `FyersClient.exit_position()` and disconnects the WebSocket.
5.  The main `run()` loop will then resume on its next scheduled cycle.

---

## 5. Operational Considerations

### 5.1. Error Handling
- The `FyersClient` will implement retry logic for transient API errors (e.g., HTTP 5xx).
- All major operations within `LiveTrader` will be wrapped in `try...except` blocks to log errors gracefully without crashing the bot.
- A special exception will be raised for authentication failures, as this is a critical, non-recoverable error.

### 5.2. Logging
- A central logger will be initialized in `run_live_bot.py` and passed to all components.
- Logs will be written to both the console and a rotating file (`logs/live_trading.log`).
- **Log Format:** `[TIMESTAMP] [LEVEL] [COMPONENT] - MESSAGE` (e.g., `[2025-07-18 10:05:01] [INFO] [LiveTrader] - Signal 1 received. Entering long position.`).

### 5.3. Security
- All API credentials (`client_id`, `secret_key`, `pin`, etc.) MUST NOT be hardcoded. They will be loaded from environment variables or a secure configuration file that is excluded from version control via `.gitignore`.
