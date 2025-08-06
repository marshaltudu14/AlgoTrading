---

### **Story 4.3.2: Implement Trade Metrics Calculation and Storage**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to implement the calculation of key trading metrics (win rate, total trades, P&L) and store them in `metrics.json`, so that the frontend dashboard can display real-time performance.

**Acceptance Criteria:**
1.  A new Python class `MetricsCalculator` is created in `src/utils/metrics_calculator.py` (new file).
2.  The `MetricsCalculator` has methods to:
    *   Load existing metrics from `metrics.json`.
    *   Update metrics based on a new `Trade` object.
    *   Calculate `totalTrades`, `winRate`, `totalPnl`, and `averagePnlPerTrade`.
    *   Persist updated metrics back to `metrics.json`.
3.  The `LiveTradingService` calls `MetricsCalculator` after each trade is logged by the `TradeLogger`.
4.  The `metrics.json` file is updated with the latest calculated metrics.
5.  The `Backend API`'s `/api/metrics` endpoint serves the updated `metrics.json` content.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Create `src/utils/metrics_calculator.py`.
-   `[ ]` **Backend:** Implement `MetricsCalculator` class with methods: `load_metrics()`, `update_metrics(trade_data)`, `save_metrics()`.
-   `[ ]` **Backend:** Ensure `load_metrics` handles the case where `metrics.json` does not exist.
-   `[ ]` **Backend:** Implement the logic within `update_metrics` to correctly calculate `totalTrades`, `winRate` (based on `status`), and `totalPnl`.
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, after calling `TradeLogger.log_trade`, instantiate `MetricsCalculator` and call `update_metrics` with the same `trade_data`.
-   `[ ]` **Backend:** Ensure `MetricsCalculator.save_metrics()` is called after updating.
-   `[ ]` **Backend:** Verify that the existing `/api/metrics` endpoint in `backend/main.py` correctly reads from `metrics.json`.
-   `[ ]` **Testing:** Create a new test file `tests/test_utils/test_metrics_calculator.py`.
-   `[ ]` **Testing:** Write unit tests for `MetricsCalculator` to verify correct calculation of metrics for various trade scenarios (wins, losses, breakeven).
-   `[ ]` **Testing:** Test persistence of metrics to `metrics.json`.

---