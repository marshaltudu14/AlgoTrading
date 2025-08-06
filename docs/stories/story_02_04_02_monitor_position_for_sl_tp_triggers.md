---

### **Story 2.4.2: Monitor Position for SL/TP Triggers**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to implement continuous monitoring of the active position against its calculated stop-loss (SL) and target price (TP) levels, so that the system can automatically exit the trade when either level is hit.

**Acceptance Criteria:**
1.  In the `LiveTradingService`'s main loop, after fetching new data, the current price of the instrument is compared against the active position's `stopLoss` and `targetPrice`.
2.  For a long position:
    *   If `current_price <= stopLoss`, a stop-loss trigger is detected.
    *   If `current_price >= targetPrice`, a target trigger is detected.
3.  For a short position:
    *   If `current_price >= stopLoss`, a stop-loss trigger is detected.
    *   If `current_price <= targetPrice`, a target trigger is detected.
4.  Upon detection of a trigger, the `LiveTradingService` initiates an order to close the position.
5.  The system logs the trigger event (SL hit or TP hit) and the current price.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, within the main trading loop, add a check for an active position.
-   `[ ]` **Backend:** If a position is active, retrieve its `stopLoss`, `targetPrice`, and `direction`.
-   `[ ]` **Backend:** Get the `current_price` from the latest fetched data.
-   `[ ]` **Backend:** Implement the conditional logic to check for SL or TP triggers based on the `direction`.
-   `[ ]` **Backend:** If a trigger is detected, call a new internal method (e.g., `_close_position`) to handle the exit.
-   `[ ]` **Backend:** Add logging to record the trigger event.
-   `[ ]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for SL/TP monitoring.
-   `[ ]` **Testing:** Write tests that simulate a long position and then provide data that hits the SL and TP, asserting that the `_close_position` method is called.
-   `[ ]` **Testing:** Repeat the above for a short position.

---