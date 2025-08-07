---

### **Story 4.2.2: Implement Manual Trade Logic in LiveTradingService**

**Status:** `Completed`

**Story:**
As a developer, I need to implement the core logic for handling manual trades within the `LiveTradingService`, including validation, execution, and pausing the automated model.

**Acceptance Criteria:**
1.  A new method `initiate_manual_trade(instrument, direction, quantity, sl, tp, user_id)` is added to `LiveTradingService`.
2.  This method first checks if an automated trade is currently active. If so, it raises an error.
3.  It performs margin and risk validation for the manual trade using existing or new utility functions.
4.  If validation passes, it calls `FyersClient.place_order` to execute the manual trade.
5.  Upon successful order placement, it creates a `Position` object for the manual trade, marking its `tradeType` as "Manual".
6.  The `LiveTradingService` pauses its time-based data fetching and model inference loop while a manual trade is active.
7.  The system logs the manual trade details and the pause/resume of the automated loop.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, add the `initiate_manual_trade` method.
-   `[ ]` **Backend:** Implement the check for active automated trade and raise an appropriate exception if one exists.
-   `[ ]` **Backend:** Integrate margin and risk validation (reusing `CapitalAwareQuantity` or creating new validation logic).
-   `[ ]` **Backend:** Call `self.fyers_client.place_order` for the manual trade.
-   `[ ]` **Backend:** Create and store the `Position` object with `tradeType = "Manual"`.
-   `[ ]` **Backend:** Implement a flag or state variable in `LiveTradingService` to pause the automated loop.
-   `[ ]` **Backend:** Modify the main loop to check this flag and pause execution when a manual trade is active.
-   `[ ]` **Backend:** Add comprehensive logging for manual trade initiation, validation, and loop pausing.
-   `[ ]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for manual trade logic.
-   `[ ]` **Testing:** Write a test that attempts to place a manual trade when an automated trade is active and asserts an error is raised.
-   `[ ]` **Testing:** Write a test that simulates a successful manual trade and asserts that the automated loop is paused.
-   `[ ]` **Testing:** Write a test for margin/risk validation failures.

---