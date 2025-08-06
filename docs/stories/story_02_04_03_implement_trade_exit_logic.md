---

### **Story 2.4.3: Implement Trade Exit Logic**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to implement the trade exit logic within the `LiveTradingService`, so that the system can close an active position either due to an SL/TP trigger or an explicit "close" signal from the model.

**Acceptance Criteria:**
1.  A dedicated internal method `_close_position(reason)` is created in `LiveTradingService` to handle all trade exits.
2.  This method calls the `exit_position` method of the `FyersClient` to place an order to close the active position.
3.  The `LiveTradingService` clears the active `Position` object after the exit order is successfully placed.
4.  The system logs the exit details, including the reason for exit (e.g., "SL Hit", "TP Hit", "Model Close").
5.  If the exit order placement fails, the system logs the error and attempts a retry (e.g., up to 3 times).
6.  Upon successful exit, the `LiveTradingService` updates the `Backend API` to reflect that there is no longer an active position.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, create the `_close_position(self, reason: str)` method.
-   `[ ]` **Backend:** Inside `_close_position`, call `self.fyers_client.exit_position(self.active_position.symbol)`.
-   `[ ]` **Backend:** Implement a retry mechanism for `exit_position` calls.
-   `[ ]` **Backend:** After a successful exit, set `self.active_position = None`.
-   `[ ]` **Backend:** Add logging for the exit reason and success/failure of the exit order.
-   `[ ]` **Backend:** Implement the logic to send a WebSocket update to the frontend indicating that the position is closed.
-   `[ ]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for trade exit.
-   `[ ]` **Testing:** Mock the `FyersClient.exit_position` method to simulate successful and failed exits.
-   `[ ]` **Testing:** Assert that the active `Position` is cleared after a successful exit.
-   `[ ]` **Testing:** Assert that the system attempts retries on failed exit orders.

---