---

### **Story 2.4.3: Implement Trade Exit Logic**

**Status:** `Completed`

**Story:**
As a developer, I need to implement the trade exit logic within the `LiveTradingService`, so that the system can close an active position either due to an SL/TP trigger or an explicit "close" signal from the model.

**Acceptance Criteria:**
1.  ✅ A dedicated internal method `_close_position(reason)` is created in `LiveTradingService` to handle all trade exits.
2.  ✅ This method calls the `exit_position` method of the `FyersClient` to place an order to close the active position.
3.  ✅ The `LiveTradingService` clears the active `Position` object after the exit order is successfully placed.
4.  ✅ The system logs the exit details, including the reason for exit (e.g., "SL Hit", "TP Hit", "Model Close").
5.  ✅ If the exit order placement fails, the system logs the error and attempts a retry (e.g., up to 3 times).
6.  ✅ Upon successful exit, the `LiveTradingService` updates the `Backend API` to reflect that there is no longer an active position.

**Tasks / Subtasks:**
-   `[x]` **Backend:** In `src/trading/live_trading_service.py`, create the `_close_position(self, reason: str)` method.
-   `[x]` **Backend:** Inside `_close_position`, call `self.fyers_client.exit_position(self.active_position.symbol)`.
-   `[x]` **Backend:** Implement a retry mechanism for `exit_position` calls.
-   `[x]` **Backend:** After a successful exit, set `self.active_position = None`.
-   `[x]` **Backend:** Add logging for the exit reason and success/failure of the exit order.
-   `[x]` **Backend:** Implement the logic to send a WebSocket update to the frontend indicating that the position is closed.
-   `[x]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for trade exit.
-   `[x]` **Testing:** Mock the `FyersClient.exit_position` method to simulate successful and failed exits.
-   `[x]` **Testing:** Assert that the active `Position` is cleared after a successful exit.
-   `[x]` **Testing:** Assert that the system attempts retries on failed exit orders.

### **Dev Agent Record**

**Debug Log:**
- Identified that the `_close_position` method was already implemented but was being called incorrectly from a synchronous function (`_check_sl_tp_triggers`).
- The synchronous call was creating a coroutine but not awaiting it, preventing the async exit logic from executing.
- Corrected the invocation by using `asyncio.create_task()` to properly schedule the `_close_position` coroutine on the event loop.
- Added a test suite (`test_trade_exit_logic.py`) to verify the fix and test the exit logic, including success, retry, and failure scenarios.
- Discovered a missing feature during testing: the system did not notify the frontend if closing a position failed after all retries.
- Implemented the missing logic to broadcast an error message to the frontend via WebSocket upon ultimate failure.
- Re-ran tests to confirm all logic, including the new error notification, works as expected.

**Completion Notes:**
- Successfully implemented and verified the trade exit logic in `LiveTradingService`.
- The `_close_position` method now correctly handles placing exit orders, clearing the active position, and logging the exit reason.
- A robust retry mechanism (up to 3 attempts) is in place for exit order placement.
- The system now correctly broadcasts position updates to the frontend via WebSocket upon successful exit.
- An error notification is now sent to the frontend if the system fails to close a position after all retry attempts.
- Corrected a bug where the asynchronous `_close_position` method was not being properly called from the synchronous `_check_sl_tp_triggers` method.
- All SL/TP triggers now correctly initiate the asynchronous exit process.
- Added a comprehensive test suite to validate all aspects of the trade exit functionality.

**File List:**
- `src/trading/live_trading_service.py` - Corrected the async call in `_check_sl_tp_triggers` and added error broadcasting on final exit failure.
- `tests/test_trading/test_trade_exit_logic.py` - New test suite to verify all trade exit scenarios.

**Change Log:**
1.  Modified `_check_sl_tp_triggers` to correctly invoke the async `_close_position` method using `asyncio.create_task`.
2.  Added logic to `_close_position` to broadcast a WebSocket error message if exiting the position fails after all retries.
3.  Created `tests/test_trading/test_trade_exit_logic.py` with 4 test cases covering successful exit, retries, final failure, and correct async invocation from triggers.
4.  All tests are passing, confirming the logic is sound.

---