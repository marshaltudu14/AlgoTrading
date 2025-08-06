---

### **Story 1.1.4: Implement WebSocket for Position Updates**

**Status:** `Ready for Dev`

**Story:**
As a developer, I need to implement a mechanism for the `LiveTradingService` to push structured position updates to the frontend via the WebSocket, so that the user can see their live trade information (entry, exit, SL/TP) on the UI.

**Acceptance Criteria:**
1.  A new message `type` is added to the WebSocket protocol for position updates (e.g., `"position_update"`).
2.  The `LiveTradingService` sends a `position_update` message whenever a new position is opened, a position is closed, or the SL/TP levels of an existing position are modified.
3.  The `data` payload of the `position_update` message conforms to the `Position` data model defined in the Architecture document.
4.  The payload includes the `instrument`, `direction`, `entryPrice`, `quantity`, `stopLoss`, `targetPrice`, `currentPnl`, and `tradeType`.
5.  When a position is closed, the message includes the `exitPrice` and final `pnl`.
6.  The `Backend API`'s WebSocket manager correctly receives these updates from the `LiveTradingService` and broadcasts them to the appropriate clients.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `LiveTradingService`, identify all the places where a position's state changes (e.g., `_enter_trade`, `_exit_trade`, `_update_sl_tp`).
-   `[ ]` **Backend:** At each of these points, create a dictionary that matches the `Position` data model.
-   `[ ]` **Backend:** Implement a method to send this dictionary to the WebSocket manager in `backend/main.py`, clearly marking it as a `"position_update"`.
-   `[ ]` **Backend:** In `backend/main.py`, enhance the WebSocket manager to differentiate between `"tick"` messages and `"position_update"` messages.
-   `[ ]` **Backend:** Ensure `"position_update"` messages are broadcast to all connected clients for the relevant `user_id`.
-   `[ ]` **Testing:** In `tests/test_api/test_websocket_tick_streaming.py` (or a new file), add tests for this functionality.
-   `[ ]` **Testing:** Write a test that simulates the `LiveTradingService` sending a `position_update` and asserts that the WebSocket client receives the correctly formatted JSON message.
-   `[ ]` **Testing:** Write separate tests for position open, position close, and SL/TP update messages.

---