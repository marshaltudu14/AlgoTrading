---

### **Story 1.1.3: Refactor WebSocket for Tick Streaming**

**Status:** `Ready for Review`

**Story:**
As a developer, I need to refactor the existing WebSocket endpoint to primarily stream raw tick data to the frontend, so that the UI's candlestick chart can be updated in real-time for visualization purposes.

**Acceptance Criteria:**
1.  The WebSocket endpoint at `/ws/live/{user_id}` in `backend/main.py` is reviewed and refactored for clarity and efficiency.
2.  The `LiveTradingService` is modified to receive raw tick data from the `FyersClient`.
3.  The `LiveTradingService` immediately relays this raw tick data to the `Backend API`'s WebSocket manager.
4.  The WebSocket sends messages to connected clients with a specific `type` for tick data (e.g., `{ "type": "tick", "data": { ...tick_data... } }`).
5.  The tick data payload contains the necessary information for the frontend chart (e.g., `price`, `volume`, `timestamp`).
6.  The WebSocket connection includes a keep-alive mechanism (ping/pong) to ensure a stable connection.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `LiveTradingService`, create a callback function to handle incoming tick data from the `FyersClient` WebSocket.
-   `[ ]` **Backend:** This callback should not perform any processing; it should simply pass the tick data to the WebSocket manager in `backend/main.py`.
-   `[ ]` **Backend:** In `backend/main.py`, modify the `/ws/live/{user_id}` endpoint to handle the incoming tick data from the `LiveTradingService`.
-   `[ ]` **Backend:** Implement a broadcast mechanism to send the tick data to all connected clients for that `user_id`.
-   `[ ]` **Backend:** Define the exact JSON structure for the `"tick"` message type.
-   `[ ]` **Testing:** In `tests/test_api/`, create `test_websocket_tick_streaming.py`.
-   `[ ]` **Testing:** Write a test that simulates a `LiveTradingService` sending tick data and asserts that the WebSocket client receives the correctly formatted JSON message.
-   `[ ]` **Testing:** Write a test to verify the ping/pong mechanism maintains the connection.

---