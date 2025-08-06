---

### **Story 4.2.1: Create Manual Trade API Endpoint**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to create a new API endpoint in the backend to receive and process manual trade requests from the frontend, so that users can initiate trades directly.

**Acceptance Criteria:**
1.  A new POST endpoint is created at `/api/manual-trade` in `backend/main.py`.
2.  The endpoint requires JWT authentication using the existing `get_current_user` dependency.
3.  The endpoint accepts a JSON request body conforming to the `ManualTradeRequest` schema (instrument, direction, quantity, optional SL/TP).
4.  The endpoint performs initial validation of the input data (e.g., quantity is positive, instrument is valid).
5.  The endpoint calls a new method in `LiveTradingService` (e.g., `initiate_manual_trade`) to handle the trade logic.
6.  If validation fails, the endpoint returns a `400 Bad Request` with a descriptive error message.
7.  If the trade initiation is successful, the endpoint returns a `200 OK` with a success message.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `backend/main.py`, define the `ManualTradeRequest` Pydantic model.
-   `[ ]` **Backend:** Create a new `async def` function for the `/api/manual-trade` route, decorated with `@app.post("/api/manual-trade")`.
-   `[ ]` **Backend:** Add `current_user: dict = Depends(get_current_user)` to the function signature.
-   `[ ]` **Backend:** Implement basic input validation for the `ManualTradeRequest` data.
-   `[ ]` **Backend:** Call a new method `self.live_trading_service.initiate_manual_trade(...)` (passing the request data and user info).
-   `[ ]` **Backend:** Implement error handling for the `initiate_manual_trade` call and return appropriate `HTTPException`s.
-   `[ ]` **Testing:** In `tests/test_api/`, create `test_manual_trade_endpoint.py`.
-   `[ ]` **Testing:** Write a test that sends a valid `ManualTradeRequest` and asserts a `200 OK` response.
-   `[ ]` **Testing:** Write tests for invalid inputs (e.g., negative quantity, invalid instrument) and assert `400 Bad Request`.
-   `[ ]` **Testing:** Write a test for authentication failure (`401 Unauthorized`).

---