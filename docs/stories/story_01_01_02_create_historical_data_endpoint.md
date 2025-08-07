---

### **Story 1.1.2: Create Historical Data API Endpoint**

**Status:** `Ready for Review`

**Story:**
As a developer, I need to create a new, secure API endpoint that provides historical candlestick data to the frontend, so that the chart can be populated with the initial data set for a selected instrument and timeframe.

**Acceptance Criteria:**
1.  A new GET endpoint is created at `/api/historical-data` in `backend/main.py`.
2.  The endpoint requires JWT authentication using the existing `get_current_user` dependency.
3.  The endpoint accepts two required query parameters: `instrument` (string) and `timeframe` (string).
4.  If `instrument` or `timeframe` are missing, the endpoint returns a `422 Unprocessable Entity` error.
5.  The endpoint calls the `fetch_and_process_data` method from the `RealtimeDataLoader` in `src/utils/realtime_data_loader.py`, passing the instrument and timeframe.
6.  The fetched data (a pandas DataFrame) is converted to a JSON array of objects and returned in the response body with a `200 OK` status.
7.  Each object in the JSON array corresponds to a candle and has the keys `time`, `open`, `high`, `low`, `close`, `volume`.
8.  If the `RealtimeDataLoader` fails to fetch data, the endpoint returns a `500 Internal Server Error` with a descriptive message.

**Tasks / Subtasks:**
-   `[x]` **Backend:** In `backend/main.py`, define a new Pydantic model or use `Depends` for the `instrument` and `timeframe` query parameters.
-   `[x]` **Backend:** Create a new `async def` function for the `/api/historical-data` route, decorated with `@app.get("/api/historical-data")`.
-   `[x]` **Backend:** Add `current_user: dict = Depends(get_current_user)` to the function signature to enforce authentication.
-   `[x]` **Backend:** Inside the function, instantiate `RealtimeDataLoader`.
-   `[x]` **Backend:** Call `loader.fetch_and_process_data(instrument=instrument, timeframe=timeframe)`.
-   `[x]` **Backend:** Convert the resulting pandas DataFrame to JSON using `df.to_json(orient='records')`.
-   `[x]` **Backend:** Implement a try-except block to catch potential exceptions from the data loader and return appropriate `HTTPException`s.
-   `[x]` **Testing:** In `tests/test_api/`, create `test_historical_data_endpoint.py`.
-   `[x]` **Testing:** Write a test that mocks the `RealtimeDataLoader` and asserts that the endpoint returns a `200` status and correctly formatted JSON data.
-   `[x]` **Testing:** Write a test for the authentication, asserting a `401 Unauthorized` error is returned without a valid token.
-   `[x]` **Testing:** Write a test to check for the `422` error when query parameters are missing.
-   `[x]` **Testing:** Write a test that simulates a failure in the `RealtimeDataLoader` and asserts a `500` error is returned.

---