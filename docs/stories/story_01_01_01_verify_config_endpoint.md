---

### **Story 1.1.1: Verify Instrument Config Endpoint**

**Status:** `Ready for Dev`

**Story:**
As a developer, I need to verify that the existing FastAPI endpoint for configuration correctly serves the instrument and timeframe data from the `config/instruments.yaml` file, so that the frontend has a reliable source for populating its selection dropdowns.

**Acceptance Criteria:**
1.  When the backend server is running, a GET request to `/api/config` successfully returns a `200 OK` status.
2.  The JSON response from `/api/config` contains two keys: `instruments` and `timeframes`.
3.  The value of the `instruments` key in the response exactly matches the content of the `instruments` list in `C:\AlgoTrading\config\instruments.yaml`.
4.  The value of the `timeframes` key in the response exactly matches the content of the `timeframes` list in `C:\AlgoTrading\config\instruments.yaml`.
5.  If `config/instruments.yaml` is missing or malformed, the endpoint returns a `500 Internal Server Error` with a clear error message in the response body.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Locate the function in `backend/main.py` that handles the `/api/config` route.
-   `[ ]` **Backend:** Review the code that reads and parses `C:\AlgoTrading\config\instruments.yaml`.
-   `[ ]` **Backend:** Ensure a try-except block is present to handle `FileNotFoundError` and `yaml.YAMLError`.
-   `[ ]` **Testing:** Create a new test file `tests/test_api/test_config_endpoint.py`.
-   `[ ]` **Testing:** Write a unit test that uses a test client to make a GET request to `/api/config`.
-   `[ ]` **Testing:** In the test, assert that the response status code is `200`.
-   `[ ]` **Testing:** In the test, assert that the JSON response body matches a predefined, expected dictionary that mirrors the contents of a test `instruments.yaml` file.
-   `[ ]` **Testing:** Write a separate test to simulate a missing `instruments.yaml` file and assert that the endpoint returns a `500` status code.

---