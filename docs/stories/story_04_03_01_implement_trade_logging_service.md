---

### **Story 4.3.1: Implement Trade Logging Service**

**Status:** `Completed`

**Story:**
As a developer, I need to implement a dedicated service for logging all trades (both automated and manual) to a local JSON file, so that a comprehensive record is maintained for analysis and auditing.

**Acceptance Criteria:**
1.  A new Python class `TradeLogger` is created in `src/utils/trade_logger.py` (new file).
2.  The `TradeLogger` class has a method `log_trade(trade_data)` that accepts a dictionary conforming to the `Trade` data model.
3.  The `log_trade` method appends the `trade_data` to a JSON file named `tradelog.json` in the project's root directory.
4.  If `tradelog.json` does not exist, it is created with an empty JSON array.
5.  The `LiveTradingService` calls `TradeLogger.log_trade` whenever a position is closed (either automated or manual).
6.  The `trade_data` includes all the attributes defined in the `Trade` data model (trade ID, instrument, timeframe, entry/exit details, P&L, etc.).
7.  The system handles potential file I/O errors gracefully (e.g., file not found, permission issues).

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Create `src/utils/trade_logger.py`.
-   `[ ]` **Backend:** Implement the `TradeLogger` class with the `log_trade` method.
-   `[ ]` **Backend:** Use Python's `json` module to read and write to `tradelog.json`.
-   `[ ]` **Backend:** Ensure the `log_trade` method correctly appends to the JSON array.
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, after a position is closed (in `_close_position`), construct the `trade_data` dictionary.
-   `[ ]` **Backend:** Call `TradeLogger().log_trade(trade_data)`.
-   `[ ]` **Backend:** Add error handling for file operations.
-   `[ ]` **Testing:** Create a new test file `tests/test_utils/test_trade_logger.py`.
-   `[ ]` **Testing:** Write a unit test that calls `log_trade` with sample data and asserts that `tradelog.json` is created/updated correctly.
-   `[ ]` **Testing:** Write a test to verify that multiple trades are correctly appended to the JSON array.
-   `[ ]` **Testing:** Test error handling for file permissions or invalid JSON data.

---