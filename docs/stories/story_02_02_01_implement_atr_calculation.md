---

### **Story 2.2.1: Implement ATR Calculation**

**Status:** `Draft`

**Story:**
As a developer, I need to implement the logic to calculate the Average True Range (ATR) for the current instrument, so that this value can be used to set dynamic stop-loss and target price levels.

**Acceptance Criteria:**
1.  A new function or method is created, likely within `src/data_processing/feature_generator.py` or a new utility file, to calculate the ATR.
2.  The function takes a pandas DataFrame of historical data as input.
3.  The function correctly calculates the True Range (TR) for each candle.
4.  The function then calculates the ATR over a specified period (e.g., 14 periods, which should be configurable).
5.  The function returns the latest ATR value as a float.
6.  The `LiveTradingService` calls this function after fetching fresh historical data at each interval.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Create a new function `calculate_atr(df, period=14)` in a relevant utility file.
-   `[ ]` **Backend:** Inside the function, calculate the `high - low`, `abs(high - close.shift())`, and `abs(low - close.shift())` to determine the True Range.
-   `[ ]` **Backend:** Use the `pandas.DataFrame.ewm()` method to calculate the exponential moving average of the True Range, which gives the ATR.
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, after fetching data, call the new `calculate_atr` function.
-   `[ ]` **Backend:** Store the returned ATR value in a variable within the `LiveTradingService` instance.
-   `[ ]` **Testing:** Create a new test file `tests/test_utils/test_atr_calculation.py`.
-   `[ ]` **Testing:** Write a unit test with a sample DataFrame of candlestick data and assert that the `calculate_atr` function returns the correct, pre-calculated ATR value.
-   `[ ]` **Testing:** Write a test to handle edge cases, such as a DataFrame with insufficient data to calculate the ATR.

---