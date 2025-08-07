---

### **Story 2.3.1: Select Nearest ITM Strike Price**

**Status:** `Completed`

**Story:**
As a developer, I need to implement a function that, given an underlying spot price, can identify and select the nearest In-the-Money (ITM) strike price for options trading, so that the system can correctly choose the appropriate option contract.

**Acceptance Criteria:**
1.  A new utility function `get_nearest_itm_strike(spot_price, available_strikes, option_type)` is created, likely in `src/utils/option_utils.py` (new file).
2.  The function takes the current `spot_price` of the underlying, a list of `available_strikes` (e.g., `[45000, 45100, 45200]`), and the `option_type` ('CE' for Call, 'PE' for Put).
3.  For a Call option ('CE'), the function correctly identifies the nearest strike price that is less than or equal to the `spot_price`.
4.  For a Put option ('PE'), the function correctly identifies the nearest strike price that is greater than or equal to the `spot_price`.
5.  The function returns the selected ITM strike price.
6.  The function handles edge cases, such as an empty list of `available_strikes` or `spot_price` being outside the range of available strikes.

**Tasks / Subtasks:**
-   `[x]` **Backend:** Create a new file `src/utils/option_utils.py`.
-   `[x]` **Backend:** Implement the `get_nearest_itm_strike` function within `src/utils/option_utils.py`.
-   `[x]` **Backend:** Ensure the logic correctly differentiates between Call and Put options for ITM calculation.
-   `[x]` **Backend:** Add error handling or return `None` if no suitable strike is found.
-   `[x]` **Testing:** Create a new test file `tests/test_utils/test_option_utils.py`.
-   `[x]` **Testing:** Write unit tests for `get_nearest_itm_strike` with various `spot_price` values, `available_strikes` lists, and `option_type`s.
-   `[x]` **Testing:** Test edge cases like `spot_price` exactly on a strike, `spot_price` between strikes, and `available_strikes` being empty.

---