---

### **Story 2.3.2: Select Nearest Options Expiry**

**Status:** `Draft`

**Story:**
As a developer, I need to implement a function that selects the nearest expiry date for an options contract, prioritizing weekly expiries, so that the system can automatically choose the most liquid and relevant options series.

**Acceptance Criteria:**
1.  A new utility function `get_nearest_expiry(available_expiries, prefer_weekly=True)` is created in `src/utils/option_utils.py`.
2.  The function takes a list of `available_expiries` (e.g., `["2025-08-07", "2025-08-14", "2025-08-28"]`) as input.
3.  The function correctly identifies the nearest expiry date from the current date.
4.  If `prefer_weekly` is `True`, the function prioritizes weekly expiries over monthly expiries if they are equally near.
5.  The function returns the selected expiry date as a string in "YYYY-MM-DD" format.
6.  The function handles edge cases, such as an empty list of `available_expiries`.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/utils/option_utils.py`, implement the `get_nearest_expiry` function.
-   `[ ]` **Backend:** Use Python's `datetime` module to parse and compare dates.
-   `[ ]` **Backend:** Implement logic to determine if an expiry is weekly (e.g., by checking if it's the last Thursday of the month for Nifty/Bank Nifty, or simply by its proximity to the current date if `prefer_weekly` is true).
-   `[ ]` **Backend:** Add error handling or return `None` if no suitable expiry is found.
-   `[ ]` **Testing:** In `tests/test_utils/test_option_utils.py`, add unit tests for `get_nearest_expiry` with various lists of `available_expiries` and `prefer_weekly` settings.
-   `[ ]` **Testing:** Test scenarios where weekly and monthly expiries are present, and ensure the correct one is selected based on proximity and preference.
-   `[ ]` **Testing:** Test with an empty list of expiries.

---