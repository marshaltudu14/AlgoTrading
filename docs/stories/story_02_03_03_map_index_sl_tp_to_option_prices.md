---

### **Story 2.3.3: Map Index SL/TP to Option Prices**

**Status:** `Draft`

**Story:**
As a developer, I need to implement a function that translates the stop-loss (SL) and target price (TP) levels of an underlying index/stock to the corresponding option contract's price, so that options positions can be managed effectively based on the underlying's movement.

**Acceptance Criteria:**
1.  A new utility function `map_underlying_to_option_price(underlying_price, option_strike, option_type, current_option_price)` is created in `src/utils/option_utils.py`.
2.  The function takes the `underlying_price` (which could be the SL or TP of the index), the `option_strike`, the `option_type` ('CE' or 'PE'), and the `current_option_price` (at the time of mapping).
3.  The function calculates the expected option price based on the underlying price, strike, and type. This will involve a simplified delta-based calculation or a direct mapping based on the difference between underlying and strike.
4.  The function returns the mapped option price.
5.  The function handles cases where the underlying price moves significantly, ensuring the mapped option price remains realistic.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/utils/option_utils.py`, implement the `map_underlying_to_option_price` function.
-   `[ ]` **Backend:** For simplicity, initially implement a linear mapping: `mapped_option_price = current_option_price + (underlying_price - current_underlying_price) * delta`. (Note: `delta` would need to be estimated or fetched).
-   `[ ]` **Backend:** Consider adding a more sophisticated options pricing model (e.g., Black-Scholes simplified version) if a simple linear mapping proves insufficient.
-   `[ ]` **Testing:** In `tests/test_utils/test_option_utils.py`, add unit tests for `map_underlying_to_option_price` with various underlying prices, strikes, and option types.
-   `[ ]` **Testing:** Test the function with both in-the-money and out-of-the-money scenarios.

---