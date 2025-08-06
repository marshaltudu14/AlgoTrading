---

### **Story 2.4.4: Handle Options Position Management**

**Status:** `Draft`

**Story:**
As a developer, I need to ensure that when trading options, the position management logic correctly maps underlying index/stock movements to the options contract, and that closing the underlying position correctly triggers the options position closure.

**Acceptance Criteria:**
1.  When an options trade is initiated, the `LiveTradingService` uses the `option_utils` functions (from Stories 2.3.1, 2.3.2, 2.3.3) to select the correct options contract (symbol, expiry, strike, type).
2.  The `LiveTradingService` stores the selected options contract details (e.g., `fyers_symbol`) as part of the `Position` object.
3.  The `LiveTradingService` continuously monitors the underlying instrument's price.
4.  When the underlying instrument's price hits the calculated SL or TP, the `LiveTradingService` triggers the `_close_position` method for the options contract.
5.  The system logs the options contract details and the underlying price that triggered the exit.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, when an options strategy is selected, integrate calls to `get_nearest_itm_strike` and `get_nearest_expiry` from `src/utils/option_utils.py` to determine the specific options contract to trade.
-   `[ ]` **Backend:** Store the full Fyers symbol for the options contract in the `Position` object.
-   `[ ]` **Backend:** Modify the SL/TP monitoring logic (from Story 2.4.2) to use the underlying instrument's price for comparison, even when an options position is active.
-   `[ ]` **Backend:** When an SL/TP is hit on the underlying, ensure the `_close_position` method is called with the options contract's Fyers symbol.
-   `[ ]` **Backend:** Add logging to clearly indicate when an options position is opened and closed, and the underlying price that caused the exit.
-   `[ ]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for options position management.
-   `[ ]` **Testing:** Mock the `option_utils` functions to return specific strike and expiry.
-   `[ ]` **Testing:** Simulate an options trade and then simulate underlying price movements that trigger SL/TP, asserting that the correct options contract is exited.

---