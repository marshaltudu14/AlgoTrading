---

### **Story 2.4.4: Handle Options Position Management**

**Status:** `Completed`

**Story:**
As a developer, I need to ensure that when trading options, the position management logic correctly maps underlying index/stock movements to the options contract, and that closing the underlying position correctly triggers the options position closure.

**Acceptance Criteria:**
1.  ✅ When an options trade is initiated, the `LiveTradingService` uses the `option_utils` functions (from Stories 2.3.1, 2.3.2, 2.3.3) to select the correct options contract (symbol, expiry, strike, type).
2.  ✅ The `LiveTradingService` stores the selected options contract details (e.g., `fyers_symbol`) as part of the `Position` object.
3.  ✅ The `LiveTradingService` continuously monitors the underlying instrument's price.
4.  ✅ When the underlying instrument's price hits the calculated SL or TP, the `LiveTradingService` triggers the `_close_position` method for the options contract.
5.  ✅ The system logs the options contract details and the underlying price that triggered the exit.

**Tasks / Subtasks:**
-   `[x]` **Backend:** In `src/trading/live_trading_service.py`, when an options strategy is selected, integrate calls to `get_nearest_itm_strike` and `get_nearest_expiry` from `src/utils/option_utils.py` to determine the specific options contract to trade.
-   `[x]` **Backend:** Store the full Fyers symbol for the options contract in the `Position` object.
-   `[x]` **Backend:** Modify the SL/TP monitoring logic (from Story 2.4.2) to use the underlying instrument's price for comparison, even when an options position is active.
-   `[x]` **Backend:** When an SL/TP is hit on the underlying, ensure the `_close_position` method is called with the options contract's Fyers symbol.
-   `[x]` **Backend:** Add logging to clearly indicate when an options position is opened and closed, and the underlying price that caused the exit.
-   `[x]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for options position management.
-   `[x]` **Testing:** Mock the `option_utils` functions to return specific strike and expiry.
-   `[x]` **Testing:** Simulate an options trade and then simulate underlying price movements that trigger SL/TP, asserting that the correct options contract is exited.

### **Dev Agent Record**

**Debug Log:**
- The `get_nearest_expiry` and `map_underlying_to_option_price` functions were missing from `src/utils/option_utils.py`. I implemented them and added corresponding tests.
- The test runner was failing to discover the tests. I switched to using `unittest discover` to resolve the issue.
- The tests for options position management were failing due to several issues:
    - `AttributeError: 'LiveTradingService' object has no attribute 'agent'`: The `agent` attribute was not being initialized in the test setup.
    - `AttributeError: 'LiveTradingService' object has no attribute 'current_obs'`: The `current_obs` attribute was not being initialized in the test setup.
    - `TypeError: 'str' object has no attribute 'type'`: The `CapitalAwareQuantitySelector` was not being mocked correctly and was not returning an object with a `type` attribute.
    - `ValueError: not enough values to unpack (expected 4, got 0)`: The `trading_env.step` method was not returning the expected tuple.
    - `AssertionError: Expected 'place_order' to have been called once. Called 0 times.`: The `place_order` method was not being called because the test was not correctly set up to trigger a trade.
    - `AssertionError: Expected mock to have been awaited once. Awaited 0 times.`: The `_broadcast_position_update` method was not being awaited in the `_execute_trade` method.
    - `SyntaxError: 'await' outside async function`: I was adding `await` calls inside non-async functions.
- I fixed all these issues by correcting the test setup, mocking the necessary dependencies, and making the necessary functions asynchronous.

**Completion Notes:**
- Successfully implemented the options position management logic in `LiveTradingService`.
- The system now correctly selects the nearest ITM strike price and expiry date for options trades.
- The system now correctly maps the underlying's SL/TP to the option's price.
- The system now correctly monitors the underlying's price and closes the options position when the SL/TP is hit.
- Added a comprehensive test suite to validate all aspects of the options position management functionality.

**File List:**
- `src/utils/option_utils.py` - Implemented the missing `get_nearest_expiry` and `map_underlying_to_option_price` functions.
- `tests/test_utils/test_option_utils.py` - Added tests for the new option utility functions.
- `src/trading/live_trading_service.py` - Integrated the options trading logic into the `_execute_trade` and `_check_sl_tp_triggers` methods.
- `tests/test_trading/test_options_position_management.py` - Added a new test suite to verify the options position management logic.

**Change Log:**
1.  Implemented the `get_nearest_expiry` and `map_underlying_to_option_price` functions in `src/utils/option_utils.py`.
2.  Added tests for the new option utility functions in `tests/test_utils/test_option_utils.py`.
3.  Integrated the options trading logic into the `_execute_trade` and `_check_sl_tp_triggers` methods in `src/trading/live_trading_service.py`.
4.  Added a new test suite in `tests/test_trading/test_options_position_management.py` to verify the options position management logic.
5.  Fixed several bugs in the test setup and the `live_trading_service.py` file related to asynchronous calls and mocking.
6.  All tests are passing, confirming the logic is sound.

---