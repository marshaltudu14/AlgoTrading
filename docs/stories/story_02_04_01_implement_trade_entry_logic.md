---

### **Story 2.4.1: Implement Trade Entry Logic**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to implement the trade entry logic within the `LiveTradingService`, so that the system can automatically place orders based on the model's signals and the calculated quantity.

**Acceptance Criteria:**
1.  When the model generates a "BUY" or "SELL" signal with a valid quantity, the `LiveTradingService` initiates an order placement.
2.  The `LiveTradingService` calls the `place_order` method of the `FyersClient` with the correct `symbol`, `side`, `quantity`, and `productType`.
3.  The `productType` is determined based on the instrument (e.g., "INTRADAY" for index options/futures, "CNC" for stocks).
4.  The system logs the order details (symbol, quantity, side, order ID) upon successful placement.
5.  If the order placement fails (e.g., due to API error, insufficient margin), the system logs the error and does not proceed with position tracking.
6.  Upon successful order placement, a new `Position` object is created and stored within the `LiveTradingService` to track the active trade. This `Position` object includes the calculated `stopLoss` and `targetPrice`.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, identify the section where the model's signal is processed.
-   `[ ]` **Backend:** Implement a conditional block to check for "BUY" or "SELL" signals and a valid quantity.
-   `[ ]` **Backend:** Call `self.fyers_client.place_order(...)` with the appropriate parameters.
-   `[ ]` **Backend:** Implement error handling for the `place_order` call.
-   `[ ]` **Backend:** Upon successful order placement, create an instance of the `Position` data model (as defined in the Architecture document) and store it in a class variable within `LiveTradingService`.
-   `[ ]` **Backend:** Ensure the `Position` object includes the `stopLoss` and `targetPrice` calculated in Story 2.2.2.
-   `[ ]` **Backend:** Add comprehensive logging for order placement success/failure and position creation.
-   `[ ]` **Testing:** In `tests/test_trading/test_live_trading_service_model_integration.py` (or a new file), add tests for trade entry.
-   `[ ]` **Testing:** Mock the `FyersClient.place_order` method to simulate successful and failed order placements.
-   `[ ]` **Testing:** Assert that a `Position` object is created and correctly populated upon successful order placement.
-   `[ ]` **Testing:** Assert that no `Position` object is created if order placement fails.

---