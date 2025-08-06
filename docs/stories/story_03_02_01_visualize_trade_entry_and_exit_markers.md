---

### **Story 3.2.1: Visualize Trade Entry and Exit Markers**

**Status:** `Draft`

**Story:**
As a user, I want to see clear visual markers on the candlestick chart indicating where a trade was entered and exited, so that I can easily review the historical performance of each trade.

**Acceptance Criteria:**
1.  The `TradingChart` component's `tradeMarkers` prop is utilized to display entry and exit points.
2.  When a trade is entered, a marker is placed on the entry candle using the `createTradeMarker` helper function.
    *   For a long entry, an `arrowUp` shape with a green color is used.
    *   For a short entry, an `arrowDown` shape with a red color is used.
3.  When a trade is exited, a marker is placed on the exit candle.
    *   For a long exit, a `circle` shape with a blue color is used.
    *   For a short exit, a `circle` shape with an orange color is used.
4.  The markers include a text label indicating "Entry" or "Exit" and the corresponding price.
5.  The `LiveTradingService` sends the necessary data (entry/exit time, price, direction) via WebSocket for the frontend to create these markers.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, when a trade is entered, include the `entryTime`, `entryPrice`, and `direction` in the WebSocket `position_update` message.
-   `[ ]` **Backend:** When a trade is exited, include the `exitTime` and `exitPrice` in the WebSocket `position_update` message.
-   `[ ]` **Frontend:** In `app/dashboard/page.tsx`, maintain a state variable (e.g., `tradeMarkers`) that is an array of `TradeMarker` objects.
-   `[ ]` **Frontend:** When a `position_update` message is received indicating a new trade, use the `createTradeMarker` helper function to generate an entry marker and add it to the `tradeMarkers` state.
-   `[ ]` **Frontend:** When a `position_update` message indicates a trade closure, generate an exit marker and add it to the `tradeMarkers` state.
-   `[ ]` **Frontend:** Pass the `tradeMarkers` state to the `TradingChart` component.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that verifies entry and exit markers are correctly displayed based on the `tradeMarkers` prop.
-   `[ ]` **Testing:** Write an integration test that simulates a trade lifecycle (entry and exit) and asserts that the correct markers appear on the chart.

---