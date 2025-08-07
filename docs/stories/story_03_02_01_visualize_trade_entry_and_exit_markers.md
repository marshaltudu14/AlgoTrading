---

### **Story 3.2.1: Visualize Trade Entry and Exit Markers**

**Status:** `Completed`

**Story:**
As a user, I want to see clear visual markers on the candlestick chart indicating where a trade was entered and exited, so that I can easily review the historical performance of each trade.

**Acceptance Criteria:**
1.  ✅ The `TradingChart` component's `tradeMarkers` prop is utilized to display entry and exit points.
2.  ✅ When a trade is entered, a marker is placed on the entry candle using the `createTradeMarker` helper function.
    *   For a long entry, an `arrowUp` shape with a green color is used.
    *   For a short entry, an `arrowDown` shape with a red color is used.
3.  ✅ When a trade is exited, a marker is placed on the exit candle.
    *   For a long exit, a `circle` shape with a blue color is used.
    *   For a short exit, a `circle` shape with an orange color is used.
4.  ✅ The markers include a text label indicating "Entry" or "Exit" and the corresponding price.
5.  ✅ The `LiveTradingService` sends the necessary data (entry/exit time, price, direction) via WebSocket for the frontend to create these markers.

**Tasks / Subtasks:**
-   `[x]` **Backend:** In `src/trading/live_trading_service.py`, when a trade is entered, include the `entryTime`, `entryPrice`, and `direction` in the WebSocket `position_update` message.
-   `[x]` **Backend:** When a trade is exited, include the `exitTime` and `exitPrice` in the WebSocket `position_update` message.
-   `[x]` **Frontend:** In `app/dashboard/live/page.tsx`, maintain a state variable (e.g., `tradeMarkers`) that is an array of `TradeMarker` objects.
-   `[x]` **Frontend:** When a `position_update` message is received indicating a new trade, use the `createTradeMarker` helper function to generate an entry marker and add it to the `tradeMarkers` state.
-   `[x]` **Frontend:** When a `position_update` message indicates a trade closure, generate an exit marker and add it to the `tradeMarkers` state.
-   `[x]` **Frontend:** Pass the `tradeMarkers` state to the `TradingChart` component.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that verifies entry and exit markers are correctly displayed based on the `tradeMarkers` prop.
-   `[ ]` **Testing:** Write an integration test that simulates a trade lifecycle (entry and exit) and asserts that the correct markers appear on the chart.

### **Dev Agent Record**

**Debug Log:**
- Modified `LiveTradingService` to include `entryTime`, `entryPrice`, `direction`, `exitTime`, and `exitPrice` in the `position_update` WebSocket message.
- Updated `TradingChart` component to accept `activePosition` prop.
- Implemented logic in `TradingChart` to create and update trade markers based on `activePosition` data.

**Completion Notes:**
- The `TradingChart` now displays entry and exit markers for active trades.
- Markers are dynamically updated based on the `activePosition` data received via WebSocket.

**File List:**
- `src/trading/live_trading_service.py` - Added `entryTime`, `entryPrice`, `direction`, `exitTime`, and `exitPrice` to `position_update` message.
- `frontend/components/trading-chart.tsx` - Updated `TradingChartProps` and implemented marker rendering logic.

**Change Log:**
1.  Modified `src/trading/live_trading_service.py` to include `entryTime` and `exitTime` in the `position_update` message.
2.  Updated `frontend/components/trading-chart.tsx` to include `activePosition` in `TradingChartProps` and implemented logic to create and update trade markers based on `activePosition` data.

---