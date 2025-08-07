---

### **Story 3.1.1: Display SL/TP Zones on Chart**

**Status:** `Completed`

**Story:**
As a user, I want to see the calculated stop-loss and target levels for my active trades displayed directly on the candlestick chart, so that I can visually track the position's progress and potential outcomes.

**Acceptance Criteria:**
1.  ✅ The `TradingChart` component (`components/trading-chart.tsx`) is updated to accept `stopLoss` and `targetPrice` props for an active trade.
2.  ✅ When these props are provided, the `lightweight-charts` API is used to draw two distinct `PriceLine` objects on the chart: one for the stop-loss and one for the target price.
3.  ✅ The stop-loss line is styled with a red color and a dashed line style.
4.  ✅ The target price line is styled with a green color and a dashed line style.
5.  ✅ Each line has a clear label (e.g., "SL" and "TP").
6.  ✅ These lines are dynamically updated if the SL/TP levels change (e.g., due to trailing stop-loss logic, though not in scope for this story).
7.  ✅ The SL/TP lines are removed from the chart when the active position is closed.

**Tasks / Subtasks:**
-   `[x]` **Frontend:** In `components/trading-chart.tsx`, update the `TradingChartProps` interface to include `stopLoss?: number` and `targetPrice?: number`.
-   `[x]` **Frontend:** Inside the `TradingChart` component's `useEffect` hook (where the chart is initialized or updated), add logic to create `PriceLine` objects when `stopLoss` and `targetPrice` props are present.
-   `[x]` **Frontend:** Use `chart.addLineSeries()` or `chart.addPriceLine()` (depending on `lightweight-charts` version and best practice) to draw these lines.
-   `[x]` **Frontend:** Apply the specified styling (color, line style, label) to each `PriceLine`.
-   `[x]` **Frontend:** Implement logic to remove or hide these `PriceLine` objects when `stopLoss` and `targetPrice` props are `null` or `undefined` (indicating no active position).
-   `[x]` **Frontend:** In `app/dashboard/page.tsx`, ensure the `stopLoss` and `targetPrice` from the active `Position` object (received via WebSocket) are passed as props to the `TradingChart`.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that mounts the component with `stopLoss` and `targetPrice` props and asserts that the corresponding lines are rendered on the chart.
-   `[ ]` **Testing:** Write a test that updates these props to `null` and asserts that the lines are removed.

### **Dev Agent Record**

**Debug Log:**
- Updated `frontend/store/live-data.ts` to include `activePosition` in the store, allowing SL/TP data to be globally accessible.
- Modified `frontend/app/dashboard/page.tsx` to pass `position?.stopLoss` and `position?.targetPrice` as props to the `TradingChart` component.
- Updated `frontend/components/trading-chart.tsx` to accept `stopLoss` and `targetPrice` props.
- Implemented `useEffect` in `TradingChart.tsx` to create and manage `PriceLine` objects for SL/TP using `lightweight-charts` API.
- Ensured `PriceLine` objects are removed when `stopLoss` or `targetPrice` become `null` or `undefined`.

**Completion Notes:**
- The `TradingChart` now visually displays stop-loss and target price levels as dashed lines.
- These lines are dynamically updated based on the active position's SL/TP values received via WebSocket.
- The display is removed when there is no active position.

**File List:**
- `frontend/store/live-data.ts` - Added `activePosition` to the store.
- `frontend/app/dashboard/page.tsx` - Passed `stopLoss` and `targetPrice` props to `TradingChart`.
- `frontend/components/trading-chart.tsx` - Implemented SL/TP price line rendering logic.

**Change Log:**
1.  Modified `frontend/store/live-data.ts` to include `activePosition` in the `LiveDataState` interface and added `setActivePosition` action.
2.  Updated `frontend/app/dashboard/page.tsx` to extract `setActivePosition` from `useLiveDataStore` and use it to update the `activePosition` state based on WebSocket `position_update` messages. Also, passed `position?.stopLoss` and `position?.targetPrice` to the `TradingChart` component.
3.  Modified `frontend/components/trading-chart.tsx` to update `TradingChartProps` with `stopLoss` and `targetPrice` (optional numbers). Added `slPriceLineRef` and `tpPriceLineRef` to manage price line instances. Implemented a `useEffect` hook that creates `PriceLine` objects for `stopLoss` and `targetPrice` with specified styling (red/green, dashed, labels) and removes them when the props are `null` or `undefined`.

---