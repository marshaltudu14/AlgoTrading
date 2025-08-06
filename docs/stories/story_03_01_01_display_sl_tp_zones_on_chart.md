---

### **Story 3.1.1: Display SL/TP Zones on Chart**

**Status:** `Ready for dev`

**Story:**
As a user, I want to see the calculated stop-loss and target levels for my active trades displayed directly on the candlestick chart, so that I can visually track the position's progress and potential outcomes.

**Acceptance Criteria:**
1.  The `TradingChart` component (`components/trading-chart.tsx`) is updated to accept `stopLoss` and `targetPrice` props for an active trade.
2.  When these props are provided, the `lightweight-charts` API is used to draw two distinct `PriceLine` objects on the chart: one for the stop-loss and one for the target price.
3.  The stop-loss line is styled with a red color and a dashed line style.
4.  The target price line is styled with a green color and a dashed line style.
5.  Each line has a clear label (e.g., "SL" and "TP").
6.  These lines are dynamically updated if the SL/TP levels change (e.g., due to trailing stop-loss logic, though not in scope for this story).
7.  The SL/TP lines are removed from the chart when the active position is closed.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** In `components/trading-chart.tsx`, update the `TradingChartProps` interface to include `stopLoss?: number` and `targetPrice?: number`.
-   `[ ]` **Frontend:** Inside the `TradingChart` component's `useEffect` hook (where the chart is initialized or updated), add logic to create `PriceLine` objects when `stopLoss` and `targetPrice` props are present.
-   `[ ]` **Frontend:** Use `chart.addLineSeries()` or `chart.addPriceLine()` (depending on `lightweight-charts` version and best practice) to draw these lines.
-   `[ ]` **Frontend:** Apply the specified styling (color, line style, label) to each `PriceLine`.
-   `[ ]` **Frontend:** Implement logic to remove or hide these `PriceLine` objects when `stopLoss` and `targetPrice` props are `null` or `undefined` (indicating no active position).
-   `[ ]` **Frontend:** In `app/dashboard/page.tsx`, ensure the `stopLoss` and `targetPrice` from the active `Position` object (received via WebSocket) are passed as props to the `TradingChart`.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that mounts the component with `stopLoss` and `targetPrice` props and asserts that the corresponding lines are rendered on the chart.
-   `[ ]` **Testing:** Write a test that updates these props to `null` and asserts that the lines are removed.

---