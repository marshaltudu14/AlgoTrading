---

### **Story 1.3.1: Integrate Real-time Data into Trading Chart**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to connect the `TradingChart` component to the real-time data stream from the WebSocket, so that the chart updates live with new tick data.

**Acceptance Criteria:**
1.  The `TradingChart` component (`components/trading-chart.tsx`) is subscribed to the global state management store that holds the WebSocket data.
2.  The component receives the initial historical data via props, as implemented in a previous story.
3.  The component receives real-time tick updates from the state management store.
4.  For each new tick received, the `update` method of the `lightweight-charts` candlestick series is called with the new price data.
5.  The chart's time scale automatically scrolls to show the latest tick, keeping the most recent price action in view.
6.  The component's performance is optimized to handle a high frequency of tick updates without causing UI lag.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** In `components/trading-chart.tsx`, import and use the state management store (e.g., `useLiveDataStore`).
-   `[ ]` **Frontend:** Create a `useEffect` hook that listens for changes to the `latestTick` in the store.
-   `[ ]` **Frontend:** Inside this `useEffect` hook, call the `candlestickSeriesRef.current.update()` method with the data from the new tick.
-   `[ ]` **Frontend:** Ensure the `time` property of the tick data is correctly formatted as a `UTCTimestamp` before being passed to the chart.
-   `[ ]` **Frontend:** Use the `chartRef.current.timeScale().scrollToRealTime()` method to ensure the chart stays focused on the latest data.
-   `[ ]` **Frontend:** Profile the component's performance using React DevTools to ensure there are no unnecessary re-renders.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that simulates receiving new tick data from the store and asserts that the chart's `update` method is called with the correct data.

---