---

### **Story 3.2.2: Create TradingView-Style Position Visualization**

**Status:** `Completed`

**Story:**
As a user, I want to see a visual representation of my active trade on the chart, similar to TradingView's long/short position tool, so that I can easily understand the current state and potential profit/loss of the position.

**Acceptance Criteria:**
1.  The `TradingChart` component is enhanced to draw a semi-transparent, colored rectangle on the chart representing the active position.
2.  The rectangle's vertical extent is defined by the entry price and the current price.
3.  The rectangle's color indicates the position's profitability:
    *   Green for a profitable long position.
    *   Red for a losing long position.
    *   Green for a profitable short position.
    *   Red for a losing short position.
4.  The rectangle's horizontal extent spans from the entry candle to the current candle.
5.  The visualization updates in real-time as new tick data arrives.
6.  The visualization is removed when the position is closed.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** In `components/trading-chart.tsx`, add new props to control the position visualization (e.g., `activePosition?: { entryPrice: number; currentPrice: number; direction: 'long' | 'short'; entryTime: number; }`).
-   `[ ]` **Frontend:** Inside `TradingChart`, use `lightweight-charts` drawing primitives (e.g., `chart.addLineSeries` with `setMarkers` or custom overlays if available) to draw the rectangle.
-   `[ ]` **Frontend:** Implement logic to calculate the color of the rectangle based on `entryPrice`, `currentPrice`, and `direction`.
-   `[ ]` **Frontend:** Ensure the rectangle updates with each new tick.
-   `[ ]` **Frontend:** Implement logic to clear the rectangle when `activePosition` prop is `null`.
-   `[ ]` **Frontend:** In `app/dashboard/page.tsx`, pass the active position data (from WebSocket updates) to the `TradingChart` component.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that mounts the component with `activePosition` props and asserts that the colored rectangle is rendered and updates correctly with new data.
-   `[ ]` **Testing:** Test both long and short positions, and profitable/losing scenarios.

---