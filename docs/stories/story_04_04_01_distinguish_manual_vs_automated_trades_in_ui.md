---

### **Story 4.4.1: Distinguish Manual vs. Automated Trades in UI**

**Status:** `Completed`

**Story:**
As a user, I want the user interface to clearly distinguish between manual and automated trades on the chart, so that I can easily identify the origin of each position and analyze their performance separately.

**Acceptance Criteria:**
1.  The `Position` object (sent via WebSocket) includes a `tradeType` field ("Automated" or "Manual").
2.  The `TradingChart` component (`components/trading-chart.tsx`) receives and interprets this `tradeType`.
3.  Automated trades are visualized on the chart using the default green/red color scheme for position rectangles and markers.
4.  Manual trades are visualized on the chart using a distinct color scheme (e.g., shades of blue) for position rectangles and markers.
5.  The trade details tooltip (on hover) explicitly states whether the trade was "Automated" or "Manual".

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, ensure that when a `Position` object is created (for both automated and manual trades), the `tradeType` attribute is correctly set.
-   `[ ]` **Backend:** Verify that the `tradeType` field is included in the WebSocket `position_update` messages.
-   `[ ]` **Frontend:** In `components/trading-chart.tsx`, modify the logic for drawing position rectangles and markers to conditionally apply colors based on the `tradeType` prop.
-   `[ ]` **Frontend:** Define the specific color palette for manual trades (e.g., blue for long, darker blue for short).
-   `[ ]` **Frontend:** Update the tooltip generation logic to include the `tradeType` in the displayed information.
-   `[ ]` **Testing:** Write a component test for `TradingChart` that mounts the component with both "Automated" and "Manual" trade data and asserts that the correct color schemes are applied.
-   `[ ]` **Testing:** Verify that the tooltip correctly displays the `tradeType`.

---