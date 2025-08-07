---

### **Story 3.3.1: Fetch and Display Real-time Metrics**

**Status:** `Completed`

**Story:**
As a user, I want to see real-time trading performance metrics on the dashboard, so that I can monitor my profitability and other key indicators at a glance.

**Acceptance Criteria:**
1.  The `DashboardPage` component (`app/dashboard/live/page.tsx`) fetches the latest metrics from the `/api/metrics` endpoint.
2.  The `AnimatedNumber` component correctly displays the `todayPnL`, `totalTrades`, and `winRate` from the fetched metrics.
3.  The metrics are updated periodically (e.g., every 5-10 seconds) or upon receiving a specific WebSocket message indicating a trade closure.
4.  The `todayPnL` value is formatted correctly with currency symbol and color-coded (green for positive, red for negative).
5.  The `winRate` is displayed as a percentage.
6.  A loading indicator is shown while metrics are being fetched.
7.  Error handling is implemented to display a toast notification if fetching metrics fails.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Ensure the `/api/metrics` endpoint in `backend/main.py` correctly reads and serves the `metrics.json` file.
-   `[ ]` **Frontend:** In `frontend/lib/api.ts`, add a new `getMetrics` async function to fetch data from `/api/metrics`.
-   `[ ]` **Frontend:** In `app/dashboard/live/page.tsx`, modify the `useEffect` hook to call `getMetrics` periodically or based on a WebSocket trigger.
-   `[ ]` **Frontend:** Update the `DashboardUserData` interface to include `totalTrades`, `winRate`, and `lastTradeTime`.
-   `[ ]` **Frontend:** Pass the fetched metrics to the `AnimatedNumber` components.
-   `[ ]` **Frontend:** Implement the logic to update the `todayPnL` color based on its value.
-   `[ ]` **Frontend:** Add a loading state for the metrics cards.
-   `[ ]` **Frontend:** Implement error handling with `Sonner` for failed metric fetches.
-   `[ ]` **Testing:** Write a unit test for the `getMetrics` API client function.
-   `[ ]` **Testing:** Write a component test for `DashboardPage` to verify that metrics are displayed and updated correctly.

---