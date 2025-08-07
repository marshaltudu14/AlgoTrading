---

### **Story 3.3.2: Implement Next Data Fetch Countdown Timer**

**Status:** `Completed`

**Story:**
As a user, I want to see a countdown timer on the frontend indicating when the next data fetch for the model will occur, so that I am aware of the system's operational rhythm.

**Acceptance Criteria:**
1.  The backend (`LiveTradingService`) provides the frontend with the configured data fetch interval (e.g., 5 minutes).
2.  The frontend displays a countdown timer (MM:SS format) that resets at the beginning of each data fetch interval.
3.  During the actual data fetching process, the timer temporarily changes to "Fetching..." or a similar status.
4.  If a data fetch fails, an error message is displayed near the timer.
5.  The timer is displayed prominently on the dashboard, as per the UX Vision.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/trading/live_trading_service.py`, ensure the configured `timeframe` (which dictates the fetch interval) is accessible and can be sent to the frontend.
-   `[ ]` **Backend:** Modify the WebSocket `position_update` message or create a new WebSocket message type (e.g., `"system_status"`) to include the `fetchIntervalSeconds` and `nextFetchTimestamp`.
-   `[ ]` **Frontend:** In `app/dashboard/page.tsx`, add state variables to manage the countdown timer's value and status (e.g., `countdown`, `fetchStatus`).
-   `[ ]` **Frontend:** Implement a `useEffect` hook that sets up an interval timer to update the countdown every second.
-   `[ ]` **Frontend:** The timer should calculate the remaining time until the `nextFetchTimestamp` based on the `fetchIntervalSeconds`.
-   `[ ]` **Frontend:** Implement logic to display "Fetching..." when a data fetch is in progress (e.g., based on a WebSocket message from the backend).
-   `[ ]` **Frontend:** Display any fetch errors received via WebSocket near the timer.
-   `[ ]` **Frontend:** Integrate the countdown timer UI element into the `DashboardPage` layout.
-   `[ ]` **Testing:** Write a component test for the countdown timer UI element, simulating different `fetchIntervalSeconds` and `nextFetchTimestamp` values.
-   `[ ]` **Testing:** Test the "Fetching..." and error display states.

---