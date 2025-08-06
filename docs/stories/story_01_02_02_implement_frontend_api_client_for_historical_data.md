---

### **Story 1.2.2: Implement Frontend API Client for Historical Data**

**Status:** `Draft`

**Story:**
As a developer, I need to implement the client-side logic to fetch historical data from the backend when a user selects an instrument and timeframe, so that the trading chart can be populated with the initial data set.

**Acceptance Criteria:**
1.  A new function is created in `frontend/lib/api.ts` to handle the GET request to the `/api/historical-data` endpoint.
2.  This function accepts `instrument` and `timeframe` as arguments and passes them as query parameters.
3.  The function is asynchronous and returns a typed array of candlestick data objects.
4.  In `frontend/app/dashboard/page.tsx`, an event handler is created for the instrument and timeframe `Select` components.
5.  When the user makes a selection, this event handler calls the new API function.
6.  The fetched historical data is stored in the component's state and passed as a prop to the `TradingChart` component.
7.  A loading state is displayed on the `TradingChart` component while the data is being fetched.
8.  If the API call fails, a `Sonner` toast notification is displayed with a user-friendly error message.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** In `frontend/lib/api.ts`, add a new `getHistoricalData` async function that takes `instrument` and `timeframe` as parameters.
-   `[ ]` **Frontend:** Use the `fetch` API to make the GET request to `/api/historical-data` with the correct query parameters.
-   `[ ]` **Frontend:** Define a TypeScript interface for the candlestick data that matches the `CandlestickData` interface in `TradingChart.tsx`.
-   `[ ]` **Frontend:** In `frontend/app/dashboard/page.tsx`, add state variables for `historicalData`, `isChartLoading`, and `chartError`.
-   `[ ]` **Frontend:** Create a function (e.g., `handleSelectionChange`) that is triggered by the `onValueChange` event of the `Select` components.
-   `[ ]` **Frontend:** Inside `handleSelectionChange`, set `isChartLoading` to true and call `getHistoricalData`.
-   `[ ]` **Frontend:** On a successful response, update the `historicalData` state and set `isChartLoading` to false.
-   `[ ]` **Frontend:** Pass the `historicalData` and `isChartLoading` state variables as props to the `TradingChart` component.
-   `[ ]` **Frontend:** In the `catch` block, call `toast.error()` to display any errors.
-   `[ ]` **Testing:** Write a unit test for the `getHistoricalData` API client function.
-   `[ ]` **Testing:** Write a component test for the `DashboardPage` to verify that the `TradingChart` is updated when the user selects a new instrument or timeframe.

---