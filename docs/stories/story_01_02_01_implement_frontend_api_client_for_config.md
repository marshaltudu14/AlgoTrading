---

### **Story 1.2.1: Implement Frontend API Client for Configuration**

**Status:** `Ready for Review`

**Story:**
As a developer, I need to implement the client-side logic to fetch the system configuration from the backend, so that the instrument and timeframe dropdowns can be populated with data from the server.

**Acceptance Criteria:**
1.  A new function is created in `frontend/lib/api.ts` (or a similar API client file) to handle the GET request to the `/api/config` endpoint.
2.  This function is asynchronous and handles the JSON response.
3.  The function returns a typed object, e.g., `{ instruments: string[], timeframes: string[] }`.
4.  The `DashboardPage` component in `frontend/app/dashboard/page.tsx` calls this new function within a `useEffect` hook on component mount.
5.  The fetched instruments and timeframes are stored in the component's state.
6.  The instrument and timeframe `Select` components (from Radix UI) are correctly populated with the data from the state.
7.  The UI displays a loading state while the configuration is being fetched.
8.  If the API call fails, a user-friendly error message is displayed using the `Sonner` toast notification component.

**Tasks / Subtasks:**
-   `[x]` **Frontend:** In `frontend/lib/api.ts`, add a new `getConfig` async function.
-   `[x]` **Frontend:** Use the `fetch` API or a library like `axios` to make the GET request to `/api/config`.
-   `[x]` **Frontend:** Define a TypeScript interface for the configuration data.
-   `[x]` **Frontend:** In `frontend/app/dashboard/page.tsx`, add state variables for `instruments`, `timeframes`, `isLoadingConfig`, and `configError`.
-   `[x]` **Frontend:** Implement the `useEffect` hook to call `getConfig` and update the state.
-   `[x]` **Frontend:** Map over the `instruments` and `timeframes` state variables to render the `SelectItem` components within the dropdowns.
-   `[x]` **Frontend:** Use conditional rendering to show a loading spinner when `isLoadingConfig` is true.
-   `[x]` **Frontend:** In the `catch` block of the `useEffect` hook, call `toast.error()` from the `Sonner` library to display any errors.
-   `[x]` **Testing:** Write a unit test for the `getConfig` API client function, mocking the `fetch` call.
-   `[x]` **Testing:** Write a component test for the `DashboardPage` to verify that the dropdowns are populated correctly after the API call.

---