---

### **Story 1.2.3: Establish Frontend WebSocket Connection**

**Status:** `Completed`

**Story:**
As a developer, I need to establish a persistent WebSocket connection from the frontend to the backend, so that the application can receive real-time tick and position updates.

**Acceptance Criteria:**
1.  A new WebSocket client service is created in the frontend (e.g., `frontend/lib/websocket.ts`).
2.  This service manages the lifecycle of the WebSocket connection (connecting, disconnecting, handling messages).
3.  The `DashboardPage` component initializes the WebSocket service when the component mounts.
4.  The WebSocket connects to the `/ws/live/{user_id}` endpoint, where `{user_id}` is the ID of the authenticated user.
5.  The service includes robust reconnection logic with an exponential backoff strategy in case the connection is lost.
6.  A global state management solution (e.g., React Context or Zustand) is used to make the WebSocket data available to all necessary components.
7.  The service correctly parses incoming messages and dispatches them to the state management store based on their `type` (`"tick"` or `"position_update"`).
8.  The UI displays a clear visual indicator of the WebSocket connection status (e.g., a colored dot in the header).

**Tasks / Subtasks:**
-   `[x]` **Frontend:** Create a new file `frontend/lib/websocket.ts`.
-   `[x]` **Frontend:** Implement a `WebSocketService` class or object with `connect`, `disconnect`, and `onMessage` methods.
-   `[x]` **Frontend:** Implement the reconnection logic within the `connect` method's `onclose` event handler.
-   `[x]` **Frontend:** Create a new state management store (e.g., `frontend/store/live-data.ts`) using Zustand or React Context.
-   `[x]` **Frontend:** The store should hold the latest tick data and the current position information.
-   `[x]` **Frontend:** In the `WebSocketService`, the `onMessage` handler should parse the incoming JSON and call the appropriate actions on the state management store.
-   `[x]` **Frontend:** In `frontend/app/dashboard/page.tsx`, import and initialize the `WebSocketService` in a `useEffect` hook.
-   `[x]` **Frontend:** Create a `ConnectionStatus` component that subscribes to the WebSocket connection state and displays an appropriate indicator.
-   `[ ]` **Testing:** Write unit tests for the `WebSocketService`, mocking the native WebSocket object.
-   `[ ]` **Testing:** Write tests for the state management store to ensure it updates correctly based on dispatched actions.

---