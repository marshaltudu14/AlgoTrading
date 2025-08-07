---

### **Story 4.2.3: Implement Manual Trade Confirmation**

**Status:** `Completed`

**Story:**
As a user, I want a confirmation step before a manual trade is executed, so that I can review the details and prevent accidental trades.

**Acceptance Criteria:**
1.  When the user submits the `ManualTradeForm`, a confirmation dialog appears on the frontend.
2.  The confirmation dialog displays all the details of the proposed trade (instrument, direction, quantity, SL/TP).
3.  The dialog has "Confirm" and "Cancel" buttons.
4.  If the user clicks "Confirm", the trade request is sent to the backend.
5.  If the user clicks "Cancel", the dialog closes and the trade is not sent.
6.  The backend API for manual trades (`/api/manual-trade`) is updated to expect a confirmation token or a two-step process to prevent re-submission.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** In `frontend/components/manual-trade-form.tsx`, implement a state variable to control the visibility of a confirmation dialog.
-   `[ ]` **Frontend:** Use a Radix UI `Dialog` component for the confirmation dialog.
-   `[ ]` **Frontend:** Populate the dialog with the trade details from the form's state.
-   `[ ]` **Frontend:** Implement `onClick` handlers for the "Confirm" and "Cancel" buttons.
-   `[ ]` **Frontend:** When "Confirm" is clicked, call the API endpoint to place the trade.
-   `[ ]` **Backend:** In `backend/main.py`, modify the `/api/manual-trade` endpoint to accept an optional `confirmation_token` or implement a two-step process (e.g., first request returns a token, second request sends the token with the trade).
-   `[ ]` **Testing:** Write a component test for `ManualTradeForm` to verify that the confirmation dialog appears with the correct details upon submission.
-   `[ ]` **Testing:** Write an integration test to verify the full confirmation flow, including successful trade placement and cancellation.

---