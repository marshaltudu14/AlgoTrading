---

### **Story 4.1.1: Implement Manual Trade Entry Form UI**

**Status:** `Completed`

**Story:**
As a user, I want a form on the frontend to manually enter trades when no automated position is active, so that I have full control over my trading when needed.

**Acceptance Criteria:**
1.  A new React component `ManualTradeForm.tsx` is created in `frontend/components/`.
2.  This component includes input fields for:
    *   Instrument (dropdown, populated from `/api/config`)
    *   Direction (Buy/Sell radio buttons or toggle)
    *   Quantity (number input)
    *   Stop-Loss (optional number input)
    *   Target (optional number input)
3.  The form has a submit button.
4.  The `ManualTradeForm` component is integrated into the `DashboardPage` (`app/dashboard/page.tsx`), ideally within a new `Card` component.
5.  The entire `ManualTradeForm` is disabled (e.g., using a `disabled` prop on the form elements) when an active automated position is detected (via WebSocket updates).
6.  A clear message or tooltip is displayed to the user explaining why the form is disabled when an automated trade is active.

**Tasks / Subtasks:**
-   `[ ]` **Frontend:** Create `frontend/components/manual-trade-form.tsx`.
-   `[ ]` **Frontend:** Design the form layout using existing UI components (e.g., Radix UI `Select`, `RadioGroup`, `Input`, `Button`).
-   `[ ]` **Frontend:** Implement state management within `ManualTradeForm` for each input field.
-   `[ ]` **Frontend:** Pass the `instruments` list from `DashboardPage` to `ManualTradeForm` to populate the instrument dropdown.
-   `[ ]` **Frontend:** In `app/dashboard/page.tsx`, import `ManualTradeForm` and render it within a `Card` component.
-   `[ ]` **Frontend:** Implement logic in `DashboardPage` to determine if an automated trade is active (by checking the `Position` object from the WebSocket state).
-   `[ ]` **Frontend:** Pass a `isDisabled` prop to `ManualTradeForm` based on the active trade status.
-   `[ ]` **Frontend:** In `ManualTradeForm`, apply the `disabled` attribute to form elements and display a tooltip or message when disabled.
-   `[ ]` **Testing:** Write a component test for `ManualTradeForm` to verify that all input fields are rendered correctly.
-   `[ ]` **Testing:** Write a test to assert that the form is disabled when `isDisabled` prop is true.

---