---

### **Story 4.3.3: Implement User Action Logging for Audit**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to implement a system for logging all significant user actions (e.g., login, logout, manual trade initiation, start/stop live trading) for audit purposes and to enhance security.

**Acceptance Criteria:**
1.  A new Python class `UserActionLogger` is created in `src/utils/user_action_logger.py` (new file).
2.  The `UserActionLogger` has a method `log_action(user_id, action_type, details)` that records the action.
3.  Logged actions include a timestamp, `user_id`, `action_type` (e.g., "LOGIN", "LOGOUT", "MANUAL_TRADE_INITIATED", "LIVE_TRADING_STARTED", "LIVE_TRADING_STOPPED"), and `details` (a dictionary for context).
4.  Logs are written to a separate file (e.g., `logs/user_actions.log`) in a structured format (e.g., JSON lines or a custom delimited format).
5.  The `Backend API` integrates calls to `UserActionLogger` at appropriate points for each user action.
6.  Sensitive information (e.g., passwords, API keys) is never logged.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** Create `src/utils/user_action_logger.py`.
-   `[ ]` **Backend:** Implement `UserActionLogger` class with `log_action` method.
-   `[ ]` **Backend:** Configure the logger to write to `logs/user_actions.log`.
-   `[ ]` **Backend:** In `backend/main.py`:
    -   Add `UserActionLogger.log_action` call in `/api/login` after successful authentication.
    -   Add `UserActionLogger.log_action` call in `/api/logout`.
    -   Add `UserActionLogger.log_action` call in `/api/live/start`.
    -   Add `UserActionLogger.log_action` call in `/api/live/stop`.
    -   Add `UserActionLogger.log_action` call in `/api/manual-trade` (after confirmation, before execution).
-   `[ ]` **Backend:** Ensure `details` dictionary provides sufficient context for each action.
-   `[ ]` **Backend:** Implement sanitization to prevent logging sensitive data.
-   `[ ]` **Testing:** Create a new test file `tests/test_utils/test_user_action_logger.py`.
-   `[ ]` **Testing:** Write unit tests for `UserActionLogger` to verify correct log format and content.
-   `[ ]` **Testing:** Write integration tests for API endpoints to assert that `log_action` is called with correct parameters.

---