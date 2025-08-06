---

### **Story 4.5.1: Secure API Keys and Sensitive Data**

**Status:** `Draft`

**Story:**
As a developer, I need to ensure that all API keys and sensitive user data are securely handled and encrypted, so that the system adheres to security best practices and protects user information.

**Acceptance Criteria:**
1.  API keys (Fyers `app_id`, `secret_key`, `totp_secret`) are loaded from environment variables, not hardcoded in the source code.
2.  The `JWT_SECRET` used for session management is loaded from an environment variable.
3.  Sensitive user data (e.g., Fyers `pin`, `totp_secret` during login) is handled securely and not exposed in logs or unencrypted storage.
4.  The `fyers_auth_service.py` module is reviewed to ensure secure handling of credentials during the authentication flow.
5.  All communication involving sensitive data (e.g., login requests) uses HTTPS. (This is typically handled by the deployment environment, but code should not prevent it).

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `backend/main.py`, replace hardcoded `JWT_SECRET` with `os.getenv("JWT_SECRET")`.
-   `[ ]` **Backend:** In `src/auth/fyers_auth_service.py`, ensure `app_id`, `secret_key`, and `totp_secret` are accessed via environment variables (e.g., `os.getenv`).
-   `[ ]` **Backend:** Review all logging statements in `backend/main.py` and `src/` directory to ensure no sensitive data is accidentally logged.
-   `[ ]` **Backend:** Implement a mechanism to mask or redact sensitive information in logs if it must pass through them temporarily.
-   `[ ]` **Testing:** Write a unit test that attempts to load API keys from environment variables and asserts that hardcoded values are not used.
-   `[ ]` **Testing:** Write a test that attempts to log sensitive data and asserts that it is either masked or not present in the logs.

---