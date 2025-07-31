# Frontend Development Stories

**Project:** AlgoTrading Web UI
**Date:** 2025-07-31
**Author:** Bob, Scrum Master

This document provides a detailed breakdown of user stories for the development team, incorporating specific technical and design choices.

---

## Epic 1: Foundation & Core User Experience

**Goal:** Establish the project's technical foundation, implement user authentication, and create a polished application shell with a functional dashboard.

### Story 1.1: Backend API and Project Setup

*   **As a** Developer,
*   **I want to** set up a basic Python FastAPI backend service in the `/backend` directory with a health-check endpoint,
*   **so that** we have a running server to build upon.

**Acceptance Criteria:**
1.  A new FastAPI application is created in the `/backend` directory.
2.  The project has a `requirements.txt` file with `fastapi` and `uvicorn`.
3.  A `/api/health` endpoint exists and returns `{"status": "ok"}`.
4.  The server can be started with `uvicorn main:app --reload` from the `/backend` directory.

### Story 1.2: Frontend Application Shell & UI Library Setup

*   **As a** Developer,
*   **I want to** set up the Next.js application with Shadcn/UI, Framer Motion, and GSAP, and create the main layout,
*   **so that** we have a working frontend environment with our core design and animation libraries installed.

**Acceptance Criteria:**
1.  Shadcn/UI is initialized in the `/frontend` project.
2.  Framer Motion and GSAP are added as dependencies.
3.  A main layout component is created that includes a sidebar for desktop and a bottom navigation bar for mobile.
4.  The layout uses Shadcn components (e.g., `Button`, `Sheet` for the mobile sidebar).
5.  Placeholder pages for Dashboard, Backtest, and Live Trade are created and linked in the navigation components.

### Story 1.3: Refactor Fyers Authentication Logic

*   **As a** Developer,
*   **I want to** refactor the hardcoded `src/auth/fyers_auth.py` script into a reusable, parameter-driven function,
*   **so that** it can be securely called by the backend API.

**Acceptance Criteria:**
1.  A function `authenticate_fyers_user(fy_id, pin, totp_secret)` is created in a module within `/backend`.
2.  The function encapsulates the Fyers login logic, and all hardcoded credentials are removed.
3.  The function returns an `access_token` on success or raises a specific exception on failure.

### Story 1.4: Create Secure Login API Endpoint

*   **As a** Backend Developer,
*   **I want to** create a `/api/login` endpoint,
*   **so that** the frontend can securely authenticate a user.

**Acceptance Criteria:**
1.  A `POST /api/login` endpoint is created in the FastAPI app.
2.  It calls the `authenticate_fyers_user` function with credentials from the request body.
3.  On success, it returns a secure HttpOnly cookie containing a JWT session token.
4.  On failure, it returns a 401 Unauthorized status.

### Story 1.5: Build Animated Frontend Login UI

*   **As a** Frontend Developer,
*   **I want to** build a polished login page with fluid animations,
*   **so that** the user has an engaging first impression.

**Acceptance Criteria:**
1.  A login page is created using Shadcn `Input`, `Button`, and `Card` components.
2.  On page load, the form elements animate into view using Framer Motion or GSAP.
3.  Clicking the "Login" button sends a request to the `POST /api/login` endpoint.
4.  On success, the user is redirected to the dashboard. On failure, an error message (a Shadcn `Alert` or `Toast`) is displayed.

### Story 1.6: Display User Data on Dashboard

*   **As a** Frontend Developer,
*   **I want to** fetch and display the user's profile data on the dashboard with a subtle animation,
*   **so that** the user can confirm they are logged in.

**Acceptance Criteria:**
1.  The dashboard page makes a `GET /api/profile` request (a new endpoint will be needed).
2.  The user's name and available capital are displayed in Shadcn `Card` components.
3.  The numbers for the capital animate counting up to the final value when they first appear, using GSAP.
4.  If the API call fails, the user is redirected to the login page.

---

## Epic 2: Backtesting MVP

**Goal:** Implement the complete end-to-end backtesting feature with high-quality data visualizations.

### Story 2.1: Backtesting API & WebSocket Setup

*   **As a** Backend Developer,
*   **I want to** create an API endpoint to start a backtest and a WebSocket to stream its results,
*   **so that** the frontend can run and monitor backtests.

**Acceptance Criteria:**
1.  A `POST /api/backtest` endpoint is created that accepts instrument, timeframe, and duration.
2.  A WebSocket endpoint `/ws/backtest/{backtest_id}` is created.
3.  The API call triggers the backtest in a background thread and returns a `backtest_id`.
4.  The backend streams final results (chart data, metrics) over the WebSocket upon completion.

### Story 2.2: Backtesting Configuration UI

*   **As a** Frontend Developer,
*   **I want to** build the UI for configuring and launching a backtest,
*   **so that** the user can easily set up their test parameters.

**Acceptance Criteria:**
1.  The backtest page uses Shadcn components (`Select`, `Input`, `Button`) for the configuration form.
2.  The instrument `Select` is populated from `config/instruments.yaml`.
3.  Clicking "Run Backtest" calls the API and establishes the WebSocket connection.
4.  The form is disabled and a loading animation is shown while the backtest is running.

### Story 2.3: Implement Backtest Results Chart

*   **As a** Frontend Developer,
*   **I want to** integrate and display backtest results using a candlestick chart,
*   **so that** users can visually analyze the trading strategy's performance.

**Acceptance Criteria:**
1.  The `lightweight-charts` library is added to the frontend project.
2.  A new chart component is created to display the backtest results received from the WebSocket.
3.  The chart must display candlesticks for the price data.
4.  Trade entry and exit points must be marked on the chart with distinct icons (e.g., up/down arrows).

### Story 2.4: Display Backtest Metrics

*   **As a** Frontend Developer,
*   **I want to** display the final performance metrics of a backtest,
*   **so that** users have a clear, quantitative summary of the results.

**Acceptance Criteria:**
1.  A section below the chart displays key metrics (e.g., Total P&L, Win Rate, Max Drawdown).
2.  Each metric is displayed in a Shadcn `Card`.
3.  The P&L value is color-coded (green for profit, red for loss).
4.  A separate, smaller chart (using `recharts` or a similar library) specifically visualizes the portfolio drawdown over time.

