# AlgoTrading Frontend Product Requirements Document (PRD)

## Goals and Background Context

### Goals

*   To create a web-based frontend for a Python algorithmic trading bot.
*   To provide a user-friendly interface for backtesting trading strategies against historical data.
*   To enable the execution and monitoring of live trades using the trained model.
*   To ensure the initial version is robust for personal use while being architected for future multi-user scalability.
*   To build user trust through real-time data visualization, clear status indicators, and transparent error logging.

### Background Context

The existing project is a functional Python-based algorithmic trading bot that has a trained model (`universal_final_model.pth`). Currently, interacting with the bot (for backtesting or live trading) requires direct backend access. This is inefficient and not user-friendly.

This PRD outlines the requirements for a Next.js web application that will act as the primary interface for the trading bot. The frontend will not handle model training, but will allow users to leverage the existing model to run backtests and initiate live trades. A key challenge is to create a seamless user experience that abstracts the backend complexity, particularly in handling different financial instruments like stocks and index options.

### Change Log

| Date       | Version | Description                                   | Author                |
| :--------- | :------ | :-------------------------------------------- | :-------------------- |
| 2025-07-31 | 1.0     | Initial draft based on brainstorming session. | John, Product Manager |

## Requirements

### Functional

1.  **FR1: User Authentication:** The system shall allow a user to log in using their Fyers API credentials to establish a secure session.
2.  **FR2: Dashboard Display:** Upon login, the system shall display a dashboard showing the user's Fyers account information and total available capital.
3.  **FR3: Backtest Parameter Selection:** The system shall allow a user to configure a backtest by selecting an instrument from `config/instruments.yaml`, a timeframe, and a duration in days.
4.  **FR4: Backtest Execution:** The system shall execute a backtest using the selected parameters against the `universal_final_model.pth`.
5.  **FR5: Backtest Visualization:** The system shall display backtest results, including a performance chart and key metrics (e.g., Win Rate, Total Trades).
6.  **FR6: Live Trade Configuration:** The system shall allow a user to configure a live trading session by selecting an instrument and timeframe.
7.  **FR7: Options Strategy Selection:** If an "index" type instrument is selected for a live trade, the system shall allow the user to select an options strategy (ITM, ATM, OTM), with ITM as the default.
8.  **FR8: Live Trade Execution:** The system shall allow a user to start and stop the live trading bot.
9.  **FR9: Real-time Monitoring:** The system shall display a real-time chart of the selected instrument, the model's predictions, and the bot's activity during a live session.
10. **FR10: Instrument Lock:** Once a live trading session is started, the system shall prevent the user from changing the selected instrument until the session is stopped.

### Non-Functional

1.  **NFR1: Real-time Feedback:** The UI shall update in near real-time during live trading and backtesting visualization, using WebSockets or a similar technology.
2.  **NFR2: Configuration-Driven:** The list of available instruments in the UI shall be dynamically populated from the `config/instruments.yaml` file.
3.  **NFR3: API Connection Resilience:** The system shall automatically attempt to re-establish a dropped Fyers API connection. The UI must display the current connection status.
4.  **NFR4: Scalability:** The initial architecture should be designed to support multiple users in the future, even though the MVP is for a single user.
5.  **NFR5: Security:** All communication between the frontend and backend, especially involving API keys and user data, must be encrypted using HTTPS.

## User Interface Design Goals

### Overall UX Vision

The user experience should be clean, intuitive, and focused on building trust. The user must feel in control and have a clear, real-time understanding of what the trading bot is doing at all times. The design should prioritize clarity and data visualization over dense, cluttered interfaces.

### Key Interaction Paradigms

*   **Navigation:** A persistent sidebar (on desktop) and a bottom navigation bar (on mobile) will provide access to the three core sections: Home/Dashboard, Backtest, and Live Trade.
*   **Data Visualization:** Interactive charts are central to the experience. They should clearly display price action, trade entries/exits, and model predictions.
*   **Status Indication:** Clear, color-coded indicators should be used for API connection status, bot running status, and trade outcomes (profit/loss).
*   **Configuration:** Forms for configuring backtests and live trading sessions should be simple and uncluttered, with sensible defaults.

### Core Screens and Views

*   **Login Screen:** A simple form to enter Fyers API credentials.
*   **Main Dashboard:** The landing page after login, showing user/capital info.
*   **Backtesting Screen:** A view with configuration inputs and a results display area (chart and metrics).
*   **Live Trading Screen:** A view for configuring, starting/stopping, and monitoring live bot performance.

### Accessibility: WCAG AA

The application should adhere to Web Content Accessibility Guidelines (WCAG) 2.1 Level AA to ensure it is usable by people with disabilities.

### Branding

There are no specific branding requirements at this stage. The focus is on a clean, professional, and modern "fintech" aesthetic. A dark mode theme is recommended.

### Target Device and Platforms: Web Responsive

The application must be a responsive web app that works seamlessly on both desktop and mobile browsers.

## Technical Assumptions

### Repository Structure: Monorepo

A monorepo is recommended to manage both the Python backend and the Next.js frontend in a single repository. This simplifies dependency management and cross-service development.

### Service Architecture

The architecture will consist of two main services:
1.  **Frontend:** A Next.js application responsible for all UI rendering and user interaction.
2.  **Backend:** A Python (FastAPI is recommended) application that exposes a REST API. This backend will handle all business logic, interact with the Fyers API, and manage the trading bot's lifecycle. All Fyers API communication (auth, data fetching, execution) MUST be handled by the backend.

### Testing Requirements: Unit + Integration

The project should include:
*   **Unit Tests:** For individual components and functions in both the frontend and backend.
*   **Integration Tests:** To verify the interactions between the frontend and backend services.

### Additional Technical Assumptions and Requests

*   **Frontend Framework:** Next.js (as specified).
*   **Backend API:** A RESTful API is required. FastAPI is recommended for its performance and automatic documentation features.
*   **Real-time Communication:** WebSockets will be used for real-time updates between the backend and frontend.
*   **Deployment:** The initial deployment can be manual, but the architecture should allow for future containerization (e.g., using Docker).
*   **Authentication Refactoring:** The existing `src/auth/fyers_auth.py` script **must** be refactored. The hardcoded credentials will be removed, and the logic will be encapsulated in a function that accepts user credentials as parameters. This function will be exposed via a secure backend API endpoint.

## Epic List

1.  **Epic 1: Foundation & Core User Experience:** Establish the project's technical foundation, implement user authentication, and create the basic application shell with a functional dashboard.
2.  **Epic 2: Backtesting MVP:** Implement the complete end-to-end backtesting feature, allowing users to configure, run, and visualize the results of a backtest.
3.  **Epic 3: Live Trading & Real-Time Monitoring:** Implement the live trading functionality, including the options strategy abstraction, real-time monitoring, and user controls.

## Epic 1: Foundation & Core User Experience

**Goal:** To establish the project's technical foundation, implement a secure user authentication flow, and create the basic application shell with a functional dashboard that displays the user's capital. This epic delivers the minimum viable product for a user to log in and see that their account is connected.

### Story 1.1: Backend API and Project Setup
As a Developer, I want to set up a basic Python backend service with a health-check endpoint, so that we have a running server to build upon.
**Acceptance Criteria:**
1. A new FastAPI application is created.
2. The project has a `requirements.txt` file with the necessary libraries.
3. There is a `/api/health` endpoint that returns a `{"status": "ok"}` JSON response.
4. The server can be started with a simple command.

### Story 1.2: Frontend Application Shell Setup
As a Developer, I want to set up a basic Next.js application with placeholder pages, so that we have a working frontend environment.
**Acceptance Criteria:**
1. A new Next.js project is created.
2. The project includes a basic layout with a sidebar and bottom navigation bar.
3. Placeholder pages for "Dashboard," "Backtest," and "Live Trade" are created and linked in the navigation.
4. The frontend development server can be started with `npm run dev`.

### Story 1.3: Refactor Fyers Authentication Logic
As a Developer, I want to refactor the hardcoded `fyers_auth.py` script into a reusable function, so that it can be called with dynamic credentials from an API.
**Acceptance Criteria:**
1. A function (e.g., `authenticate_fyers_user`) is created that encapsulates the Fyers login logic.
2. The function accepts user credentials (`fy_id`, `pin`, `totp_secret`) as parameters.
3. All hardcoded credentials are removed from the file.
4. The function returns an `access_token` on successful authentication or an error on failure.

### Story 1.4: Create Secure Login API Endpoint
As a Backend Developer, I want to create a `/api/login` endpoint, so that the frontend can securely send user credentials for authentication.
**Acceptance Criteria:**
1. A POST endpoint at `/api/login` is created in the Python backend.
2. It accepts `fy_id`, `pin`, and `totp_secret` in the request body.
3. It calls the `authenticate_fyers_user` function with the provided credentials.
4. On success, it returns a secure session token (e.g., a JWT) to the client.
5. On failure, it returns an appropriate HTTP error code (e.g., 401 Unauthorized).

### Story 1.5: Build Frontend Login UI
As a Frontend Developer, I want to build a login page with a form, so that users can enter their Fyers credentials.
**Acceptance Criteria:**
1. A new page is created at `/login`.
2. The page contains a form with input fields for Fyers ID, PIN, and TOTP secret.
3. A "Login" button is present.
4. Basic form validation is implemented.

### Story 1.6: Connect Login UI to Backend API
As a Frontend Developer, I want to connect the login form to the backend's `/api/login` endpoint, so that users can authenticate and their session is managed.
**Acceptance Criteria:**
1. Clicking the "Login" button sends the form data to the `POST /api/login` endpoint.
2. On a successful response, the received session token is securely stored.
3. After successful login, the user is redirected to the main dashboard (`/`).
4. On a failure response, an error message is displayed to the user.

### Story 1.7: Create Authenticated User Profile Endpoint
As a Backend Developer, I want to create a secure `/api/profile` endpoint, so that the frontend can fetch data for an authenticated user.
**Acceptance Criteria:**
1. A GET endpoint at `/api/profile` is created.
2. The endpoint requires a valid session token for access.
3. It uses the Fyers API to fetch the user's profile and capital information.
4. It returns the profile and capital data as a JSON object.
5. If the token is invalid or missing, it returns a 401 Unauthorized error.

### Story 1.8: Display User Data on Dashboard
As a Frontend Developer, I want to fetch and display the user's profile data on the dashboard, so that the user can confirm they are logged in and see their available capital.
**Acceptance Criteria:**
1. The dashboard page calls the `GET /api/profile` endpoint.
2. The user's name and available capital are displayed on the page.
3. If the API call fails, the user is redirected back to the login page.

---

## Epic 2: Backtesting MVP

**Goal:** To implement the complete end-to-end backtesting feature. This will allow users to configure a backtest, run it against the core trading model, and visualize the performance results, providing a safe way to evaluate strategies.

### Story 2.1: Backend API for Backtesting
As a Backend Developer, I want to create a `/api/backtest` endpoint, so that the frontend can initiate a backtesting run.
**Acceptance Criteria:**
1. A POST endpoint at `/api/backtest` is created.
2. It accepts `instrument`, `timeframe`, and `duration` as parameters.
3. It validates the input parameters.
4. It triggers the backtesting process in the Python environment.
5. It returns a unique `backtest_id` to track the session.

### Story 2.2: Backend WebSocket for Backtest Progress
As a Backend Developer, I want to set up a WebSocket connection, so that I can stream real-time progress and results of a backtest to the frontend.
**Acceptance Criteria:**
1. A WebSocket endpoint is created (e.g., `/ws/backtest/{backtest_id}`).
2. Once a backtest starts, the backend sends progress updates (e.g., percentage complete) over the WebSocket.
3. Upon completion, the backend sends the final results (chart data, metrics) over the WebSocket.

### Story 2.3: Frontend UI for Backtest Configuration
As a Frontend Developer, I want to build the UI for the backtesting page, so that users can set up and start a new backtest.
**Acceptance Criteria:**
1. The backtesting page contains a form for selecting an instrument, timeframe, and duration.
2. The instrument dropdown is populated dynamically from `config/instruments.yaml`.
3. A "Run Backtest" button is present.
4. Clicking the button calls the `POST /api/backtest` endpoint and establishes a WebSocket connection.

### Story 2.4: Frontend UI for Backtest Visualization
As a Frontend Developer, I want to display the results of the backtest, so that users can analyze the performance of the model.
**Acceptance Criteria:**
1. An interactive chart is displayed on the page.
2. The chart visualizes the portfolio's value over the backtest period, updated in real-time via the WebSocket if possible.
3. Key metrics (Win Rate, Total Trades, P&L) are displayed clearly below the chart once the backtest is complete.
4. A loading indicator or progress bar is shown while the backtest is running.

---

## Epic 3: Live Trading & Real-Time Monitoring

**Goal:** To implement the full live trading functionality. This includes starting and stopping the bot, handling the complexity of options trading, and providing a rich, real-time monitoring experience to build user trust and control.

### Story 3.1: Backend Logic for Live Trading Control
As a Backend Developer, I want to create API endpoints to start and stop the live trading bot, so that the user has full control over the session.
**Acceptance Criteria:**
1. A `POST /api/live/start` endpoint is created that initiates the live trading loop.
2. A `POST /api/live/stop` endpoint is created that gracefully stops the trading loop and closes any open positions.
3. Both endpoints require a valid user session.

### Story 3.2: Implement Options Strategy Abstraction
As a Backend Developer, I want to implement the logic to find and trade option contracts (ITM, ATM, OTM) based on the index signals from the core environment, so that the bot can trade derivatives.
**Acceptance Criteria:**
1. A function is created that, given an index and a strategy (ITM/ATM/OTM), finds the appropriate option contract using the Fyers API.
2. The live trading loop uses this function to place trades when an index instrument is selected.
3. Error handling is implemented for cases where a suitable contract cannot be found.

### Story 3.3: Backend WebSocket for Live Data
As a Backend Developer, I want to stream all relevant live trading data via a WebSocket, so that the frontend can provide a real-time view of the bot's activity.
**Acceptance Criteria:**
1. A WebSocket endpoint is created (e.g., `/ws/live`).
2. The backend streams real-time market data for the selected instrument.
3. The backend streams the model's predictions as they are generated.
4. The backend streams events for all trades (entry, exit, profit/loss).
5. The backend streams changes in connection status or other system-level events.

### Story 3.4: Frontend UI for Live Trading
As a Frontend Developer, I want to build the complete UI for the live trading screen, so that users can configure, run, and monitor the bot.
**Acceptance Criteria:**
1. The UI includes controls to select an instrument, timeframe, and (if applicable) options strategy.
2. "Start" and "Stop" buttons are present and connected to the backend APIs.
3. A real-time interactive chart displays market data, trade markers, and predictions from the WebSocket feed.
4. A log panel displays a time-stamped feed of all events from the WebSocket.
5. The instrument selection is disabled while a session is active.
