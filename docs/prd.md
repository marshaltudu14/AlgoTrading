# Live Trading System Product Requirements Document (PRD)

## 1. Goals and Background Context

### Goals

*   Develop a real-time trading system that supports both automated and manual trading.
*   Provide users with the flexibility to select instruments and timeframes.
*   Ensure accurate and timely data fetching and processing for reliable trade execution.
*   Implement robust position management with automated stop-loss and target calculations.
*   Offer a clear and intuitive frontend for data visualization and manual trade entry.
*   Maintain comprehensive logging and metrics for performance analysis.
*   Guarantee system stability and security through robust error handling and user authentication.

### Background Context

This project aims to create a sophisticated yet user-friendly trading platform for the Indian market. The system will empower traders by combining the speed and discipline of automated, model-driven trading with the flexibility and control of manual intervention. By providing real-time data, advanced visualizations, and comprehensive performance tracking, the platform will enable users to make more informed and timely trading decisions.

### Change Log

| Date | Version | Description | Author |
| :--- | :--- | :--- | :--- |
| 2025-08-06 | 1.0 | Initial draft | John, PM |

## 2. Requirements

### Functional Requirements

*   **FR1:** The system shall allow users to select a trading instrument from a predefined list (e.g., Nifty, Bank Nifty, stocks) through a frontend dropdown menu.
*   **FR2:** The system shall allow users to select a timeframe (e.g., 1m, 2m, 5m, 15m, 1h) for candlestick data through a frontend dropdown menu.
*   **FR3:** The system shall fetch 30 days of historical candle data for the selected instrument and timeframe upon initialization.
*   **FR4:** The system shall fetch real-time data at intervals aligned with the selected timeframe, rounded to the nearest valid interval.
*   **FR5:** The system shall process real-time data and feed it to a predictive model for trading signals (buy, sell, hold, close long, close short) and quantity.
*   **FR6:** The system shall validate all trading actions against predefined margin limits and other risk parameters.
*   **FR7:** The system shall continuously feed historical data to the environment in the backend for automated positions.
*   **FR8:** The system shall pause the feed of historical data to the model during manual trading positions.
*   **FR9:** The environment shall manage positions by calculating stop-loss and target levels based on a fixed risk-reward ratio and ATR-based points.
*   **FR10:** For options trades, the system shall map the index's stop-loss and target to corresponding option prices.
*   **FR11:** The system shall use external functions to select the nearest options expiry (preferring weekly) and the nearest ITM strike based on the spot price.
*   **FR12:** The system shall use external functions to handle order placement and exit via the Fyers API.
*   **FR13:** The system shall allow users to enter manual trades (instrument, direction, quantity, optional SL/target) through a frontend form when no automated trade is active.
*   **FR14:** The system shall validate manual trade inputs for margin and risk, requiring user confirmation before execution.
*   **FR15:** The frontend shall display a countdown timer for the next data fetch.
*   **FR16:** The frontend shall display a candlestick chart with real-time tick data from the Fyers WebSocket.
*   **FR17:** The candlestick chart shall display stop-loss and target levels as shaded zones.
*   **FR18:** The frontend shall display a TradingView-style long/short graph marking entry/exit points, SL/target zones, and distinguishing between manual and automated trades.
*   **FR19:** The frontend shall include a dashboard with real-time performance metrics.
*   **FR20:** The system shall log all trades in a local JSON file with detailed trade information.
*   **FR21:** The system shall calculate and display metrics such as win rate, total trades, and P&L, aggregated and filterable by instrument and timeframe.

### Non-Functional Requirements

*   **NFR1:** The system shall handle data fetch delays or failures with a retry mechanism.
*   **NFR2:** The system shall implement WebSocket reconnection logic to ensure a stable connection.
*   **NFR3:** The system shall provide alerts for critical errors.
*   **NFR4:** All API keys and sensitive user data shall be encrypted.
*   **NFR5:** The system shall require user authentication for all manual trading actions.
*   **NFR6:** All user actions shall be logged for audit purposes.

## 3. User Interface Design Goals

### Overall UX Vision

The user interface should be clean, intuitive, and data-rich, providing traders with the information they need to make quick and informed decisions. The design should prioritize clarity and ease of use, even with the complexity of the underlying trading system. The overall feel should be professional and modern, similar to popular trading platforms like TradingView.

### Key Interaction Paradigms

*   **Real-time Updates:** The UI should update in real-time to reflect market changes and system status, without requiring manual refreshes.
*   **Interactive Charts:** The candlestick chart should be the central element of the UI, allowing for easy navigation (zooming, panning) and clear visualization of trading data.
*   **Clear Calls to Action:** Buttons and forms for manual trading should be clearly labeled and positioned for easy access.
*   **Visual Feedback:** The system should provide clear visual feedback for all actions, such as trade execution, errors, and connection status.

### Core Screens and Views

*   **Main Dashboard:** A single-screen view that combines the candlestick chart, real-time metrics, and manual trading controls.
*   **Trade History:** A separate view or a section on the dashboard to display the log of all past trades.
*   **Settings:** A screen for configuring user preferences and API keys.

### Accessibility: WCAG AA

The application should adhere to WCAG 2.1 AA standards to ensure it is usable by people with a wide range of disabilities.

### Branding

The branding should be professional and trustworthy, with a color scheme that is easy on the eyes for long periods of use. The use of color should be consistent and meaningful (e.g., green for long positions, red for short positions).

### Target Device and Platforms: Web Responsive

The primary platform will be a responsive web application that works seamlessly on both desktop and mobile browsers.

## 4. Technical Assumptions

### Repository Structure: Monorepo

A monorepo structure is recommended to simplify dependency management and ensure consistency between the frontend and backend code, which will be managed in the same repository.

### Service Architecture: Local Server

The system will run on a local server. The backend will be a self-contained application that serves the frontend and handles all trading logic.

### Testing Requirements: Unit + Integration

A combination of unit and integration tests will be necessary to ensure the reliability of the system. Unit tests will verify the functionality of individual components, while integration tests will ensure that the different parts of the system work together as expected in the local environment.

### Additional Technical Assumptions and Requests

*   **Backend:** Python with a framework like FastAPI or Flask, given the existing Python-based components in the project.
*   **Frontend:** An existing frontend application will be integrated with the backend.
*   **Database:** A simple, file-based solution like SQLite for the trade log, as the requirements specify local storage.
*   **Real-time Communication:** WebSockets will be used for real-time data updates between the local backend and the frontend.
*   **Deployment:** The system is intended for local execution only. No cloud deployment is required.

## 5. Epic List

*   **Epic 1: API and Frontend Integration:** Integrate the existing backend services with the frontend, establish a WebSocket connection for real-time updates, and ensure the frontend can display data from the backend.
*   **Epic 2: Core Trading Logic & Position Management:** Integrate the predictive model to generate trading signals and implement the environment for managing automated positions, including stop-loss and target calculations for both indices/stocks and options.
*   **Epic 3: Frontend Integration & Visualization:** Enhance the frontend to visualize real-time market data, trading signals, and position information on interactive charts, creating a rich user experience.
*   **Epic 4: Manual Trading & Complete Functionality:** Implement full manual trading capabilities, a comprehensive performance dashboard, and robust trade logging to complete the trading system.

## 6. Epic Details

### Epic 1: API and Frontend Integration
**Goal:** Integrate the existing backend services with the frontend, establish a WebSocket connection for real-time updates, and ensure the frontend can display data from the backend.

---

#### Story 1.1: Enhance API for Frontend Needs
*As a developer, I want to enhance the existing FastAPI backend to provide all necessary data for the frontend, including instrument configuration, historical data, and real-time updates.*

**Acceptance Criteria:**
1.  The `/api/config` endpoint is confirmed to provide the list of available instruments and timeframes from `config/instruments.yaml`.
2.  An API endpoint is created to serve historical data, leveraging the existing `FyersClient` and `RealtimeDataLoader`.
3.  The WebSocket endpoint (`/ws/live/{user_id}`) is adapted to stream processed real-time data to the frontend.
4.  The API includes robust error handling and provides clear error messages to the frontend.

---

#### Story 1.2: Frontend and Backend Integration
*As a developer, I want to connect the frontend to the backend API to fetch initial data and establish a WebSocket connection for real-time updates.*

**Acceptance Criteria:**
1.  The frontend makes an API call to `/api/config` to populate the instrument and timeframe selection dropdowns.
2.  Upon selection, the frontend requests historical data from the backend.
3.  The frontend establishes a WebSocket connection to `/ws/live/{user_id}` to receive real-time data.
4.  The frontend correctly handles connection errors and displays appropriate messages to the user.

---

#### Story 1.3: Real-time Chart Visualization
*As a user, I want to see a real-time candlestick chart on the frontend that updates with every new tick of data from the WebSocket.*

**Acceptance Criteria:**
1.  The frontend uses a charting library (e.g., TradingView, Chart.js) to display the candlestick chart.
2.  The chart is populated with the initial historical data.
3.  The chart updates in real-time as new data is received from the WebSocket.
4.  The chart is interactive, allowing users to zoom and pan.

---

### Epic 2: Core Trading Logic & Position Management
**Goal:** Integrate the predictive model to generate trading signals and implement the environment for managing automated positions, including stop-loss and target calculations for both indices/stocks and options.

---

#### Story 2.1: Integrate Predictive Model for Trading Signals
*As a developer, I want to integrate the predictive model with the `LiveTradingService` to generate real-time trading signals and quantities.*

**Acceptance Criteria:**
1.  The `LiveTradingService` feeds the processed real-time data into the loaded predictive model (e.g., `universal_final_model.pth`).
2.  The model outputs an action (buy, sell, hold, close long, close short) and a quantity.
3.  The `LiveTradingService` correctly interprets the model's output.
4.  The quantity prediction is validated against the available capital using the logic in `src/utils/capital_aware_quantity.py`.

---

#### Story 2.2: Implement ATR-based Stop-Loss and Target
*As a developer, I want to enhance the `TradingEnv` to automatically calculate stop-loss (SL) and target price (TP) using the Average True Range (ATR) and a fixed risk-reward ratio.*

**Acceptance Criteria:**
1.  The ATR is calculated correctly based on the historical and incoming real-time data.
2.  The `TradingEnv` uses the ATR to calculate SL and TP points for new positions.
3.  The risk-reward ratio is configurable (e.g., in a settings file).
4.  The calculated SL and TP levels are stored as part of the position's state within the environment.

---

#### Story 2.3: Develop Options Mapping and Selection Logic
*As a developer, I want to create helper functions to select appropriate options contracts and map index/stock price levels to options prices.*

**Acceptance Criteria:**
1.  A function is created that, given an underlying spot price, selects the nearest In-the-Money (ITM) strike price.
2.  A function is created that selects the nearest expiry date, with a preference for weekly expiries.
3.  A function is created to translate the ATR-based SL and TP points from the underlying instrument to the selected option's price.
4.  These functions are integrated into the `LiveTradingService` to be used when an options strategy is selected.

---

#### Story 2.4: Enhance Position and Order Management
*As a developer, I want to update the `LiveTradingService` to manage the full lifecycle of a trade, from entry to exit, based on model signals or SL/TP triggers.*

**Acceptance Criteria:**
1.  The `LiveTradingService` places an order using `FyersClient` when the model signals an entry.
2.  The service monitors the active position against the calculated SL and TP levels.
3.  The service automatically places an exit order if the price hits the SL or TP.
4.  The service also exits a position if the model generates an explicit "close" signal.
5.  When an options position is managed, closing the position is correctly triggered by events in the underlying instrument.

---

### Epic 3: Frontend Integration & Visualization
**Goal:** Enhance the frontend to visualize real-time market data, trading signals, and position information on interactive charts, creating a rich user experience.

---

#### Story 3.1: Display SL/TP Zones on Chart
*As a user, I want to see the calculated stop-loss and target levels for my active trades displayed directly on the candlestick chart, so that I can visually track the position's progress.*

**Acceptance Criteria:**
1.  The backend WebSocket stream is updated to include the SL and TP price levels for any active position.
2.  The frontend consumes the SL and TP data from the WebSocket.
3.  The chart displays the SL and TP levels as distinct horizontal lines or shaded zones.
4.  The visualization is clear and easy to understand, using different colors or styles for SL and TP (e.g., red for stop-loss, green for target).
5.  The SL/TP visualization is removed from the chart once the position is closed.

---

#### Story 3.2: Create Trading-Specific Visualizations
*As a user, I want to see a visual representation of my trades on the chart, similar to TradingView's long/short position tool, so that I can easily analyze my entry and exit points.*

**Acceptance Criteria:**
1.  When a trade is executed, a visual marker is placed on the entry candle.
2.  The chart displays a colored zone (e.g., green for long, red for short) that extends from the entry price to the SL and TP levels.
3.  When the trade is closed, another marker is placed on the exit candle.
4.  The system can display multiple, non-overlapping trades on the chart simultaneously.
5.  Hovering over a trade visualization displays a tooltip with key details (entry/exit price, P&L, etc.).

---

#### Story 3.3: Develop Real-time Metrics Dashboard
*As a user, I want a dashboard on the frontend that displays my real-time trading performance, so that I can monitor my profitability and other key metrics at a glance.*

**Acceptance Criteria:**
1.  The backend provides an API endpoint (e.g., `/api/metrics`) that serves the contents of `metrics.json`.
2.  The frontend fetches data from this endpoint to populate the dashboard.
3.  The dashboard displays key metrics, including:
    *   Total P&L
    *   Win Rate
    *   Total Trades
    *   Average Profit/Loss per trade
4.  The metrics are updated in near real-time as trades are closed.
5.  The dashboard allows filtering metrics by instrument and timeframe.

---

### Epic 4: Manual Trading & Complete Functionality
**Goal:** Implement full manual trading capabilities, a comprehensive performance dashboard, and robust trade logging to complete the trading system.

---

#### Story 4.1: Implement Manual Trade Entry Form
*As a user, I want a form on the frontend to manually enter trades when no automated position is active, so that I have full control over my trading when needed.*

**Acceptance Criteria:**
1.  A "Manual Trade" form is added to the frontend.
2.  The form includes fields for:
    *   Instrument (dropdown)
    *   Direction (Buy/Sell)
    *   Quantity
    *   Stop-Loss (optional)
    *   Target (optional)
3.  The form is only enabled when there are no active automated positions.
4.  Submitting the form sends the trade details to a new backend endpoint (e.g., `/api/manual-trade`).

---

#### Story 4.2: Backend Logic for Manual Trading
*As a developer, I want to implement the backend logic to handle manual trade requests, including validation and execution.*

**Acceptance Criteria:**
1.  A new endpoint (`/api/manual-trade`) is created to receive manual trade requests.
2.  The backend validates the manual trade against margin and risk rules.
3.  A confirmation step is implemented (e.g., the API returns a confirmation token that the frontend must send back) before executing the trade.
4.  The `LiveTradingService` is updated to pause the model's data feed when a manual trade is active.
5.  The manual trade is executed using the existing `FyersClient`.
6.  The model's data feed resumes after the manual position is closed.

---

#### Story 4.3: Comprehensive Trade Logging
*As a developer, I want to ensure all trades (both automated and manual) are logged in a structured JSON file with all required details, so that we have a complete record for analysis.*

**Acceptance Criteria:**
1.  A logging service is created that can be used by both the automated and manual trading flows.
2.  Each trade log entry includes:
    *   Trade ID
    *   Instrument and Timeframe
    *   Trade Type (Automated/Manual)
    *   Direction (Long/Short)
    *   Entry/Exit Times and Prices
    *   SL and Target Prices
    *   Quantity
    *   Final Profit/Loss
    *   Holding Time
    *   Risk-Reward Ratio
    *   Status (Win/Loss/Breakeven)
3.  The logs are written to a local JSON file (`tradelog.json`).

---

#### Story 4.4: Distinguish Manual vs. Automated Trades in UI
*As a user, I want the user interface to clearly distinguish between manual and automated trades, so that I can easily identify the origin of each position.*

**Acceptance Criteria:**
1.  The backend WebSocket stream includes the trade type (Automated/Manual) for each position.
2.  The frontend uses this information to visually differentiate the trades on the chart.
3.  For example, automated trades could be represented with green/red zones, while manual trades use a different color scheme (e.g., blue).
4.  The trade details tooltip also explicitly states the trade type.
