**Session Date:** 2025-07-31
**Facilitator:** Business Analyst Mary
**Participant:** User

# Executive Summary
**Topic:** Planning and Brainstorming for a Next.js frontend for a Python-based Algorithmic Trading Bot.

**Session Goals:** To define the core features for a personal-use dashboard with future scalability in mind, focusing on backtesting and live trading functionalities.

**Techniques Used:** Progressive Flow (Crazy Eights, Feature Brainstorming, "What If?" Scenario Analysis), Idea Categorization.

**Key Themes Identified:**
- **Clear Separation of Concerns:** The frontend is purely for interaction (backtesting, live trading), while the core model training remains a backend process.
- **User-Centric Workflow:** The application flow is designed around a simple, intuitive user journey: Login -> Dashboard -> Select Action (Backtest/Live).
- **Real-time Visualization:** A strong emphasis on real-time charts and metrics for both backtesting and live trading to provide immediate feedback and build user confidence.
- **Abstraction for Complexity:** The need to create a seamless user experience that hides the complexity of handling different instrument types (stocks vs. index options).
- **Robust Error Handling:** Proactive planning for potential failure points like API disconnections and trade execution errors.

# Idea Categorization

### Immediate Opportunities
*Ideas ready to implement now*

1.  **Core Application Shell**
    -   **Description:** A Next.js application with a basic layout including a sidebar and a bottom navigation bar.
    -   **Why immediate:** This is the foundational structure required for all other features.
    -   **Resources needed:** Next.js, a component library (e.g., Material-UI or Shadcn/UI).

2.  **Fyers Authentication**
    -   **Description:** A login screen that uses the Fyers API to create a user session.
    -   **Why immediate:** This is the entry point to the application and is required to access any data or functionality.
    -   **Resources needed:** Fyers API credentials, frontend logic for handling OAuth or token-based auth.

3.  **Basic Dashboard**
    -   **Description:** A "Home" screen that displays the logged-in user's information and their available trading capital, fetched via the Fyers API.
    -   **Why immediate:** Provides the user with a landing page and confirms their session is active.
    -   **Resources needed:** API endpoints to fetch user data.

### Future Innovations
*Ideas requiring development/research*

1.  **Backtesting Module**
    -   **Description:** A dedicated screen for running backtests. Users can select an instrument (from `config/instruments.yaml`), a timeframe, and a duration. The module will run the backtest against the `universal_final_model.pth` and display results.
    -   **Development needed:**
        -   UI components for instrument, timeframe, and date selection.
        -   Backend API to trigger the backtest process.
        -   Websocket or polling mechanism to send progress/results to the frontend.
        -   Charting library (e.g., Chart.js, D3, or a trading-specific library) to visualize results.
    -   **Timeline estimate:** Medium Term.

2.  **Live Trading Module**
    -   **Description:** A screen for initiating and monitoring live trades. Includes instrument/timeframe selection, start/stop controls, and real-time visualization.
    -   **Development needed:**
        -   Backend logic to manage the live trading loop.
        -   The abstraction layer to handle index options (ATM/ITM/OTM) while the core environment trades the index.
        -   Robust error handling and reconnection logic for the Fyers API.
        -   Real-time chart and metrics display.
    -   **Timeline estimate:** Medium-to-Long Term.

3.  **Dynamic Options Strategy Selector**
    -   **Description:** In the Live Trading view, if an index is selected, allow the user to choose between ITM, ATM, and OTM strategies on-the-fly. The bot will use the latest selection for its next trade.
    -   **Development needed:** UI component for the selector and backend logic to apply the selection to the trade execution logic.
    -   **Timeline estimate:** Medium Term.

### Insights & Learnings
*Key realizations from the session*

- **The Options Abstraction is Key:** The logic to translate the trading environment's simple "index trade" signals into real-world "option trades" is a critical and complex piece of the architecture that lives outside the core model. The UI must support this by allowing strategy selection (ITM/ATM/OTM).
- **UI Must Build Trust:** Since this is an automated trading system, the UI's primary role is to provide confidence. This means clear status indicators, real-time feedback, and transparent error logging are not just nice-to-have, but essential features.
- **Configuration-Driven UI:** The instrument selection should be dynamically populated from the `config/instruments.yaml` file, making the frontend adaptable to backend changes without requiring code modifications.

# Action Planning

### Top 3 Priority Ideas

1.  **#1 Priority: Build the Core App Shell & Authentication**
    -   **Rationale:** This is the non-negotiable first step. Without a logged-in user, no other feature can be built or tested.
    -   **Next steps:**
        1.  Set up a new Next.js project.
        2.  Implement the Fyers login flow.
        3.  Create the basic sidebar/bottom-nav layout.
        4.  Create the placeholder Dashboard, Backtest, and Live Trade pages.
    -   **Timeline:** Short Term.

2.  **#2 Priority: Develop the Backtesting Module**
    -   **Rationale:** Backtesting is a safer, lower-risk way to validate the entire pipeline (frontend -> backend -> model) before committing real capital. It allows for testing the charting and metrics display in a controlled environment.
    -   **Next steps:**
        1.  Design the UI for backtest parameter selection.
        2.  Build the API endpoint to receive backtest requests.
        3.  Implement the backtesting logic on the backend.
        4.  Integrate a charting library and display the results.
    -   **Timeline:** Medium Term.

3.  **#3 Priority: Implement the Live Trading View (Read-Only First)**
    -   **Rationale:** Start by building the monitoring aspects of the live trading screen without enabling the actual trade execution. This separates the risk of UI development from the risk of financial loss.
    -   **Next steps:**
        1.  Build the UI for the Live Trading screen (instrument selection, status indicators, chart).
        2.  Connect the UI to a real-time data feed from the backend.
        3.  Display the model's predictions on the chart in real-time.
        4.  Implement the "Start/Stop" buttons but have them only control the data feed initially.
    -   **Timeline:** Medium Term.
