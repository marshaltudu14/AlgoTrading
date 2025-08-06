# UI/UX Vision & Enhancement Plan

This document outlines the user interface and experience vision for the Live Trading System. It builds upon the existing frontend application, focusing on targeted enhancements to deliver the full functionality described in the PRD.

## 1. Core Principles

- **Clarity and Intuitiveness:** The interface must be easy to understand and use, even with the underlying complexity.
- **Data-Forward Design:** Prioritize the clear and accurate visualization of trading data.
- **Confidence and Control:** The design should empower users, giving them a sense of confidence and control over their trading.
- **Consistency:** Maintain a consistent design language throughout the application.

## 2. UI/UX Enhancement Plan

### 2.1. Enhance the `TradingChart` Component (`components/trading-chart.tsx`)

The `TradingChart` component is the centerpiece of the application. The following enhancements will be made:

- **Implement SL/TP Zones:**
    - **Action:** Modify the `TradingChart` props to accept `stopLoss` and `targetPrice` levels for an active trade.
    - **Implementation:** Use the `lightweight-charts` API to draw two `PriceLine` objects for the SL and TP levels.
        - **Styling:** Use distinct colors (e.g., red for SL, green for TP), dashed lines, and clear labels.
    - **Rationale:** Addresses **FR17** and provides users with immediate visual reference points for their trades.

- **Create TradingView-Style Position Visualization:**
    - **Action:** Implement a more sophisticated visualization for active trades, beyond simple markers.
    - **Implementation:** Draw a semi-transparent, colored rectangle on the chart to represent the active position. The rectangle's color will change based on the trade's profitability (e.g., green for profit, red for loss).
    - **Rationale:** Fulfills **FR18** by providing a clear, TradingView-style visual representation of the active trade's status.

- **Differentiate Manual vs. Automated Trades:**
    - **Action:** Update the `tradeMarkers` prop and the `createTradeMarker` function to include a `tradeType` ('automated' or 'manual').
    - **Implementation:** Use the `tradeType` to apply different color schemes. Manual trades will use a distinct color (e.g., blue) for all their visual elements.
    - **Rationale:** Addresses the need to visually distinguish between trade types, preventing user confusion.

### 2.2. Enhance the Dashboard Page (`app/dashboard/page.tsx`)

The dashboard will be enhanced to include manual trading functionality and more detailed status information.

- **Integrate a `ManualTradeForm` Component:**
    - **Action:** Create a new reusable component, `ManualTradeForm.tsx`.
    - **Implementation:** This component will contain the form as specified in **FR13** (Instrument, Direction, Quantity, SL/TP). It will be placed within the `DashboardPage` and will be disabled when an automated trade is active.
    - **Rationale:** Provides the required manual trading functionality in a logical and accessible location.

- **Add "Next Fetch" Countdown Timer:**
    - **Action:** Add a UI element to the dashboard's header or a status bar.
    - **Implementation:** This element will display a countdown (MM:SS) to the next data fetch. It will also display the current status ("Fetching...", "Error", etc.).
    - **Rationale:** Fulfills **FR15** and provides users with important system status information.

### 2.3. Data Flow and State Management

- **Action:** The data payload sent from the backend over the WebSocket will be updated to include the following fields for each trade: `stopLoss`, `targetPrice`, and `tradeType`.
- **Rationale:** This ensures the frontend has all the necessary data to render the enhanced visualizations and component states correctly.

This document provides a clear and actionable plan for the frontend development team.

