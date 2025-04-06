# Active Context: AlgoTrading System (2025-04-06 ~4:55 PM)

**Last Major Task:** Implemented and tested core real-time options trading logic.

**Key Outcomes:**
*   Refactored Fyers API V3 WebSocket and REST integration according to documentation.
*   Implemented options trading flow:
    *   Signal generation on underlying index.
    *   Option chain fetching and ITM strike selection (`options_utils.py`).
    *   Order placement for CE/PE based on signal and `TRADING_MODE` ("OptionBuy" default).
    *   Index-based SL/TP calculation and storage in metadata.
    *   Subscription/unsubscription of option market data based on trade fills.
    *   Exit logic based on underlying index price hitting SL/TP levels.
    *   Real-time logging of open option position status.
*   Centralized strategy parameters in `config.py`.
*   Updated relevant modules (`RealtimeStrategyExecutor`, `fyers_auth.py`, `signals.py`, `run_realtime_bot.py`, etc.).
*   Successfully tested the core pipeline using a forced entry signal (order rejected due to market closed, but failure handled correctly).

**Current State:**
*   Core real-time options trading infrastructure is in place.
*   Requires implementation of `TODOs` in callback handlers (`_handle_order_update`, `_handle_trade_update`, `_handle_position_update`) for robust state management.
*   Requires verification of `PositionManager` for handling option symbols and quantities.
*   Requires further testing during market hours.
