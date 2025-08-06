---

### **Story 2.2.2: Calculate ATR-based Stop-Loss and Target**

**Status:** `Ready for dev`

**Story:**
As a developer, I need to integrate the ATR calculation into the `TradingEnv` to automatically determine stop-loss (SL) and target price (TP) levels for new positions based on a configurable risk-reward ratio.

**Acceptance Criteria:**
1.  The `TradingEnv`'s method for initiating a new position (e.g., `enter_trade`) accepts the current ATR value and the entry price.
2.  A configurable risk-reward ratio (e.g., `1:2` for 1 unit of risk to 2 units of reward) is defined in a configuration file (e.g., `src/config/config.py`).
3.  The `TradingEnv` calculates the absolute SL and TP price levels using the entry price, ATR, and the risk-reward ratio.
    *   For a long position: `SL = EntryPrice - (ATR * SL_Factor)`, `TP = EntryPrice + (ATR * TP_Factor)`.
    *   For a short position: `SL = EntryPrice + (ATR * SL_Factor)`, `TP = EntryPrice - (ATR * TP_Factor)`.
    *   The `SL_Factor` and `TP_Factor` are derived from the risk-reward ratio.
4.  The calculated SL and TP values are stored as attributes of the active `Position` object within the `TradingEnv`.
5.  The system logs the calculated SL and TP values when a new position is opened.

**Tasks / Subtasks:**
-   `[ ]` **Backend:** In `src/config/config.py`, add a new configuration variable for `RISK_REWARD_RATIO` (e.g., `RISK_REWARD_RATIO = {"risk_units": 1, "reward_units": 2}`).
-   `[ ]` **Backend:** In `src/backtesting/environment.py` (or wherever `TradingEnv` is defined), modify the `enter_trade` or equivalent method to accept `current_atr` and `entry_price` as parameters.
-   `[ ]` **Backend:** Implement the logic within `TradingEnv` to calculate `SL_Factor` and `TP_Factor` based on `RISK_REWARD_RATIO`.
-   `[ ]` **Backend:** Implement the SL and TP calculation formulas based on the `direction` of the trade.
-   `[ ]` **Backend:** Update the `Position` object (or internal state representing the position) to store the calculated `stopLoss` and `targetPrice`.
-   `[ ]` **Backend:** Add logging to output the calculated SL and TP values.
-   `[ ]` **Testing:** In `tests/test_backtesting/`, create `test_trading_env_sl_tp.py`.
-   `[ ]` **Testing:** Write a unit test that initializes `TradingEnv` with a mock `current_atr` and `entry_price`.
-   `[ ]` **Testing:** Assert that for both long and short positions, the calculated `stopLoss` and `targetPrice` are correct based on the `RISK_REWARD_RATIO`.
-   `[ ]` **Testing:** Test with different `current_atr` values and ensure the SL/TP scales correctly.

---