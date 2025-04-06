import pandas as pd
import numpy as np # Import numpy
import pandas_ta as ta
from backtesting import Strategy
from backtesting.lib import crossover # Re-import crossover
# Assuming src.data_handler might be needed for type hints or constants later
# try:
#     from .data_handler import FullFeaturePipeline # Example if needed
# except ImportError:
import src.config as config # Import config to potentially access constants if needed elsewhere

# --- Removed MACDCrossoverStrategy ---

class InsideCandleStrategy(Strategy):
    """
    Trades breakouts following an inside candle pattern using ATR-based SL/TP.
    Buys on upward breakout of the mother candle's high after an inside candle.
    Sells on downward breakout of the mother candle's low after an inside candle.
    Uses ATR for stop-loss and take-profit calculation.
    """
    # --- Strategy Parameters ---
    stop_loss_atr_multiplier = 1.5 # ATR multiplier for stop loss
    risk_reward_ratio = 2.0      # Risk:Reward ratio for take profit
    lot_size = 50                # Default lot size (will be overridden by run_backtest)
    atr_period = 14              # Default ATR period (used for column name)
    # n_bars_exit = 5 # Removed fixed exit, relying on SL/TP
    max_debug_prints = 10        # Limit debug prints

    # --- Strategy Initialization ---
    def init(self):
        """
        Initialize the strategy, access pre-calculated ATR, and set up debug counters.
        """
        self._long_print_count = 0
        self._short_print_count = 0
        # Access pre-calculated ATR (assuming it exists in the data passed)
        self.atr_col = f'atr_{self.atr_period}'
        if not hasattr(self.data, self.atr_col):
            # This should ideally not happen if run_backtest.py pre-calculates correctly
            raise ValueError(f"Required ATR column ({self.atr_col}) not found in input data.")
        self.atr = getattr(self.data, self.atr_col)

    # --- Trading Logic ---
    def next(self):
        """
        Define the trading logic for each timestep (candle).
        """
        # Need at least 3 bars for inside bar check
        if len(self.data.Close) < 3:
            return

        # --- Inside Bar Detection ---
        previous_high = self.data.High[-2]
        previous_low = self.data.Low[-2]
        pre_previous_high = self.data.High[-3]
        pre_previous_low = self.data.Low[-3]

        is_inside_bar = (previous_high < pre_previous_high and
                         previous_low > pre_previous_low)

        # --- Debug Print for Inside Bar ---
        # if is_inside_bar:
        #     print(f"{self.data.index[-1]}: Inside bar detected (Prev H: {previous_high}, Prev L: {previous_low} | PrePrev H: {pre_previous_high}, PrePrev L: {pre_previous_low})")

        # --- Exit Logic (Handled by SL/TP) ---
        # No explicit exit logic needed if relying solely on SL/TP set during entry

        # --- Entry Logic ---
        # Only enter if not already in a position and the *previous* bar was an inside bar
        if not self.position and is_inside_bar:
            current_high = self.data.High[-1]
            current_low = self.data.Low[-1]
            current_atr = self.atr[-1] # Get the ATR of the *previous* bar (the inside bar)

            # Check for valid ATR
            if current_atr is None or np.isnan(current_atr) or current_atr <= 0:
                return # Cannot set SL/TP without valid ATR

            sl_distance = current_atr * self.stop_loss_atr_multiplier
            tp_distance = sl_distance * self.risk_reward_ratio

            # --- Long Entry ---
            # If current bar breaks above the *mother* candle's high (pre_previous_high)
            if current_high > pre_previous_high:
                 # SL/TP relative to the actual entry price (next bar's open)
                 entry_price = self.data.Open[-1] # Next bar's open is the default entry
                 sl_price = entry_price - sl_distance
                 tp_price = entry_price + tp_distance
                 # Check if SL/TP are valid relative to the anticipated entry price
                 if sl_price < entry_price < tp_price:
                     self.buy(sl=sl_price, tp=tp_price, size=self.lot_size)
                 # else: # Optional: Log skipped trades if needed, but removing for now
                 #     pass


             # --- Short Entry ---
            # If current bar breaks below the *mother* candle's low (pre_previous_low)
            elif current_low < pre_previous_low:
                 # SL/TP relative to the actual entry price (next bar's open)
                 entry_price = self.data.Open[-1] # Next bar's open is the default entry
                 sl_price = entry_price + sl_distance
                 tp_price = entry_price - tp_distance
                 # Check if SL/TP are valid relative to the anticipated entry price
                 if tp_price < entry_price < sl_price:
                     self.sell(sl=sl_price, tp=tp_price, size=self.lot_size)
                 # else: # Optional: Log skipped trades if needed, but removing for now
                 #     pass
             # elif current_low < previous_low:
             #     sl_price = previous_low + sl_distance
            #     tp_price = previous_low - tp_distance
            #     # Place a sell stop order at the breakdown level
            # If current bar breaks below the inside bar's low
            # elif current_low < previous_low:
            #     sl_price = previous_low + sl_distance
            #     tp_price = previous_low - tp_distance
            #     # Place a sell stop order at the breakdown level
            #     self.sell(stop=previous_low, sl=sl_price, tp=tp_price, size=self.lot_size)