import numpy as np
from numba import njit

@njit
def label_signals_jit(close_arr, high_arr, low_arr, target_arr, stoploss_arr):
    """
    Labels signals based on future price movements hitting target or stoploss.

    Args:
        close_arr (np.ndarray): Array of closing prices.
        high_arr (np.ndarray): Array of high prices.
        low_arr (np.ndarray): Array of low prices.
        target_arr (np.ndarray): Array of target points/values for each candle.
        stoploss_arr (np.ndarray): Array of stoploss points/values for each candle.

    Returns:
        tuple: Contains:
            - signals (np.ndarray): Signal labels (1: Buy Target, 2: Buy SL, 3: Sell Target, 4: Sell SL, 0: No trigger).
            - entry_prices (np.ndarray): Entry price for each signal.
            - exit_prices (np.ndarray): Exit price where the signal triggered.
            - candles_to_profit (np.ndarray): Number of candles until profit target hit.
            - candles_to_loss (np.ndarray): Number of candles until stoploss hit.
    """
    n = len(close_arr)
    signals = np.zeros(n, dtype=np.float32)
    entry_prices = np.zeros(n, dtype=np.float32)
    exit_prices = np.zeros(n, dtype=np.float32)
    candles_to_profit = np.zeros(n, dtype=np.float32)
    candles_to_loss = np.zeros(n, dtype=np.float32)

    for i in range(n - 1): # Iterate up to the second to last candle
        entry_price = close_arr[i]
        target = target_arr[i]
        stop_loss = stoploss_arr[i]

        # Define potential exit levels
        buy_target_price = entry_price + target
        buy_sl_price = entry_price - stop_loss
        sell_target_price = entry_price - target
        sell_sl_price = entry_price + stop_loss

        signal_found = False
        # Look ahead from the next candle onwards
        for offset in range(i + 1, n):
            future_high = high_arr[offset]
            future_low = low_arr[offset]

            triggers = []
            # Check Buy triggers
            if future_high >= buy_target_price:
                triggers.append((1, offset - i))  # buy target hit
            if future_low <= buy_sl_price:
                triggers.append((2, offset - i))  # buy stoploss hit
            # Check Sell triggers
            if future_low <= sell_target_price:
                triggers.append((3, offset - i))  # sell target hit
            if future_high >= sell_sl_price:
                triggers.append((4, offset - i))  # sell stoploss hit

            # If any triggers fired in this future candle
            if triggers:
                # Prioritize based on trigger type (1=Buy Tgt, 2=Buy SL, 3=Sell Tgt, 4=Sell SL)
                triggers.sort(key=lambda x: x[0])
                first_trigger, candle_count = triggers[0]

                signals[i] = float(first_trigger)
                entry_prices[i] = entry_price

                # Record exit price and candle count based on the first trigger
                if first_trigger == 1:  # buy target
                    exit_prices[i] = buy_target_price
                    candles_to_profit[i] = float(candle_count)
                elif first_trigger == 2:  # buy stoploss
                    exit_prices[i] = buy_sl_price
                    candles_to_loss[i] = float(candle_count)
                elif first_trigger == 3:  # sell target
                    exit_prices[i] = sell_target_price
                    candles_to_profit[i] = float(candle_count)
                elif first_trigger == 4:  # sell stoploss
                    exit_prices[i] = sell_sl_price
                    candles_to_loss[i] = float(candle_count)

                signal_found = True
                break # Stop looking ahead once the first signal is found for candle 'i'

        # If no signal was found after checking all future candles
        if not signal_found:
            signals[i] = 0.0 # No trigger reached
            entry_prices[i] = entry_price # Still record entry price
            # Exit price, ctp, ctl remain 0

    return signals, entry_prices, exit_prices, candles_to_profit, candles_to_loss
