import pandas as pd
import numpy as np

# Removed get_inside_candle_signals function - moved to src/signals.py


def run_custom_backtest(df: pd.DataFrame,
                        signals: pd.Series, # Added signals as input
                        initial_cash: float = 100_000.0,
                        lot_size: int = 50,
                        commission_per_trade: float = 0.0, # Example: Fixed commission per trade
                        stop_loss_atr_multiplier: float = 1.5,
                        risk_reward_ratio: float = 2.0,
                        atr_period: int = 14):
    """
    Runs a custom backtest loop based on provided signals.

    Args:
        df: DataFrame containing OHLC data and pre-calculated ATR.
            Requires columns: 'Open', 'High', 'Low', 'Close', f'atr_{atr_period}'.
        signals: Pandas Series containing the trading signals (1: Buy, -1: Sell, 0: Hold).
                 Index must align with df.
        initial_cash: Starting capital.
        lot_size: Number of units per trade.
        commission_per_trade: Fixed commission cost for entry and exit leg (applied once per trade).
        stop_loss_atr_multiplier: ATR multiplier for SL.
        risk_reward_ratio: RR ratio for TP.
        atr_period: Period used for ATR calculation (for column name).

    Returns:
        A tuple containing:
        - pandas.DataFrame: Record of executed trades.
        - dict: Performance metrics.
    """
    if f'atr_{atr_period}' not in df.columns:
        raise ValueError(f"Required column 'atr_{atr_period}' not found in DataFrame.")

    # --- Initialization ---
    cash = initial_cash
    position = 0  # 0: Flat, 1: Long, -1: Short
    entry_price = np.nan
    stop_loss_price = np.nan
    take_profit_price = np.nan
    trades = []
    equity_curve = [initial_cash] # Track equity over time
    max_points_captured_overall = 0.0 # Initialize overall max points captured
    max_points_lost_overall = 0.0 # Initialize overall max points lost (will be negative)
    current_trade_peak_pnl_points = 0.0 # Track peak profit points during current trade
    current_trade_valley_pnl_points = 0.0 # Track peak loss points during current trade

    # --- Validate Inputs ---
    if not isinstance(signals, pd.Series):
        raise TypeError("signals must be a pandas Series.")
    if not df.index.equals(signals.index):
        raise ValueError("DataFrame index and signals Series index do not match.")
    if not signals.isin([0, 1, -1]).all():
        raise ValueError("signals Series must contain only 0, 1, or -1.")


    # --- Backtesting Loop ---
    for i in range(1, len(df)): # Start from 1 to access previous bar data
        current_open = df['Open'].iloc[i]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]
        current_close = df['Close'].iloc[i]
        current_time = df.index[i]
        signal = signals.iloc[i]
        # ATR from the *previous* bar (signal bar) for SL/TP calculation
        current_atr = df[f'atr_{atr_period}'].iloc[i-1]

        # --- Track intra-trade PnL points ---
        if position == 1:
             # Max potential profit points if exited at current high
            current_trade_peak_pnl_points = max(current_trade_peak_pnl_points, current_high - entry_price)
             # Max potential loss points if exited at current low (will be negative)
            current_trade_valley_pnl_points = min(current_trade_valley_pnl_points, current_low - entry_price)
        elif position == -1:
             # Max potential profit points if exited at current low (price diff is negative, multiply by -1)
            current_trade_peak_pnl_points = max(current_trade_peak_pnl_points, (entry_price - current_low))
             # Max potential loss points if exited at current high (price diff is positive, multiply by -1)
            current_trade_valley_pnl_points = min(current_trade_valley_pnl_points, (entry_price - current_high))


        # --- Check for SL/TP Hit ---
        exit_price = np.nan
        exit_reason = None
        if position == 1: # Long position
            if current_low <= stop_loss_price:
                exit_price = stop_loss_price # Assume SL hit exactly
                exit_reason = "SL"
            elif current_high >= take_profit_price:
                exit_price = take_profit_price # Assume TP hit exactly
                exit_reason = "TP"
        elif position == -1: # Short position
            if current_high >= stop_loss_price:
                exit_price = stop_loss_price # Assume SL hit exactly
                exit_reason = "SL"
            elif current_low <= take_profit_price:
                exit_price = take_profit_price # Assume TP hit exactly
                exit_reason = "TP"

        # --- Process Exit ---
        if not np.isnan(exit_price):
            trade_pnl_points = (exit_price - entry_price) * position # PnL in points for this trade
            pnl = trade_pnl_points * lot_size - (2 * commission_per_trade) # Commission on entry & exit
            cash += pnl

            # Update overall max points captured/lost
            max_points_captured_overall = max(max_points_captured_overall, trade_pnl_points)
            max_points_lost_overall = min(max_points_lost_overall, trade_pnl_points)

            trades.append({
                "EntryTime": entry_time,
                "ExitTime": current_time,
                "Direction": "Long" if position == 1 else "Short",
                "EntryPrice": entry_price,
                "ExitPrice": exit_price,
                "StopLoss": stop_loss_price_initial, # Record initial SL
                "TakeProfit": take_profit_price_initial, # Record initial TP
                "ExitReason": exit_reason,
                "PnL": pnl,
                "Commission": 2 * commission_per_trade,
                "Equity": cash,
                "Points": trade_pnl_points # Add points captured/lost for this trade
            })
            # print(f"{current_time}: EXIT {trades[-1]['Direction']} at {exit_price:.2f} ({exit_reason}). PnL: {pnl:.2f}. Cash: {cash:.2f}")
            position = 0
            entry_price = np.nan
            stop_loss_price = np.nan
            take_profit_price = np.nan
            current_trade_peak_pnl_points = 0.0 # Reset for next trade
            current_trade_valley_pnl_points = 0.0 # Reset for next trade

        # --- Process Entry ---
        if position == 0 and signal != 0:
            if np.isnan(current_atr) or current_atr <= 0:
                # print(f"{current_time}: Signal {signal} ignored due to invalid ATR: {current_atr}")
                continue # Skip entry if ATR is invalid

            sl_distance = current_atr * stop_loss_atr_multiplier
            tp_distance = sl_distance * risk_reward_ratio
            entry_time = current_time # Entry happens on this bar's open

            if signal == 1: # Buy Signal
                entry_price = current_open # Enter at open
                stop_loss_price = entry_price - sl_distance
                take_profit_price = entry_price + tp_distance
                # Basic validation
                if not (stop_loss_price < entry_price < take_profit_price):
                    print(f"{current_time}: Long entry SKIPPED. Invalid SL/TP: SL={stop_loss_price:.2f}, Entry={entry_price:.2f}, TP={take_profit_price:.2f}")
                    continue
                position = 1
                cash -= commission_per_trade # Commission on entry
                stop_loss_price_initial = stop_loss_price # Store for record
                take_profit_price_initial = take_profit_price # Store for record
                current_trade_peak_pnl_points = 0.0 # Reset for new trade
                current_trade_valley_pnl_points = 0.0 # Reset for new trade
                # print(f"{current_time}: ENTER LONG at {entry_price:.2f}. SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}. Cash: {cash:.2f}")

            elif signal == -1: # Sell Signal
                entry_price = current_open # Enter at open
                stop_loss_price = entry_price + sl_distance
                take_profit_price = entry_price - tp_distance
                 # Basic validation
                if not (take_profit_price < entry_price < stop_loss_price):
                    print(f"{current_time}: Short entry SKIPPED. Invalid SL/TP: TP={take_profit_price:.2f}, Entry={entry_price:.2f}, SL={stop_loss_price:.2f}")
                    continue
                position = -1
                cash -= commission_per_trade # Commission on entry
                stop_loss_price_initial = stop_loss_price # Store for record
                take_profit_price_initial = take_profit_price # Store for record
                current_trade_peak_pnl_points = 0.0 # Reset for new trade
                current_trade_valley_pnl_points = 0.0 # Reset for new trade
                # print(f"{current_time}: ENTER SHORT at {entry_price:.2f}. SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}. Cash: {cash:.2f}")

        equity_curve.append(cash) # Record equity at end of bar

    # --- Final Position Close (if any) ---
    if position != 0:
        exit_price = df['Close'].iloc[-1] # Close at last bar's close
        trade_pnl_points = (exit_price - entry_price) * position # PnL in points for this trade
        pnl = trade_pnl_points * lot_size - (2 * commission_per_trade)
        cash += pnl

        # Update overall max points captured/lost
        max_points_captured_overall = max(max_points_captured_overall, trade_pnl_points)
        max_points_lost_overall = min(max_points_lost_overall, trade_pnl_points)

        trades.append({
            "EntryTime": entry_time,
            "ExitTime": df.index[-1],
            "Direction": "Long" if position == 1 else "Short",
            "EntryPrice": entry_price,
            "ExitPrice": exit_price,
            "StopLoss": stop_loss_price_initial,
            "TakeProfit": take_profit_price_initial,
            "ExitReason": "EndOfData",
            "PnL": pnl,
            "Commission": 2 * commission_per_trade,
            "Equity": cash,
            "Points": trade_pnl_points # Add points captured/lost for this trade
        })
        # print(f"{df.index[-1]}: FORCE EXIT {trades[-1]['Direction']} at {exit_price:.2f}. PnL: {pnl:.2f}. Cash: {cash:.2f}")
        equity_curve[-1] = cash # Update last equity point

    # --- Calculate Metrics ---
    trades_df = pd.DataFrame(trades)
    # Pass new metrics to the calculation function
    metrics = calculate_performance_metrics(
        trades_df, initial_cash, equity_curve, df.index[0], df.index[-1], lot_size,
        max_points_captured_overall, max_points_lost_overall # Pass new metrics
    )

    return trades_df, metrics

def calculate_performance_metrics(trades_df: pd.DataFrame, initial_cash: float, equity_curve: list,
                                start_date, end_date, lot_size: int,
                                max_points_captured: float, max_points_lost: float) -> dict: # Added new metrics
    """Calculates performance metrics from trades."""
    metrics = {}
    num_trades = len(trades_df)
    metrics["Lot Size"] = lot_size # Added Lot Size metric
    metrics["Start Date"] = start_date
    metrics["End Date"] = end_date
    metrics["Duration"] = end_date - start_date
    metrics["Initial Cash"] = initial_cash
    metrics["Final Equity"] = equity_curve[-1]
    metrics["Total Return [%]"] = ((metrics["Final Equity"] / initial_cash) - 1) * 100
    metrics["Number of Trades"] = num_trades

    if num_trades > 0:
        wins = trades_df[trades_df['PnL'] > 0]
        losses = trades_df[trades_df['PnL'] <= 0] # Include break-even as loss for win rate

        metrics["Number of Wins"] = len(wins)
        metrics["Number of Losses"] = len(losses)
        metrics["Win Rate [%]"] = (len(wins) / num_trades) * 100 if num_trades > 0 else 0
        metrics["Average Win PnL"] = wins['PnL'].mean() if len(wins) > 0 else 0
        metrics["Average Loss PnL"] = losses['PnL'].mean() if len(losses) > 0 else 0
        metrics["Average Win Points"] = wins['Points'].mean() if len(wins) > 0 else 0 # Added Points metric
        metrics["Average Loss Points"] = losses['Points'].mean() if len(losses) > 0 else 0 # Added Points metric
        metrics["Max Points Captured"] = max_points_captured # Added new metric
        metrics["Max Points Lost"] = max_points_lost # Added new metric
        metrics["Profit Factor"] = abs(wins['PnL'].sum() / losses['PnL'].sum()) if losses['PnL'].sum() != 0 else np.inf
        metrics["Expectancy PnL"] = trades_df['PnL'].mean()
        metrics["Expectancy Points"] = trades_df['Points'].mean() # Added Points metric
        metrics["Total PnL"] = trades_df['PnL'].sum()
        metrics["Total Points"] = trades_df['Points'].sum() # Added Points metric
        metrics["Total Commission"] = trades_df['Commission'].sum()

        # Drawdown calculation (simplified)
        equity_s = pd.Series(equity_curve)
        peak = equity_s.expanding(min_periods=1).max()
        drawdown = (equity_s - peak) / peak
        metrics["Max Drawdown [%]"] = abs(drawdown.min()) * 100
        # metrics["Avg. Drawdown [%]"] = abs(drawdown[drawdown < 0].mean()) * 100 # More complex

    else:
        metrics["Number of Wins"] = 0
        metrics["Number of Losses"] = 0
        metrics["Win Rate [%]"] = 0
        metrics["Average Win PnL"] = 0
        metrics["Average Loss PnL"] = 0
        metrics["Average Win Points"] = 0
        metrics["Average Loss Points"] = 0
        metrics["Max Points Captured"] = 0
        metrics["Max Points Lost"] = 0
        metrics["Profit Factor"] = np.nan
        metrics["Expectancy PnL"] = 0
        metrics["Expectancy Points"] = 0
        metrics["Total PnL"] = 0
        metrics["Total Points"] = 0
        metrics["Total Commission"] = 0
        metrics["Max Drawdown [%]"] = 0

    return metrics

# Example Usage (can be run from another script)
if __name__ == '__main__':
    # This is placeholder example data. Replace with actual data loading.
    print("Running example usage (requires data loading)...")
    # Example: Load data (replace with your actual data loading)
    # try:
    #     data_path = Path("../data/historical_processed/Nifty_5.parquet") # Adjust path if needed
    #     df_data = pd.read_parquet(data_path)
    #     # Ensure correct columns exist (Open, High, Low, Close, atr_14)
    #     required_cols = ['Open', 'High', 'Low', 'Close', 'atr_14']
    #     # Rename if necessary (assuming lowercase from processing)
    #     rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}
    #     df_data = df_data.rename(columns={k: v for k, v in rename_map.items() if k in df_data.columns})
    #
    #     if all(col in df_data.columns for col in required_cols):
    #         trades_log, performance = run_custom_backtest(
    #             df_data,
    #             initial_cash=100_000,
    #             lot_size=75,
    #             commission_per_trade=5.0, # Example fixed commission
    #             stop_loss_atr_multiplier=1.5,
    #             risk_reward_ratio=2.0,
    #             atr_period=14
    #         )
    #         print("\n--- Custom Backtest Results ---")
    #         for key, value in performance.items():
    #             print(f"{key}: {value}")
    #         print("\n--- Trades Log ---")
    #         print(trades_log.to_string())
    #     else:
    #         missing = set(required_cols) - set(df_data.columns)
    #         print(f"Error: Missing required columns in data: {missing}")
    #
    # except FileNotFoundError:
    #      print(f"Error: Data file not found at {data_path}")
    # except Exception as e:
    #      print(f"An error occurred during example run: {e}")
    #      import traceback
    #      traceback.print_exc()
