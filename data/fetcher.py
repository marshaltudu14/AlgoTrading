"""
Fetches historical and real-time market data from Fyers and other sources.
Provides functions for both training data and live data fetching.
"""

import time
from datetime import date, timedelta

import pandas as pd

# TODO: Import or pass fyers API client instance
# from auth.fyers_auth import fyers

# TODO: Define or pass active_order_sleep duration (in seconds)
# active_order_sleep = 2


def fetch_candle_data(days_back, index_symbol, interval_minutes, fyers, active_order_sleep):
    """
    Fetch recent historical candle data for a given index symbol.

    Args:
        days_back (int): Number of days back from today to fetch.
        index_symbol (str): Fyers symbol string.
        interval_minutes (int): Candle interval in minutes.
        fyers: Authenticated Fyers API client.
        active_order_sleep (int): Sleep duration on error (seconds).

    Returns:
        pd.DataFrame: DataFrame with columns ['datetime', 'open', 'high', 'low', 'close', 'volume'].
    """
    while True:
        try:
            today = date.today()
            start_date = today - timedelta(days=days_back)

            data = {
                "symbol": index_symbol,
                "resolution": interval_minutes,
                "date_format": "1",
                "range_from": start_date.strftime("%Y-%m-%d"),
                "range_to": today.strftime("%Y-%m-%d"),
                "cont_flag": "1",
            }

            result = fyers.history(data=data)

            if result and "candles" in result:
                df = pd.DataFrame(
                    result["candles"],
                    columns=["datetime", "open", "high", "low", "close", "volume"],
                )
                return df

        except Exception as e:
            print(f"Error fetching candle data: {e}")
            time.sleep(active_order_sleep)


def fetch_train_candle_data(total_days, index_symbol, interval_minutes, fyers, active_order_sleep):
    """
    Fetch extended historical candle data in chunks for ML training.

    Args:
        total_days (int): Total number of days of data to fetch.
        index_symbol (str): Fyers symbol string.
        interval_minutes (int): Candle interval in minutes.
        fyers: Authenticated Fyers API client.
        active_order_sleep (int): Sleep duration on error (seconds).

    Returns:
        pd.DataFrame: Concatenated DataFrame of all fetched candles.
    """
    combined_df = pd.DataFrame()
    chunk_size = 100  # Fyers API limit per request (approximate)

    while True:
        try:
            date_increment = 0
            while date_increment < total_days:
                end_date = date.today() - timedelta(days=date_increment)
                start_date = end_date - timedelta(days=chunk_size)

                data = {
                    "symbol": index_symbol,
                    "resolution": interval_minutes,
                    "date_format": "1",
                    "range_from": start_date.strftime("%Y-%m-%d"),
                    "range_to": end_date.strftime("%Y-%m-%d"),
                    "cont_flag": "1",
                }

                result = fyers.history(data=data)

                if result and "candles" in result:
                    temp_df = pd.DataFrame(
                        result["candles"],
                        columns=["datetime", "open", "high", "low", "close", "volume"],
                    )
                    combined_df = pd.concat([temp_df, combined_df], ignore_index=True)

                date_increment += chunk_size

            return combined_df

        except Exception as e:
            print(f"Error fetching training candle data: {e}")
            time.sleep(active_order_sleep)
