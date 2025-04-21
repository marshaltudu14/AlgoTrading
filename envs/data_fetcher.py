import time
from datetime import date, timedelta
import pandas as pd


def fetch_candle_data(fyers, days, index_symbol, interval_minutes, sleep_interval=1):
    """
    Fetch historical candle data for the past `days` days.
    """
    while True:
        try:
            today = date.today()
            start_date = today - timedelta(days=days)
            data = {
                "symbol": index_symbol,
                "resolution": interval_minutes,
                "date_format": "1",
                "range_from": start_date,
                "range_to": today,
                "cont_flag": "1"
            }
            result = fyers.history(data=data)
            if result and 'candles' in result:
                df = pd.DataFrame(result['candles'],
                                  columns=['datetime','open','high','low','close','volume'])
                return df
        except Exception as e:
            print(f"Error fetching Candle Data: {e}")
            time.sleep(sleep_interval)


def fetch_train_candle_data(fyers, total_days, index_symbol, interval_minutes, batch_size=100, sleep_interval=1):
    """
    Fetch and concatenate candle data in batches of `batch_size` days until `total_days` covered.
    """
    combined = []
    fetched = 0
    while fetched < total_days:
        try:
            end = date.today() - timedelta(days=fetched)
            start = end - timedelta(days=batch_size)
            data = {
                "symbol": index_symbol,
                "resolution": interval_minutes,
                "date_format": "1",
                "range_from": start,
                "range_to": end,
                "cont_flag": "1"
            }
            result = fyers.history(data=data)
            if result and 'candles' in result:
                temp = pd.DataFrame(result['candles'],
                                    columns=['datetime','open','high','low','close','volume'])
                combined.append(temp)
            fetched += batch_size
        except Exception as e:
            print(f"Error fetching train data: {e}")
            time.sleep(sleep_interval)
    if combined:
        return pd.concat(combined, ignore_index=True)
    return pd.DataFrame()
