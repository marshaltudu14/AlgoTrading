import requests
import pandas as pd
import datetime
import re
from config import OPTION_MASTER_FO, STRIKE_STEP, TIMEFRAMES, INSTRUMENTS
from envs.data_fetcher import fetch_candle_data


def fetch_option_master():
    """Download and return the FO symbol master as a DataFrame."""
    data = requests.get(OPTION_MASTER_FO).json()
    return pd.DataFrame(data)


def nearest_ITM_strike(master_df: pd.DataFrame, underlying_price: float, instrument: str, option_type: str) -> str:
    """
    Select the nearest ITM strike symbol for given option_type ('CE' or 'PE').
    """
    # Underlying code (e.g. 'NIFTY50' from 'NSE:NIFTY50-INDEX')
    underlying_symbol = INSTRUMENTS[instrument]
    code = underlying_symbol.split(':')[1].split('-')[0]

    # Filter by instrument and option type
    df = master_df[master_df['symbol'].str.contains(code) & master_df['symbol'].str.endswith(option_type)]
    # Extract strike via regex before CE/PE
    df['strike'] = df['symbol'].apply(lambda s: int(re.search(r"(\d+)(?=" + option_type + r")", s).group(1)))
    # Filter to multiples of STRIKE_STEP
    df = df[df['strike'] % STRIKE_STEP == 0]

    # For CE: strikes <= price; for PE: strikes >= price
    if option_type == 'CE':
        candidates = df[df['strike'] <= underlying_price]
    else:
        candidates = df[df['strike'] >= underlying_price]

    # Find closest
    idx = (candidates['strike'] - underlying_price).abs().idxmin()
    return candidates.loc[idx, 'symbol']


def get_current_index_price(fyers, index_symbol: str) -> float:
    """Fetch latest close price for index via one candle."""
    today = datetime.date.today()
    df = fetch_candle_data(fyers, 1, index_symbol, TIMEFRAMES[0])
    # last row close
    return float(df['close'].iloc[-1])
