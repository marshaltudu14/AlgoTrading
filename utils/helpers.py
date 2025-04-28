"""
Helper functions for AlgoTrading.
Utility functions used across the application.
"""
import os
import json
import requests
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

from core.logging_setup import get_logger
from core.config import (
    OPTION_MASTER_FO,
    STRIKE_STEP,
    INSTRUMENTS,
    TIMEFRAMES
)

logger = get_logger(__name__)


def fetch_option_master() -> pd.DataFrame:
    """
    Download and return the FO symbol master as a DataFrame.
    
    Returns:
        DataFrame with option master data
    """
    logger.info(f"Fetching option master from {OPTION_MASTER_FO}")
    
    try:
        response = requests.get(OPTION_MASTER_FO)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame.from_dict(data, orient='index')
            logger.info(f"Fetched option master with {len(df)} symbols")
            return df
        else:
            logger.error(f"Failed to fetch option master: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching option master: {e}")
        return pd.DataFrame()


def nearest_ITM_strike(
    master_df: pd.DataFrame,
    underlying_price: float,
    instrument: str,
    option_type: str
) -> str:
    """
    Select the nearest ITM strike symbol for given option_type ('CE' or 'PE').
    
    Args:
        master_df: Option master DataFrame
        underlying_price: Current price of the underlying
        instrument: Instrument name
        option_type: Option type ('CE' or 'PE')
        
    Returns:
        Selected option symbol
    """
    logger.info(f"Finding nearest ITM strike for {instrument} @ {underlying_price} ({option_type})")
    
    try:
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
        symbol = candidates.loc[idx, 'symbol']
        
        logger.info(f"Selected strike: {symbol}")
        return symbol
    except Exception as e:
        logger.error(f"Error finding nearest ITM strike: {e}")
        return ""


def get_current_index_price(fyers, index_symbol: str) -> float:
    """
    Fetch latest close price for index via one candle.
    
    Args:
        fyers: Fyers API client
        index_symbol: Index symbol
        
    Returns:
        Current index price
    """
    logger.info(f"Getting current price for {index_symbol}")
    
    try:
        from data.fetcher import fetch_candle_data
        
        # Fetch one day of data
        df = fetch_candle_data(fyers, 1, index_symbol, TIMEFRAMES[0])
        
        # Get last row close
        price = float(df['close'].iloc[-1])
        
        logger.info(f"Current price for {index_symbol}: {price}")
        return price
    except Exception as e:
        logger.error(f"Error getting current index price: {e}")
        return 0.0


def calculate_days_to_expiry(date_str: str, format_str: str = '%y%m%d') -> int:
    """
    Calculate days to expiry for a given date string.
    
    Args:
        date_str: Date string
        format_str: Date format string
        
    Returns:
        Days to expiry
    """
    try:
        expiry_date = datetime.strptime(date_str, format_str)
        today = datetime.now()
        delta = expiry_date - today
        return max(0, delta.days)
    except Exception as e:
        logger.error(f"Error calculating days to expiry: {e}")
        return 0


def format_price(price: float, tick_size: float = 0.05) -> float:
    """
    Format price to nearest tick size.
    
    Args:
        price: Price to format
        tick_size: Tick size
        
    Returns:
        Formatted price
    """
    return round(price / tick_size) * tick_size


def calculate_margin_required(
    symbol: str,
    quantity: int,
    price: float,
    instrument: Optional[str] = None
) -> float:
    """
    Calculate margin required for a position.
    
    Args:
        symbol: Symbol to trade
        quantity: Quantity to trade
        price: Current price
        instrument: Instrument name (optional)
        
    Returns:
        Margin required
    """
    # Extract instrument from symbol if not provided
    if not instrument:
        for instr, index_symbol in INSTRUMENTS.items():
            if index_symbol.split(':')[1].split('-')[0] in symbol:
                instrument = instr
                break
    
    # Get margin requirement
    margin_pct = 1.0
    if instrument in INSTRUMENTS:
        from core.config import MARGIN_REQUIREMENTS
        margin_pct = MARGIN_REQUIREMENTS.get(instrument, 1.0)
    
    # Calculate margin
    return price * quantity * margin_pct


def is_market_open() -> bool:
    """
    Check if market is open based on current time.
    
    Returns:
        True if market is open
    """
    from core.config import MARKET_OPEN_TIME, MARKET_CLOSE_TIME
    import pytz
    
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist).time()
    
    # Parse market hours
    open_time = datetime.strptime(MARKET_OPEN_TIME, '%H:%M').time()
    close_time = datetime.strptime(MARKET_CLOSE_TIME, '%H:%M').time()
    
    # Check if current time is within market hours
    return open_time <= now <= close_time


def get_expiry_dates(instrument: str) -> List[str]:
    """
    Get list of available expiry dates for an instrument.
    
    Args:
        instrument: Instrument name
        
    Returns:
        List of expiry dates
    """
    try:
        # Fetch option master
        master_df = fetch_option_master()
        
        # Get underlying symbol
        underlying_symbol = INSTRUMENTS[instrument]
        code = underlying_symbol.split(':')[1].split('-')[0]
        
        # Filter by instrument
        df = master_df[master_df['symbol'].str.contains(code)]
        
        # Extract expiry dates
        expiry_dates = []
        for symbol in df['symbol']:
            match = re.search(r"(\d{6})(?=CE|PE)", symbol)
            if match:
                expiry_dates.append(match.group(1))
        
        # Remove duplicates and sort
        expiry_dates = sorted(list(set(expiry_dates)))
        
        return expiry_dates
    except Exception as e:
        logger.error(f"Error getting expiry dates: {e}")
        return []
