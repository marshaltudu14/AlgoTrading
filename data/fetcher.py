"""
Data fetching module for AlgoTrading.
Handles fetching historical and real-time data from Fyers API.
"""
import time
import json
import requests
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

from core.logging_setup import get_logger
from core.constants import (
    FYERS_EXCHANGE_NSE,
    FYERS_EXCHANGE_BSE,
    FYERS_SEGMENT_FO,
    OptionType
)
from core.config import (
    OPTION_MASTER_FO,
    STRIKE_STEP,
    MAX_DAYS_TO_EXPIRY,
    MIN_DAYS_TO_EXPIRY,
    PREFER_ATM
)

logger = get_logger(__name__)


def fetch_candle_data(
    fyers, 
    days: int, 
    index_symbol: str, 
    interval_minutes: int, 
    sleep_interval: float = 1,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch historical candle data for the past `days` days.
    
    Args:
        fyers: Fyers API client
        days: Number of days to fetch
        index_symbol: Symbol to fetch data for
        interval_minutes: Candle interval in minutes
        sleep_interval: Sleep interval between retries
        max_retries: Maximum number of retries
        
    Returns:
        DataFrame with candle data
    """
    retries = 0
    while retries < max_retries:
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
            logger.info(f"Fetching candle data for {index_symbol} @ {interval_minutes}min from {start_date} to {today}")
            result = fyers.history(data=data)
            
            if result and 'candles' in result and result['candles']:
                df = pd.DataFrame(
                    result['candles'],
                    columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Convert datetime to pandas datetime
                df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
                
                # Check if we have enough data
                if len(df) < 10:  # Arbitrary minimum number of candles
                    logger.warning(f"Not enough data for {index_symbol}: only {len(df)} candles")
                    retries += 1
                    time.sleep(sleep_interval)
                    continue
                
                logger.info(f"Fetched {len(df)} candles for {index_symbol}")
                return df
            else:
                logger.warning(f"No candles in response for {index_symbol}: {result}")
                retries += 1
                time.sleep(sleep_interval)
        except Exception as e:
            logger.error(f"Error fetching candle data for {index_symbol}: {e}")
            retries += 1
            time.sleep(sleep_interval)
    
    # If we get here, we've exhausted all retries
    logger.error(f"Failed to fetch candle data for {index_symbol} after {max_retries} retries")
    return pd.DataFrame()


def fetch_train_candle_data(
    fyers, 
    total_days: int, 
    index_symbol: str, 
    interval_minutes: int, 
    batch_size: int = 100, 
    sleep_interval: float = 1,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch and concatenate candle data in batches of `batch_size` days until `total_days` covered.
    
    Args:
        fyers: Fyers API client
        total_days: Total number of days to fetch
        index_symbol: Symbol to fetch data for
        interval_minutes: Candle interval in minutes
        batch_size: Number of days per batch
        sleep_interval: Sleep interval between retries
        max_retries: Maximum number of retries
        
    Returns:
        DataFrame with concatenated candle data
    """
    combined = []
    fetched = 0
    
    logger.info(f"Fetching {total_days} days of training data for {index_symbol} @ {interval_minutes}min")
    
    while fetched < total_days:
        retries = 0
        success = False
        
        while retries < max_retries and not success:
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
                
                logger.info(f"Fetching batch from {start} to {end} for {index_symbol}")
                result = fyers.history(data=data)
                
                if result and 'candles' in result and result['candles']:
                    temp = pd.DataFrame(
                        result['candles'],
                        columns=['datetime', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Convert datetime to pandas datetime
                    temp['datetime'] = pd.to_datetime(temp['datetime'], unit='s')
                    
                    combined.append(temp)
                    logger.info(f"Fetched {len(temp)} candles for batch")
                    success = True
                else:
                    logger.warning(f"No candles in response for batch: {result}")
                    retries += 1
                    time.sleep(sleep_interval)
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                retries += 1
                time.sleep(sleep_interval)
        
        if not success:
            logger.warning(f"Failed to fetch batch after {max_retries} retries, continuing with next batch")
        
        fetched += batch_size
    
    if combined:
        result = pd.concat(combined, ignore_index=True)
        
        # Remove duplicates
        result = result.drop_duplicates(subset=['datetime'])
        
        # Sort by datetime
        result = result.sort_values('datetime')
        
        logger.info(f"Total fetched: {len(result)} candles for {index_symbol}")
        return result
    
    logger.error(f"Failed to fetch any data for {index_symbol}")
    return pd.DataFrame()


def fetch_option_chain(
    symbol: str,
    expiry_date: Optional[str] = None,
    strike_step: int = STRIKE_STEP,
    option_master_url: str = OPTION_MASTER_FO,
    max_retries: int = 3
) -> Dict[str, Dict]:
    """
    Fetch option chain for a given symbol and expiry date.
    
    Args:
        symbol: Symbol to fetch options for (e.g., 'NIFTY')
        expiry_date: Expiry date in format 'YYMMDD' (default: nearest expiry)
        strike_step: Step size for strikes
        option_master_url: URL for option master file
        max_retries: Maximum number of retries
        
    Returns:
        Dictionary of option symbols mapped to their details
    """
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Fetching option chain for {symbol}")
            response = requests.get(option_master_url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Filter options for the given symbol
                options = {}
                for symbol_key, details in data.items():
                    # Check if this is an option for our symbol
                    if (details.get('underSym') == symbol and 
                        details.get('optType') in ('CE', 'PE')):
                        
                        # If expiry date is specified, filter by it
                        if expiry_date:
                            option_expiry = datetime.fromtimestamp(
                                int(details.get('expiryDate'))
                            ).strftime('%y%m%d')
                            
                            if option_expiry != expiry_date:
                                continue
                        
                        # Add to options dictionary
                        options[symbol_key] = details
                
                if not options:
                    logger.warning(f"No options found for {symbol}")
                    retries += 1
                    time.sleep(1)
                    continue
                
                logger.info(f"Found {len(options)} options for {symbol}")
                return options
            else:
                logger.warning(f"Failed to fetch option chain: {response.status_code}")
                retries += 1
                time.sleep(1)
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            retries += 1
            time.sleep(1)
    
    logger.error(f"Failed to fetch option chain after {max_retries} retries")
    return {}


def get_nearest_expiry_date(
    symbol: str,
    option_master_url: str = OPTION_MASTER_FO,
    max_days: int = MAX_DAYS_TO_EXPIRY,
    min_days: int = MIN_DAYS_TO_EXPIRY
) -> Optional[str]:
    """
    Get the nearest expiry date for options.
    
    Args:
        symbol: Symbol to fetch options for (e.g., 'NIFTY')
        option_master_url: URL for option master file
        max_days: Maximum days to expiry
        min_days: Minimum days to expiry
        
    Returns:
        Expiry date in format 'YYMMDD' or None if not found
    """
    try:
        logger.info(f"Finding nearest expiry date for {symbol}")
        response = requests.get(option_master_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get all expiry dates for the symbol
            expiry_dates = set()
            for details in data.values():
                if details.get('underSym') == symbol and 'expiryDate' in details:
                    expiry_dates.add(int(details['expiryDate']))
            
            if not expiry_dates:
                logger.warning(f"No expiry dates found for {symbol}")
                return None
            
            # Convert to datetime and filter by min/max days
            today = datetime.now()
            valid_expiries = []
            
            for expiry_timestamp in expiry_dates:
                expiry_date = datetime.fromtimestamp(expiry_timestamp)
                days_to_expiry = (expiry_date - today).days
                
                if min_days <= days_to_expiry <= max_days:
                    valid_expiries.append((expiry_date, days_to_expiry))
            
            if not valid_expiries:
                logger.warning(f"No valid expiry dates found for {symbol} within {min_days}-{max_days} days")
                return None
            
            # Sort by days to expiry and get the nearest
            valid_expiries.sort(key=lambda x: x[1])
            nearest_expiry = valid_expiries[0][0]
            
            # Format as YYMMDD
            expiry_str = nearest_expiry.strftime('%y%m%d')
            logger.info(f"Nearest expiry date for {symbol}: {expiry_str}")
            return expiry_str
        else:
            logger.warning(f"Failed to fetch option master: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error finding nearest expiry date: {e}")
        return None


def select_option_strike(
    current_price: float,
    action: str,
    option_chain: Dict[str, Dict],
    strike_step: int = STRIKE_STEP,
    prefer_atm: bool = PREFER_ATM
) -> Optional[str]:
    """
    Select the appropriate option strike based on current price and action.
    
    Args:
        current_price: Current price of the underlying
        action: Action to take ('BUY_CE' or 'BUY_PE')
        option_chain: Dictionary of option symbols mapped to their details
        strike_step: Step size for strikes
        prefer_atm: Whether to prefer At-The-Money options
        
    Returns:
        Selected option symbol or None if not found
    """
    try:
        logger.info(f"Selecting option strike for {action} at price {current_price}")
        
        # Determine option type
        option_type = None
        if action == "BUY_CE":
            option_type = OptionType.CALL.value
        elif action == "BUY_PE":
            option_type = OptionType.PUT.value
        else:
            logger.warning(f"Invalid action: {action}")
            return None
        
        # Round current price to nearest strike step
        rounded_price = round(current_price / strike_step) * strike_step
        
        # Find all available strikes
        available_strikes = {}
        for symbol, details in option_chain.items():
            if details.get('optType') == option_type:
                strike = details.get('strikePrice')
                if strike:
                    available_strikes[symbol] = float(strike)
        
        if not available_strikes:
            logger.warning(f"No {option_type} options found")
            return None
        
        # Select strike based on strategy
        if prefer_atm:
            # Find the strike closest to current price
            closest_symbol = min(
                available_strikes.items(),
                key=lambda x: abs(x[1] - current_price)
            )[0]
            logger.info(f"Selected ATM option: {closest_symbol}")
            return closest_symbol
        else:
            # For calls, select slightly ITM; for puts, select slightly ITM
            if option_type == OptionType.CALL.value:
                # Find the highest strike below current price
                itm_strikes = {s: k for s, k in available_strikes.items() if k <= current_price}
                if itm_strikes:
                    selected = max(itm_strikes.items(), key=lambda x: x[1])[0]
                    logger.info(f"Selected ITM call option: {selected}")
                    return selected
            else:  # PUT
                # Find the lowest strike above current price
                itm_strikes = {s: k for s, k in available_strikes.items() if k >= current_price}
                if itm_strikes:
                    selected = min(itm_strikes.items(), key=lambda x: x[1])[0]
                    logger.info(f"Selected ITM put option: {selected}")
                    return selected
            
            # If no ITM options found, fall back to ATM
            closest_symbol = min(
                available_strikes.items(),
                key=lambda x: abs(x[1] - current_price)
            )[0]
            logger.info(f"No suitable ITM options, using ATM: {closest_symbol}")
            return closest_symbol
    except Exception as e:
        logger.error(f"Error selecting option strike: {e}")
        return None


def verify_data_sufficiency(
    df: pd.DataFrame,
    min_rows: int = 100,
    required_columns: List[str] = ['open', 'high', 'low', 'close']
) -> bool:
    """
    Verify that the dataframe has sufficient data for processing.
    
    Args:
        df: DataFrame to verify
        min_rows: Minimum number of rows required
        required_columns: List of required columns
        
    Returns:
        True if data is sufficient, False otherwise
    """
    try:
        # Check if dataframe is empty
        if df.empty:
            logger.warning("DataFrame is empty")
            return False
        
        # Check if dataframe has enough rows
        if len(df) < min_rows:
            logger.warning(f"DataFrame has only {len(df)} rows, minimum required is {min_rows}")
            return False
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values in required columns
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Null values found in columns: {null_counts[null_counts > 0].to_dict()}")
            return False
        
        # Check for zero values in required columns (except volume)
        zero_counts = (df[required_columns] == 0).sum()
        if zero_counts.sum() > 0:
            logger.warning(f"Zero values found in columns: {zero_counts[zero_counts > 0].to_dict()}")
            return False
        
        logger.info(f"Data verification passed: {len(df)} rows with all required columns")
        return True
    except Exception as e:
        logger.error(f"Error verifying data sufficiency: {e}")
        return False
