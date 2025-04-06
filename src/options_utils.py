# src/options_utils.py
"""
Utility functions for handling Fyers option chain data and selecting option symbols.
"""
import logging
import time
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

def get_option_chain(fyers_instance, underlying_symbol: str) -> pd.DataFrame | None:
    """
    Fetches the option chain for a given underlying symbol.

    Args:
        fyers_instance: Authenticated FyersModel instance.
        underlying_symbol (str): The symbol of the underlying index/stock (e.g., "NSE:NIFTYBANK-INDEX").

    Returns:
        pd.DataFrame containing the option chain data, or None on error.
    """
    logger.debug(f"Fetching option chain for {underlying_symbol}...")
    try:
        # Note: Fyers API v3 might use 'symbol' instead of 'underlying_symbol' here.
        # The documentation example uses 'symbol'. Let's stick to that.
        # Strikecount determines how many ITM/OTM strikes around ATM are returned.
        # A higher count might be needed if the nearest ITM isn't returned with a low count.
        data = {
            "symbol": underlying_symbol,
            "strikecount": 10 # Fetch a decent number of strikes around ATM
        }
        response = fyers_instance.optionchain(data=data)

        if response and response.get("s") == "ok" and "optionsChain" in response.get("data", {}):
            oc_df = pd.DataFrame(response['data']['optionsChain'])
            # Ensure strike_price is numeric, handling potential errors
            oc_df['strike_price'] = pd.to_numeric(oc_df['strike_price'], errors='coerce')
            # Filter out rows where strike_price couldn't be converted or is invalid (-1)
            oc_df = oc_df[oc_df['strike_price'].notna() & (oc_df['strike_price'] > 0)]
            logger.debug(f"Successfully fetched and parsed option chain for {underlying_symbol}. Shape: {oc_df.shape}")
            return oc_df
        else:
            logger.error(f"Failed to fetch option chain for {underlying_symbol}. Response: {response}")
            return None
    except Exception as e:
        logger.error(f"Exception fetching option chain for {underlying_symbol}: {e}", exc_info=True)
        return None

def select_itm_strike(oc_df: pd.DataFrame, underlying_ltp: float, option_type: str) -> float | None:
    """
    Selects the nearest In-The-Money (ITM) strike price.

    Args:
        oc_df (pd.DataFrame): DataFrame containing the option chain.
        underlying_ltp (float): The current Last Traded Price of the underlying.
        option_type (str): "CE" for Call options, "PE" for Put options.

    Returns:
        The selected ITM strike price (float), or None if no suitable strike is found.
    """
    if oc_df is None or oc_df.empty or underlying_ltp <= 0:
        logger.warning("Cannot select ITM strike: Invalid input DataFrame, LTP, or empty chain.")
        return None

    # Filter for the desired option type
    filtered_df = oc_df[oc_df['option_type'] == option_type].copy()
    if filtered_df.empty:
        logger.warning(f"No {option_type} options found in the provided chain.")
        return None

    # Find nearest ITM strike
    itm_strike = None
    if option_type == "CE":
        # ITM Calls: strike < underlying_ltp. Find the highest strike among these.
        itm_options = filtered_df[filtered_df['strike_price'] < underlying_ltp]
        if not itm_options.empty:
            itm_strike = itm_options['strike_price'].max()
    elif option_type == "PE":
        # ITM Puts: strike > underlying_ltp. Find the lowest strike among these.
        itm_options = filtered_df[filtered_df['strike_price'] > underlying_ltp]
        if not itm_options.empty:
            itm_strike = itm_options['strike_price'].min()

    if itm_strike is None:
        logger.warning(f"Could not find suitable ITM {option_type} strike for underlying LTP {underlying_ltp}.")
        # Fallback? Maybe select ATM instead? For now, return None.
        # atm_strike = filtered_df.iloc[(filtered_df['strike_price'] - underlying_ltp).abs().argsort()[:1]]['strike_price'].iloc[0]
        # return atm_strike
        return None
    else:
        logger.info(f"Selected ITM {option_type} strike: {itm_strike} for underlying LTP {underlying_ltp}")
        return float(itm_strike) # Ensure float type

def get_option_symbol_by_strike(oc_df: pd.DataFrame, strike_price: float, option_type: str) -> str | None:
    """
    Finds the full option symbol for a given strike and type from the option chain DataFrame.

    Args:
        oc_df (pd.DataFrame): DataFrame containing the option chain.
        strike_price (float): The desired strike price.
        option_type (str): "CE" or "PE".

    Returns:
        The full Fyers option symbol string, or None if not found.
    """
    if oc_df is None or oc_df.empty:
        return None
    try:
        # Find the row matching the strike and type
        target_option = oc_df[
            (oc_df['strike_price'] == strike_price) &
            (oc_df['option_type'] == option_type)
        ]
        if not target_option.empty:
            symbol = target_option['symbol'].iloc[0]
            logger.debug(f"Found symbol {symbol} for strike {strike_price} {option_type}")
            return symbol
        else:
            logger.warning(f"Could not find symbol for strike {strike_price} {option_type} in option chain.")
            return None
    except Exception as e:
        logger.error(f"Error finding option symbol for strike {strike_price} {option_type}: {e}", exc_info=True)
        return None

# Example Usage (for testing)
if __name__ == "__main__":
    # This requires mocking fyers_instance or having live credentials
    # and ensuring the market is open for option chain data.
    print("Testing options_utils (requires authenticated fyers instance and market hours)")
    # Example:
    # from src import config, fyers_auth
    # token_info = fyers_auth.get_fyers_access_token()
    # if token_info:
    #     fyers = fyersModel.FyersModel(client_id=config.APP_ID, token=token_info['access_token'])
    #     underlying = "NSE:NIFTYBANK-INDEX"
    #     ltp = 51500 # Example LTP
    #
    #     oc_data = get_option_chain(fyers, underlying)
    #     if oc_data is not None:
    #         print("\nOption Chain Head:")
    #         print(oc_data.head())
    #
    #         # Test ITM CE
    #         itm_ce_strike = select_itm_strike(oc_data, ltp, "CE")
    #         if itm_ce_strike:
    #             itm_ce_symbol = get_option_symbol_by_strike(oc_data, itm_ce_strike, "CE")
    #             print(f"\nSelected ITM CE Strike: {itm_ce_strike}, Symbol: {itm_ce_symbol}")
    #
    #         # Test ITM PE
    #         itm_pe_strike = select_itm_strike(oc_data, ltp, "PE")
    #         if itm_pe_strike:
    #             itm_pe_symbol = get_option_symbol_by_strike(oc_data, itm_pe_strike, "PE")
    #             print(f"Selected ITM PE Strike: {itm_pe_strike}, Symbol: {itm_pe_symbol}")
    # else:
    #     print("Authentication failed, cannot run tests.")
    pass
