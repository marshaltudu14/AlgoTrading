from typing import List, Optional
from datetime import datetime

def get_nearest_itm_strike(spot_price: float, available_strikes: List[float], option_type: str) -> Optional[float]:
    """
    Identifies and selects the nearest In-the-Money (ITM) strike price.

    Args:
        spot_price (float): The current spot price of the underlying asset.
        available_strikes (List[float]): A list of available strike prices.
        option_type (str): The type of option, either 'CE' for Call or 'PE' for Put.

    Returns:
        Optional[float]: The nearest ITM strike price, or None if no suitable strike is found.
    """
    if not available_strikes:
        return None

    if option_type.upper() == 'CE':
        itm_strikes = [s for s in available_strikes if s <= spot_price]
        return max(itm_strikes) if itm_strikes else None
    elif option_type.upper() == 'PE':
        itm_strikes = [s for s in available_strikes if s >= spot_price]
        return min(itm_strikes) if itm_strikes else None
    else:
        return None

def get_nearest_expiry(available_expiries: List[str], prefer_weekly: bool = True) -> Optional[str]:
    """
    Selects the nearest expiry date for an options contract.

    Args:
        available_expiries (List[str]): A list of available expiry dates in "YYYY-MM-DD" format.
        prefer_weekly (bool): If True, prioritizes weekly expiries (not fully implemented, defaults to nearest).

    Returns:
        Optional[str]: The nearest expiry date, or None if the list is empty.
    """
    if not available_expiries:
        return None

    today = datetime.now().date()
    
    try:
        # Parse dates and filter for future expiries
        future_expiries = sorted([datetime.strptime(d, "%Y-%m-%d").date() for d in available_expiries if datetime.strptime(d, "%Y-%m-%d").date() >= today])
    except ValueError:
        # Handle potential parsing errors
        return None

    if not future_expiries:
        return None

    # For now, we just return the absolute nearest expiry. 
    # The logic for distinguishing weekly/monthly can be added later.
    return future_expiries[0].strftime("%Y-%m-%d")

def map_underlying_to_option_price(underlying_target_price: float, current_underlying_price: float, current_option_price: float, option_type: str) -> float:
    """
    Translates the SL/TP of an underlying to the corresponding option price using a simplified linear mapping.

    Args:
        underlying_target_price (float): The target price of the underlying (e.g., SL or TP).
        current_underlying_price (float): The current price of the underlying at the time of mapping.
        current_option_price (float): The current price of the option.
        option_type (str): The type of option, 'CE' or 'PE'.

    Returns:
        float: The mapped option price.
    """
    price_change = underlying_target_price - current_underlying_price

    if option_type.upper() == 'CE':
        # For calls, option price moves in the same direction as the underlying
        mapped_price = current_option_price + price_change
    elif option_type.upper() == 'PE':
        # For puts, option price moves in the opposite direction
        mapped_price = current_option_price - price_change
    else:
        mapped_price = current_option_price

    # Ensure the option price doesn't go below zero
    return max(0, mapped_price)