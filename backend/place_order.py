"""
Place Order Script for Index Options Trading
Based on closing price, finds nearest ITM/ATM/OTM option and places order with SL and target
"""

import sys
import os
import asyncio
import argparse
import logging
from typing import Dict, Any, Optional
from fyers_apiv3 import fyersModel

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth.fyers_auth_service import get_access_token, create_fyers_model, FyersAuthenticationError
from src.config.fyers_config import (
    APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP,
    DEFAULT_LOT_SIZE, DEFAULT_PRODUCT_TYPE, DEFAULT_ORDER_TYPE, DEFAULT_EXCHANGE,
    INDEX_SYMBOLS, STRIKE_PRICE_INTERVAL
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_option_price_range(range_str: str) -> tuple:
    """Parse option price range string into min and max values"""
    try:
        if '-' in range_str:
            min_ltp, max_ltp = map(float, range_str.split('-'))
            return min_ltp, max_ltp
        else:
            # If single value provided, use it as max with 0 as min
            max_ltp = float(range_str)
            return 0, max_ltp
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid option price range format: {range_str}. Use format like '300-500'")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Place index option order")

    parser.add_argument("--symbol", type=str, required=True,
                        help="Index symbol (e.g., NSE:BANKNIFTY-INDEX, NSE:NIFTY50-INDEX)")
    parser.add_argument("--closing_price", type=float, required=True,
                        help="Index closing price")
    parser.add_argument("--direction", type=str, required=True,
                        choices=['BUY', 'SELL'],
                        help="Trade direction (BUY/SELL)")
    parser.add_argument("--option_price_range", type=str, required=True,
                        help="Option LTP range to filter strikes (e.g., 300-500)")
    parser.add_argument("--quantity", type=int, required=True,
                        help="Order quantity (lot size)")
    parser.add_argument("--sl_price", type=float, required=True,
                        help="Stop loss price in points")
    parser.add_argument("--target_price", type=float, required=True,
                        help="Target price in points")
    parser.add_argument("--option_type", type=str, default="CE",
                        choices=['CE', 'PE'],
                        help="Option type (CE/PE)")

    args = parser.parse_args()

    # Parse the option price range
    try:
        args.min_ltp, args.max_ltp = parse_option_price_range(args.option_price_range)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    return args

async def get_option_chain(fyers, index_symbol: str, strike_count: int = 20) -> dict:
    """
    Get option chain data for the given index using Fyers API

    Args:
        fyers: Fyers API instance
        index_symbol: Index symbol (e.g., NSE:BANKNIFTY-INDEX)
        strike_count: Number of strikes to fetch (default: 20)

    Returns:
        Dictionary containing option chain data
    """
    try:
        # Use Fyers Option Chain API
        import requests
        from datetime import datetime

        # Construct the API URL
        base_url = "https://api-t1.fyers.in/data/options-chain-v3"

        # Use the symbol as provided - no mapping needed
        option_chain_symbol = index_symbol

        # URL encode the symbol
        import urllib.parse
        encoded_symbol = urllib.parse.quote(option_chain_symbol)

        api_url = f"{base_url}?symbol={encoded_symbol}&strikecount={strike_count}"

        # Get the access token for authorization
        access_token = fyers.token

        # Make the API request
        headers = {
            'Authorization': f"{access_token}"
        }

        logger.info(f"Fetching option chain from: {api_url}")
        response = requests.get(api_url, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            if response_data.get('s') == 'ok' and 'data' in response_data:
                option_data = response_data['data']
                logger.info(f"Successfully fetched option chain for {option_chain_symbol}")
                return option_data
            else:
                logger.error(f"API returned error: {response_data}")
                return {}
        else:
            logger.error(f"Failed to fetch option chain: {response.status_code} - {response.text}")
            return {}

    except Exception as e:
        logger.error(f"Error fetching option chain: {e}")
        return {}

async def get_expanded_option_chain(fyers, index_symbol: str, closing_price: float,
                                   price_range: float, max_attempts: int = 3) -> dict:
    """
    Get expanded option chain data with sufficient strikes around the closing price

    Args:
        fyers: Fyers API instance
        index_symbol: Index symbol (e.g., NSE:BANKNIFTY-INDEX)
        closing_price: Index closing price
        price_range: Desired price range around closing price
        max_attempts: Maximum number of attempts to expand the option chain

    Returns:
        Dictionary containing option chain data
    """
    base_strike_count = 20
    for attempt in range(max_attempts):
        strike_count = base_strike_count * (attempt + 1)
        logger.info(f"Attempt {attempt + 1}: Fetching {strike_count} strikes")

        option_data = await get_option_chain(fyers, index_symbol, strike_count)

        if option_data and 'optionsChain' in option_data:
            options_chain = option_data['optionsChain']
            if options_chain:
                # Filter out non-strike entries (like the index itself)
                valid_strikes = [option.get('strike_price') for option in options_chain
                               if option.get('strike_price') and option.get('strike_price') > 0]

                if valid_strikes:
                    min_strike = min(valid_strikes)
                    max_strike = max(valid_strikes)

                    logger.info(f"Strike range: {min_strike} - {max_strike}")
                    logger.info(f"Desired range: {closing_price - price_range} - {closing_price + price_range}")

                    if (min_strike <= closing_price - price_range and
                        max_strike >= closing_price + price_range):
                        logger.info(f"Found suitable option chain with {len(valid_strikes)} valid strikes")
                        return option_data
                    else:
                        logger.info(f"Strike range insufficient, expanding...")
                else:
                    logger.warning(f"No valid strike prices found in options chain")
            else:
                logger.warning(f"Empty options chain returned")
        else:
            logger.warning(f"Failed to fetch option chain: {option_data}")

    # If we couldn't find the ideal range, return the best we have
    logger.info(f"Returning the largest option chain available")
    return option_data if option_data else {}

def find_nearest_option(closing_price: float, min_ltp: float, max_ltp: float,
                       option_chain_data: dict, option_type: str = "CE") -> Optional[Dict]:
    """
    Find option within specified LTP range, closest to ATM

    Args:
        closing_price: Index closing price
        min_ltp: Minimum LTP to consider
        max_ltp: Maximum LTP to consider
        option_chain_data: Option chain data from Fyers API
        option_type: Option type (CE/PE)

    Returns:
        Dictionary with option details or None if not found
    """
    try:
        if not option_chain_data or 'optionsChain' not in option_chain_data:
            logger.error("No option chain data available")
            return None

        options_chain = option_chain_data['optionsChain']
        if not options_chain:
            logger.error("Empty options chain")
            return None

        # First, find all options within the LTP range
        suitable_options = []

        # Debug: Print some sample LTP values
        sample_ltps = []
        all_options = []
        for option in options_chain:
            if not option.get('strike_price') or option.get('strike_price') <= 0:
                continue

            # Each option has an 'option_type' field indicating CE or PE
            current_option_type = option.get('option_type', '')
            if current_option_type == option_type:
                ltp = option.get('ltp', 0)
                all_options.append((option.get('strike_price'), ltp))
                if ltp > 0:
                    sample_ltps.append((option.get('strike_price'), ltp))

        # Sort by LTP and print some samples
        sample_ltps.sort(key=lambda x: x[1])
        all_options.sort(key=lambda x: x[1])
        logger.info(f"ALL {option_type} options (strike, ltp): {all_options[:20]}")
        logger.info(f"Sample {option_type} option LTPs > 0 (strike, ltp): {sample_ltps[:10]}")

        for option in options_chain:
            # Skip non-strike entries (like the index itself)
            if not option.get('strike_price') or option.get('strike_price') <= 0:
                continue

            # Each option has an 'option_type' field indicating CE or PE
            current_option_type = option.get('option_type', '')
            if current_option_type == option_type:
                ltp = option.get('ltp', 0)
                strike_price = option.get('strike_price')

                # Check if LTP is within the specified range
                if min_ltp <= ltp <= max_ltp and strike_price:
                    suitable_options.append({
                        'symbol': option.get('symbol', ''),
                        'strike': strike_price,
                        'ltp': ltp,
                        'option_type': option_type,
                        'distance_from_atm': abs(strike_price - closing_price)
                    })

        if not suitable_options:
            logger.error(f"No {option_type} options found in LTP range {min_ltp}-{max_ltp}")
            return None

        # Sort by distance from ATM (closest first)
        suitable_options.sort(key=lambda x: x['distance_from_atm'])

        # Select the option closest to ATM
        selected_option = suitable_options[0]

        logger.info(f"Found {len(suitable_options)} options in LTP range {min_ltp}-{max_ltp}")
        logger.info(f"Selected option: {selected_option['symbol']}")
        logger.info(f"Strike: {selected_option['strike']}, LTP: {selected_option['ltp']}")
        logger.info(f"Distance from ATM: {selected_option['distance_from_atm']}")

        return {
            "symbol": selected_option['symbol'],
            "strike": selected_option['strike'],
            "option_type": selected_option['option_type'],
            "ltp": selected_option['ltp']
        }

    except Exception as e:
        logger.error(f"Error finding nearest option: {e}")
        return None

async def get_option_ltp(fyers, option_symbol: str) -> float:
    """
    Get the last traded price for an option

    Args:
        fyers: Fyers API instance
        option_symbol: Option symbol

    Returns:
        Last traded price
    """
    try:
        # Quote API to get LTP
        quote_response = fyers.quotes({"symbols": option_symbol})

        if quote_response.get("s") == "ok" and quote_response.get("d"):
            data = quote_response["d"][0]
            if data and "v" in data and data["v"]:
                ltp = data["v"].get("lp", 0)
                logger.info(f"Option {option_symbol} LTP: {ltp}")
                return ltp

        logger.error(f"Failed to get LTP for {option_symbol}: {quote_response}")
        return 0

    except Exception as e:
        logger.error(f"Error getting option LTP: {e}")
        return 0

async def place_order(fyers, option_symbol: str, direction: str, quantity: int,
                     sl_price: float, target_price: float, ltp: float) -> Dict[str, Any]:
    """
    Place order with calculated SL and target prices

    Args:
        fyers: Fyers API instance
        option_symbol: Option symbol to trade
        direction: Trade direction (BUY/SELL)
        quantity: Order quantity
        sl_price: Stop loss in points
        target_price: Target in points
        ltp: Current LTP of the option

    Returns:
        Order response
    """
    try:
        # Calculate actual SL and target prices
        if direction.upper() == "BUY":
            actual_sl = ltp - sl_price
            actual_target = ltp + target_price
        else:  # SELL
            actual_sl = ltp + sl_price
            actual_target = ltp - target_price

        logger.info(f"Placing {direction} order for {option_symbol}")
        logger.info(f"Quantity: {quantity}, LTP: {ltp}")
        logger.info(f"SL: {actual_sl}, Target: {actual_target}")

        # Prepare order data
        order_data = {
            "symbol": option_symbol,
            "qty": quantity,
            "type": 2,  # Market order
            "side": 1 if direction.upper() == "BUY" else -1,
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "orderTag": "OptionOrder"
        }

        # Place the order
        order_response = fyers.place_order(order_data)

        logger.info(f"Order response: {order_response}")

        if order_response.get("s") == "ok":
            logger.info(f"Order placed successfully. Order ID: {order_response.get('id')}")
            return order_response
        else:
            logger.error(f"Order placement failed: {order_response}")
            return order_response

    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return {"error": str(e)}

async def main():
    """Main function to place order"""
    args = parse_arguments()

    logger.info("Starting order placement process...")
    logger.info(f"Parameters: Closing Price={args.closing_price}, Direction={args.direction}")
    logger.info(f"Option Range={args.option_price_range}, Quantity={args.quantity}")
    logger.info(f"SL Points={args.sl_price}, Target Points={args.target_price}")

    try:
        # Get access token (cached or fresh)
        logger.info("Getting access token...")
        access_token = await get_access_token(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )
        logger.info("Authentication successful!")

        # Set environment variable for create_fyers_model function
        os.environ["FYERS_APP_ID"] = APP_ID

        # Create Fyers model instance
        fyers = create_fyers_model(access_token)

        # Use provided symbol
        index_symbol = args.symbol

        # Get option chain with expansion if needed
        logger.info(f"Fetching option chain for {index_symbol}...")
        # For LTP range filtering, we need a larger strike range to find options in the LTP range
        # Use closing price as a proxy for the range needed
        strike_range = max(args.closing_price * 0.1, 1000)  # Use 10% of price or 1000, whichever is larger
        option_chain = await get_expanded_option_chain(
            fyers, index_symbol, args.closing_price, strike_range
        )

        # Find nearest option within LTP range
        logger.info(f"Finding option within LTP range {args.min_ltp}-{args.max_ltp}...")
        selected_option = find_nearest_option(
            args.closing_price,
            args.min_ltp,
            args.max_ltp,
            option_chain,
            args.option_type
        )

        if not selected_option:
            logger.error("No suitable option found")
            return

        # Use the LTP from option chain data, or fetch it separately if needed
        option_ltp = selected_option.get("ltp", 0)
        if option_ltp <= 0:
            logger.info("LTP not available in option chain, fetching separately...")
            option_ltp = await get_option_ltp(fyers, selected_option["symbol"])

        if option_ltp <= 0:
            logger.error("Failed to get option LTP")
            return

        # Place the order
        logger.info("Placing order...")
        order_result = await place_order(
            fyers,
            selected_option["symbol"],
            args.direction,
            args.quantity,
            args.sl_price,
            args.target_price,
            option_ltp
        )

        if order_result.get("s") == "ok":
            logger.info("Order placed successfully!")
        else:
            logger.error("Order placement failed!")

    except FyersAuthenticationError as e:
        logger.error(f"Authentication error: {str(e)}")
    except Exception as e:
        logger.error(f"Error during order placement: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())