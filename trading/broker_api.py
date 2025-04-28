"""
Broker API module for AlgoTrading.
Handles authentication and communication with the Fyers API.
"""
import base64
import requests
import pyotp
import json
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs
import pytz
from typing import Dict, List, Tuple, Optional, Union, Any

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from core.logging_setup import get_logger
from core.constants import (
    FYERS_ORDER_TYPE_MARKET,
    FYERS_ORDER_TYPE_LIMIT,
    FYERS_ORDER_SIDE_BUY,
    FYERS_ORDER_SIDE_SELL,
    FYERS_PRODUCT_TYPE_INTRADAY,
    FYERS_VALIDITY_DAY
)

logger = get_logger(__name__)
ist_timezone = pytz.timezone("Asia/Kolkata")


def get_encoded_string(s: str) -> str:
    """
    Encode a string in base64.
    
    Args:
        s: String to encode
        
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(str(s).encode("ascii")).decode("ascii")


def authenticate_fyers(
    app_id: str,
    secret_key: str,
    redirect_uri: str,
    fyers_user: str,
    fyers_pin: str,
    fyers_totp: str,
    response_type: str = "code",
    grant_type: str = "authorization_code",
    max_retries: int = 3
) -> Tuple[fyersModel.FyersModel, data_ws.FyersDataSocket]:
    """
    Authenticate with Fyers API and return API client and websocket.
    
    Args:
        app_id: Fyers app ID
        secret_key: Fyers app secret key
        redirect_uri: Redirect URI for authentication
        fyers_user: Fyers user ID
        fyers_pin: Fyers PIN
        fyers_totp: Fyers TOTP secret
        response_type: Response type for authentication
        grant_type: Grant type for authentication
        max_retries: Maximum number of retries
        
    Returns:
        Tuple of (fyers_client, fyers_socket)
    """
    logger.info(f"Authenticating with Fyers API for user {fyers_user}")
    
    retries = 0
    while retries < max_retries:
        try:
            # Create session model
            session = fyersModel.SessionModel(
                client_id=app_id,
                secret_key=secret_key,
                redirect_uri=redirect_uri,
                response_type=response_type,
                grant_type=grant_type
            )
            
            # Generate auth code
            session.generate_authcode()
            logger.info("Generated auth code")
            
            # Send login OTP
            url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
            res = requests.post(
                url=url_send_login_otp,
                json={"fy_id": get_encoded_string(fyers_user), "app_id": "2"}
            ).json()
            
            if "request_key" not in res:
                logger.error(f"Failed to send login OTP: {res}")
                retries += 1
                time.sleep(2)
                continue
                
            logger.info("Sent login OTP")
            
            # Wait if needed to ensure TOTP is valid
            if datetime.now(ist_timezone).second % 30 > 27:
                logger.info("Waiting for TOTP to refresh")
                time.sleep(5)
            
            # Verify OTP
            url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
            res2 = requests.post(
                url=url_verify_otp,
                json={
                    "request_key": res.get("request_key"),
                    "otp": pyotp.TOTP(fyers_totp).now()
                }
            ).json()
            
            if "request_key" not in res2:
                logger.error(f"Failed to verify OTP: {res2}")
                retries += 1
                time.sleep(2)
                continue
                
            logger.info("Verified OTP")
            
            # Verify PIN
            ses = requests.Session()
            url_verify_pin = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
            payload2 = {
                "request_key": res2.get("request_key"),
                "identity_type": "pin",
                "identifier": get_encoded_string(fyers_pin)
            }
            res3 = ses.post(url=url_verify_pin, json=payload2).json()
            
            if "data" not in res3 or "access_token" not in res3["data"]:
                logger.error(f"Failed to verify PIN: {res3}")
                retries += 1
                time.sleep(2)
                continue
                
            logger.info("Verified PIN")
            
            # Update session headers
            ses.headers.update({'authorization': f"Bearer {res3['data']['access_token']}"})
            
            # Get auth code
            tokenurl = "https://api-t1.fyers.in/api/v3/token"
            payload3 = {
                "fyers_id": fyers_user,
                "app_id": app_id[:-4],
                "redirect_uri": redirect_uri,
                "appType": app_id.split('-')[-1],
                "code_challenge": "",
                "state": "None",
                "scope": "",
                "nonce": "",
                "response_type": response_type,
                "create_cookie": True
            }
            res4 = ses.post(url=tokenurl, json=payload3).json()
            
            if "Url" not in res4:
                logger.error(f"Failed to get token URL: {res4}")
                retries += 1
                time.sleep(2)
                continue
                
            # Extract auth code from URL
            url = res4.get('Url', '')
            parsed = urlparse(url)
            auth_code = parse_qs(parsed.query).get('auth_code', [''])[0]
            
            if not auth_code:
                logger.error(f"Failed to extract auth code from URL: {url}")
                retries += 1
                time.sleep(2)
                continue
                
            logger.info("Got auth code")
            
            # Generate token
            session.set_token(auth_code)
            auth_response = session.generate_token()
            
            if "access_token" not in auth_response:
                logger.error(f"Failed to generate token: {auth_response}")
                retries += 1
                time.sleep(2)
                continue
                
            access_token = auth_response.get("access_token")
            logger.info("Generated access token")
            
            # Create Fyers client
            fyers = fyersModel.FyersModel(client_id=app_id, token=access_token)
            
            # Create websocket token
            ws_token = f"{app_id}:{access_token}"
            fyers_socket = data_ws.FyersDataSocket(access_token=ws_token, log_path="")
            
            logger.info("Authentication successful")
            return fyers, fyers_socket
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            retries += 1
            time.sleep(2)
    
    logger.error(f"Authentication failed after {max_retries} retries")
    raise Exception(f"Authentication failed after {max_retries} retries")


def place_order(
    fyers: fyersModel.FyersModel,
    symbol: str,
    qty: int,
    side: int = FYERS_ORDER_SIDE_BUY,
    order_type: int = FYERS_ORDER_TYPE_MARKET,
    limit_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    product_type: str = FYERS_PRODUCT_TYPE_INTRADAY,
    validity: str = FYERS_VALIDITY_DAY,
    disclosed_qty: int = 0,
    offline_order: bool = False
) -> Dict[str, Any]:
    """
    Place an order via Fyers REST API.
    
    Args:
        fyers: Fyers API client
        symbol: Symbol to trade
        qty: Quantity to trade
        side: Order side (1 for buy, -1 for sell)
        order_type: Order type (1 for limit, 2 for market)
        limit_price: Limit price for limit orders
        stop_price: Stop price for stop orders
        stop_loss: Stop loss price
        take_profit: Take profit price
        product_type: Product type (INTRADAY, MARGIN, CNC)
        validity: Order validity (DAY, IOC)
        disclosed_qty: Disclosed quantity
        offline_order: Whether to place offline order
        
    Returns:
        Order response
    """
    logger.info(f"Placing order: {symbol}, qty={qty}, side={side}, type={order_type}")
    
    # Prepare order data
    data = {
        "symbol": symbol,
        "qty": qty,
        "type": order_type,
        "side": side,
        "productType": product_type,
        "validity": validity,
        "disclosedQty": disclosed_qty,
        "offlineOrder": offline_order
    }
    
    # Add limit price for limit orders
    if order_type == FYERS_ORDER_TYPE_LIMIT and limit_price is not None:
        data["limitPrice"] = limit_price
    
    # Add stop price for stop orders
    if order_type in [3, 4] and stop_price is not None:
        data["stopPrice"] = stop_price
    
    # Add stop loss and take profit if provided
    if stop_loss is not None:
        data["stopLoss"] = stop_loss
    
    if take_profit is not None:
        data["takeProfit"] = take_profit
    
    # Place order
    try:
        response = fyers.place_order(data=data)
        
        if response.get('s') == 'ok':
            order_id = response.get('id', 'unknown')
            logger.info(f"Order placed successfully: ID={order_id}")
        else:
            logger.error(f"Order placement failed: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        return {"s": "error", "message": str(e)}


def get_order_status(
    fyers: fyersModel.FyersModel,
    order_id: str
) -> Dict[str, Any]:
    """
    Fetch status of an existing order.
    
    Args:
        fyers: Fyers API client
        order_id: Order ID to check
        
    Returns:
        Order status response
    """
    logger.info(f"Getting status for order: {order_id}")
    
    try:
        params = {"id": order_id}
        response = fyers.order_book(data=params)
        
        if response.get('s') == 'ok':
            logger.info(f"Got order status for {order_id}")
        else:
            logger.warning(f"Failed to get order status: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error getting order status: {e}")
        return {"s": "error", "message": str(e)}


def cancel_order(
    fyers: fyersModel.FyersModel,
    order_id: str
) -> Dict[str, Any]:
    """
    Cancel an existing order.
    
    Args:
        fyers: Fyers API client
        order_id: Order ID to cancel
        
    Returns:
        Cancel order response
    """
    logger.info(f"Cancelling order: {order_id}")
    
    try:
        params = {"id": order_id}
        response = fyers.cancel_order(data=params)
        
        if response.get('s') == 'ok':
            logger.info(f"Order {order_id} cancelled successfully")
        else:
            logger.warning(f"Failed to cancel order: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        return {"s": "error", "message": str(e)}


def get_positions(
    fyers: fyersModel.FyersModel
) -> Dict[str, Any]:
    """
    Get current positions.
    
    Args:
        fyers: Fyers API client
        
    Returns:
        Positions response
    """
    logger.info("Getting current positions")
    
    try:
        response = fyers.positions()
        
        if response.get('s') == 'ok':
            positions = response.get('netPositions', [])
            logger.info(f"Got {len(positions)} positions")
        else:
            logger.warning(f"Failed to get positions: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        return {"s": "error", "message": str(e)}


def get_funds(
    fyers: fyersModel.FyersModel
) -> Dict[str, Any]:
    """
    Get available funds.
    
    Args:
        fyers: Fyers API client
        
    Returns:
        Funds response
    """
    logger.info("Getting available funds")
    
    try:
        response = fyers.funds()
        
        if response.get('s') == 'ok':
            logger.info("Got funds information")
        else:
            logger.warning(f"Failed to get funds: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error getting funds: {e}")
        return {"s": "error", "message": str(e)}


def subscribe_ticks(
    fyers_socket: data_ws.FyersDataSocket,
    symbols: List[str],
    on_message: callable
) -> None:
    """
    Subscribe to real-time ticks for given symbols.
    
    Args:
        fyers_socket: Fyers websocket client
        symbols: List of symbols to subscribe to
        on_message: Callback function for tick data
    """
    logger.info(f"Subscribing to ticks for {len(symbols)} symbols")
    
    try:
        payload = {"symbol": symbols}
        fyers_socket.subscribe(payload)
        fyers_socket.on_message = on_message
        logger.info(f"Subscribed to: {symbols}")
    except Exception as e:
        logger.error(f"Error subscribing to ticks: {e}")


def unsubscribe_ticks(
    fyers_socket: data_ws.FyersDataSocket,
    symbols: List[str]
) -> None:
    """
    Unsubscribe from real-time ticks.
    
    Args:
        fyers_socket: Fyers websocket client
        symbols: List of symbols to unsubscribe from
    """
    logger.info(f"Unsubscribing from ticks for {len(symbols)} symbols")
    
    try:
        payload = json.dumps({"T": "MSUB", "symbols": symbols, "SLIST": [], "SUB_T": -1})
        fyers_socket.send(payload)
        logger.info(f"Unsubscribed from: {symbols}")
    except Exception as e:
        logger.error(f"Error unsubscribing from ticks: {e}")


def get_market_status(
    fyers: fyersModel.FyersModel
) -> Dict[str, Any]:
    """
    Get current market status.
    
    Args:
        fyers: Fyers API client
        
    Returns:
        Market status response
    """
    logger.info("Getting market status")
    
    try:
        response = fyers.market_status()
        
        if response.get('s') == 'ok':
            logger.info("Got market status")
        else:
            logger.warning(f"Failed to get market status: {response}")
        
        return response
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return {"s": "error", "message": str(e)}
