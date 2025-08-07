"""
Refactored Fyers Authentication Service
Converts the hardcoded authentication script into reusable functions
"""
import base64
import requests
import pyotp
import time
import asyncio
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
import pandas as pd
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FyersAuthenticationError(Exception):
    """Custom exception for Fyers authentication errors"""
    pass

def getEncodedString(string: str) -> str:
    """Encode string to base64"""
    string = str(string)
    base64_bytes = base64.b64encode(string.encode("ascii"))
    return base64_bytes.decode("ascii")

async def authenticate_fyers_user(
    redirect_uri: str,
    fy_id: str,
    pin: str
) -> str:
    """
    Authenticate user with Fyers API and return access token
    
    Args:
        app_id: Fyers application ID
        secret_key: Fyers secret key
        redirect_uri: Redirect URI for OAuth
        fy_id: Fyers user ID
        pin: User PIN
        totp_secret: TOTP secret for 2FA
        
    Returns:
        str: Access token for API calls
        
    Raises:
        FyersAuthenticationError: If authentication fails
    """
    try:
        logger.info(f"Starting authentication for user: {fy_id}")
        
        app_id = os.getenv("FYERS_APP_ID")
        secret_key = os.getenv("FYERS_SECRET_KEY")
        totp_secret = os.getenv("FYERS_TOTP_SECRET")

        if not all([app_id, secret_key, totp_secret]):
            raise FyersAuthenticationError("Fyers API credentials (APP_ID, SECRET_KEY, TOTP_SECRET) must be set as environment variables.")

        # Create session model
        session = fyersModel.SessionModel(
            client_id=app_id,
            secret_key=secret_key,
            redirect_uri=redirect_uri,
            response_type="code",
            grant_type="authorization_code"
        )
        
        if session is None:
            raise FyersAuthenticationError("Failed to create session model")
        
        # Generate auth code
        session.generate_authcode()
        
        # Step 1: Send login OTP
        url_send_login_otp = "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2"
        otp_payload = {
            "fy_id": getEncodedString(fy_id),
            "app_id": "2"
        }
        
        logger.info("Sending login OTP...")
        res = requests.post(url=url_send_login_otp, json=otp_payload).json()
        
        if not res.get("request_key"):
            raise FyersAuthenticationError(f"Failed to send OTP: {res}")
        
        # Wait if we're too close to TOTP expiry
        if datetime.now().second % 30 > 27:
            logger.info("Waiting for TOTP refresh...")
            await asyncio.sleep(5)
        
        # Step 2: Verify OTP
        url_verify_otp = "https://api-t2.fyers.in/vagator/v2/verify_otp"
        totp_code = pyotp.TOTP(totp_secret).now()
        
        otp_verify_payload = {
            "request_key": res["request_key"],
            "otp": totp_code
        }
        
        logger.info("Verifying OTP...")
        res2 = requests.post(url=url_verify_otp, json=otp_verify_payload).json()
        
        if not res2.get("request_key"):
            raise FyersAuthenticationError(f"Failed to verify OTP: {res2}")
        
        # Step 3: Verify PIN
        ses = requests.Session()
        url_verify_pin = "https://api-t2.fyers.in/vagator/v2/verify_pin_v2"
        pin_payload = {
            "request_key": res2["request_key"],
            "identity_type": "pin",
            "identifier": getEncodedString(pin)
        }
        
        logger.info("Verifying PIN...")
        res3 = ses.post(url=url_verify_pin, json=pin_payload).json()
        
        if not res3.get("data", {}).get("access_token"):
            raise FyersAuthenticationError(f"Failed to verify PIN: {res3}")
        
        # Update session headers
        ses.headers.update({
            'authorization': f"Bearer {res3['data']['access_token']}"
        })
        
        # Step 4: Get authorization code
        tokenurl = "https://api-t1.fyers.in/api/v3/token"
        token_payload = {
            "fyers_id": fy_id,
            "app_id": app_id[:-4],  # Remove last 4 characters
            "redirect_uri": redirect_uri,
            "appType": "100",
            "code_challenge": "",
            "state": "None",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True
        }
        
        logger.info("Getting authorization code...")
        res4 = ses.post(url=tokenurl, json=token_payload).json()
        
        if not res4.get("Url"):
            raise FyersAuthenticationError(f"Failed to get authorization URL: {res4}")
        
        # Parse authorization code from URL
        url = res4['Url']
        parsed = urlparse(url)
        auth_code = parse_qs(parsed.query)['auth_code'][0]
        
        # Step 5: Generate access token
        session.set_token(auth_code)
        auth_response = session.generate_token()
        
        if not auth_response.get("access_token"):
            raise FyersAuthenticationError(f"Failed to generate access token: {auth_response}")
        
        access_token = auth_response["access_token"]
        
        logger.info(f"Authentication successful for user: {fy_id}")
        return access_token
        
    except Exception as e:
        logger.error(f"Authentication failed for user {fy_id}: {str(e)}")
        if isinstance(e, FyersAuthenticationError):
            raise
        else:
            raise FyersAuthenticationError(f"Unexpected error during authentication: {str(e)}")

async def get_user_profile(access_token: str) -> Dict[str, Any]:
    """
    Get user profile information from Fyers API
    
    Args:
        access_token: Valid access token
        app_id: Fyers application ID
        
    Returns:
        Dict containing user profile data
        
    Raises:
        FyersAuthenticationError: If profile fetch fails
    """
    try:
        logger.info("Fetching user profile...")
        
        app_id = os.getenv("FYERS_APP_ID")
        if not app_id:
            raise FyersAuthenticationError("Fyers APP_ID must be set as an environment variable.")

        # Create Fyers model instance
        fyers = fyersModel.FyersModel(client_id=app_id, token=access_token)
        
        # Get profile data
        profile_response = fyers.get_profile()
        
        if profile_response.get("code") != 200:
            raise FyersAuthenticationError(f"Failed to fetch profile: {profile_response}")
        
        profile_data = profile_response.get("data", {})
        
        # Get funds information
        funds_response = fyers.funds()
        funds_data = {}
        capital = 0
        available_balance = 0

        if funds_response.get("code") == 200:
            fund_limit = funds_response.get("fund_limit", [])

            # Find "Available Balance" from fund_limit array
            for fund_item in fund_limit:
                if fund_item.get("title") == "Available Balance":
                    available_balance = fund_item.get("equityAmount", 0)
                elif fund_item.get("title") == "Total Balance":
                    capital = fund_item.get("equityAmount", 0)

            # If no specific balance found, use first item
            if capital == 0 and fund_limit:
                capital = fund_limit[0].get("equityAmount", 0)
            if available_balance == 0:
                available_balance = capital

        # Combine profile and funds data
        result = {
            "name": profile_data.get("name", "User"),
            "email": profile_data.get("email_id", ""),
            "mobile": profile_data.get("mobile_number", ""),
            "capital": capital,
            "available_balance": available_balance,
            "profile_data": profile_data,
            "funds_data": funds_response.get("fund_limit", [])
        }
        
        logger.info("Profile fetched successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch profile: {str(e)}")
        if isinstance(e, FyersAuthenticationError):
            raise
        else:
            raise FyersAuthenticationError(f"Unexpected error fetching profile: {str(e)}")

def create_fyers_model(access_token: str) -> fyersModel.FyersModel:
    """
    Create a Fyers model instance for API calls
    
    Args:
        access_token: Valid access token
        app_id: Fyers application ID
        
    Returns:
        FyersModel instance
    """
    app_id = os.getenv("FYERS_APP_ID")
    if not app_id:
        raise FyersAuthenticationError("Fyers APP_ID must be set as an environment variable.")
    return fyersModel.FyersModel(client_id=app_id, token=access_token)

def create_fyers_websocket(access_token: str, log_path: str = "") -> data_ws.FyersDataSocket:
    """
    Create a Fyers WebSocket instance for real-time data
    
    Args:
        access_token: Valid access token
        app_id: Fyers application ID
        log_path: Path for WebSocket logs
        
    Returns:
        FyersDataSocket instance
    """
    app_id = os.getenv("FYERS_APP_ID")
    if not app_id:
        raise FyersAuthenticationError("Fyers APP_ID must be set as an environment variable.")
    ws_token = f"{app_id}:{access_token}"
    return data_ws.FyersDataSocket(access_token=ws_token, log_path=log_path)

# For backward compatibility, keep the original global variables as None
# These will be set by the calling code if needed
fyers = None
fyers_socket = None
