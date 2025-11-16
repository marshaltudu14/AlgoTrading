"""
Token Management Utility
Handles caching and reuse of Fyers access tokens
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class TokenManager:
    """Manages Fyers access token caching and refresh"""

    def __init__(self, token_file: str = "fyers_token.json"):
        self.token_file = token_file
        self.access_token = None
        self.token_expiry = None

    def load_token(self) -> Optional[str]:
        """Load token from file if it exists and is not expired"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)

                # Check if token is expired (tokens typically last for 24 hours)
                expiry_time = datetime.fromisoformat(token_data.get('expiry'))
                if datetime.now() < expiry_time:
                    self.access_token = token_data.get('access_token')
                    self.token_expiry = expiry_time
                    logger.info(f"Loaded valid token from cache, expires at: {expiry_time}")
                    return self.access_token
                else:
                    logger.info("Token expired, will fetch new token")
                    return None
            else:
                logger.info("No token file found, will fetch new token")
                return None

        except Exception as e:
            logger.error(f"Error loading token: {e}")
            return None

    def save_token(self, access_token: str, expiry_hours: int = 24) -> None:
        """Save token to file with expiry time"""
        try:
            expiry_time = datetime.now() + timedelta(hours=expiry_hours)
            token_data = {
                'access_token': access_token,
                'expiry': expiry_time.isoformat()
            }

            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)

            self.access_token = access_token
            self.token_expiry = expiry_time
            logger.info(f"Saved token to cache, expires at: {expiry_time}")

        except Exception as e:
            logger.error(f"Error saving token: {e}")

    def get_valid_token(self) -> Optional[str]:
        """Get valid token, loading from cache or returning None if expired"""
        if not self.access_token or (self.token_expiry and datetime.now() >= self.token_expiry):
            return self.load_token()
        return self.access_token

    def validate_token(self, access_token: str) -> bool:
        """Validate token by making a profile API call"""
        try:
            from fyers_apiv3 import fyersModel

            # Get app_id from environment
            app_id = os.getenv("FYERS_APP_ID")
            if not app_id:
                logger.error("FYERS_APP_ID not found in environment")
                return False

            # Create Fyers model and test the token
            fyers = fyersModel.FyersModel(client_id=app_id, token=access_token)
            response = fyers.get_profile()

            # Check if the response is successful
            if response.get("code") == 200 and response.get("s") == "ok":
                logger.info("Token validation successful")
                return True
            else:
                logger.warning(f"Token validation failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False

    def clear_token(self) -> None:
        """Clear cached token"""
        self.access_token = None
        self.token_expiry = None
        if os.path.exists(self.token_file):
            os.remove(self.token_file)
            logger.info("Cleared cached token")

# Global token manager instance
token_manager = TokenManager()

def get_cached_token(auto_refresh: bool = True) -> Optional[str]:
    """Get cached access token, with optional auto-refresh and validation"""
    token = token_manager.get_valid_token()

    # If we have a token, validate it
    if token:
        # Set app_id in environment for validation
        try:
            from ..config.fyers_config import APP_ID
            os.environ["FYERS_APP_ID"] = APP_ID
        except ImportError:
            pass

        # Validate the token
        if not token_manager.validate_token(token):
            logger.info("Token validation failed, clearing cached token")
            token_manager.clear_token()
            token = None

    # If no valid token and auto_refresh is enabled, get a new one
    if not token and auto_refresh:
        logger.info("Token expired/invalid/missing, attempting auto-refresh...")
        try:
            # Import here to avoid circular imports
            from ..config.fyers_config import APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP
            import asyncio
            from .fyers_auth_service import authenticate_fyers_user

            # Get new token
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                new_token = loop.run_until_complete(authenticate_fyers_user(
                    app_id=APP_ID,
                    secret_key=SECRET_KEY,
                    redirect_uri=REDIRECT_URI,
                    fy_id=FYERS_USER,
                    pin=FYERS_PIN,
                    totp_secret=FYERS_TOTP
                ))
                # Save new token
                save_token_to_cache(new_token)
                logger.info("Successfully auto-refreshed token")
                return new_token
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to auto-refresh token: {e}")
            return None
    return token

def save_token_to_cache(access_token: str, expiry_hours: int = 24) -> None:
    """Save access token to cache"""
    token_manager.save_token(access_token, expiry_hours)

def clear_cached_token() -> None:
    """Clear cached token"""
    token_manager.clear_token()