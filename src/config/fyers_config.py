"""
Fyers API Configuration
Contains all hardcoded credentials and settings for Fyers API access
"""

# Fyers API Credentials
APP_ID = "TS79V3NXK1-100"
SECRET_KEY = "KQCPB0FJ74"
REDIRECT_URI = "https://google.com"
FYERS_USER = "XM22383"
FYERS_PIN = "4628"
FYERS_TOTP = "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"

# Trading Configuration
DEFAULT_LOT_SIZE = 50
DEFAULT_PRODUCT_TYPE = "INTRADAY"
DEFAULT_ORDER_TYPE = "MARKET"
DEFAULT_EXCHANGE = "NFO"  # National F&O Exchange

# Index Configuration
INDEX_SYMBOLS = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX"
}

# Option Strike Price Configuration
STRIKE_PRICE_INTERVAL = 50  # Standard interval for Nifty options

# API Configuration
API_BASE_URL = "https://api-t1.fyers.in"
API_VERSION = "v3"