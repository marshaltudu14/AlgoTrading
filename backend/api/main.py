"""
FastAPI Backend for AlgoTrading Platform
Connects Python authentication services with Next.js frontend
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os
import asyncio
from typing import Dict, Any
from pydantic import BaseModel, Field
import logging

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AlgoTrading API",
    description="API for algorithmic trading platform with Fyers integration",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class AuthRequest(BaseModel):
    app_id: str = Field(..., description="Fyers Application ID")
    secret_key: str = Field(..., description="Fyers Secret Key")
    redirect_uri: str = Field(..., description="Redirect URI for OAuth")
    fy_id: str = Field(..., description="Fyers User ID")
    pin: str = Field(..., description="User PIN")
    totp_secret: str = Field(..., description="TOTP Secret for 2FA")

class AuthResponse(BaseModel):
    success: bool
    access_token: str = None
    profile: Dict[str, Any] = None
    error: str = None

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(status="healthy", message="FastAPI backend is running")

@app.post("/auth/fyers/login", response_model=AuthResponse)
async def login(auth_data: AuthRequest):
    """
    Authenticate user with Fyers API

    Args:
        auth_data: Authentication credentials

    Returns:
        AuthResponse: Contains access token and user profile or error message
    """
    try:
        logger.info(f"Starting authentication for user: {auth_data.fy_id}")

        # Authenticate user
        access_token = await authenticate_fyers_user(
            app_id=auth_data.app_id,
            secret_key=auth_data.secret_key,
            redirect_uri=auth_data.redirect_uri,
            fy_id=auth_data.fy_id,
            pin=auth_data.pin,
            totp_secret=auth_data.totp_secret
        )

        # Get user profile
        profile = await get_user_profile(access_token, auth_data.app_id)

        logger.info(f"Authentication successful for user: {auth_data.fy_id}")

        return AuthResponse(
            success=True,
            access_token=access_token,
            profile=profile
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Authentication failed for user {auth_data.fy_id}: {error_msg}")

        return AuthResponse(
            success=False,
            error=error_msg
        )

@app.get("/user/profile")
async def get_profile(access_token: str, app_id: str):
    """
    Get user profile information

    Args:
        access_token: Valid access token
        app_id: Fyers application ID

    Returns:
        User profile information
    """
    try:
        profile = await get_user_profile(access_token, app_id)
        return {"success": True, "profile": profile}
    except Exception as e:
        logger.error(f"Failed to fetch profile: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/logout")
async def logout():
    """
    Logout user by clearing session data

    Returns:
        Success message for logout
    """
    try:
        # In a real application, you might want to:
        # - Invalidate the token on the broker's side
        # - Clear session data from database
        # - Log the logout activity

        logger.info("User logged out successfully")
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return {"success": True, "message": "Logged out successfully"}  # Always return success for logout

@app.get("/candle-data/{symbol}/{timeframe}")
async def get_candle_data(symbol: str, timeframe: str, start_date: str = None, end_date: str = None):
    """
    Get historical candle data for a symbol and timeframe

    Args:
        symbol: Trading symbol (e.g., NSE:NIFTY50-INDEX)
        timeframe: Timeframe (e.g., 1, 5, 15, 60)
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)

    Returns:
        Historical candle data with OHLCV format
    """
    try:
        import os
        import sys
        import asyncio
        from datetime import datetime, timedelta
        import pandas as pd

        # Add backend to Python path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

        from backend.fetch_candle_data import fetch_candles, get_access_token, create_fyers_model
        from backend.config.fyers_config import APP_ID, SECRET_KEY, REDIRECT_URI, FYERS_USER, FYERS_PIN, FYERS_TOTP

        logger.info(f"Fetching candle data for {symbol} {timeframe}")

        # Get access token (cached or fresh)
        access_token = await get_access_token(
            app_id=APP_ID,
            secret_key=SECRET_KEY,
            redirect_uri=REDIRECT_URI,
            fy_id=FYERS_USER,
            pin=FYERS_PIN,
            totp_secret=FYERS_TOTP
        )

        # Create Fyers model instance
        fyers = create_fyers_model(access_token)

        # Parse dates if provided
        start_dt = None
        end_dt = None

        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Fetch candle data
        df = fetch_candles(fyers, symbol, timeframe, start_dt, end_dt)

        if df.empty:
            return {
                "success": False,
                "error": "No data available for the specified parameters",
                "data": []
            }

        # Convert DataFrame to list of objects (proper format for frontend)
        # Fyers returns: [timestamp, open, high, low, close, volume]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        # Convert to candle data format expected by lightweight-charts
        candle_data = []
        for _, row in df.iterrows():
            candle_data.append({
                time: int(row['datetime']),  # Convert to Unix timestamp
                open: float(row['open']),
                high: float(row['high']),
                low: float(row['low']),
                close: float(row['close'])
            })

        # Sort by time (oldest first for proper chart display)
        candle_data.sort(key=lambda x: x['time'])

        logger.info(f"Successfully fetched {len(candle_data)} candles for {symbol} {timeframe}")

        return {
            "success": True,
            "data": candle_data,
            "count": len(candle_data)
        }

    except Exception as e:
        logger.error(f"Error fetching candle data: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "success": False,
            "error": f"Failed to fetch candle data: {str(e)}",
            "data": []
        }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)