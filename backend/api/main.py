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