"""
Authentication routes for AlgoTrading System
Handles login, logout, and user profile endpoints
"""
import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add the backend directory to Python path to enable absolute imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models.api_models import LoginRequest
from core.dependencies import (
    get_current_user,
    create_jwt_token,
    get_user_action_logger
)
from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["authentication"])


@router.post("/login")
async def login(request: LoginRequest):
    """Authenticate user with Fyers credentials"""
    try:
        logger.info(f"Login attempt for user: {request.fy_id}")
        
        # Call the refactored authentication function
        access_token = await authenticate_fyers_user(
            app_id=request.app_id,
            secret_key=request.secret_key,
            redirect_uri=request.redirect_uri,
            fy_id=request.fy_id,
            pin=request.pin,
            totp_secret=request.totp_secret
        )
        
        # Create JWT session token
        session_token = create_jwt_token(request.fy_id, access_token, request.app_id)
        
        user_logger = get_user_action_logger()
        if user_logger:
            user_logger.log_action(request.fy_id, "LOGIN", {"status": "success"})
        logger.info(f"Login successful for user: {request.fy_id}")

        # Create response with HTTP-only cookie
        response = JSONResponse(content={
            "success": True,
            "message": "Login successful"
        })

        # Set HTTP-only cookie for authentication
        response.set_cookie(
            key="auth_token",
            value=session_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            max_age=86400  # 24 hours
        )

        return response

    except Exception as e:
        logger.error(f"Login failed for user {request.fy_id}: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )


@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user and clear session"""
    try:
        user_id = current_user["user_id"]

        # Create response and clear cookie
        response = JSONResponse(content={
            "success": True,
            "message": "Logged out successfully"
        })

        # Clear the auth cookie
        response.delete_cookie(key="auth_token")

        user_logger = get_user_action_logger()
        if user_logger:
            user_logger.log_action(user_id, "LOGOUT", {"status": "success"})
        logger.info(f"User {user_id} logged out successfully")
        return response

    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Logout failed: {str(e)}"
        )


@router.get("/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile and capital information"""
    try:
        user_id = current_user["user_id"]
        access_token = current_user["access_token"]
        app_id = current_user["app_id"]
        
        # Get user profile from Fyers API
        profile_data = await get_user_profile(
            access_token=access_token,
            app_id=app_id
        )
        
        return {
            "user_id": user_id,
            "name": profile_data.get("name", "User"),
            "capital": profile_data.get("capital", 0),
            "login_time": datetime.utcnow().isoformat()  # Use current time as login time for display
        }
        
    except Exception as e:
        logger.error(f"Failed to get profile for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profile: {str(e)}"
        )