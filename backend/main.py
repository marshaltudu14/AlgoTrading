"""
FastAPI Backend for AlgoTrading System
"""
import multiprocessing
import os

if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('forkserver', force=True)
    except RuntimeError:
        pass # Already set

import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException, Depends, status, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import jwt
import asyncio
import json
from datetime import datetime, timedelta
import logging

from fyers_apiv3 import fyersModel

# Import our refactored auth module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile
from trading.backtest_service import BacktestService
from trading.live_trading_service import LiveTradingService

# Import existing trading components
from backtesting.environment import TradingEnv, TradingMode
from utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlgoTrading API",
    description="Backend API for AlgoTrading System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = "your-secret-key-change-in-production"  # TODO: Use environment variable
JWT_ALGORITHM = "HS256"
active_backtests: Dict[str, BacktestService] = {}
active_live_sessions: Dict[str, LiveTradingService] = {}

# Pydantic models
class LoginRequest(BaseModel):
    app_id: str
    secret_key: str
    redirect_uri: str
    fy_id: str
    pin: str
    totp_secret: str

class BacktestRequest(BaseModel):
    instrument: str
    timeframe: str
    duration: int
    initial_capital: float = 100000

class LiveTradingRequest(BaseModel):
    instrument: str
    timeframe: str
    option_strategy: Optional[str] = "ITM"

# Helper functions
def create_jwt_token(user_id: str, access_token: str, app_id: str) -> str:
    """Create JWT token for session management"""
    payload = {
        "user_id": user_id,
        "access_token": access_token,
        "app_id": app_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(request: Request):
    """Dependency to get current authenticated user from cookie"""
    # Get token from cookie
    token = request.cookies.get("auth_token")

    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")

    payload = verify_jwt_token(token)
    user_id = payload.get("user_id")
    access_token = payload.get("access_token")
    app_id = payload.get("app_id")

    if not user_id or not access_token or not app_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    return {"user_id": user_id, "access_token": access_token, "app_id": app_id}

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/funds")
async def get_funds(current_user: dict = Depends(get_current_user)):
    """Get funds information, including realized P&L"""
    try:
        user_id = current_user["user_id"]
        access_token = current_user["access_token"]
        app_id = current_user["app_id"]

        fyers = fyersModel.FyersModel(
            client_id=app_id,
            token=access_token,
            is_async=True,
        )
        
        funds_response = await fyers.funds()
        
        realized_pnl = 0
        total_balance = 0
        if funds_response and funds_response.get("s") == "ok" and "fund_limit" in funds_response:
            for item in funds_response["fund_limit"]:
                if item.get("title") == "Realized Profit and Loss":
                    realized_pnl = item.get("equityAmount", 0)
                elif item.get("title") == "Total Balance":
                    total_balance = item.get("equityAmount", 0)
        
        return {"todayPnL": realized_pnl, "totalFunds": total_balance}

    except Exception as e:
        logger.error(f"Failed to get funds for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch funds: {str(e)}"
        )

@app.get("/api/metrics")
async def get_metrics():
    """Get trading metrics from metrics.json"""
    metrics_file_path = Path("metrics.json")
    if not metrics_file_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    
    try:
        with open(metrics_file_path, "r") as f:
            metrics_data = json.load(f)
        return metrics_data
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding metrics.json")
    except Exception as e:
        logger.error(f"Failed to read metrics.json: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to read metrics: {str(e)}")

@app.post("/api/login")
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

@app.post("/api/logout")
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

        logger.info(f"User {user_id} logged out successfully")
        return response

    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Logout failed: {str(e)}"
        )

@app.get("/api/profile")
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user profile and capital information"""
    try:
        user_id = current_user["user_id"]
        access_token = current_user["access_token"]
        
        # Get user profile from Fyers API
        profile_data = await get_user_profile(
            access_token=access_token,
            app_id=current_user["app_id"]
        )
        
        return {
            "user_id": user_id,
            "name": profile_data.get("name", "User"),
            "capital": profile_data.get("capital", 0),
            "login_time": datetime.utcnow().isoformat() # Use current time as login time for display
        }
        
    except Exception as e:
        logger.error(f"Failed to get profile for user {current_user['user_id']}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch profile: {str(e)}"
        )

@app.post("/api/backtest")
async def start_backtest(
    request: BacktestRequest,
    current_user: dict = Depends(get_current_user)
):
    """Start a new backtest"""
    try:
        user_id = current_user["user_id"]
        access_token = current_user["access_token"]
        app_id = current_user["app_id"]

        # Create backtest service
        backtest_service = BacktestService(
            user_id=user_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            duration=request.duration,
            initial_capital=request.initial_capital,
            access_token=access_token,
            app_id=app_id
        )
        
        # Generate backtest ID
        backtest_id = f"{user_id}_{datetime.utcnow().timestamp()}"
        
        # Store active backtest
        active_backtests[backtest_id] = backtest_service
        
        # Start backtest in background
        asyncio.create_task(backtest_service.run())
        
        logger.info(f"Started backtest {backtest_id} for user {user_id}")
        
        return {
            "backtest_id": backtest_id,
            "status": "started",
            "message": "Backtest initiated successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start backtest: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start backtest: {str(e)}"
        )

@app.post("/api/live/start")
async def start_live_trading(
    request: LiveTradingRequest,
    current_user: dict = Depends(get_current_user)
):
    """Start live trading session"""
    try:
        user_id = current_user["user_id"]
        access_token = current_user["access_token"]
        app_id = current_user["app_id"]
        
        # Check if user already has an active live session
        if user_id in active_live_sessions:
            raise HTTPException(
                status_code=400,
                detail="Live trading session already active"
            )
        
        # Create live trading service
        live_service = LiveTradingService(
            user_id=user_id,
            access_token=access_token,
            app_id=app_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            option_strategy=request.option_strategy
        )
        
        # Store active session
        active_live_sessions[user_id] = live_service
        
        # Start live trading in background
        asyncio.create_task(live_service.run())
        
        logger.info(f"Started live trading for user {user_id}")
        
        return {
            "status": "started",
            "message": "Live trading started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start live trading: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start live trading: {str(e)}"
        )

@app.post("/api/live/stop")
async def stop_live_trading(current_user: dict = Depends(get_current_user)):
    """Stop live trading session"""
    try:
        user_id = current_user["user_id"]
        
        # Get active session
        live_service = active_live_sessions.get(user_id)
        if not live_service:
            raise HTTPException(
                status_code=400,
                detail="No active live trading session"
            )
        
        # Stop the service
        await live_service.stop()
        
        # Remove from active sessions
        del active_live_sessions[user_id]
        
        logger.info(f"Stopped live trading for user {user_id}")
        
        return {
            "status": "stopped",
            "message": "Live trading stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop live trading: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop live trading: {str(e)}"
        )

# WebSocket endpoints

@app.websocket("/ws/backtest/{backtest_id}")
async def websocket_backtest(websocket: WebSocket, backtest_id: str):
    """WebSocket endpoint for backtest progress updates"""
    await websocket.accept()

    try:
        # Get backtest service
        backtest_service = active_backtests.get(backtest_id)
        if not backtest_service:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Backtest not found"
            }))
            return

        # Add client to backtest service
        backtest_service.add_websocket_client(websocket)

        logger.info(f"WebSocket connected for backtest {backtest_id}")

        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "WebSocket connected successfully",
            "backtest_id": backtest_id
        }))

        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (ping/pong)
                message = await websocket.receive_text()

                # Handle ping/pong or other client messages
                if message == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for backtest {backtest_id}: {str(e)}")
        import traceback
        logger.error(f"WebSocket traceback: {traceback.format_exc()}")
    finally:
        # Remove client from backtest service
        if backtest_service:
            backtest_service.remove_websocket_client(websocket)
        logger.info(f"WebSocket disconnected for backtest {backtest_id}")

@app.websocket("/ws/live/{user_id}")
async def websocket_live_trading(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for live trading updates"""
    await websocket.accept()

    try:
        # Get live trading service
        live_service = active_live_sessions.get(user_id)
        if not live_service:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Live trading session not found"
            }))
            return

        # Add client to live service
        live_service.add_websocket_client(websocket)

        logger.info(f"WebSocket connected for live trading {user_id}")

        # Send current status
        status = live_service.get_status()
        await websocket.send_text(json.dumps({
            "type": "status",
            "data": status
        }))

        # Keep connection alive
        while True:
            try:
                # Wait for messages from client
                message = await websocket.receive_text()

                # Handle ping/pong or other client messages
                if message == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error for live trading {user_id}: {e}")
    finally:
        # Remove client from live service
        if live_service:
            live_service.remove_websocket_client(websocket)
        logger.info(f"WebSocket disconnected for live trading {user_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
