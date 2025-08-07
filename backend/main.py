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
import yaml

from fyers_apiv3 import fyersModel

# Import our refactored auth module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from auth.fyers_auth_service import authenticate_fyers_user, get_user_profile
from trading.backtest_service import BacktestService
from trading.live_trading_service import LiveTradingService

# Import existing trading components
from backtesting.environment import TradingEnv, TradingMode
from utils.data_loader import DataLoader
from utils.user_action_logger import UserActionLogger
from utils.realtime_data_loader import RealtimeDataLoader

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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://*.vercel.app",
        "https://vercel.app"
    ],  # Frontend URLs for local development and Vercel deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
active_backtests: Dict[str, BacktestService] = {}
active_live_sessions: Dict[str, LiveTradingService] = {}
user_action_logger = UserActionLogger()

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

class ManualTradeRequest(BaseModel):
    instrument: str
    direction: str
    quantity: int
    stopLoss: Optional[float] = None
    target: Optional[float] = None

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
    except jwt.PyJWTError:
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

@app.get("/api/config")
async def get_config():
    """Get configuration data including instruments and timeframes"""
    try:
        # Get the path to the config file (relative to project root)
        config_path = Path(__file__).parent.parent / "config" / "instruments.yaml"
        
        # Read and parse the YAML file
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        # Validate that required keys exist
        if 'instruments' not in config_data:
            raise HTTPException(
                status_code=500,
                detail="Invalid configuration: 'instruments' key missing from config file"
            )
        
        if 'timeframes' not in config_data:
            raise HTTPException(
                status_code=500,
                detail="Invalid configuration: 'timeframes' key missing from config file"
            )
        
        return {
            "instruments": config_data["instruments"],
            "timeframes": config_data["timeframes"]
        }
        
    except FileNotFoundError:
        logger.error("Configuration file not found: config/instruments.yaml")
        raise HTTPException(
            status_code=500,
            detail="Configuration file not found"
        )
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse configuration file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Invalid YAML configuration: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error reading configuration: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read configuration: {str(e)}"
        )

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
        
        user_action_logger.log_action(request.fy_id, "LOGIN", {"status": "success"})
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

        user_action_logger.log_action(user_id, "LOGOUT", {"status": "success"})
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
        
        user_action_logger.log_action(user_id, "LIVE_TRADING_STARTED", {"instrument": request.instrument, "timeframe": request.timeframe})
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
        
        user_action_logger.log_action(user_id, "LIVE_TRADING_STOPPED", {"status": "success"})
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

@app.get("/api/historical-data")
async def get_historical_data(
    instrument: str,
    timeframe: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get historical candlestick data for a specified instrument and timeframe.
    Requires JWT authentication.
    """
    try:
        # Validate required parameters
        if not instrument:
            raise HTTPException(
                status_code=422,
                detail="Instrument parameter is required"
            )
        
        if not timeframe:
            raise HTTPException(
                status_code=422,
                detail="Timeframe parameter is required"
            )
        
        logger.info(f"Fetching historical data for user {current_user['user_id']}: {instrument}, {timeframe}")
        
        # Initialize RealtimeDataLoader
        data_loader = RealtimeDataLoader()
        
        # Fetch and process data using the instrument and timeframe parameters
        historical_data = data_loader.fetch_and_process_data(
            symbol=instrument,
            timeframe=timeframe
        )
        
        if historical_data is None or historical_data.empty:
            logger.error(f"No historical data found for {instrument} with timeframe {timeframe}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch historical data for {instrument}"
            )
        
        # Convert DataFrame to JSON format expected by frontend
        # Reset index to include datetime as a column for conversion
        historical_data_reset = historical_data.reset_index()
        
        # Ensure we have the required OHLCV columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in historical_data_reset.columns]
        
        if not available_columns:
            logger.error(f"No OHLCV columns found in data for {instrument}")
            raise HTTPException(
                status_code=500,
                detail="Historical data missing required OHLCV columns"
            )
        
        # Create response data with time, open, high, low, close, volume
        response_data = []
        for _, row in historical_data_reset.iterrows():
            candle = {
                "time": row.get('datetime', row.name).isoformat() if hasattr(row.get('datetime', row.name), 'isoformat') else str(row.get('datetime', row.name)),
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0))
            }
            response_data.append(candle)
        
        logger.info(f"Successfully fetched {len(response_data)} candles for {instrument}")
        
        return response_data
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Failed to fetch historical data for {instrument}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while fetching historical data: {str(e)}"
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

@app.post("/api/manual-trade")
async def manual_trade(
    request: ManualTradeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Initiate a manual trade"""
    try:
        user_id = current_user["user_id"]
        live_service = active_live_sessions.get(user_id)

        if not live_service:
            raise HTTPException(
                status_code=400,
                detail="Live trading session not active"
            )

        await live_service.initiate_manual_trade(
            instrument=request.instrument,
            direction=request.direction,
            quantity=request.quantity,
            sl=request.stopLoss,
            tp=request.target,
            user_id=user_id
        )

        user_action_logger.log_action(user_id, "MANUAL_TRADE_INITIATED", {"instrument": request.instrument, "direction": request.direction, "quantity": request.quantity})
        return {"status": "success", "message": "Manual trade initiated"}

    except Exception as e:
        logger.error(f"Failed to initiate manual trade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate manual trade: {str(e)}"
        )

@app.websocket("/ws/live/{user_id}")
async def websocket_live_trading(websocket: WebSocket, user_id: str):
    """
    WebSocket endpoint for live trading updates and tick data streaming.
    
    Handles different message types:
    - "tick": Raw market tick data for frontend charting
    - "position_update": Position state changes (open, close, SL/TP updates)
    - "trade_executed": Trade execution notifications
    - "stats_update": Trading statistics updates
    - "ping"/"pong": Keep-alive mechanism
    """
    await websocket.accept()
    
    # WebSocket manager for this connection
    last_ping_time = asyncio.get_event_loop().time()

    try:
        # Get live trading service
        live_service = active_live_sessions.get(user_id)
        if not live_service:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Live trading session not found"
            }))
            return

        # Add client to live service for tick data broadcasting
        live_service.add_websocket_client(websocket)

        logger.info(f"WebSocket connected for live trading {user_id}")

        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "WebSocket connected successfully",
            "user_id": user_id
        }))

        # Send current status
        status = live_service.get_status()
        await websocket.send_text(json.dumps({
            "type": "status",
            "data": status
        }))

        # Create keep-alive task
        async def keep_alive():
            """Send periodic pings to keep connection alive"""
            while True:
                try:
                    await asyncio.sleep(30)  # Send ping every 30 seconds
                    await websocket.send_text(json.dumps({
                        "type": "ping",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                except Exception as e:
                    logger.warning(f"Keep-alive ping failed for {user_id}: {e}")
                    break

        # Start keep-alive task
        keep_alive_task = asyncio.create_task(keep_alive())

        # Keep connection alive and handle client messages
        while True:
            try:
                # Set timeout for receiving messages (60 seconds)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
                
                # Update last ping time
                last_ping_time = asyncio.get_event_loop().time()

                # Handle different message types
                if message == "ping":
                    await websocket.send_text("pong")
                elif message == "pong":
                    # Client responded to our ping
                    logger.debug(f"Received pong from {user_id}")
                else:
                    try:
                        # Try to parse as JSON for structured messages
                        msg_data = json.loads(message)
                        msg_type = msg_data.get("type", "unknown")
                        
                        if msg_type == "pong":
                            logger.debug(f"Received structured pong from {user_id}")
                        else:
                            logger.debug(f"Received message type '{msg_type}' from {user_id}")
                    except json.JSONDecodeError:
                        # Handle non-JSON messages
                        logger.debug(f"Received non-JSON message from {user_id}: {message}")

            except asyncio.TimeoutError:
                # Check if connection is still alive
                current_time = asyncio.get_event_loop().time()
                if current_time - last_ping_time > 90:  # 90 seconds timeout
                    logger.warning(f"WebSocket timeout for {user_id}")
                    break
                continue
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message for {user_id}: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for live trading {user_id}: {e}")
    finally:
        # Cancel keep-alive task
        if 'keep_alive_task' in locals():
            keep_alive_task.cancel()
            
        # Remove client from live service
        if 'live_service' in locals() and live_service:
            live_service.remove_websocket_client(websocket)
        logger.info(f"WebSocket disconnected for live trading {user_id}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
