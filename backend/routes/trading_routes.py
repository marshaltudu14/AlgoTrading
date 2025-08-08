"""
Trading routes for AlgoTrading System
Handles backtest, live trading, and manual trade endpoints
"""
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add the backend directory to Python path to enable absolute imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from models.api_models import BacktestRequest, LiveTradingRequest, ManualTradeRequest, TradingMode
from core.dependencies import (
    get_current_user,
    active_backtests,
    active_live_sessions,
    get_user_action_logger
)
# Lazy import trading services to avoid circular imports
def get_backtest_service():
    from trading.backtest_service import BacktestService
    return BacktestService

def get_live_trading_service():
    from trading.live_trading_service import LiveTradingService
    return LiveTradingService

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["trading"])


@router.post("/backtest")
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
        BacktestService = get_backtest_service()
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


@router.post("/live/start")
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
        LiveTradingService = get_live_trading_service()
        live_service = LiveTradingService(
            user_id=user_id,
            access_token=access_token,
            app_id=app_id,
            instrument=request.instrument,
            timeframe=request.timeframe,
            option_strategy=request.option_strategy,
            trading_mode=request.trading_mode
        )
        
        # Store active session
        active_live_sessions[user_id] = live_service
        
        # Start live trading in background
        asyncio.create_task(live_service.run())
        
        user_logger = get_user_action_logger()
        if user_logger:
            user_logger.log_action(user_id, "LIVE_TRADING_STARTED", {
                "instrument": request.instrument, 
                "timeframe": request.timeframe,
                "trading_mode": request.trading_mode.value
            })
        logger.info(f"Started {request.trading_mode.value} trading for user {user_id}")
        
        return {
            "status": "started",
            "message": f"{request.trading_mode.value.title()} trading started successfully",
            "trading_mode": request.trading_mode.value.upper()
        }
        
    except Exception as e:
        logger.error(f"Failed to start live trading: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start live trading: {str(e)}"
        )


@router.post("/live/stop")
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
        
        user_logger = get_user_action_logger()
        if user_logger:
            user_logger.log_action(user_id, "LIVE_TRADING_STOPPED", {"status": "success"})
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


@router.post("/manual-trade")
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
            user_id=user_id,
            trading_mode=request.trading_mode
        )

        user_logger = get_user_action_logger()
        if user_logger:
            user_logger.log_action(user_id, "MANUAL_TRADE_INITIATED", {
                "instrument": request.instrument, 
                "direction": request.direction, 
                "quantity": request.quantity,
                "trading_mode": request.trading_mode.value
            })
        return {
            "status": "success", 
            "message": f"Manual {request.trading_mode.value} trade initiated",
            "trading_mode": request.trading_mode.value.upper()
        }

    except Exception as e:
        logger.error(f"Failed to initiate manual trade: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate manual trade: {str(e)}"
        )