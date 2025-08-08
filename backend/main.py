"""
FastAPI Backend for AlgoTrading System - Main Orchestrator
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

# Add the backend directory to Python path to enable absolute imports
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import our refactored auth module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modular components
from routes.auth_routes import router as auth_router
from routes.config_routes import router as config_router
from routes.trading_routes import router as trading_router
from websocket_handlers.live_websocket import (
    websocket_backtest_handler,
    websocket_live_trading_handler
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
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

# Register route modules
app.include_router(auth_router)
app.include_router(config_router)
app.include_router(trading_router)

# WebSocket endpoints
@app.websocket("/ws/backtest/{backtest_id}")
async def websocket_backtest(websocket: WebSocket, backtest_id: str):
    """WebSocket endpoint for backtest progress updates"""
    await websocket_backtest_handler(websocket, backtest_id)

@app.websocket("/ws/live/{user_id}")
async def websocket_live_trading(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for live trading updates and tick data streaming"""
    await websocket_live_trading_handler(websocket, user_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
