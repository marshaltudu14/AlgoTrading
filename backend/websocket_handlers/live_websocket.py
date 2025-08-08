"""
WebSocket handlers for AlgoTrading System
Handles live trading and backtest WebSocket connections
"""
import json
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from fastapi import WebSocket, WebSocketDisconnect

# Add the backend directory to Python path to enable absolute imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.dependencies import active_backtests, active_live_sessions

# Configure logging
logger = logging.getLogger(__name__)


async def websocket_backtest_handler(websocket: WebSocket, backtest_id: str):
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


async def websocket_live_trading_handler(websocket: WebSocket, user_id: str):
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
        logger.debug(f"WebSocket connected for user {user_id}")

        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "WebSocket connected successfully",
            "user_id": user_id
        }))

        # Get live trading service if it exists
        live_service = active_live_sessions.get(user_id)
        if live_service:
            # Add client to live service for tick data broadcasting
            live_service.add_websocket_client(websocket)
            
            # Send current status
            status = live_service.get_status()
            await websocket.send_text(json.dumps({
                "type": "status",
                "data": status
            }))
        else:
            # Send status indicating no active session
            await websocket.send_text(json.dumps({
                "type": "status",
                "data": {
                    "is_active": False,
                    "message": "No active live trading session"
                }
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
                # Set timeout for receiving messages (30 seconds to align with ping interval)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Update last ping time when we receive any message
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
                # Check if connection is still alive - only log warning if truly timed out
                current_time = asyncio.get_event_loop().time()
                if current_time - last_ping_time > 120:  # 120 seconds total timeout (4 missed pings)
                    logger.warning(f"WebSocket timeout for {user_id} - no activity for {current_time - last_ping_time:.1f} seconds")
                    break
                # No need to log or continue - just retry receiving messages
                pass
                
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
        logger.debug(f"WebSocket disconnected for live trading {user_id}")