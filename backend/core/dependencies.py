"""
Core dependencies for AlgoTrading System
JWT authentication, global state management, and shared dependencies
"""
import os
import sys
import jwt
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import these services only when needed to avoid circular imports
# from trading.backtest_service import BacktestService
# from trading.live_trading_service import LiveTradingService
# from utils.user_action_logger import UserActionLogger

# Configure logging
logger = logging.getLogger(__name__)

# Security configuration
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"

# Global state management - using Any type temporarily to avoid import issues
active_backtests: Dict[str, Any] = {}
active_live_sessions: Dict[str, Any] = {}

# Initialize user_action_logger when first needed
_user_action_logger = None

def get_user_action_logger():
    """Lazy load the user action logger to avoid import issues"""
    global _user_action_logger
    if _user_action_logger is None:
        try:
            from utils.user_action_logger import UserActionLogger
            _user_action_logger = UserActionLogger()
        except ImportError:
            # Fallback for testing
            _user_action_logger = None
    return _user_action_logger

# Provide access to user_action_logger for backward compatibility
user_action_logger = get_user_action_logger()


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