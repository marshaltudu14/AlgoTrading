"""
Configuration and data routes for AlgoTrading System
Handles config, historical data, funds, and metrics endpoints
"""
import json
import sys
import logging
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from fyers_apiv3 import fyersModel

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add the backend directory to Python path to enable absolute imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from core.dependencies import get_current_user
from utils.realtime_data_loader import RealtimeDataLoader

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["configuration"])


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/config")
async def get_config():
    """Get configuration data including instruments and timeframes"""
    try:
        # Get the path to the config file (relative to project root)
        config_path = Path(__file__).parent.parent.parent / "config" / "instruments.yaml"
        
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


@router.get("/funds")
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


@router.get("/metrics")
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


@router.get("/historical-data")
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

        # Initialize RealtimeDataLoader with authentication
        data_loader = RealtimeDataLoader(
            access_token=current_user["access_token"],
            app_id=current_user["app_id"]
        )

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
            # Use datetime_epoch if available (preferred for charts), otherwise datetime_readable, then fallback
            time_value = None
            if 'datetime_epoch' in row and pd.notna(row['datetime_epoch']):
                time_value = int(row['datetime_epoch'])  # Use epoch timestamp for lightweight-charts
            elif 'datetime_readable' in row and pd.notna(row['datetime_readable']):
                # Convert datetime_readable to epoch timestamp
                try:
                    dt = pd.to_datetime(row['datetime_readable'])
                    time_value = int(dt.timestamp())
                except:
                    time_value = None
            elif 'datetime' in row and pd.notna(row['datetime']):
                # Fallback to datetime column
                try:
                    if hasattr(row['datetime'], 'timestamp'):
                        time_value = int(row['datetime'].timestamp())
                    else:
                        dt = pd.to_datetime(row['datetime'])
                        time_value = int(dt.timestamp())
                except:
                    time_value = None

            # Skip rows with invalid timestamps
            if time_value is None or time_value <= 0:
                logger.warning(f"Skipping row with invalid timestamp: {row.get('datetime_epoch', row.get('datetime_readable', 'unknown'))}")
                continue

            candle = {
                "time": time_value,  # Use epoch timestamp
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