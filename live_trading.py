import time
import datetime
import logging
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from envs.data_fetcher import fetch_candle_data
from broker.broker_api import authenticate_fyers
from feature_engine import build_live_features
from models.inference import predict_action
from order_manager import select_and_place
from config import (
    INSTRUMENTS, APP_ID, SECRET_KEY, REDIRECT_URI,
    FYERS_USER, FYERS_PIN, FYERS_TOTP, LIVE_FETCH_DAYS,
    QUANTITIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Authenticate and connect
fyers, fyers_socket = authenticate_fyers(
    APP_ID, SECRET_KEY,
    REDIRECT_URI, FYERS_USER,
    FYERS_PIN, FYERS_TOTP
)
# Connect websocket for ticks
try:
    fyers_socket.connect()
    logger.info("WebSocket connected")
except Exception as e:
    logger.error(f"WebSocket connection failed: {e}")

# Preload model
from models.inference import get_model_instance
get_model_instance(use_moe=True)

# TEST SETTINGS: only Nifty at 2-minute interval
TEST_INSTRUMENTS = {'Nifty': INSTRUMENTS['Nifty']}
TEST_TIMEFRAME = 2
TEST_FETCH_DAYS = LIVE_FETCH_DAYS
# Track open positions to prevent new entries until exit
open_positions = {instr: False for instr in TEST_INSTRUMENTS}

tz = pytz.timezone('Asia/Kolkata')
# Scheduler with IST timezone
scheduler = BlockingScheduler(timezone=tz)

# Track position details for each instrument
position_details = {
    instr: {
        'position': 0,
        'entry_price': 0.0,
        'position_duration': 0,
        'entry_time': None
    } for instr in TEST_INSTRUMENTS
}

def calculate_unrealized_pnl(instrument, current_price):
    """Calculate unrealized PnL for the current position."""
    details = position_details[instrument]
    if details['position'] == 0:
        return 0.0

    quantity = QUANTITIES.get(instrument, 0)
    return (current_price - details['entry_price']) * quantity

def run_cycle():
    for instrument, index_symbol in TEST_INSTRUMENTS.items():
        try:
            # fetch enough days for feature computation
            df = fetch_candle_data(fyers, TEST_FETCH_DAYS, index_symbol, TEST_TIMEFRAME)

            # Get current price
            current_price = df['close'].iloc[-1]

            # Get position details
            details = position_details[instrument]

            # Calculate unrealized PnL
            unrealized_pnl = calculate_unrealized_pnl(instrument, current_price)

            # Build features with position information
            features = build_live_features(
                df,
                instrument,
                TEST_TIMEFRAME,
                position=details['position'],
                position_duration=details['position_duration'],
                entry_price=details['entry_price'],
                unrealized_pnl=unrealized_pnl
            )

            # Get prediction using MoE model
            action = predict_action(features, instrument, TEST_TIMEFRAME, use_moe=True)
            logger.info(f"{instrument} prediction: {action}")

            # Handle actions
            if action.startswith("BUY_") and details['position'] == 0:
                # Enter position
                details['position'] = 1
                details['entry_price'] = current_price
                details['position_duration'] = 0
                details['entry_time'] = datetime.datetime.now(tz)

                # Place order
                atr = df['ATR'].iloc[-1]
                select_and_place(fyers, fyers_socket, instrument, action, atr)

                logger.info(f"{instrument}: Entered position at {current_price}")

            elif action == "HOLD" and details['position'] == 1:
                # Update position duration
                details['position_duration'] += 1
                logger.info(f"{instrument}: Holding position, duration={details['position_duration']}, PnL={unrealized_pnl:.2f}")

            elif details['position'] == 1:
                # Exit position
                details['position'] = 0
                details['position_duration'] = 0
                details['entry_price'] = 0.0
                details['entry_time'] = None

                # Handle exit (this would be handled by your order management system)
                logger.info(f"{instrument}: Exited position at {current_price}, PnL={unrealized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error in run_cycle for {instrument}: {e}")

        time.sleep(0.5)

# Schedule run_cycle at nearest round times starting 9:15 IST with given interval
start_time = datetime.datetime.now(tz).replace(hour=9, minute=15, second=0, microsecond=0)
trigger = IntervalTrigger(minutes=TEST_TIMEFRAME, start_date=start_time, timezone=tz)
scheduler.add_job(run_cycle, trigger)

if __name__ == '__main__':
    logger.info("Starting live trading scheduler...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")
        try:
            fyers_socket.close()
        except AttributeError:
            logger.info("fyers_socket has no close method, skipping cleanup")
