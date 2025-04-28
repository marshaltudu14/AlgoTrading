import time
import datetime
import logging
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from envs.data_fetcher import fetch_candle_data
from broker.broker_api import authenticate_fyers
from feature_engine import build_live_features
from model_inference import load_model, predict_action
from order_manager import select_and_place
from config import (
    INSTRUMENTS, APP_ID, SECRET_KEY, REDIRECT_URI,
    FYERS_USER, FYERS_PIN, FYERS_TOTP, LIVE_FETCH_DAYS
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
load_model()

# TEST SETTINGS: only Nifty at 2-minute interval
TEST_INSTRUMENTS = {'Nifty': INSTRUMENTS['Nifty']}
TEST_TIMEFRAME = 2
TEST_FETCH_DAYS = LIVE_FETCH_DAYS
# Track open positions to prevent new entries until exit
open_positions = {instr: False for instr in TEST_INSTRUMENTS}

tz = pytz.timezone('Asia/Kolkata')
# Scheduler with IST timezone
scheduler = BlockingScheduler(timezone=tz)

def run_cycle():
    for instrument, index_symbol in TEST_INSTRUMENTS.items():
        # skip if a position is already open
        if open_positions[instrument]:
            logger.info(f"{instrument}: position open, skipping new entry")
            continue
        try:
            # fetch enough days for feature computation
            df = fetch_candle_data(fyers, TEST_FETCH_DAYS, index_symbol, TEST_TIMEFRAME)
            features = build_live_features(df, instrument, TEST_TIMEFRAME)
            action = predict_action(features)
            logger.info(f"{instrument} prediction: {action}")
            if action.startswith("BUY_"):
                open_positions[instrument] = True
                # get last ATR for SL/TP
                atr = df['ATR'].iloc[-1]
                select_and_place(fyers, fyers_socket, instrument, action, atr)
                # reset after exit
                open_positions[instrument] = False
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
        fyers_socket.close()
