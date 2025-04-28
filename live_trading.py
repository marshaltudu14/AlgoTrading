"""
Live trading script for AlgoTrading.
Runs the trading system in live mode with real-time data and order execution.
"""
import time
import datetime
import pytz
from typing import Dict, Any
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from core.logging_setup import get_logger
from core.config import (
    INSTRUMENTS,
    QUANTITIES,
    APP_ID,
    SECRET_KEY,
    REDIRECT_URI,
    FYERS_USER,
    FYERS_PIN,
    FYERS_TOTP,
    LIVE_FETCH_DAYS,
    MARKET_OPEN_TIME,
    MARKET_CLOSE_TIME,
    INITIAL_CAPITAL
)
from data.fetcher import (
    fetch_candle_data,
    verify_data_sufficiency
)
from features.feature_engine import (
    build_live_features,
    verify_features
)
from models.inference import (
    predict_action,
    get_model_instance
)
from trading.broker_api import (
    authenticate_fyers,
    get_positions,
    get_funds
)
from trading.order_manager import select_and_place
from trading.position_manager import (
    PositionManager,
    sync_positions_with_broker
)
from trading.risk_manager import RiskManager
from utils.helpers import is_market_open

# Configure logging
logger = get_logger("live_trading")

# Initialize timezone
tz = pytz.timezone('Asia/Kolkata')

# Initialize scheduler with IST timezone
scheduler = BlockingScheduler(timezone=tz)

# Global variables
fyers = None
fyers_socket = None
position_manager = None
risk_manager = None

# Trading settings
TRADING_INSTRUMENTS = {'Nifty': INSTRUMENTS['Nifty']}
TRADING_TIMEFRAME = 2
TRADING_FETCH_DAYS = LIVE_FETCH_DAYS
USE_ENHANCED_FEATURES = True
USE_MOE_MODEL = True


def initialize_system():
    """Initialize the trading system."""
    global fyers, fyers_socket, position_manager, risk_manager

    logger.info("Initializing trading system")

    # Authenticate and connect to Fyers
    try:
        fyers, fyers_socket = authenticate_fyers(
            APP_ID, SECRET_KEY,
            REDIRECT_URI, FYERS_USER,
            FYERS_PIN, FYERS_TOTP
        )

        # Connect websocket for ticks
        fyers_socket.connect()
        logger.info("WebSocket connected")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise

    # Preload model
    logger.info("Preloading model")
    get_model_instance(use_moe=USE_MOE_MODEL)

    # Initialize position manager
    position_manager = PositionManager(initial_capital=INITIAL_CAPITAL)

    # Initialize risk manager
    risk_manager = RiskManager(position_manager)

    # Sync positions with broker
    sync_positions_with_broker(position_manager, fyers)

    logger.info("Trading system initialized successfully")

def run_trading_cycle():
    """Run a single trading cycle."""
    global fyers, fyers_socket, position_manager, risk_manager

    # Skip if market is closed
    if not is_market_open():
        logger.info("Market is closed, skipping trading cycle")
        return

    logger.info("Starting trading cycle")

    # Update risk manager
    risk_manager.update_daily_pnl()

    # Check if trading is allowed
    can_trade, reason = risk_manager.can_trade(fyers)
    if not can_trade:
        logger.info(f"Trading not allowed: {reason}")
        return

    # Check if we should exit all positions
    should_exit, exit_reason = risk_manager.should_exit_all()
    if should_exit:
        logger.warning(f"Exiting all positions: {exit_reason}")
        # TODO: Implement exit all positions
        return

    # Process each instrument
    for instrument, index_symbol in TRADING_INSTRUMENTS.items():
        try:
            # Check if trading is allowed for this instrument
            can_trade_instrument, reason = risk_manager.can_trade_instrument(instrument)
            if not can_trade_instrument:
                logger.info(f"Trading not allowed for {instrument}: {reason}")
                continue

            # Fetch historical data
            df = fetch_candle_data(fyers, TRADING_FETCH_DAYS, index_symbol, TRADING_TIMEFRAME)

            # Verify data sufficiency
            if not verify_data_sufficiency(df):
                logger.warning(f"Insufficient data for {instrument}")
                continue

            # Get current price
            current_price = df['close'].iloc[-1]

            # Get position details
            positions = position_manager.get_positions_for_instrument(instrument)
            position = positions[0] if positions else None

            # Set position features
            position_value = 1 if position else 0
            position_duration = position.duration if position else 0
            entry_price = position.entry_price if position else 0.0
            unrealized_pnl = position.unrealized_pnl if position else 0.0

            # Build features
            features = build_live_features(
                df,
                instrument,
                TRADING_TIMEFRAME,
                position=position_value,
                position_duration=position_duration,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                use_enhanced_features=USE_ENHANCED_FEATURES
            )

            # Verify features
            if not verify_features(features):
                logger.warning(f"Invalid features for {instrument}")
                continue

            # Get prediction
            action = predict_action(
                features,
                instrument,
                TRADING_TIMEFRAME,
                use_moe=USE_MOE_MODEL
            )

            logger.info(f"{instrument} prediction: {action}")

            # Get risk assessment
            risk_assessment = risk_manager.get_risk_assessment(instrument, action)
            logger.info(f"Risk assessment: {risk_assessment}")

            # Handle actions
            if action.startswith("BUY_") and not position:
                # Check risk score
                if risk_assessment['risk_score'] > 0.7:
                    logger.warning(f"Risk too high for {instrument}: {risk_assessment['risk_score']:.2f}")
                    continue

                # Place order
                atr = df['ATR'].iloc[-1]
                order_result = select_and_place(fyers, fyers_socket, instrument, action, atr)

                if order_result.get('s') == 'ok':
                    logger.info(f"{instrument}: Entered position at {current_price}")

                    # Add position to position manager
                    symbol = order_result.get('symbol')
                    filled_price = order_result.get('filled_price')
                    sl = order_result.get('sl')
                    tp = order_result.get('tp')

                    # Parse option details
                    from trading.position_manager import parse_option_symbol
                    option_details = parse_option_symbol(symbol)

                    if option_details:
                        position_manager.add_position(
                            symbol=symbol,
                            instrument=instrument,
                            entry_price=filled_price,
                            quantity=QUANTITIES[instrument],
                            entry_time=datetime.datetime.now(tz),
                            option_type=option_details['option_type'],
                            strike_price=option_details['strike_price'],
                            expiry_date=option_details['expiry_date'],
                            stop_loss=sl,
                            take_profit=tp
                        )
                else:
                    logger.error(f"Order failed: {order_result}")

            elif action == "HOLD" and position:
                # Update position
                position.update(current_price, datetime.datetime.now(tz))
                logger.info(f"{instrument}: Holding position, duration={position.duration:.1f}, PnL={position.unrealized_pnl:.2f}")

            elif position:
                # Exit position
                logger.info(f"{instrument}: Exiting position at {current_price}")
                position_manager.exit_position(
                    symbol=position.symbol,
                    exit_price=current_price,
                    exit_time=datetime.datetime.now(tz),
                    reason="MODEL_EXIT"
                )

        except Exception as e:
            logger.error(f"Error in trading cycle for {instrument}: {e}")

        # Sleep between instruments
        time.sleep(0.5)

    # Log position summary
    summary = position_manager.get_summary()
    logger.info(f"Position summary: {summary}")

def schedule_trading():
    """Schedule trading cycles."""
    # Schedule run_cycle at nearest round times starting at market open
    market_open = datetime.datetime.now(tz).replace(
        hour=int(MARKET_OPEN_TIME.split(':')[0]),
        minute=int(MARKET_OPEN_TIME.split(':')[1]),
        second=0,
        microsecond=0
    )

    # If market already open, start immediately
    if datetime.datetime.now(tz) > market_open:
        market_open = datetime.datetime.now(tz)

    # Create trigger
    trigger = IntervalTrigger(
        minutes=TRADING_TIMEFRAME,
        start_date=market_open,
        timezone=tz
    )

    # Add job
    scheduler.add_job(run_trading_cycle, trigger)
    logger.info(f"Scheduled trading cycles every {TRADING_TIMEFRAME} minutes starting at {market_open}")


def main():
    """Main function."""
    try:
        # Initialize system
        initialize_system()

        # Schedule trading
        schedule_trading()

        # Start scheduler
        logger.info("Starting live trading scheduler...")
        scheduler.start()

    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down...")

        # Close websocket
        if fyers_socket:
            try:
                fyers_socket.close()
            except AttributeError:
                logger.info("fyers_socket has no close method, skipping cleanup")

        # Log final summary
        if position_manager:
            summary = position_manager.get_summary()
            logger.info(f"Final position summary: {summary}")


if __name__ == '__main__':
    main()
