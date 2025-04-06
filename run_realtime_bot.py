# run_realtime_bot.py
"""
Main script to initialize and run the real-time trading bot.
"""

import logging
import time
import signal # For handling Ctrl+C
import sys
from fyers_apiv3 import fyersModel # Added missing import

# Assuming src is in the same parent directory or PYTHONPATH is set
try:
    from src import config
    # Use the function directly, not a class
    from src.fyers_auth import get_fyers_access_token
    # Import updated/new handlers
    from src.realtime_data_handler import RealtimeMarketDataHandler
    from src.order_update_handler import RealtimeOrderUpdateHandler
    from src.order_manager import OrderManager
    from src.position_manager import PositionManager
    from src.realtime_strategy_executor import RealtimeStrategyExecutor
    # Import signals module to dynamically load generator class
    from src import signals as signal_module
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Error: Ensure 'src' directory is accessible or in PYTHONPATH.")
    exit(1)

# --- Logging Setup ---
# TODO: Configure logging properly (file handler, formatting)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables ---
executor = None # To hold the executor instance for signal handling

# --- Signal Handler ---
def handle_signal(signum, frame):
    """Gracefully shuts down the bot on Ctrl+C."""
    logger.warning(f"Received signal {signum}. Initiating shutdown...")
    if executor:
        executor.stop()
    # Allow some time for cleanup before exiting
    time.sleep(2)
    logger.info("Shutdown complete.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal) # Ctrl+C
signal.signal(signal.SIGTERM, handle_signal) # Termination signal

# --- Main Execution ---
def main():
    global executor # Allow modification of the global variable

    logger.info("--- Initializing Real-Time Trading Bot ---")

    # 1. Authentication
    logger.info("Authenticating with Fyers...")
    token_info = None
    fyers_instance = None
    try:
        token_info = get_fyers_access_token() # Returns dict with 'access_token' and 'ws_token'
        if not token_info or 'access_token' not in token_info or 'ws_token' not in token_info:
            logger.error("Authentication failed or did not return expected token format. Exiting.")
            return
        # Initialize fyersModel with the raw token for REST calls
        fyers_instance = fyersModel.FyersModel(client_id=config.APP_ID, token=token_info['access_token'])
        # Optional: Verify connection
        profile = fyers_instance.get_profile()
        logger.info(f"Fyers authentication successful. Profile FY_ID: {profile.get('data', {}).get('fy_id')}")
    except Exception as e:
        logger.error(f"Fyers authentication failed: {e}", exc_info=True)
        return

    # 2. Initialize Components
    logger.info("Initializing components...")
    # Extract tokens for clarity
    ws_access_token = token_info['ws_token'] # Formatted token for WebSockets

    # --- Create Executor first (to pass its method as callback) ---
    # Placeholder for executor instance before data_handler is created
    temp_executor_ref = {"instance": None}

    def tick_callback_wrapper(tick_data):
        """Wrapper to call the executor's handler method."""
        if temp_executor_ref["instance"]:
            temp_executor_ref["instance"]._handle_tick(tick_data)
        else:
            logger.warning("Executor not yet initialized when tick callback called.")

    # --- Callback Wrappers ---
    # Need wrappers because handler methods need 'self' but callbacks don't pass it
    def tick_callback_wrapper(tick_data):
        if executor: executor._handle_tick(tick_data)
        else: logger.warning("Executor not yet initialized for tick callback.")

    def order_callback_wrapper(order_data):
        if executor: executor._handle_order_update(order_data)
        else: logger.warning("Executor not yet initialized for order callback.")

    def trade_callback_wrapper(trade_data):
        if executor: executor._handle_trade_update(trade_data)
        else: logger.warning("Executor not yet initialized for trade callback.")

    def position_callback_wrapper(position_data):
        if executor: executor._handle_position_update(position_data)
        else: logger.warning("Executor not yet initialized for position callback.")

    # --- Initialize Handlers ---
    market_data_handler = RealtimeMarketDataHandler(
        access_token=ws_access_token, # Use formatted token
        log_path=config.LOGS_DIR,
        on_tick_callback=tick_callback_wrapper
    )
    order_update_handler = RealtimeOrderUpdateHandler(
        access_token=ws_access_token, # Use formatted token
        log_path=config.LOGS_DIR,
        on_order_callback=order_callback_wrapper,
        on_trade_callback=trade_callback_wrapper,
        on_position_callback=position_callback_wrapper
        # on_general_callback=general_callback_wrapper # Add if needed
    )

    # --- Initialize Managers ---
    order_manager = OrderManager(fyers_instance) # Uses REST API instance
    position_manager = PositionManager(order_manager) # Assuming PositionManager needs OrderManager

    # --- Load Active Strategy Config ---
    active_strategy_name = config.ACTIVE_STRATEGY_NAME
    if active_strategy_name not in config.STRATEGY_CONFIGS:
        logger.error(f"Active strategy '{active_strategy_name}' not found in STRATEGY_CONFIGS. Exiting.")
        return
    strategy_params = config.STRATEGY_CONFIGS[active_strategy_name]
    logger.info(f"Loading configuration for active strategy: {active_strategy_name}")

    # --- Dynamically Initialize Signal Generator ---
    signal_generator_class_name = strategy_params.get("signal_generator_class")
    if not signal_generator_class_name:
        logger.error(f"No 'signal_generator_class' defined for strategy '{active_strategy_name}'. Exiting.")
        return

    try:
        SignalGeneratorClass = getattr(signal_module, signal_generator_class_name)
        # Instantiate with parameters from the strategy config
        signal_generator = SignalGeneratorClass(
            timeframe_minutes=strategy_params["timeframe_minutes"],
            atr_period=strategy_params["atr_period"]
            # Add other necessary params if the generator class requires them
        )
        logger.info(f"Initialized signal generator: {signal_generator_class_name}")
    except AttributeError:
        logger.error(f"Signal generator class '{signal_generator_class_name}' not found in src.signals module. Exiting.")
        return
    except Exception as e:
        logger.error(f"Error initializing signal generator '{signal_generator_class_name}': {e}", exc_info=True)
        return


    # 3. Initialize Executor
    logger.info("Initializing strategy executor...")
    executor = RealtimeStrategyExecutor(
        strategy_config=strategy_params, # Pass the specific strategy config
        market_data_handler=market_data_handler,
        order_update_handler=order_update_handler,
        order_manager=order_manager,
        position_manager=position_manager,
        signal_generator=signal_generator
    )
    # No need for temp_executor_ref anymore, callbacks check global executor

    # 4. Start Executor
    logger.info("Starting executor...")
    try:
        executor.start() # This starts the background threads

        # Keep the main thread alive while the executor is running
        logger.info("Main thread waiting for executor to stop (Press Ctrl+C to exit)...")
        while executor.running:
            # Perform periodic checks or just sleep
            time.sleep(1) # Check every second

    except KeyboardInterrupt:
        # This might not be reached if the signal handler catches SIGINT first
        logger.warning("KeyboardInterrupt received in main. Stopping executor...")
        if executor:
            executor.stop()
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        if executor:
            executor.stop() # Attempt graceful shutdown
    finally:
        logger.info("--- Real-Time Trading Bot Finished ---")


if __name__ == "__main__":
    main()
