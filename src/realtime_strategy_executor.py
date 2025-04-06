# src/realtime_strategy_executor.py
"""
Main engine for the live trading bot.
Connects data handler, order manager, position manager, and strategy logic.
Receives data, gets signals, places orders, and manages exits.
"""

import logging
import time
import threading
import math # For P/L calculation rounding
# Updated imports for handlers
from src.realtime_data_handler import RealtimeMarketDataHandler
from src.order_update_handler import RealtimeOrderUpdateHandler
from src.order_manager import OrderManager
from src.position_manager import PositionManager
# Import the new options utils
from src import options_utils
# Assuming the real-time signal generator class is in src.signals
# from src.signals import InsideCandleRealtimeSignalGenerator # No longer needed if loaded dynamically

logger = logging.getLogger(__name__)

class RealtimeStrategyExecutor:
    # Changed config to strategy_config and removed type hint for signal_generator
    def __init__(self, strategy_config: dict, market_data_handler: RealtimeMarketDataHandler, order_update_handler: RealtimeOrderUpdateHandler, order_manager: OrderManager, position_manager: PositionManager, signal_generator):
        """
        Initializes the RealtimeStrategyExecutor.

        Args:
            strategy_config (dict): Configuration specific to the active strategy.
            market_data_handler: Instance of RealtimeMarketDataHandler.
            order_update_handler: Instance of RealtimeOrderUpdateHandler.
            order_manager: Instance of OrderManager.
            position_manager: Instance of PositionManager.
            signal_generator: Instance of the dynamically loaded signal generator class.
        """
        self.strategy_config = strategy_config # Store strategy-specific config
        self.market_data_handler = market_data_handler
        self.order_update_handler = order_update_handler
        # Store fyers_instance for option chain calls etc.
        self.fyers_instance = order_manager.fyers # Assuming OrderManager holds it
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.signal_generator = signal_generator
        self.running = False
        self.lock = threading.Lock() # Lock for managing state changes from callbacks
        # Get symbols from the strategy config if defined, else fallback or raise error
        self.symbols_to_trade = self.strategy_config.get("symbols_to_trade", []) # Assuming strategy config might define symbols
        if not self.symbols_to_trade:
             # Fallback to general config or raise error if strategy needs specific symbols
             from src import config as global_config # Import global config only if needed
             self.symbols_to_trade = global_config.REALTIME_SYMBOLS
             logger.warning(f"Using global REALTIME_SYMBOLS for strategy {strategy_config.get('name', 'Unnamed')}")
        self.active_orders = {} # Store details of pending entry/exit orders {symbol: order_id}
        # Store metadata associated with positions {option_symbol: {'sl': index_sl, 'tp': index_tp, 'entry_order_id': id, 'underlying_symbol': ul_sym, 'option_trade_side': side, 'entry_price': price, 'original_signal': sig}}
        self.position_meta = {}

        # Assign callbacks
        self.market_data_handler.on_tick_callback = self._handle_tick
        self.order_update_handler.on_order_callback = self._handle_order_update
        self.order_update_handler.on_trade_callback = self._handle_trade_update
        self.order_update_handler.on_position_callback = self._handle_position_update
        # self.order_update_handler.on_general_callback = self._handle_general_update # Optional

    def start(self):
        """Starts the main trading loop and connects data handler."""
        if self.running:
            logger.warning("Executor already running.")
            return

        self.running = True
        logger.info("Starting Realtime Strategy Executor...")

        # Connect handlers
        logger.info("Connecting Market Data Handler...")
        self.market_data_handler.connect()
        logger.info("Connecting Order Update Handler...")
        self.order_update_handler.connect()

        # Wait briefly for connections
        time.sleep(3) # Adjust as needed

        # Subscribe after connection checks
        market_data_connected = self.market_data_handler.is_connected
        order_updates_connected = self.order_update_handler.is_connected

        if market_data_connected:
            if self.symbols_to_trade: # Only subscribe if symbols are defined
                logger.info(f"Subscribing Market Data Handler to symbols: {self.symbols_to_trade}")
                # Subscribe to SymbolUpdate by default, add DepthUpdate if needed
                self.market_data_handler.subscribe(self.symbols_to_trade, data_type="SymbolUpdate")
            else:
                logger.warning("No symbols defined in strategy config or global config to subscribe market data.")
        else:
            logger.error("Market Data Handler failed to connect.")

        if order_updates_connected:
            order_sub_types = ["OnOrders", "OnTrades", "OnPositions"] # Add "OnGeneral" if needed
            logger.info(f"Subscribing Order Update Handler to types: {order_sub_types}")
            self.order_update_handler.subscribe(order_sub_types)
        else:
             logger.error("Order Update Handler failed to connect.")

        if not market_data_connected or not order_updates_connected:
             logger.error("One or more handlers failed to connect. Stopping executor.")
             self.stop() # Attempt graceful stop of any connected handlers
             return

        logger.info("Executor started. Waiting for updates...")
        # Main loop might just keep the thread alive, logic driven by callbacks
        # self._main_loop() # Keep commented out or simple keep-alive if logic is in callbacks

    def stop(self):
        """Stops the trading loop gracefully."""
        if not self.running:
            logger.warning("Executor not running.")
            return

        self.running = False
        logger.info("Stopping Realtime Strategy Executor...")
        # Stop handlers
        if self.market_data_handler:
            logger.info("Stopping Market Data Handler...")
            # Unsubscribe from all tracked option symbols first
            option_symbols_to_unsubscribe = list(self.position_meta.keys())
            if option_symbols_to_unsubscribe:
                 logger.info(f"Unsubscribing market data for options: {option_symbols_to_unsubscribe}")
                 self.market_data_handler.unsubscribe(symbols=option_symbols_to_unsubscribe, data_type="SymbolUpdate")
            # Unsubscribe from underlying symbols
            if self.symbols_to_trade:
                 self.market_data_handler.unsubscribe(symbols=self.symbols_to_trade, data_type="SymbolUpdate")
            self.market_data_handler.stop()
        if self.order_update_handler:
             logger.info("Stopping Order Update Handler...")
             # Unsubscribe if method exists/needed
             # self.order_update_handler.unsubscribe(["OnOrders", "OnTrades", "OnPositions"])
             self.order_update_handler.stop()
        logger.info("Executor stopped.")

    # def _main_loop(self):
    #     """If logic is primarily callback-driven, this loop might just keep the process alive."""
    #     while self.running:
    #         try:
    #             # Can perform periodic tasks here if needed (e.g., position sync)
    #             time.sleep(60) # Example: Check positions every minute
    #             self.position_manager.update_positions()
    #         except Exception as e:
    #             logger.error(f"Error in periodic main loop task: {e}", exc_info=True)


    def _handle_tick(self, tick_data):
        """Callback function to handle incoming ticks from RealtimeDataHandler."""
        if not self.running:
            return # Ignore ticks if stopping

        symbol = tick_data.get('symbol')
        ltp = tick_data.get('ltp')
        if not symbol or ltp is None:
            logger.warning(f"Received incomplete tick data: {tick_data}")
            return

        # logger.debug(f"Handling tick for {symbol}: LTP={ltp}")

        with self.lock: # Ensure thread safety for state changes
            # 1. Update signal generator if the tick is for an underlying symbol
            signal = 0
            if symbol in self.symbols_to_trade:
                signal = self.signal_generator.update(tick_data)

            # 2. Check for exits based on the tick (could be underlying or option)
            self._check_exits(symbol, ltp) # Pass current tick symbol and price

            # 3. Process any new signal generated for the underlying
            if signal != 0 and symbol in self.symbols_to_trade: # Only process signals for configured underlyings
                self._process_signal(symbol, signal, tick_data) # Pass underlying symbol and tick data

            # 4. Log current status of open OPTION positions
            self._log_open_positions_status()


    def _check_exits(self, tick_symbol, tick_ltp):
        """
        Checks if the current tick triggers an exit for any open OPTION positions
        based on the UNDERLYING index's SL/TP levels.
        """
        # This check should happen within the lock from _handle_tick

        # Iterate through all tracked option positions and their metadata
        for option_symbol, meta in list(self.position_meta.items()): # Use list to allow deletion during iteration
            underlying_symbol = meta.get('underlying_symbol')
            index_sl_price = meta.get('sl')
            index_tp_price = meta.get('tp')
            option_trade_side = meta.get('option_trade_side') # 1 if bought option, -1 if sold
            original_signal = meta.get('original_signal') # Signal that triggered the entry

            # Only check if the current tick is for the relevant underlying index
            if tick_symbol != underlying_symbol:
                continue

            # Check if we have an actual position in this option (fill confirmed)
            option_position = self.position_manager.get_position(option_symbol)
            if not option_position:
                continue # Entry order might not be filled yet

            # Check if already processing an exit order for this option
            if option_symbol in self.active_orders and self.active_orders[option_symbol] != meta.get('entry_order_id'):
                logger.debug(f"Skipping Index SL/TP check for {option_symbol}, already processing exit order {self.active_orders[option_symbol]}.")
                continue

            exit_triggered = False
            exit_reason = ""

            # Determine exit condition based on the underlying's price vs stored index SL/TP
            if option_trade_side == 1: # Long Option (CE or PE)
                if index_sl_price is not None and \
                   ((original_signal == 1 and tick_ltp <= index_sl_price) or \
                    (original_signal == -1 and tick_ltp >= index_sl_price)):
                    exit_reason = "UNDERLYING SL"
                    exit_triggered = True
                elif index_tp_price is not None and \
                     ((original_signal == 1 and tick_ltp >= index_tp_price) or \
                      (original_signal == -1 and tick_ltp <= index_tp_price)):
                    exit_reason = "UNDERLYING TP"
                    exit_triggered = True
            elif option_trade_side == -1: # Short Option (CE or PE)
                 if index_tp_price is not None and \
                    ((original_signal == 1 and tick_ltp >= index_tp_price) or \
                     (original_signal == -1 and tick_ltp <= index_tp_price)):
                     exit_reason = "UNDERLYING TP (Short Option Exit)"
                     exit_triggered = True
                 elif index_sl_price is not None and \
                      ((original_signal == 1 and tick_ltp <= index_sl_price) or \
                       (original_signal == -1 and tick_ltp >= index_sl_price)):
                      exit_reason = "UNDERLYING SL (Short Option Exit)"
                      exit_triggered = True

            if exit_triggered:
                logger.info(f"{exit_reason} triggered for option {option_symbol} based on underlying {underlying_symbol} at {tick_ltp} (Index SL: {index_sl_price}, Index TP: {index_tp_price})")
                # Use the helper method to place the exit order for the *option*
                self._exit_position(option_symbol, option_position)


    def _process_signal(self, underlying_symbol, signal, tick_data):
        """
        Processes a trading signal for the underlying index, fetches the option chain,
        selects an ITM option, and places an order based on TRADING_MODE.
        """
        # This check should happen within the lock from _handle_tick
        underlying_ltp = tick_data.get('ltp') # Price of the underlying index
        if not underlying_ltp:
             logger.warning(f"Missing LTP in tick data for {underlying_symbol}. Cannot process signal.")
             return

        logger.info(f"Processing signal {signal} for underlying {underlying_symbol} at LTP {underlying_ltp}.")

        # --- Options Logic ---
        # 1. Fetch Option Chain
        oc_df = options_utils.get_option_chain(self.fyers_instance, underlying_symbol)
        if oc_df is None or oc_df.empty:
            logger.error(f"Could not fetch or parse option chain for {underlying_symbol}. Skipping signal.")
            return

        # 2. Determine Target Option Type and Trade Side
        trading_mode = self.strategy_config.get("trading_mode", "OptionBuy") # Default to Buy
        option_type = None
        option_trade_side = None

        if trading_mode == "OptionBuy":
            option_type = "CE" if signal == 1 else "PE" if signal == -1 else None
            option_trade_side = 1 # Always Buy
        elif trading_mode == "OptionSell":
            option_type = "PE" if signal == 1 else "CE" if signal == -1 else None
            option_trade_side = -1 # Always Sell
        else:
            logger.error(f"Invalid TRADING_MODE '{trading_mode}'. Use 'OptionBuy' or 'OptionSell'. Skipping.")
            return

        if option_type is None:
            logger.debug(f"Signal is 0 for {underlying_symbol}, no option trade needed.")
            return

        # 3. Select Strike Price (Nearest ITM)
        strike_selection_mode = self.strategy_config.get("strike_selection_mode", "NearestITM")
        selected_strike = None
        if strike_selection_mode == "NearestITM":
            selected_strike = options_utils.select_itm_strike(oc_df, underlying_ltp, option_type)
        # Add other modes like ATM if needed later
        # elif strike_selection_mode == "ATM":
        #     selected_strike = options_utils.select_atm_strike(oc_df, underlying_ltp)

        if selected_strike is None:
            logger.warning(f"Could not select suitable {option_type} strike for {underlying_symbol} at LTP {underlying_ltp}. Skipping signal.")
            return

        # 4. Get Full Option Symbol
        option_symbol = options_utils.get_option_symbol_by_strike(oc_df, selected_strike, option_type)
        if option_symbol is None:
            logger.error(f"Could not find option symbol for strike {selected_strike} {option_type}. Skipping signal.")
            return

        # 5. Check Existing Position/Orders for the *Option* Symbol
        current_option_position = self.position_manager.get_position(option_symbol)
        if current_option_position:
            logger.info(f"Skipping signal for {underlying_symbol}: Already have position in {option_symbol}.")
            return
        if option_symbol in self.active_orders:
            logger.info(f"Skipping signal for {underlying_symbol}: Active order exists for {option_symbol} ({self.active_orders[option_symbol]}).")
            return

        # 6. Calculate Index-Based SL/TP Levels
        sl_multiplier = self.strategy_config.get("sl_atr_multiplier", 1.0)
        rr_ratio = self.strategy_config.get("rr_ratio", 2.0)
        index_sl_price = None
        index_tp_price = None

        if hasattr(self.signal_generator, 'get_last_atr'):
            atr_value = self.signal_generator.get_last_atr(underlying_symbol) # Use underlying symbol's ATR
            if atr_value and atr_value > 0 and underlying_ltp:
                sl_distance = atr_value * sl_multiplier
                tp_distance = sl_distance * rr_ratio
                if signal == 1: # Index Buy Signal
                    index_sl_price = round(underlying_ltp - sl_distance, 2)
                    index_tp_price = round(underlying_ltp + tp_distance, 2)
                elif signal == -1: # Index Sell Signal
                    index_sl_price = round(underlying_ltp + sl_distance, 2)
                    index_tp_price = round(underlying_ltp - tp_distance, 2)
                logger.info(f"Calculated Index SL={index_sl_price}, Index TP={index_tp_price} for {underlying_symbol} based on ATR={atr_value:.2f}")
            else:
                logger.warning(f"Could not get valid ATR or price for {underlying_symbol} to calculate Index SL/TP. ATR={atr_value}, Price={underlying_ltp}")
        else:
            logger.warning(f"Signal generator {type(self.signal_generator).__name__} does not have 'get_last_atr' method. Cannot calculate Index SL/TP.")

        # 7. Place Option Order
        order_qty_map = self.strategy_config.get("order_quantity_map", {})
        # Use underlying symbol to get quantity, assuming it maps to option lot size needs
        order_qty = order_qty_map.get(underlying_symbol)
        if not order_qty:
             logger.warning(f"Order quantity not defined for underlying symbol {underlying_symbol} in strategy config. Skipping signal.")
             return

        product_type = self.strategy_config.get("product_type", "INTRADAY")
        order_type = self.strategy_config.get("order_type", 2) # Default to Market

        logger.info(f"Attempting {trading_mode} entry: Side={option_trade_side} for {order_qty} contracts of {option_symbol} at market.")
        entry_order_response = self.order_manager.place_order(
            symbol=option_symbol,
            qty=order_qty,
            side=option_trade_side,
            order_type=order_type,
            product_type=product_type
        )

        if entry_order_response and entry_order_response.get('id'):
            order_id = entry_order_response['id']
            logger.info(f"Entry order placed successfully for {option_symbol}. Order ID: {order_id}. Underlying Index SL: {index_sl_price}, TP: {index_tp_price}")
            self.active_orders[option_symbol] = order_id # Track order using *option* symbol
            # Store metadata using *option* symbol as key
            self.position_meta[option_symbol] = {
                'sl': index_sl_price, # Store INDEX SL price
                'tp': index_tp_price, # Store INDEX TP price
                'entry_order_id': order_id,
                'underlying_symbol': underlying_symbol, # Store the related underlying
                'option_trade_side': option_trade_side, # Store if we bought or sold the option
                'original_signal': signal # Store the signal that triggered this
            }
        else:
            logger.error(f"Failed to place entry order for {option_symbol}. Response: {entry_order_response}")

        # --- Signal-based Exit Logic (for underlying) ---
        # Removed as exits are handled by _check_exits based on index SL/TP


    def _exit_position(self, option_symbol, position):
         """Helper function to place an exit order for an OPTION position."""
         if option_symbol in self.active_orders:
             logger.info(f"Skipping exit for {option_symbol}, already processing order {self.active_orders[option_symbol]}.")
             return

         # Use position details passed from _check_exits
         qty_to_exit = position.get('qty')
         side = position.get('side') # This is the side of the current position (1 for long, -1 for short)
         if qty_to_exit is None or side is None:
              logger.error(f"Cannot exit position for {option_symbol}, missing qty or side in position data: {position}")
              return

         exit_side = -1 * side # Opposite side to close the position
         product_type = self.strategy_config.get("product_type", "INTRADAY") # Use strategy product type

         logger.info(f"Placing market order to exit {abs(qty_to_exit)} of {option_symbol}...")
         exit_order_response = self.order_manager.place_order(
             symbol=option_symbol, # Use the option symbol here
             qty=abs(qty_to_exit),
             side=exit_side,
             order_type=2, # Market order exit
             product_type=product_type
         )
         if exit_order_response and exit_order_response.get('id'):
             logger.info(f"Exit order placed successfully for {option_symbol}. Order ID: {exit_order_response['id']}")
             self.active_orders[option_symbol] = exit_order_response['id'] # Track the exit order
             # SL/TP metadata will be cleared in _handle_trade_update upon confirmation
         else:
             logger.error(f"Failed to place exit order for {option_symbol}. Response: {exit_order_response}")

    # --- New Callback Handlers for Order Updates ---

    def _handle_order_update(self, order_data):
        """Callback for processing order updates from RealtimeOrderUpdateHandler."""
        with self.lock:
            # logger.info(f"Received Order Update: {order_data}") # Can be verbose
            symbol = order_data.get('symbol')
            order_id = order_data.get('id')
            status = order_data.get('status') # 5: Rejected, 1: Cancelled

            # If an order we were tracking is rejected or cancelled, remove it from active orders
            if symbol in self.active_orders and self.active_orders[symbol] == order_id:
                if status == 5: # Rejected
                    logger.error(f"Order {order_id} for {symbol} REJECTED: {order_data.get('message')}")
                    del self.active_orders[symbol]
                    # Also remove associated metadata if it was an entry order
                    if symbol in self.position_meta and self.position_meta[symbol].get('entry_order_id') == order_id:
                        del self.position_meta[symbol]
                elif status == 1: # Cancelled
                    logger.warning(f"Order {order_id} for {symbol} CANCELLED.")
                    del self.active_orders[symbol]
                    # Also remove associated metadata if it was an entry order
                    if symbol in self.position_meta and self.position_meta[symbol].get('entry_order_id') == order_id:
                        del self.position_meta[symbol]

            # TODO: Add more sophisticated order state tracking if needed

    def _handle_trade_update(self, trade_data):
        """Callback for processing trade updates from RealtimeOrderUpdateHandler."""
        with self.lock:
            # logger.info(f"Received Trade Update: {trade_data}")
            symbol = trade_data.get('symbol')
            side = trade_data.get('side')
            qty = trade_data.get('tradedQty')
            price = trade_data.get('tradePrice')
            order_number = trade_data.get('orderNumber') # Get the order ID associated with the trade

            if not all([symbol, side, qty, price, order_number]):
                logger.warning(f"Received incomplete trade update: {trade_data}")
                return

            logger.info(f"Processing trade for {symbol}: Side={side}, Qty={qty}, Price={price}, OrderID={order_number}")

            # Update Position Manager state based on the fill
            entry_price = None
            if hasattr(self.position_manager, 'update_position_on_fill'):
                 entry_price = self.position_manager.update_position_on_fill(symbol, side, qty, price) # Assume this returns avg entry price
                 logger.info(f"Position Manager updated for trade on {symbol}. Entry Price: {entry_price}")
            else:
                 logger.warning("PositionManager does not have 'update_position_on_fill' method. Cannot update position state.")

            # Check if this trade corresponds to an active order we are tracking
            if symbol in self.active_orders and self.active_orders[symbol] == order_number:
                logger.info(f"Trade confirms active order {order_number} for {symbol}. Clearing active order tracking.")
                # Store entry price in metadata if it's an entry trade confirmation
                meta = self.position_meta.get(symbol)
                if meta and meta.get('entry_order_id') == order_number and entry_price is not None:
                    meta['entry_price'] = entry_price
                    logger.info(f"Stored entry price {entry_price} in metadata for {symbol}")

                del self.active_orders[symbol] # Remove from active orders regardless

                # If this was an exit trade (check position side after update or trade side)
                current_pos = self.position_manager.get_position(symbol)
                if not current_pos or current_pos.get('netQty', 0) == 0: # Position is now flat or closed
                    if symbol in self.position_meta:
                        logger.info(f"Position for {symbol} closed by trade. Clearing SL/TP metadata.")
                        del self.position_meta[symbol]
                    # Unsubscribe from market data for the closed option position
                    logger.info(f"Exit trade confirmed for {symbol}. Unsubscribing from market data.")
                    self.market_data_handler.unsubscribe(symbols=[symbol], data_type="SymbolUpdate")

            # If this trade confirms an *entry* order, subscribe to its market data
            meta = self.position_meta.get(symbol)
            if meta and meta.get('entry_order_id') == order_number:
                 logger.info(f"Entry order {order_number} for {symbol} confirmed by trade. Subscribing to market data.")
                 self.market_data_handler.subscribe(symbols=[symbol], data_type="SymbolUpdate")
                 # SL/TP metadata is already set in _process_signal


    def _handle_position_update(self, position_data):
        """Callback for processing position updates from RealtimeOrderUpdateHandler."""
        with self.lock:
            logger.info(f"Received Position Update: {position_data}")
            # TODO: Update PositionManager state based on the full position snapshot
            # This can help reconcile state if trades/orders were missed
            # self.position_manager.sync_position(position_data)
            pass

    # Optional: Add handler for general messages if subscribed
    # def _handle_general_update(self, general_data):
    #     with self.lock:
    #         logger.info(f"Received General Update: {general_data}")
    #         pass

    def _log_open_positions_status(self):
        """Logs the status of all currently tracked open option positions."""
        if not self.position_meta:
            return # No open positions being tracked

        log_header = "\n--- Open Positions Status ---"
        log_lines = [log_header]
        has_positions_to_log = False

        for option_symbol, meta in self.position_meta.items():
            position_details = self.position_manager.get_position(option_symbol)
            option_ltp_data = self.market_data_handler.get_latest_data(option_symbol)
            current_option_ltp = option_ltp_data.get('ltp') if option_ltp_data else None

            if not position_details: # Position might not be confirmed yet or already closed internally
                # Check if we are still waiting for an entry order fill
                if meta.get('entry_order_id') and option_symbol in self.active_orders and self.active_orders[option_symbol] == meta.get('entry_order_id'):
                     log_lines.append(f"  Symbol: {option_symbol} (Waiting Entry Fill - Order: {meta['entry_order_id']})")
                     has_positions_to_log = True
                continue # Skip logging if position doesn't exist and no active entry order

            has_positions_to_log = True
            # Use stored entry price if available, otherwise try from position details
            entry_price = meta.get('entry_price')
            if entry_price is None:
                 entry_price = position_details.get('buyAvg') if position_details.get('side', 0) == 1 else position_details.get('sellAvg')

            qty = position_details.get('qty', 0)
            side = position_details.get('side', 0) # 1 for long, -1 for short
            index_sl = meta.get('sl', 'N/A')
            index_tp = meta.get('tp', 'N/A')

            pnl = 0.0
            if current_option_ltp is not None and entry_price is not None and entry_price > 0:
                if side == 1: # Long option
                    pnl = (current_option_ltp - entry_price) * qty
                elif side == -1: # Short option
                    pnl = (entry_price - current_option_ltp) * abs(qty) # Qty might be negative

            status_line = (
                f"  Symbol: {option_symbol} | "
                f"Pos: {'LONG' if side == 1 else 'SHORT' if side == -1 else 'FLAT'} {qty} | "
                f"Entry: {entry_price:.2f} | "
                f"LTP: {current_option_ltp:.2f if current_option_ltp is not None else 'N/A'} | "
                f"Idx SL: {index_sl} | Idx TP: {index_tp} | "
                f"P/L: {pnl:.2f}"
            )
            log_lines.append(status_line)

        if has_positions_to_log:
            log_lines.append("-----------------------------")
            logger.info("\n".join(log_lines))


if __name__ == '__main__':
    # Example usage or testing
    logging.basicConfig(level=logging.INFO)
    logger.info("RealtimeStrategyExecutor module")
    # Add test code here if needed (requires mocked dependencies and config)
