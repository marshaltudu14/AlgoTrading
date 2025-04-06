# src/realtime_data_handler.py
"""
Handles connection to Fyers WebSocket for real-time market data.
Subscribes to symbols, receives data, and potentially processes it.
"""

import logging
import threading
import time
# Updated import based on documentation
from fyers_apiv3.FyersWebsocket import data_ws

logger = logging.getLogger(__name__)

# Removed unused constants

class RealtimeMarketDataHandler: # Renamed class for clarity
    def __init__(self, access_token: str, log_path: str = 'logs/', on_tick_callback=None):
        """
        Initializes the RealtimeMarketDataHandler for market data (Symbol/Depth).

        Args:
            access_token (str): The access token for Fyers API V3 WebSocket. Format: client_id:access_token
            log_path (str): Path to store WebSocket logs.
            on_tick_callback (callable, optional): Function to call when a symbol tick/update is received.
                                                   Expected signature: callback(tick_data: dict)
        """
        if not access_token or ':' not in access_token:
             raise ValueError("Invalid access_token format. Expected 'client_id:access_token'")

        self.access_token = access_token
        self.log_path = log_path
        # self.data_type removed, handled per subscription
        self.fyers_ws = None
        self.ws_thread = None
        self.is_connected = False
        # Store subscribed symbols per data_type: {"SymbolUpdate": set(), "DepthUpdate": set()}
        self.subscribed_symbols = {"SymbolUpdate": set(), "DepthUpdate": set()}
        self.latest_symbol_data = {} # Store latest tick/update for each symbol
        self.lock = threading.Lock() # For thread safety when accessing shared data
        self.on_tick_callback = on_tick_callback
        # Removed on_order_update_callback

        logger.info("RealtimeMarketDataHandler initialized.")

    def _run_websocket(self):
        """Target function for the WebSocket thread."""
        if not self.fyers_ws:
            logger.error("WebSocket instance not created before running thread.")
            return
        logger.info("Starting WebSocket connection in background thread...")
        self.fyers_ws.connect()
        # Keep thread alive while WebSocket is running (connect is blocking)
        logger.info("WebSocket connection loop finished in background thread.") # Should not be reached unless connection drops

    def connect(self):
        """Establishes the WebSocket connection."""
        if self.is_connected:
            logger.warning("WebSocket is already connected.")
            return

        logger.info("Attempting to connect WebSocket...")
        try:
            # Use the correct class from the imported module
            self.fyers_ws = data_ws.FyersDataSocket(
                access_token=self.access_token,
                log_path=self.log_path,
                litemode=False, # Set to True for litemode if needed
                write_to_file=False, # Set to False as callbacks are used
                reconnect=True, # Enable auto-reconnect
                on_connect=self.on_open, # Callback on successful connection
                on_close=self.on_close, # Callback on connection close
                on_error=self.on_error, # Callback on error
                on_message=self.on_message # Callback for messages
            )

            # Start the connection in a separate thread
            self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.ws_thread.start()

            # Give some time for connection to establish (optional, can rely on on_open)
            time.sleep(2) # Adjust as needed
            # Note: is_connected flag will be set in on_open callback

        except Exception as e:
            logger.error(f"Failed to initialize or connect WebSocket: {e}", exc_info=True)
            self.is_connected = False # Ensure flag is false on error

    def on_open(self):
        """Callback function when WebSocket connection is established."""
        self.is_connected = True
        logger.info("WebSocket connection established successfully.")
        # Resubscribe to symbols if connection was dropped and re-established
        if self.subscribed_symbols:
            logger.info(f"Re-subscribing to symbols: {list(self.subscribed_symbols)}")
            # Resubscribe logic needs update after subscribe method is fixed
            # Iterate through stored subscriptions and resubscribe
            with self.lock:
                for dtype, sym_set in self.subscribed_symbols.items():
                    if sym_set:
                        logger.info(f"Re-subscribing to {dtype} for symbols: {list(sym_set)}")
                        # Call the updated subscribe method
                        self._subscribe_internal(list(sym_set), dtype)


    def subscribe(self, symbols: list[str], data_type: str = "SymbolUpdate"):
        """
        Subscribes to real-time market data for the given symbols and data type.

        Args:
            symbols (list[str]): List of symbols to subscribe to (e.g., ['NSE:SBIN-EQ']).
            data_type (str): The type of data feed ("SymbolUpdate" or "DepthUpdate"). Defaults to "SymbolUpdate".
        """
        if not self.is_connected or not self.fyers_ws:
            logger.error(f"Cannot subscribe to {data_type}, WebSocket is not connected.")
            return

        if not isinstance(symbols, list):
            logger.error("Symbols must be provided as a list.")
            return

        if data_type not in self.subscribed_symbols:
            logger.error(f"Invalid data_type '{data_type}'. Must be 'SymbolUpdate' or 'DepthUpdate'.")
            return

        # Use the internal helper method
        self._subscribe_internal(symbols, data_type)


    def _subscribe_internal(self, symbols: list[str], data_type: str):
        """Internal helper to handle subscription logic."""
        with self.lock:
            current_subs = self.subscribed_symbols.get(data_type, set())
            symbols_to_subscribe = [s for s in symbols if s not in current_subs]

        if not symbols_to_subscribe:
            logger.info(f"All requested symbols for {data_type} are already subscribed.")
            return

        logger.info(f"Subscribing to {data_type} for symbols: {symbols_to_subscribe}")
        try:
            # Use correct parameter name 'symbols' and pass data_type
            self.fyers_ws.subscribe(symbols=symbols_to_subscribe, data_type=data_type)
            # Update internal state *after* successful subscription call
            with self.lock:
                self.subscribed_symbols[data_type].update(symbols_to_subscribe)
            logger.info(f"Successfully subscribed to {data_type}. Current subscriptions: {self.subscribed_symbols}")
        except Exception as e:
            logger.error(f"Error subscribing to {data_type} for symbols {symbols_to_subscribe}: {e}", exc_info=True)


    def unsubscribe(self, symbols: list[str], data_type: str = "SymbolUpdate"):
        """
        Unsubscribes from real-time market data for the given symbols and data type.

        Args:
            symbols (list[str]): List of symbols to unsubscribe from.
            data_type (str): The type of data feed ("SymbolUpdate" or "DepthUpdate"). Defaults to "SymbolUpdate".
        """
        if not self.is_connected or not self.fyers_ws:
            logger.error(f"Cannot unsubscribe from {data_type}, WebSocket is not connected.")
            return

        if not isinstance(symbols, list):
            logger.error("Symbols must be provided as a list.")
            return

        if data_type not in self.subscribed_symbols:
            logger.error(f"Invalid data_type '{data_type}'. Must be 'SymbolUpdate' or 'DepthUpdate'.")
            return

        with self.lock:
            current_subs = self.subscribed_symbols.get(data_type, set())
            symbols_to_unsubscribe = [s for s in symbols if s in current_subs]

        if not symbols_to_unsubscribe:
            logger.info(f"None of the requested symbols for {data_type} are currently subscribed.")
            return

        logger.info(f"Unsubscribing from {data_type} for symbols: {symbols_to_unsubscribe}")
        try:
            # Use correct parameter name 'symbols' and pass data_type
            self.fyers_ws.unsubscribe(symbols=symbols_to_unsubscribe, data_type=data_type)
             # Update internal state *after* successful unsubscription call
            with self.lock:
                self.subscribed_symbols[data_type].difference_update(symbols_to_unsubscribe)
            logger.info(f"Successfully unsubscribed from {data_type}. Current subscriptions: {self.subscribed_symbols}")
        except Exception as e:
            logger.error(f"Error unsubscribing from {data_type} for symbols {symbols_to_unsubscribe}: {e}", exc_info=True)


    def on_message(self, message):
        """Callback function when a message is received from the WebSocket."""
        # logger.debug(f"WebSocket message received: {message}") # Can be very verbose

        # Market data socket only sends market data (SymbolUpdate or DepthUpdate)
        # No need to check self.data_type anymore
        if isinstance(message, dict) and 'symbol' in message:
            symbol = message['symbol']
            with self.lock:
                self.latest_symbol_data[symbol] = message # Store latest tick regardless of type (Symbol/Depth)
            # logger.debug(f"Updated latest data for {symbol}")
            # Call the tick callback if provided (can handle both SymbolUpdate/DepthUpdate)
            if self.on_tick_callback:
                try:
                    self.on_tick_callback(message)
                except Exception as e:
                    logger.error(f"Error in on_tick_callback for symbol {symbol}: {e}", exc_info=True)
        # Removed elif for DATA_TYPE_ORDER
        else:
            # Handle other message types or formats if necessary (e.g., connection messages)
            logger.debug(f"Received unhandled/non-symbol message format: {message}")


    def on_error(self, error):
        """Callback function when a WebSocket error occurs."""
        logger.error(f"WebSocket error: {error}")
        # Optional: Add more sophisticated error handling/reporting

    def on_close(self):
        # TODO: Handle WebSocket closure
        logger.warning("WebSocket connection closed.")
        self.is_connected = False
        # Optional: Implement logic for cleanup or reconnection attempts if needed

    def get_latest_data(self, symbol: str) -> dict | None:
        """
        Retrieves the latest stored data for a given symbol.

        Args:
            symbol (str): The symbol to retrieve data for.

        Returns:
            dict | None: The latest data dictionary, or None if no data exists.
        """
        with self.lock:
            return self.latest_symbol_data.get(symbol)

    def stop(self):
        """Stops the WebSocket connection."""
        logger.info("Stopping WebSocket connection...")
        self.is_connected = False # Set flag immediately
        if self.fyers_ws:
            try:
                # Use close() method similar to Order socket handler
                if hasattr(self.fyers_ws, 'close'):
                    self.fyers_ws.close()
                    logger.info("Market Data WebSocket close command issued.")
                else:
                     logger.warning("Market Data WebSocket instance has no 'close' method.")
            except AttributeError:
                 logger.warning("Market Data WebSocket instance has no 'close' method.")
            except Exception as e:
                logger.error(f"Error stopping Market Data WebSocket: {e}", exc_info=True)
        if self.ws_thread and self.ws_thread.is_alive():
             logger.info("Waiting for WebSocket thread to join...")
             self.ws_thread.join(timeout=5) # Wait for thread to finish
             if self.ws_thread.is_alive():
                  logger.warning("WebSocket thread did not join within timeout.")
        logger.info("WebSocket connection stopped.")

if __name__ == '__main__':
    # Example usage or testing
    logging.basicConfig(level=logging.INFO)
    logger.info("RealtimeDataHandler module")
    # Add test code here if needed
