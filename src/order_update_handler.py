# src/order_update_handler.py
"""
Handles connection to Fyers Order WebSocket for real-time order, trade,
and position updates.
"""

import logging
import threading
import time
from fyers_apiv3.FyersWebsocket import order_ws

logger = logging.getLogger(__name__)

class RealtimeOrderUpdateHandler:
    def __init__(self, access_token: str, log_path: str = 'logs/',
                 on_order_callback=None, on_trade_callback=None,
                 on_position_callback=None, on_general_callback=None):
        """
        Initializes the RealtimeOrderUpdateHandler.

        Args:
            access_token (str): The access token for Fyers API V3 WebSocket. Format: client_id:access_token
            log_path (str): Path to store WebSocket logs.
            on_order_callback (callable, optional): Function for order updates. Signature: callback(order_data: dict)
            on_trade_callback (callable, optional): Function for trade updates. Signature: callback(trade_data: dict)
            on_position_callback (callable, optional): Function for position updates. Signature: callback(position_data: dict)
            on_general_callback (callable, optional): Function for general messages. Signature: callback(general_data: dict)
        """
        if not access_token or ':' not in access_token:
            raise ValueError("Invalid access_token format. Expected 'client_id:access_token'")

        self.access_token = access_token
        self.log_path = log_path
        self.fyers_order_ws = None
        self.ws_thread = None
        self.is_connected = False
        self.subscribed_data_types = set() # e.g., {"OnOrders", "OnTrades"}
        self.lock = threading.Lock()

        # Store callbacks
        self.on_order_callback = on_order_callback
        self.on_trade_callback = on_trade_callback
        self.on_position_callback = on_position_callback
        self.on_general_callback = on_general_callback

        logger.info("RealtimeOrderUpdateHandler initialized.")

    def _run_websocket(self):
        """Target function for the WebSocket thread."""
        if not self.fyers_order_ws:
            logger.error("Order WebSocket instance not created before running thread.")
            return
        logger.info("Starting Order WebSocket connection in background thread...")
        self.fyers_order_ws.connect()
        logger.info("Order WebSocket connection loop finished in background thread.")

    def connect(self):
        """Establishes the Order WebSocket connection."""
        if self.is_connected:
            logger.warning("Order WebSocket is already connected.")
            return

        logger.info("Attempting to connect Order WebSocket...")
        try:
            self.fyers_order_ws = order_ws.FyersOrderSocket(
                access_token=self.access_token,
                log_path=self.log_path,
                write_to_file=False, # Callbacks are used
                on_connect=self.on_open,
                on_close=self.on_close,
                on_error=self.on_error,
                on_orders=self._on_orders_internal, # Internal handlers to call user callbacks
                on_positions=self._on_positions_internal,
                on_trades=self._on_trades_internal,
                on_general=self._on_general_internal
            )

            self.ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.ws_thread.start()
            time.sleep(2) # Allow time for connection

        except Exception as e:
            logger.error(f"Failed to initialize or connect Order WebSocket: {e}", exc_info=True)
            self.is_connected = False

    def on_open(self):
        """Callback function when Order WebSocket connection is established."""
        self.is_connected = True
        logger.info("Order WebSocket connection established successfully.")
        # Resubscribe if needed
        if self.subscribed_data_types:
            logger.info(f"Re-subscribing to order data types: {list(self.subscribed_data_types)}")
            self.subscribe(list(self.subscribed_data_types))

    def subscribe(self, data_types: list[str]):
        """
        Subscribes to real-time order updates.

        Args:
            data_types (list[str]): List of types like ["OnOrders", "OnTrades", "OnPositions", "OnGeneral"].
        """
        if not self.is_connected or not self.fyers_order_ws:
            logger.error("Cannot subscribe, Order WebSocket is not connected.")
            return

        if not isinstance(data_types, list):
            logger.error("data_types must be provided as a list.")
            return

        valid_types = {"OnOrders", "OnTrades", "OnPositions", "OnGeneral"}
        types_to_subscribe = [dt for dt in data_types if dt in valid_types and dt not in self.subscribed_data_types]

        if not types_to_subscribe:
            logger.info("All requested order data types are already subscribed.")
            return

        # FyersOrderSocket expects a single comma-separated string
        data_type_string = ",".join(types_to_subscribe)
        logger.info(f"Subscribing to order data types: {data_type_string}")

        try:
            self.fyers_order_ws.subscribe(data_type=data_type_string)
            with self.lock:
                self.subscribed_data_types.update(types_to_subscribe)
            logger.info(f"Successfully subscribed. Current order subscriptions: {list(self.subscribed_data_types)}")
        except Exception as e:
            logger.error(f"Error subscribing to order data types {data_type_string}: {e}", exc_info=True)

    def unsubscribe(self, data_types: list[str]):
        """Unsubscribes from real-time order updates."""
        if not self.is_connected or not self.fyers_order_ws:
            logger.error("Cannot unsubscribe, Order WebSocket is not connected.")
            return

        if not isinstance(data_types, list):
            logger.error("data_types must be provided as a list.")
            return

        valid_types = {"OnOrders", "OnTrades", "OnPositions", "OnGeneral"}
        types_to_unsubscribe = [dt for dt in data_types if dt in valid_types and dt in self.subscribed_data_types]

        if not types_to_unsubscribe:
            logger.info("None of the requested order data types are currently subscribed.")
            return

        data_type_string = ",".join(types_to_unsubscribe)
        logger.info(f"Unsubscribing from order data types: {data_type_string}")

        try:
            # Note: FyersOrderSocket might not have an explicit unsubscribe method in docs,
            # but the general pattern suggests it might exist or use subscribe with SUB_T=-1 internally.
            # Assuming an unsubscribe method exists for symmetry:
            # self.fyers_order_ws.unsubscribe(data_type=data_type_string)
            # If not, managing subscriptions might require reconnecting or handling internally.
            # For now, we'll just update our internal state.
            logger.warning("FyersOrderSocket unsubscribe method behavior not explicitly documented; updating internal state only.")
            with self.lock:
                self.subscribed_data_types.difference_update(types_to_unsubscribe)
            logger.info(f"Internal state updated for unsubscribe. Current order subscriptions: {list(self.subscribed_data_types)}")
        except AttributeError:
             logger.error("FyersOrderSocket does not have an 'unsubscribe' method. Manage subscriptions via connect/subscribe.")
        except Exception as e:
            logger.error(f"Error during unsubscribe attempt for {data_type_string}: {e}", exc_info=True)

    # --- Internal Callbacks ---
    def _on_orders_internal(self, message):
        # logger.debug(f"Order WS - Order update: {message}")
        if self.on_order_callback:
            try:
                self.on_order_callback(message)
            except Exception as e:
                logger.error(f"Error in on_order_callback: {e}", exc_info=True)

    def _on_trades_internal(self, message):
        # logger.debug(f"Order WS - Trade update: {message}")
        if self.on_trade_callback:
            try:
                self.on_trade_callback(message)
            except Exception as e:
                logger.error(f"Error in on_trade_callback: {e}", exc_info=True)

    def _on_positions_internal(self, message):
        # logger.debug(f"Order WS - Position update: {message}")
        if self.on_position_callback:
            try:
                self.on_position_callback(message)
            except Exception as e:
                logger.error(f"Error in on_position_callback: {e}", exc_info=True)

    def _on_general_internal(self, message):
        # logger.debug(f"Order WS - General message: {message}")
        if self.on_general_callback:
            try:
                self.on_general_callback(message)
            except Exception as e:
                logger.error(f"Error in on_general_callback: {e}", exc_info=True)

    def on_error(self, error):
        """Callback function when an Order WebSocket error occurs."""
        logger.error(f"Order WebSocket error: {error}")

    def on_close(self):
        """Callback function when Order WebSocket connection is closed."""
        logger.warning("Order WebSocket connection closed.")
        self.is_connected = False

    def stop(self):
        """Stops the Order WebSocket connection."""
        logger.info("Stopping Order WebSocket connection...")
        self.is_connected = False
        if self.fyers_order_ws:
            try:
                # FyersOrderSocket might not have stop_running, check SDK or handle closure
                # self.fyers_order_ws.stop_running()
                # Manually close if stop_running isn't available
                if hasattr(self.fyers_order_ws, 'close'):
                     self.fyers_order_ws.close()
                logger.info("Order WebSocket stop/close command issued.")
            except AttributeError:
                 logger.warning("Order WebSocket instance has no 'stop_running' or 'close' method.")
            except Exception as e:
                logger.error(f"Error stopping Order WebSocket: {e}", exc_info=True)
        if self.ws_thread and self.ws_thread.is_alive():
             logger.info("Waiting for Order WebSocket thread to join...")
             self.ws_thread.join(timeout=5)
             if self.ws_thread.is_alive():
                  logger.warning("Order WebSocket thread did not join within timeout.")
        logger.info("Order WebSocket connection stopped.")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("RealtimeOrderUpdateHandler module")
    # Add test code here if needed
