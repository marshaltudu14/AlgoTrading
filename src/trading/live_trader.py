#!/usr/bin/env python3
"""
Live Trading Orchestrator
"""
from src.trading.fyers_client import FyersClient
from src.trading.inference_engine import InferenceEngine
from src.trading.live_data_processor import LiveDataProcessor
import time
import logging
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/live_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.fyers_client = FyersClient(config)
        self.inference_engine = InferenceEngine(
            model_path="models/supervised_model.joblib",
            scaler_path="models/scaler.joblib",
            encoder_path="models/label_encoder.joblib"
        )
        self.live_data_processor = LiveDataProcessor(config)
        self.active_position = False
        self.position_symbol = None
        self.entry_price = None
        self.stop_loss_price = None
        self.target_price = None

    def run(self):
        logger.info("LiveTrader started.")
        while True:
            if not self.active_position:
                logger.info("No active position. Checking for new trade decision...")
                decision = self.get_trade_decision()
                logger.info(f"Model predicted: {decision}")

                if decision == "Long":
                    logger.info("Long signal received. Attempting to enter position.")
                    self.manage_active_trade(1) # 1 for Buy
                elif decision == "Short":
                    logger.info("Short signal received. Attempting to enter position.")
                    self.manage_active_trade(-1) # -1 for Sell
                else:
                    logger.info("Hold signal received. Waiting for next cycle.")
            else:
                logger.info(f"Active position in {self.position_symbol}. Monitoring for SL/TP.")
                # The WebSocket callback handles the exit, so the main loop just waits.

            time.sleep(self.config['trading']['check_interval_seconds']) # Wait for configured interval

    def get_trade_decision(self):
        symbol = self.config['trading']['instrument']
        timeframe = self.config['trading']['timeframe']

        logger.info(f"Fetching historical {timeframe} candles for {symbol} for the last 15 days...")
        historical_data = self.fyers_client.get_historical_data(
            symbol=symbol,
            timeframe=timeframe
        )

        if historical_data.empty:
            logger.warning("No historical data fetched. Cannot make a decision.")
            return "Hold"

        logger.info(f"Processing {len(historical_data)} historical candles...")
        processed_data = self.live_data_processor.process_data(historical_data)
        
        if processed_data.empty:
            logger.warning("Processed data is empty. Cannot make a decision.")
            return "Hold"

        # Get prediction for the latest candle
        latest_processed_candle = processed_data.iloc[-1]
        prediction = self.inference_engine.predict(latest_processed_candle)
        return prediction

    def manage_active_trade(self, side):
        symbol = self.config['trading']['instrument']
        qty = self.config['trading']['trade_quantity']

        logger.info(f"Placing { 'Buy' if side == 1 else 'Sell' } order for {qty} of {symbol}...")
        order_response = self.fyers_client.place_order(
            symbol=symbol,
            side=side,
            qty=qty
        )
        logger.info(f"Order response: {order_response}")

        if order_response and order_response.get('s') == 'ok':
            self.active_position = True
            self.position_symbol = symbol
            self.entry_price = order_response.get('tradedPrice', order_response.get('limitPrice')) # Get actual traded price
            
            # Fetch latest ATR for SL/TP calculation
            latest_data = self.fyers_client.get_historical_data(symbol, self.config['trading']['timeframe'], 1)
            if not latest_data.empty and 'atr' in latest_data.columns:
                current_atr = latest_data.iloc[-1]['atr']
                logger.info(f"Current ATR for SL/TP calculation: {current_atr}")

                risk_multiplier = self.config['trading']['risk_multiplier']
                reward_multiplier = self.config['trading']['reward_multiplier']

                if side == 1: # Long position
                    self.stop_loss_price = self.entry_price - (current_atr * risk_multiplier)
                    self.target_price = self.entry_price + (current_atr * reward_multiplier)
                else: # Short position
                    self.stop_loss_price = self.entry_price + (current_atr * risk_multiplier)
                    self.target_price = self.entry_price - (current_atr * reward_multiplier)
                
                logger.info(f"Position entered at: {self.entry_price:.2f}, SL: {self.stop_loss_price:.2f}, TP: {self.target_price:.2f}")

                # Connect to WebSocket and monitor for SL/TP
                logger.info(f"Connecting to WebSocket for {symbol} to monitor SL/TP...")
                self.fyers_client.connect_websocket(
                    symbol=symbol,
                    on_tick_callback=self._on_websocket_tick
                )
            else:
                logger.warning("Could not fetch ATR for SL/TP calculation. Position entered without real-time SL/TP monitoring.")
        else:
            logger.error(f"Failed to enter position: {order_response}")
            self.active_position = False # Reset active position if order failed

    def _on_websocket_tick(self, message):
        # This callback will be called by FyersClient for every incoming tick
        if 'ltp' in message and self.active_position:
            current_ltp = message['ltp']
            logger.debug(f"Received tick for {self.position_symbol}: LTP = {current_ltp:.2f}")

            # Check for Stop Loss hit
            if (self.entry_price > self.stop_loss_price and current_ltp <= self.stop_loss_price) or \
               (self.entry_price < self.stop_loss_price and current_ltp >= self.stop_loss_price):
                logger.info(f"STOP LOSS HIT for {self.position_symbol}! Current LTP: {current_ltp:.2f}")
                self._exit_current_position("SL Hit")

            # Check for Target Price hit
            elif (self.entry_price < self.target_price and current_ltp >= self.target_price) or \
                 (self.entry_price > self.target_price and current_ltp <= self.target_price):
                logger.info(f"TARGET PRICE HIT for {self.position_symbol}! Current LTP: {current_ltp:.2f}")
                self._exit_current_position("TP Hit")

    def _exit_current_position(self, reason):
        logger.info(f"Exiting position for {self.position_symbol} due to: {reason}")
        exit_response = self.fyers_client.exit_position(self.position_symbol)
        logger.info(f"Exit order response: {exit_response}")

        if exit_response and exit_response.get('s') == 'ok':
            logger.info(f"Position for {self.position_symbol} successfully exited.")
        else:
            logger.error(f"Failed to exit position for {self.position_symbol}: {exit_response}")
        
        self.active_position = False
        self.position_symbol = None
        self.entry_price = None
        self.stop_loss_price = None
        self.target_price = None
        # Disconnect WebSocket if connected
        if hasattr(self.fyers_client, 'fyers_socket') and self.fyers_client.fyers_socket:
            self.fyers_client.fyers_socket.close_connection()
            logger.info("WebSocket connection closed.")