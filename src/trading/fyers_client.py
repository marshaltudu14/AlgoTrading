#!/usr/bin/env python3
"""
Fyers API Client
"""

import pandas as pd
from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws
from src.auth import fyers_auth

class FyersClient:
    def __init__(self, config=None):
        self.fyers = fyers_auth.fyers

        # No hardcoded symbols - let the caller specify exact Fyers symbol

    def get_historical_data(self, symbol, timeframe, days=30, from_date=None, to_date=None):
        """
        Fetch historical data from Fyers API with configurable parameters.

        Args:
            symbol (str): Trading symbol (e.g., "NSE:BANKNIFTY23DEC46500CE")
            timeframe (str): Timeframe for candles ("1", "5", "15", "30", "60", "D")
            days (int): Number of days to fetch (default: 30)
            from_date (str): Start date in YYYY-MM-DD format (overrides days)
            to_date (str): End date in YYYY-MM-DD format (overrides days)

        Returns:
            pd.DataFrame: Historical data with OHLCV columns
        """
        from datetime import datetime, timedelta
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Use provided dates or calculate from days parameter
            if from_date is None or to_date is None:
                today = datetime.now()
                start_date = today - timedelta(days=days)
                from_date = start_date.strftime("%Y-%m-%d")
                to_date = today.strftime("%Y-%m-%d")

            logger.info(f"Fetching {days} days of {timeframe}-minute data for {symbol}")
            logger.info(f"Date range: {from_date} to {to_date}")

            data = {
                "symbol": symbol,
                "resolution": timeframe,
                "date_format": "1",
                "range_from": from_date,
                "range_to": to_date,
                "cont_flag": "1"
            }

            response = self.fyers.history(data=data)

            if not response:
                logger.error(f"No response received from Fyers API for {symbol}")
                return pd.DataFrame()

            if response.get('s') != 'ok':
                logger.error(f"Fyers API error for {symbol}: {response.get('message', 'Unknown error')}")
                return pd.DataFrame()

            if 'candles' not in response or not response['candles']:
                logger.warning(f"No candle data received for {symbol}")
                return pd.DataFrame()

            # Create DataFrame with proper column names
            # CRITICAL: Keep datetime as epoch timestamps for feature generator processing
            df = pd.DataFrame(response['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            # Do NOT convert to datetime here - let feature generator handle it
            # This preserves the raw epoch timestamps that the feature generator expects

            logger.info(f"Successfully fetched {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    # Removed unnecessary helper methods - caller should provide exact Fyers symbol

    def place_order(self, symbol, side, qty):
        data = {
            "symbol": symbol,
            "qty": qty,
            "type": 2,  # Market order
            "side": side, # 1 for Buy, -1 for Sell
            "productType": "INTRADAY",
            "limitPrice": 0,
            "stopPrice": 0,
            "validity": "DAY",
            "disclosedQty": 0,
            "offlineOrder": False,
            "stopLoss": 0,
            "takeProfit": 0
        }
        response = self.fyers.place_order(data=data)
        return response

    def exit_position(self, symbol):
        # Assuming single position per symbol
        positions = self.fyers.get_positions()
        for pos in positions['netPositions']:
            if pos['symbol'] == symbol:
                return self.fyers.exit_positions(id=pos['id'])
        return None

    def get_open_positions(self):
        return self.fyers.get_positions()

    def connect_websocket(self, symbol, on_tick_callback):
        self.fyers_socket = fyers_auth.fyers_socket
        self.fyers_socket.subscribe(symbol=symbol, data_type="symbolData")
        self.fyers_socket.on_message = on_tick_callback
