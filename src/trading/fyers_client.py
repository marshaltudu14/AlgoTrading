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

    def get_historical_data(self, symbol, timeframe, from_date=None, to_date=None):
        from datetime import datetime, timedelta

        # Use provided dates or default to 15 days
        if from_date is None or to_date is None:
            today = datetime.now()
            fifteen_days_ago = today - timedelta(days=15)
            from_date = fifteen_days_ago.strftime("%Y-%m-%d")
            to_date = today.strftime("%Y-%m-%d")

        data = {
            "symbol": symbol,
            "resolution": timeframe,
            "date_format": "1",
            "range_from": from_date,
            "range_to": to_date,
            "cont_flag": "1"
        }
        response = self.fyers.history(data=data)
        if response and response.get('s') == 'ok' and 'candles' in response:
            df = pd.DataFrame(response['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
            df.set_index('datetime', inplace=True)
            return df
        return pd.DataFrame()

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
