import time
from broker.broker_api import place_order, get_order_status, cancel_order, subscribe_ticks, unsubscribe_ticks
from utils import fetch_option_master, get_current_index_price, nearest_ITM_strike
from config import INSTRUMENTS, QUANTITIES, SL_ATR_MULT, TP_ATR_MULT, TIMEFRAMES


def select_and_place(fyers, fyers_socket, instrument: str, action: str, atr: float):
    """Select nearest ITM strike, place limit order, and monitor for SL/TP."""
    index_sym = INSTRUMENTS[instrument]
    price = get_current_index_price(fyers, index_sym)
    opt_master = fetch_option_master()
    option_type = action.split('_')[-1]
    strike_sym = nearest_ITM_strike(opt_master, price, instrument, option_type)
    qty = QUANTITIES[instrument]
    # Calculate SL and TP
    if action == 'BUY_CE':
        sl = round(price - SL_ATR_MULT * atr, 2)
        tp = round(price + TP_ATR_MULT * atr, 2)
    else:
        sl = round(price + SL_ATR_MULT * atr, 2)
        tp = round(price - TP_ATR_MULT * atr, 2)
    order_data = {
        'symbol': strike_sym,
        'qty': qty,
        'type': 2,             # LIMIT
        'side': 1,             # BUY
        'limitPrice': price,
        'stopLoss': abs(price - sl),
        'takeProfit': abs(tp - price),
        'productType': 'INTRADAY',
        'validity': 'DAY',
        'offlineOrder': False
    }
    resp = place_order(fyers, order_data)
    order_id = resp.get('id')
    # Poll until order is filled
    while True:
        status = get_order_status(fyers, order_id)
        orders = status.get('orderBook', [])
        if orders and orders[0].get('status') == 2:  # 2 -> Filled
            filled_price = orders[0].get('fillPrice')
            break
        time.sleep(1)
    # Now subscribe ticks for the option symbol
    monitor_trade(order_id, fyers, fyers_socket, sl, tp, strike_sym)


def monitor_trade(order_id: str, fyers, fyers_socket, sl: float, tp: float, strike_sym: str, timeout: int = 3600):
    """Subscribe to ticks for strike_sym and cancel on SL/TP hit."""
    active = True

    def on_tick(msg):
        nonlocal active
        lp = msg.get('lp') or msg.get('last_price')
        if lp is None:
            return
        if active and (lp <= sl or lp >= tp):
            cancel_order(fyers, order_id)
            active = False

    subscribe_ticks(fyers_socket, [strike_sym], on_tick)
    start = time.time()
    while active and time.time() - start < timeout:
        time.sleep(0.5)
    unsubscribe_ticks(fyers_socket, [strike_sym])
