"""
Order management module for AlgoTrading.
Handles order placement, monitoring, and execution.
"""
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime

from fyers_apiv3 import fyersModel
from fyers_apiv3.FyersWebsocket import data_ws

from core.logging_setup import get_logger
from core.config import (
    INSTRUMENTS, 
    QUANTITIES, 
    SL_ATR_MULT, 
    TP_ATR_MULT, 
    TIMEFRAMES,
    STRIKE_STEP
)
from core.constants import (
    FYERS_ORDER_TYPE_LIMIT,
    FYERS_ORDER_SIDE_BUY,
    FYERS_PRODUCT_TYPE_INTRADAY,
    FYERS_VALIDITY_DAY,
    OptionType
)
from trading.broker_api import (
    place_order, 
    get_order_status, 
    cancel_order, 
    subscribe_ticks, 
    unsubscribe_ticks
)
from data.fetcher import (
    fetch_option_chain,
    get_nearest_expiry_date,
    select_option_strike
)

logger = get_logger(__name__)


def select_and_place(
    fyers: fyersModel.FyersModel,
    fyers_socket: data_ws.FyersDataSocket,
    instrument: str,
    action: str,
    atr: float,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Select nearest ITM strike, place limit order, and monitor for SL/TP.
    
    Args:
        fyers: Fyers API client
        fyers_socket: Fyers websocket client
        instrument: Instrument to trade
        action: Action to take ('BUY_CE' or 'BUY_PE')
        atr: ATR value for SL/TP calculation
        max_retries: Maximum number of retries
        
    Returns:
        Order response
    """
    logger.info(f"Selecting and placing order for {instrument}: {action}")
    
    # Get index symbol
    index_sym = INSTRUMENTS[instrument]
    
    # Get current index price
    from data.fetcher import get_current_index_price
    price = get_current_index_price(fyers, index_sym)
    logger.info(f"Current price for {instrument}: {price}")
    
    # Get option type from action
    option_type = action.split('_')[-1]
    logger.info(f"Option type: {option_type}")
    
    # Get nearest expiry date
    expiry_date = get_nearest_expiry_date(instrument)
    if not expiry_date:
        logger.error(f"Failed to get expiry date for {instrument}")
        return {"s": "error", "message": "Failed to get expiry date"}
    
    logger.info(f"Nearest expiry date: {expiry_date}")
    
    # Get option chain
    option_chain = fetch_option_chain(instrument, expiry_date)
    if not option_chain:
        logger.error(f"Failed to get option chain for {instrument}")
        return {"s": "error", "message": "Failed to get option chain"}
    
    # Select strike
    strike_sym = select_option_strike(price, action, option_chain, STRIKE_STEP)
    if not strike_sym:
        logger.error(f"Failed to select strike for {instrument}")
        return {"s": "error", "message": "Failed to select strike"}
    
    logger.info(f"Selected strike: {strike_sym}")
    
    # Get quantity
    qty = QUANTITIES[instrument]
    logger.info(f"Quantity: {qty}")
    
    # Calculate SL and TP
    if action == 'BUY_CE':
        sl = round(price - SL_ATR_MULT * atr, 2)
        tp = round(price + TP_ATR_MULT * atr, 2)
    else:  # BUY_PE
        sl = round(price + SL_ATR_MULT * atr, 2)
        tp = round(price - TP_ATR_MULT * atr, 2)
    
    logger.info(f"SL: {sl}, TP: {tp}")
    
    # Place order
    order_data = {
        "symbol": strike_sym,
        "qty": qty,
        "type": FYERS_ORDER_TYPE_LIMIT,
        "side": FYERS_ORDER_SIDE_BUY,
        "limitPrice": price,
        "stopLoss": abs(price - sl),
        "takeProfit": abs(tp - price),
        "productType": FYERS_PRODUCT_TYPE_INTRADAY,
        "validity": FYERS_VALIDITY_DAY,
        "offlineOrder": False
    }
    
    # Place order with retries
    retries = 0
    order_id = None
    
    while retries < max_retries and not order_id:
        resp = place_order(fyers, **order_data)
        
        if resp.get('s') == 'ok':
            order_id = resp.get('id')
            logger.info(f"Order placed successfully: ID={order_id}")
        else:
            logger.warning(f"Order placement failed: {resp}")
            retries += 1
            time.sleep(1)
    
    if not order_id:
        logger.error(f"Failed to place order after {max_retries} retries")
        return {"s": "error", "message": "Failed to place order"}
    
    # Poll until order is filled
    filled = False
    filled_price = None
    poll_count = 0
    
    while not filled and poll_count < 30:  # Poll for up to 30 seconds
        status = get_order_status(fyers, order_id)
        orders = status.get('orderBook', [])
        
        if orders and orders[0].get('status') == 2:  # 2 -> Filled
            filled = True
            filled_price = orders[0].get('avgPrice')
            logger.info(f"Order filled at price: {filled_price}")
            break
        
        poll_count += 1
        time.sleep(1)
    
    if not filled:
        logger.warning(f"Order not filled after 30 seconds, cancelling")
        cancel_order(fyers, order_id)
        return {"s": "error", "message": "Order not filled"}
    
    # Now subscribe ticks for the option symbol
    monitor_result = monitor_trade(order_id, fyers, fyers_socket, sl, tp, strike_sym)
    
    return {
        "s": "ok",
        "order_id": order_id,
        "symbol": strike_sym,
        "filled_price": filled_price,
        "sl": sl,
        "tp": tp,
        "monitor_result": monitor_result
    }


def monitor_trade(
    order_id: str,
    fyers: fyersModel.FyersModel,
    fyers_socket: data_ws.FyersDataSocket,
    sl: float,
    tp: float,
    strike_sym: str,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Subscribe to ticks for strike_sym and cancel on SL/TP hit.
    
    Args:
        order_id: Order ID to monitor
        fyers: Fyers API client
        fyers_socket: Fyers websocket client
        sl: Stop loss price
        tp: Take profit price
        strike_sym: Strike symbol to monitor
        timeout: Timeout in seconds
        
    Returns:
        Monitoring result
    """
    logger.info(f"Monitoring trade for {strike_sym}: SL={sl}, TP={tp}")
    
    active = True
    exit_reason = None
    exit_price = None
    
    def on_tick(msg):
        nonlocal active, exit_reason, exit_price
        
        # Extract last price from tick data
        lp = msg.get('lp') or msg.get('last_price')
        
        if lp is None:
            return
        
        logger.debug(f"Tick for {strike_sym}: {lp}")
        
        if active:
            if lp <= sl:
                logger.info(f"SL hit for {strike_sym} at {lp}")
                cancel_order(fyers, order_id)
                active = False
                exit_reason = "SL"
                exit_price = lp
            elif lp >= tp:
                logger.info(f"TP hit for {strike_sym} at {lp}")
                cancel_order(fyers, order_id)
                active = False
                exit_reason = "TP"
                exit_price = lp
    
    # Subscribe to ticks
    subscribe_ticks(fyers_socket, [strike_sym], on_tick)
    
    # Monitor for timeout
    start = time.time()
    while active and time.time() - start < timeout:
        time.sleep(0.5)
    
    # Unsubscribe from ticks
    unsubscribe_ticks(fyers_socket, [strike_sym])
    
    # If still active after timeout, cancel order
    if active:
        logger.info(f"Timeout for {strike_sym}, cancelling order")
        cancel_order(fyers, order_id)
        exit_reason = "TIMEOUT"
    
    return {
        "exit_reason": exit_reason,
        "exit_price": exit_price,
        "duration": time.time() - start
    }
