import pytest
from src.backtesting.engine import BacktestingEngine
from src.config.instrument import Instrument

@pytest.fixture
def sample_instrument_stock():
    return Instrument(symbol="AAPL", type="STOCK", lot_size=1, tick_size=0.01)

@pytest.fixture
def sample_instrument_option():
    return Instrument(symbol="BANKNIFTY", type="OPTION", lot_size=25, tick_size=0.05)

@pytest.fixture
def engine_stock(sample_instrument_stock):
    return BacktestingEngine(initial_capital=100000, instrument=sample_instrument_stock)

@pytest.fixture
def engine_option(sample_instrument_option):
    return BacktestingEngine(initial_capital=100000, instrument=sample_instrument_option)

def test_initial_capital_validation():
    with pytest.raises(ValueError, match="Initial capital must be positive."):
        BacktestingEngine(initial_capital=0, instrument=Instrument("TEST", "STOCK", 1, 0.01))
    with pytest.raises(ValueError, match="Initial capital must be positive."):
        BacktestingEngine(initial_capital=-100, instrument=Instrument("TEST", "STOCK", 1, 0.01))

def test_reset(engine_stock):
    engine_stock.execute_trade("BUY_LONG", 100, 10)
    engine_stock.reset()
    state = engine_stock.get_account_state()
    assert state["capital"] == 100000
    assert state["current_position_quantity"] == 0
    assert state["realized_pnl"] == 0
    assert state["unrealized_pnl"] == 0

def test_buy_long_stock(engine_stock):
    initial_capital = engine_stock._capital
    engine_stock.execute_trade("BUY_LONG", 100, 10)
    state = engine_stock.get_account_state(current_price=100)
    expected_cost = (100 * 10 * engine_stock.instrument.lot_size) + engine_stock.BROKERAGE_ENTRY
    assert state["capital"] == initial_capital - expected_cost
    assert state["current_position_quantity"] == 10
    assert state["current_position_entry_price"] == 100
    assert state["is_position_open"] is True

def test_buy_long_option(engine_option):
    initial_capital = engine_option._capital
    proxy_premium = 5.0 # Example premium
    quantity = 2 # lots
    engine_option.execute_trade("BUY_LONG", 1000, quantity, proxy_premium=proxy_premium) # Price here is underlying, not premium
    state = engine_option.get_account_state(current_price=1000)
    expected_cost = (proxy_premium * quantity * engine_option.instrument.lot_size) + engine_option.BROKERAGE_ENTRY
    assert state["capital"] == initial_capital - expected_cost
    assert state["current_position_quantity"] == quantity
    assert state["current_position_entry_price"] == 1000 # Entry price is underlying for options
    assert state["is_position_open"] is True

def test_insufficient_capital_buy(engine_stock):
    engine_stock.execute_trade("BUY_LONG", 1000000, 10) # Attempt to buy more than capital
    state = engine_stock.get_account_state()
    assert state["capital"] == 100000 # Capital should remain unchanged
    assert state["current_position_quantity"] == 0

def test_close_long_stock(engine_stock):
    engine_stock.execute_trade("BUY_LONG", 100, 10)
    initial_capital_after_buy = engine_stock._capital
    engine_stock.execute_trade("CLOSE_LONG", 110, 10)
    state = engine_stock.get_account_state()
    expected_pnl = (110 - 100) * 10 * engine_stock.instrument.lot_size
    expected_capital = initial_capital_after_buy + expected_pnl - engine_stock.BROKERAGE_EXIT
    assert state["capital"] == expected_capital
    assert state["current_position_quantity"] == 0
    assert state["realized_pnl"] == expected_pnl
    assert state["is_position_open"] is False

def test_close_long_option(engine_option):
    # Simulate buying an option with premium 5, underlying 1000
    engine_option.execute_trade("BUY_LONG", 1000, 2, proxy_premium=5.0)
    initial_capital_after_buy = engine_option._capital

    # Simulate closing the option when underlying is 1050 (intrinsic value 50)
    # P&L = (intrinsic_value * quantity * lot_size) - (premium_paid * quantity * lot_size)
    # Max loss capped at premium paid
    close_price_underlying = 1050
    intrinsic_value_at_close = max(0.0, close_price_underlying - engine_option._current_position_entry_price)
    
    # The original premium paid per lot was 5.0
    premium_paid_per_lot = 5.0
    expected_pnl = (intrinsic_value_at_close * 2 * engine_option.instrument.lot_size) - \
                   (premium_paid_per_lot * 2 * engine_option.instrument.lot_size)
    
    # Max loss check
    max_loss = -(premium_paid_per_lot * 2 * engine_option.instrument.lot_size)
    expected_pnl = max(expected_pnl, max_loss)

    engine_option.execute_trade("CLOSE_LONG", close_price_underlying, 2, proxy_premium=5.0) # proxy_premium here is not used for PnL calc
    state = engine_option.get_account_state()
    expected_capital = initial_capital_after_buy + expected_pnl - engine_option.BROKERAGE_EXIT
    assert state["capital"] == pytest.approx(expected_capital)
    assert state["current_position_quantity"] == 0
    assert state["realized_pnl"] == pytest.approx(expected_pnl)
    assert state["is_position_open"] is False

def test_trailing_stop_long_position(engine_stock):
    # Use smaller ATR to avoid take profit interference (TP will be at 100 + 2*0.5 = 101)
    engine_stock.execute_trade("BUY_LONG", 100, 10, atr_value=0.5)
    # Price moves up, trailing stop should move up (stay below TP of 101)
    engine_stock.execute_trade("HOLD", 100.8, 0) # Price moves to 100.8 (below TP of 101)
    assert engine_stock._peak_price == 100.8
    assert engine_stock._trailing_stop_price == pytest.approx(100.8 * (1 - engine_stock.trailing_stop_percentage))

    # Price drops and hits trailing stop
    initial_capital_before_close = engine_stock._capital
    trailing_stop_price = 100.8 * (1 - engine_stock.trailing_stop_percentage)
    engine_stock.execute_trade("HOLD", trailing_stop_price - 0.01, 0) # Price drops below trailing stop
    state = engine_stock.get_account_state()
    assert state["current_position_quantity"] == 0 # Position should be closed
    # Verify PnL calculation for trailing stop close
    # PnL = (close_price - entry_price) * quantity * lot_size
    close_price = trailing_stop_price - 0.01
    expected_pnl = (close_price - 100) * 10 * engine_stock.instrument.lot_size
    expected_capital = initial_capital_before_close + expected_pnl - engine_stock.BROKERAGE_EXIT
    assert state["capital"] == pytest.approx(expected_capital)

def test_unrealized_pnl_stock(engine_stock):
    engine_stock.execute_trade("BUY_LONG", 100, 10)
    engine_stock._update_unrealized_pnl(105)
    expected_unrealized_pnl = (105 - 100) * 10 * engine_stock.instrument.lot_size
    assert engine_stock._unrealized_pnl == expected_unrealized_pnl

def test_unrealized_pnl_option(engine_option):
    engine_option.execute_trade("BUY_LONG", 1000, 2, proxy_premium=5.0)
    # Underlying moves to 1020, intrinsic value 20
    engine_option._update_unrealized_pnl(1020)
    intrinsic_value = max(0.0, 1020 - 1000)
    # Unrealized PnL = (intrinsic_value * quantity * lot_size) - (premium_paid * quantity * lot_size)
    expected_unrealized_pnl = (intrinsic_value * 2 * engine_option.instrument.lot_size) - \
                              (5.0 * 2 * engine_option.instrument.lot_size)
    assert engine_option._unrealized_pnl == pytest.approx(expected_unrealized_pnl)

