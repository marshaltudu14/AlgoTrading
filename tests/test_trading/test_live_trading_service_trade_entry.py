"""
Tests for trade entry logic in LiveTradingService.
"""
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime
from src.trading.live_trading_service import LiveTradingService
from src.trading.position import Position

@pytest.fixture
def live_trading_service():
    """Fixture for LiveTradingService."""
    with patch('src.trading.live_trading_service.create_fyers_model') as mock_create_fyers_model:
        with patch('src.trading.live_trading_service.create_fyers_websocket'):
            with patch('src.trading.live_trading_service.DataLoader'):
                with patch('agents.ppo_agent.PPOAgent') as mock_ppo_agent:
                    with patch('src.trading.live_trading_service.CapitalAwareQuantitySelector') as mock_cap_selector:
                        mock_fyers_model = MagicMock()
                        mock_create_fyers_model.return_value = mock_fyers_model
                        
                        # Mock CapitalAwareQuantitySelector
                        mock_cap_instance = MagicMock()
                        mock_cap_selector.return_value = mock_cap_instance
                        mock_cap_instance.adjust_quantity_for_capital.return_value = 1
                        
                        service = LiveTradingService(
                            user_id="test_user",
                            access_token="test_token",
                            app_id="test_app_id",
                            instrument="NIFTY50",
                            timeframe="5m",
                            option_strategy="None"  # Disable options trading for tests
                        )
                        service.fyers_model = mock_fyers_model
                        service.agent = mock_ppo_agent
                        service.trading_env = MagicMock()
                        service.current_obs = MagicMock()
                        service.current_price = 100
                        
                        # Mock trading_env.step to return proper tuple
                        service.trading_env.step.return_value = (
                            MagicMock(),  # obs
                            0.0,          # reward
                            False,        # done
                            {'trade_executed': True, 'trade_info': {'pnl': 0.0}}  # info
                        )
                        
                        # Mock _get_fyers_symbol method
                        service._get_fyers_symbol = MagicMock(return_value="NSE:NIFTY50-INDEX")
                        
                        # Mock async methods to avoid asyncio issues
                        service._broadcast_position_update = MagicMock()
                        service._broadcast_update = MagicMock()
                        
                        # Initialize required attributes
                        service.active_position = None
                        service.today_trades = 0
                        service.total_trades = 0
                        service.current_pnl = 0.0
                        service.win_count = 0
                        service.current_price = 100.0  # Set a numeric value for current_price
                        
                        yield service

@pytest.mark.asyncio
async def test_trade_entry_buy_success(live_trading_service):
    """Test successful buy trade entry."""
    live_trading_service.agent.act.return_value = (3, 1)  # BUY signal, quantity 1
    live_trading_service.fyers_model.funds.return_value = {'fund_limit': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'equityAmount': 100000}]}
    live_trading_service.fyers_model.place_order.return_value = {"s": "ok", "id": "12345"}
    live_trading_service.trading_env.engine._stop_loss_price = 95
    live_trading_service.trading_env.engine._target_profit_price = 105

    await live_trading_service._execute_trade(3)

    assert live_trading_service.active_position is not None
    assert isinstance(live_trading_service.active_position, Position)
    assert live_trading_service.active_position.direction == "Long"
    assert live_trading_service.active_position.quantity == 1

@pytest.mark.asyncio
async def test_trade_entry_sell_success(live_trading_service):
    """Test successful sell trade entry."""
    live_trading_service.agent.act.return_value = (1, 1)  # SELL signal, quantity 1
    live_trading_service.fyers_model.funds.return_value = {'fund_limit': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'equityAmount': 100000}]}
    live_trading_service.fyers_model.place_order.return_value = {"s": "ok", "id": "12345"}
    live_trading_service.trading_env.engine._stop_loss_price = 105
    live_trading_service.trading_env.engine._target_profit_price = 95

    await live_trading_service._execute_trade(1)

    assert live_trading_service.active_position is not None
    assert isinstance(live_trading_service.active_position, Position)
    assert live_trading_service.active_position.direction == "Short"

@pytest.mark.asyncio
async def test_trade_entry_order_failure(live_trading_service):
    """Test trade entry when order placement fails."""
    live_trading_service.agent.act.return_value = (3, 1)  # BUY signal, quantity 1
    live_trading_service.fyers_model.funds.return_value = {'fund_limit': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'equityAmount': 100000}]}
    live_trading_service.fyers_model.place_order.return_value = {"s": "error", "message": "Insufficient funds"}

    await live_trading_service._execute_trade(3)

    assert live_trading_service.active_position is None

@pytest.mark.asyncio
async def test_trade_entry_insufficient_capital(live_trading_service):
    """Test trade entry with insufficient capital."""
    live_trading_service.agent.act.return_value = (3, 1)  # BUY signal, quantity 1
    live_trading_service.fyers_model.funds.return_value = {'fund_limit': [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {'equityAmount': 100}]}

    await live_trading_service._execute_trade(3)

    assert live_trading_service.active_position is None
