"""
Tests for SL/TP trigger logic in LiveTradingService.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.trading.live_trading_service import LiveTradingService
from src.trading.position import Position
from datetime import datetime

@pytest.fixture
def live_trading_service():
    """Fixture for LiveTradingService."""
    with patch('src.trading.live_trading_service.create_fyers_model') as mock_create_fyers_model:
        with patch('src.trading.live_trading_service.create_fyers_websocket'):
            with patch('src.trading.live_trading_service.DataLoader'):
                with patch('agents.ppo_agent.PPOAgent') as mock_ppo_agent:
                    with patch('src.trading.live_trading_service.CapitalAwareQuantitySelector'):
                        mock_fyers_model = MagicMock()
                        mock_create_fyers_model.return_value = mock_fyers_model
                        
                        service = LiveTradingService(
                            user_id="test_user",
                            access_token="test_token",
                            app_id="test_app_id",
                            instrument="NIFTY50",
                            timeframe="5m"
                        )
                        service.fyers_model = mock_fyers_model
                        service.agent = mock_ppo_agent
                        service.trading_env = MagicMock()
                        service.current_obs = MagicMock()
                        
                        # Mock the _close_position method (note: it's called without await in actual code)
                        service._close_position = MagicMock()
                        
                        # Mock async broadcast methods
                        service._broadcast_position_update = AsyncMock()
                        service._broadcast_update = AsyncMock()
                        
                        return service

def test_long_position_sl_trigger(live_trading_service):
    """Test stop-loss trigger for a long position."""
    live_trading_service.active_position = Position(
        instrument="NIFTY50",
        direction="Long",
        entry_price=100,
        quantity=1,
        stop_loss=95,
        target_price=105,
        entry_time=datetime.now(),
        trade_type="Automated"
    )
    live_trading_service.current_price = 90
    live_trading_service._check_sl_tp_triggers()
    
    # Verify _close_position was called with correct reason
    live_trading_service._close_position.assert_called_once_with("SL Hit")

def test_long_position_tp_trigger(live_trading_service):
    """Test target-profit trigger for a long position."""
    live_trading_service.active_position = Position(
        instrument="NIFTY50",
        direction="Long",
        entry_price=100,
        quantity=1,
        stop_loss=95,
        target_price=105,
        entry_time=datetime.now(),
        trade_type="Automated"
    )
    live_trading_service.current_price = 110
    live_trading_service._check_sl_tp_triggers()
    
    # Verify _close_position was called with correct reason
    live_trading_service._close_position.assert_called_once_with("TP Hit")

def test_short_position_sl_trigger(live_trading_service):
    """Test stop-loss trigger for a short position."""
    live_trading_service.active_position = Position(
        instrument="NIFTY50",
        direction="Short",
        entry_price=100,
        quantity=1,
        stop_loss=105,
        target_price=95,
        entry_time=datetime.now(),
        trade_type="Automated"
    )
    live_trading_service.current_price = 110
    live_trading_service._check_sl_tp_triggers()
    
    # Verify _close_position was called with correct reason
    live_trading_service._close_position.assert_called_once_with("SL Hit")

def test_short_position_tp_trigger(live_trading_service):
    """Test target-profit trigger for a short position."""
    live_trading_service.active_position = Position(
        instrument="NIFTY50",
        direction="Short",
        entry_price=100,
        quantity=1,
        stop_loss=105,
        target_price=95,
        entry_time=datetime.now(),
        trade_type="Automated"
    )
    live_trading_service.current_price = 90
    live_trading_service._check_sl_tp_triggers()
    
    # Verify _close_position was called with correct reason
    live_trading_service._close_position.assert_called_once_with("TP Hit")