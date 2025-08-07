import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

# Add project root to path to allow imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.live_trading_service import LiveTradingService
from src.trading.position import Position
from src.config.instrument import Instrument

class TestOptionsPositionManagement(unittest.TestCase):

    def setUp(self):
        """Set up a mock LiveTradingService for testing."""
        self.user_id = "test_user"
        self.access_token = "test_token"
        self.app_id = "test_app_id"
        self.instrument = "NIFTYBANK-INDEX"
        self.timeframe = "5m"

        # Mock Fyers model and socket
        self.mock_fyers_model = MagicMock()
        self.mock_fyers_socket = MagicMock()

        # Patch the create_fyers_model and create_fyers_websocket functions
        self.patcher_model = patch('src.trading.live_trading_service.create_fyers_model', return_value=self.mock_fyers_model)
        self.patcher_socket = patch('src.trading.live_trading_service.create_fyers_websocket', return_value=self.mock_fyers_socket)
        self.patcher_model.start()
        self.patcher_socket.start()

        self.service = LiveTradingService(
            user_id=self.user_id,
            access_token=self.access_token,
            app_id=self.app_id,
            instrument=self.instrument,
            timeframe=self.timeframe,
            option_strategy="ITM"
        )
        self.service.fyers_model = self.mock_fyers_model
        self.service.fyers_socket = self.mock_fyers_socket
        self.service.current_price = 45000
        self.service.trading_env = MagicMock()
        # Create a mock Instrument object
        mock_instrument = Instrument(symbol="NIFTYBANK-INDEX", instrument_type="index", lot_size=15, tick_size=0.05, option_premium_range=[0.01, 0.05])
        self.service.trading_env.instrument = mock_instrument
        self.service.trading_env.engine._stop_loss_price = 44900
        self.service.trading_env.engine._target_profit_price = 45100
        self.service.agent = MagicMock()
        self.service.agent.act.return_value = (3, 10) # Mocking a BUY signal with quantity 10
        self.service.current_obs = MagicMock()
        self.service.trading_env.step.return_value = (MagicMock(), 0, False, {'trade_executed': True, 'trade_info': {'pnl': 0}})

        # Mock the funds call to return a valid structure
        self.mock_fyers_model.funds.return_value = {
            'fund_limit': [{"equityAmount": 0}] * 10 + [{"equityAmount": 100000}]
        }

    def tearDown(self):
        self.patcher_model.stop()
        self.patcher_socket.stop()

    @patch('src.trading.live_trading_service.CapitalAwareQuantitySelector')
    @patch('src.trading.live_trading_service.get_nearest_itm_strike', return_value=45000)
    @patch('src.trading.live_trading_service.get_nearest_expiry', return_value="2025-08-14")
    @patch('src.trading.live_trading_service.map_underlying_to_option_price', side_effect=[100, 200])
    def test_options_trade_entry(self, mock_map_price, mock_get_expiry, mock_get_strike, mock_capital_selector):
        """Test that an options trade is entered correctly."""
        # Arrange
        mock_selector_instance = MagicMock()
        mock_selector_instance.adjust_quantity_for_capital.return_value = 10
        mock_capital_selector.return_value = mock_selector_instance

        self.mock_fyers_model.place_order.return_value = {"s": "ok", "id": "12345"}
        self.service._broadcast_position_update = AsyncMock()

        # Act
        asyncio.run(self.service._execute_trade(3)) # BUY signal

        # Assert
        self.mock_fyers_model.place_order.assert_called_once()
        self.assertIsNotNone(self.service.active_position)
        self.assertEqual(self.service.active_position.instrument, "NSE:NFOBANK2025081445000CE")
        self.assertEqual(self.service.active_position.stop_loss, 100)
        self.assertEqual(self.service.active_position.target_price, 200)
        self.service._broadcast_position_update.assert_awaited_once()

    @patch('asyncio.create_task')
    def test_options_sl_tp_trigger(self, mock_create_task):
        """Test that SL/TP triggers on the underlying close the options position."""
        # Arrange
        self.service.active_position = Position(
            instrument="NSE:NFO2025081445000CE",
            direction="Long",
            entry_price=150,
            quantity=10,
            stop_loss=100,
            target_price=200,
            entry_time=datetime.now(),
            trade_type="Automated"
        )
        self.service.current_price = 44890 # Trigger SL on underlying

        # Act
        self.service._check_sl_tp_triggers()

        # Assert
        mock_create_task.assert_called_once()

if __name__ == '__main__':
    unittest.main()