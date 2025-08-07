import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import pandas as pd
import torch

# Add project root to path to allow imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.trading.live_trading_service import LiveTradingService
from src.trading.position import Position

class TestTradeExitLogic(unittest.TestCase):

    def setUp(self):
        """Set up a mock LiveTradingService for testing."""
        self.user_id = "test_user"
        self.access_token = "test_token"
        self.app_id = "test_app_id"
        self.instrument = "NSE:NIFTYBANK-INDEX"
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
            timeframe=self.timeframe
        )
        self.service.fyers_model = self.mock_fyers_model
        self.service.fyers_socket = self.mock_fyers_socket
        self.service.websocket_clients.append(AsyncMock())

    def tearDown(self):
        self.patcher_model.stop()
        self.patcher_socket.stop()

    def test_successful_trade_exit(self):
        """Test that a trade is exited successfully on the first attempt."""
        # Arrange
        self.service.active_position = Position(
            instrument=self.instrument,
            direction="Long",
            entry_price=45000,
            quantity=10,
            stop_loss=44900,
            target_price=45100,
            entry_time=datetime.now(),
            trade_type="Automated"
        )
        self.mock_fyers_model.exit_positions.return_value = {"s": "ok"}
        self.service._broadcast_position_update = AsyncMock()

        # Act
        asyncio.run(self.service._close_position("Test Exit"))

        # Assert
        self.mock_fyers_model.exit_positions.assert_called_once_with(id=self.instrument)
        self.assertIsNone(self.service.active_position)
        self.service._broadcast_position_update.assert_awaited_once()

    def test_trade_exit_with_retries(self):
        """Test that the system retries exiting a trade on failure."""
        # Arrange
        self.service.active_position = Position(
            instrument=self.instrument,
            direction="Long",
            entry_price=45000,
            quantity=10,
            stop_loss=44900,
            target_price=45100,
            entry_time=datetime.now(),
            trade_type="Automated"
        )
        # Simulate failure on first two calls, success on the third
        self.mock_fyers_model.exit_positions.side_effect = [
            {"s": "error", "message": "Failed"},
            {"s": "error", "message": "Failed"},
            {"s": "ok"}
        ]
        self.service._broadcast_position_update = AsyncMock()

        # Act
        asyncio.run(self.service._close_position("Test Retry Exit"))

        # Assert
        self.assertEqual(self.mock_fyers_model.exit_positions.call_count, 3)
        self.assertIsNone(self.service.active_position)
        self.service._broadcast_position_update.assert_awaited_once()

    def test_trade_exit_failure_after_retries(self):
        """Test that the system stops retrying after 3 failed attempts."""
        # Arrange
        self.service.active_position = Position(
            instrument=self.instrument,
            direction="Long",
            entry_price=45000,
            quantity=10,
            stop_loss=44900,
            target_price=45100,
            entry_time=datetime.now(),
            trade_type="Automated"
        )
        self.mock_fyers_model.exit_positions.return_value = {"s": "error", "message": "Failed"}
        self.service._broadcast_position_update = AsyncMock()
        self.service._broadcast_update = AsyncMock()

        # Act
        asyncio.run(self.service._close_position("Test Failure Exit"))

        # Assert
        self.assertEqual(self.mock_fyers_model.exit_positions.call_count, 3)
        self.assertIsNotNone(self.service.active_position) # Position should not be cleared
        self.service._broadcast_position_update.assert_not_awaited()
        self.service._broadcast_update.assert_awaited_once() # Should send an error message

    @patch('asyncio.create_task')
    def test_sl_tp_trigger_calls_close_position(self, mock_create_task):
        """Test that SL/TP triggers correctly call _close_position via asyncio.create_task."""
        # Arrange
        self.service.active_position = Position(
            instrument=self.instrument,
            direction="Long",
            entry_price=45000,
            quantity=10,
            stop_loss=44950,
            target_price=45050,
            entry_time=datetime.now(),
            trade_type="Automated"
        )
        self.service.current_price = 44940 # Trigger SL

        # Act
        self.service._check_sl_tp_triggers()

        # Assert
        mock_create_task.assert_called_once()
        # You can add more detailed assertions here if needed,
        # for example by inspecting the coroutine passed to create_task
        # but this confirms the correct async invocation.

if __name__ == '__main__':
    unittest.main()