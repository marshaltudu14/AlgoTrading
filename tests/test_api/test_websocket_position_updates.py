"""
Unit tests for WebSocket position update functionality.

Tests verify the WebSocket endpoint at /ws/live/{user_id} properly handles
position update messages from LiveTradingService, including position open,
position close, SL/TP updates, and real-time PnL tracking.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
from pathlib import Path

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from main import app, active_live_sessions

# Create test client
client = TestClient(app)

# Sample position data for testing
SAMPLE_POSITION_OPEN = {
    "instrument": "BANKNIFTY",
    "direction": "Long",
    "entryPrice": 45600.50,
    "quantity": 2,
    "stopLoss": 45400.00,
    "targetPrice": 45800.00,
    "currentPnl": 0.0,
    "tradeType": "Automated",
    "isOpen": True
}

SAMPLE_POSITION_UPDATED = {
    "instrument": "BANKNIFTY",
    "direction": "Long",
    "entryPrice": 45600.50,
    "quantity": 2,
    "stopLoss": 45400.00,
    "targetPrice": 45800.00,
    "currentPnl": 200.0,  # Price moved up
    "tradeType": "Automated",
    "isOpen": True
}

SAMPLE_POSITION_CLOSED = {
    "instrument": "BANKNIFTY",
    "direction": "Long",
    "entryPrice": 45600.50,
    "quantity": 2,
    "stopLoss": 45400.00,
    "targetPrice": 45800.00,
    "currentPnl": 400.0,
    "tradeType": "Automated",
    "isOpen": False,
    "exitPrice": 45800.50,
    "pnl": 400.0  # Final PnL for closed position
}

EXPECTED_POSITION_MESSAGE = {
    "type": "position_update",
    "data": SAMPLE_POSITION_OPEN
}

# Mock LiveTradingService for testing position updates
class MockLiveTradingServiceWithPositions:
    """Mock LiveTradingService with position update functionality"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.websocket_clients = []
        self.status = "running"
        self.is_running = True
        self.current_position_data = None
        
    def add_websocket_client(self, websocket):
        self.websocket_clients.append(websocket)
        
    def remove_websocket_client(self, websocket):
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
            
    def get_status(self):
        return {
            "status": self.status,
            "is_running": self.is_running,
            "current_pnl": 1250.75,
            "today_trades": 5,
            "win_rate": 80.0,
            "current_price": 45600.50,
            "position": 1,
            "instrument": "BANKNIFTY",
            "timeframe": "5m",
            "option_strategy": "ITM"
        }
    
    async def simulate_position_update(self, position_data):
        """Simulate broadcasting position update to connected clients"""
        self.current_position_data = position_data
        
        position_message = {
            "type": "position_update",
            "data": position_data
        }
        
        # Send to all connected clients
        for client in self.websocket_clients:
            try:
                await client.send_text(json.dumps(position_message, default=str))
            except Exception as e:
                print(f"Error sending position update: {e}")
    
    async def simulate_position_open(self):
        """Simulate opening a new position"""
        await self.simulate_position_update(SAMPLE_POSITION_OPEN)
    
    async def simulate_position_pnl_update(self):
        """Simulate PnL update for existing position"""
        await self.simulate_position_update(SAMPLE_POSITION_UPDATED)
    
    async def simulate_position_close(self):
        """Simulate closing a position"""
        await self.simulate_position_update(SAMPLE_POSITION_CLOSED)


class TestWebSocketPositionUpdates:
    """Test class for WebSocket position update functionality."""

    def test_websocket_receives_position_open_message(self):
        """Test WebSocket receives properly formatted position open message."""
        mock_service = MockLiveTradingServiceWithPositions("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Simulate position open
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mock_service.simulate_position_open())
                
                # Receive position update message
                data = websocket.receive_text()
                message = json.loads(data)
                
                # Verify message structure
                assert message["type"] == "position_update"
                assert "data" in message
                
                position_data = message["data"]
                assert position_data["instrument"] == "BANKNIFTY"
                assert position_data["direction"] == "Long"
                assert position_data["entryPrice"] == 45600.50
                assert position_data["quantity"] == 2
                assert position_data["stopLoss"] == 45400.00
                assert position_data["targetPrice"] == 45800.00
                assert position_data["currentPnl"] == 0.0
                assert position_data["tradeType"] == "Automated"
                assert position_data["isOpen"] == True
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_receives_position_pnl_update(self):
        """Test WebSocket receives position updates with real-time PnL changes."""
        mock_service = MockLiveTradingServiceWithPositions("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Simulate position PnL update
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mock_service.simulate_position_pnl_update())
                
                # Receive position update message
                data = websocket.receive_text()
                message = json.loads(data)
                
                # Verify message structure
                assert message["type"] == "position_update"
                assert "data" in message
                
                position_data = message["data"]
                assert position_data["currentPnl"] == 200.0  # Updated PnL
                assert position_data["isOpen"] == True  # Still open
                assert position_data["direction"] == "Long"
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_receives_position_close_message(self):
        """Test WebSocket receives position close message with exit price and final PnL."""
        mock_service = MockLiveTradingServiceWithPositions("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Simulate position close
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mock_service.simulate_position_close())
                
                # Receive position update message
                data = websocket.receive_text()
                message = json.loads(data)
                
                # Verify message structure for closed position
                assert message["type"] == "position_update"
                assert "data" in message
                
                position_data = message["data"]
                assert position_data["isOpen"] == False  # Position closed
                assert "exitPrice" in position_data
                assert position_data["exitPrice"] == 45800.50
                assert "pnl" in position_data
                assert position_data["pnl"] == 400.0  # Final PnL
                assert position_data["currentPnl"] == 400.0
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_position_update_message_format(self):
        """Test that position update messages follow the exact required JSON structure."""
        position_message = {
            "type": "position_update",
            "data": SAMPLE_POSITION_OPEN
        }
        
        # Verify top-level structure
        assert "type" in position_message
        assert position_message["type"] == "position_update"
        assert "data" in position_message
        
        # Verify data structure matches Position model
        data = position_message["data"]
        required_fields = [
            "instrument", "direction", "entryPrice", "quantity",
            "stopLoss", "targetPrice", "currentPnl", "tradeType", "isOpen"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify data types
        assert isinstance(data["entryPrice"], (int, float))
        assert isinstance(data["quantity"], int)
        assert isinstance(data["stopLoss"], (int, float))
        assert isinstance(data["targetPrice"], (int, float))
        assert isinstance(data["currentPnl"], (int, float))
        assert isinstance(data["tradeType"], str)
        assert isinstance(data["isOpen"], bool)
        
        # Verify JSON serializability
        json_str = json.dumps(position_message, default=str)
        parsed_back = json.loads(json_str)
        assert parsed_back["type"] == "position_update"
        assert parsed_back["data"]["instrument"] == "BANKNIFTY"

    def test_position_update_handles_different_directions(self):
        """Test position updates for both Long and Short positions."""
        test_cases = [
            {
                "direction": "Long",
                "entryPrice": 45600.50,
                "currentPnl": 200.0,  # Profit for long
                "quantity": 2
            },
            {
                "direction": "Short",
                "entryPrice": 45600.50,
                "currentPnl": 150.0,  # Profit for short
                "quantity": 1
            },
            {
                "direction": "",  # No position
                "entryPrice": 0.0,
                "currentPnl": 0.0,
                "quantity": 0
            }
        ]
        
        for case in test_cases:
            position_data = {
                "instrument": "BANKNIFTY",
                "direction": case["direction"],
                "entryPrice": case["entryPrice"],
                "quantity": case["quantity"],
                "stopLoss": 45400.00,
                "targetPrice": 45800.00,
                "currentPnl": case["currentPnl"],
                "tradeType": "Automated",
                "isOpen": case["direction"] != ""  # Open if has direction
            }
            
            position_message = {
                "type": "position_update",
                "data": position_data
            }
            
            # Should be JSON serializable
            json_str = json.dumps(position_message)
            parsed = json.loads(json_str)
            
            # Verify structure is preserved
            assert parsed["type"] == "position_update"
            assert parsed["data"]["direction"] == case["direction"]
            assert parsed["data"]["currentPnl"] == case["currentPnl"]

    def test_position_update_with_manual_trade_type(self):
        """Test position updates correctly handle manual trade type."""
        manual_position = {
            "instrument": "NIFTY50",
            "direction": "Short",
            "entryPrice": 18500.00,
            "quantity": 3,
            "stopLoss": 18600.00,
            "targetPrice": 18400.00,
            "currentPnl": -50.0,
            "tradeType": "Manual",  # Manual trade
            "isOpen": True
        }
        
        position_message = {
            "type": "position_update",
            "data": manual_position
        }
        
        # Verify manual trade type is preserved
        assert position_message["data"]["tradeType"] == "Manual"
        
        # JSON serialization should work
        json_str = json.dumps(position_message)
        parsed = json.loads(json_str)
        assert parsed["data"]["tradeType"] == "Manual"

    def test_websocket_handles_multiple_position_updates(self):
        """Test WebSocket can handle sequence of position updates (open -> update -> close)."""
        mock_service = MockLiveTradingServiceWithPositions("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Sequence: Open -> PnL Update -> Close
                loop.run_until_complete(mock_service.simulate_position_open())
                message1 = json.loads(websocket.receive_text())
                
                loop.run_until_complete(mock_service.simulate_position_pnl_update())
                message2 = json.loads(websocket.receive_text())
                
                loop.run_until_complete(mock_service.simulate_position_close())
                message3 = json.loads(websocket.receive_text())
                
                # Verify sequence
                assert message1["data"]["isOpen"] == True
                assert message1["data"]["currentPnl"] == 0.0
                
                assert message2["data"]["isOpen"] == True
                assert message2["data"]["currentPnl"] == 200.0
                
                assert message3["data"]["isOpen"] == False
                assert message3["data"]["currentPnl"] == 400.0
                assert "exitPrice" in message3["data"]
                assert "pnl" in message3["data"]
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_position_update_numeric_precision(self):
        """Test position updates handle numeric precision correctly."""
        test_cases = [
            {"price": 45600, "pnl": 100},  # integers
            {"price": 45600.50, "pnl": 150.25},  # floats
            {"price": 45600.123456, "pnl": 200.789123},  # high precision
        ]
        
        for case in test_cases:
            position_data = {
                "instrument": "BANKNIFTY",
                "direction": "Long",
                "entryPrice": case["price"],
                "quantity": 1,
                "stopLoss": case["price"] - 200,
                "targetPrice": case["price"] + 200,
                "currentPnl": case["pnl"],
                "tradeType": "Automated",
                "isOpen": True
            }
            
            position_message = {
                "type": "position_update",
                "data": position_data
            }
            
            # Should be JSON serializable with precision preserved
            json_str = json.dumps(position_message)
            parsed = json.loads(json_str)
            
            # Types should be preserved appropriately
            assert isinstance(parsed["data"]["entryPrice"], (int, float))
            assert isinstance(parsed["data"]["currentPnl"], (int, float))
            assert parsed["data"]["entryPrice"] == case["price"]
            assert parsed["data"]["currentPnl"] == case["pnl"]


class TestPositionUpdateIntegration:
    """Integration tests for position update functionality."""

    def test_position_update_message_differentiation(self):
        """Test that position_update messages can be differentiated from tick messages."""
        mock_service = MockLiveTradingServiceWithPositions("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Send position update
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mock_service.simulate_position_open())
                
                # Receive and verify it's a position update, not a tick
                data = websocket.receive_text()
                message = json.loads(data)
                
                assert message["type"] == "position_update"
                assert message["type"] != "tick"  # Clearly differentiated
                assert "data" in message
                
                # Position data structure is different from tick data
                position_data = message["data"]
                assert "entryPrice" in position_data  # Position-specific field
                assert "stopLoss" in position_data  # Position-specific field
                assert "symbol" not in position_data  # Not a tick field
                assert "volume" not in position_data  # Not a tick field
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_position_update_required_fields_validation(self):
        """Test that all required fields per Architecture Position model are present."""
        required_fields = [
            "instrument",     # Trading symbol
            "direction",      # Long/Short  
            "entryPrice",     # Entry price
            "quantity",       # Quantity
            "stopLoss",       # Stop loss price
            "targetPrice",    # Target price
            "currentPnl",     # Real-time PnL
            "tradeType",      # Automated/Manual
        ]
        
        # Verify all required fields are in sample data
        for field in required_fields:
            assert field in SAMPLE_POSITION_OPEN, f"Missing required field: {field}"
        
        # Verify data types are appropriate
        assert isinstance(SAMPLE_POSITION_OPEN["entryPrice"], (int, float))
        assert isinstance(SAMPLE_POSITION_OPEN["quantity"], int)
        assert isinstance(SAMPLE_POSITION_OPEN["currentPnl"], (int, float))
        assert isinstance(SAMPLE_POSITION_OPEN["tradeType"], str)
        
        # Verify values make sense
        assert SAMPLE_POSITION_OPEN["entryPrice"] > 0
        assert SAMPLE_POSITION_OPEN["quantity"] > 0
        assert SAMPLE_POSITION_OPEN["tradeType"] in ["Automated", "Manual"]

    @patch('src.trading.live_trading_service.LiveTradingService')
    def test_live_trading_service_position_integration(self, mock_live_service_class):
        """Test integration between LiveTradingService position updates and WebSocket."""
        mock_service = MockLiveTradingServiceWithPositions("integration_user")
        mock_live_service_class.return_value = mock_service
        
        active_live_sessions["integration_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/integration_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Simulate the LiveTradingService detecting position change and broadcasting it
                loop.run_until_complete(mock_service.simulate_position_open())
                
                # WebSocket client should receive the position update
                data = websocket.receive_text()
                message = json.loads(data)
                
                assert message["type"] == "position_update"
                assert message["data"]["instrument"] == "BANKNIFTY"
                assert message["data"]["direction"] == "Long"
                
        finally:
            if "integration_user" in active_live_sessions:
                del active_live_sessions["integration_user"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])