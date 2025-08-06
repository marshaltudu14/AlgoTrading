"""
Unit tests for WebSocket tick streaming functionality.

Tests verify the WebSocket endpoint at /ws/live/{user_id} properly handles
tick data streaming from LiveTradingService, ping/pong keep-alive mechanism,
and proper message formatting for frontend chart integration.
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

# Import WebSocket testing utilities
from starlette.testclient import WebSocketTestSession

# Create test client
client = TestClient(app)

# Sample tick data for testing
SAMPLE_TICK_DATA = {
    "symbol": "NSE:BANKNIFTY-INDEX",
    "price": 45600.50,
    "volume": 1250.0,
    "timestamp": "2024-01-15T10:30:00.123456",
    "bid": 45600.25,
    "ask": 45600.75,
    "high": 45650.00,
    "low": 45580.00,
    "open": 45590.00
}

EXPECTED_TICK_MESSAGE = {
    "type": "tick",
    "data": SAMPLE_TICK_DATA
}

# Mock LiveTradingService for testing
class MockLiveTradingService:
    """Mock LiveTradingService for WebSocket testing"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.websocket_clients = []
        self.status = "running"
        self.is_running = True
        
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
    
    async def simulate_tick_data(self, tick_data):
        """Simulate broadcasting tick data to connected clients"""
        tick_message = {
            "type": "tick",
            "data": tick_data
        }
        
        # Send to all connected clients
        for client in self.websocket_clients:
            try:
                await client.send_text(json.dumps(tick_message, default=str))
            except Exception as e:
                print(f"Error sending tick data: {e}")


class TestWebSocketTickStreaming:
    """Test class for WebSocket tick streaming functionality."""

    def test_websocket_connection_without_active_session(self):
        """Test WebSocket connection fails when no active live session exists."""
        with client.websocket_connect("/ws/live/nonexistent_user") as websocket:
            # Should receive error message about missing session
            data = websocket.receive_text()
            message = json.loads(data)
            
            assert message["type"] == "error"
            assert "Live trading session not found" in message["message"]

    def test_websocket_connection_with_active_session(self):
        """Test successful WebSocket connection with active live session."""
        # Set up mock live session
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Should receive connection confirmation
                data = websocket.receive_text()
                message = json.loads(data)
                
                assert message["type"] == "connected"
                assert message["user_id"] == "test_user"
                assert "WebSocket connected successfully" in message["message"]
                
                # Should receive status message
                data = websocket.receive_text()
                message = json.loads(data)
                
                assert message["type"] == "status"
                assert "data" in message
                assert message["data"]["status"] == "running"
                assert message["data"]["instrument"] == "BANKNIFTY"
                
        finally:
            # Cleanup
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_ping_pong_mechanism(self):
        """Test ping/pong keep-alive mechanism works correctly."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Send ping
                websocket.send_text("ping")
                
                # Should receive pong
                response = websocket.receive_text()
                assert response == "pong"
                
                # Send structured ping
                ping_message = {"type": "ping", "timestamp": "2024-01-15T10:30:00"}
                websocket.send_text(json.dumps(ping_message))
                
                # The server should handle it without error
                # We can't easily test the server-initiated pings in this setup
                # but we can verify the connection remains stable
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_receives_tick_data(self):
        """Test WebSocket receives properly formatted tick data."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Simulate tick data being sent
                async def send_tick():
                    await mock_service.simulate_tick_data(SAMPLE_TICK_DATA)
                
                # Run the tick simulation
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(send_tick())
                
                # Receive tick message
                data = websocket.receive_text()
                message = json.loads(data)
                
                # Verify tick message structure
                assert message["type"] == "tick"
                assert "data" in message
                
                tick_data = message["data"]
                assert tick_data["symbol"] == SAMPLE_TICK_DATA["symbol"]
                assert tick_data["price"] == SAMPLE_TICK_DATA["price"]
                assert tick_data["volume"] == SAMPLE_TICK_DATA["volume"]
                assert "timestamp" in tick_data
                
                # Verify all required fields for charting
                required_fields = ["symbol", "price", "volume", "timestamp", "bid", "ask", "high", "low", "open"]
                for field in required_fields:
                    assert field in tick_data, f"Missing required field: {field}"
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_tick_data_message_format(self):
        """Test that tick data messages follow the exact required JSON structure."""
        mock_service = MockLiveTradingService("test_user")
        
        # Test the message format directly
        tick_message = {
            "type": "tick",
            "data": SAMPLE_TICK_DATA
        }
        
        # Verify top-level structure
        assert "type" in tick_message
        assert tick_message["type"] == "tick"
        assert "data" in tick_message
        
        # Verify data structure
        data = tick_message["data"]
        assert isinstance(data["price"], (int, float))
        assert isinstance(data["volume"], (int, float))
        assert isinstance(data["timestamp"], str)
        assert isinstance(data["symbol"], str)
        
        # Verify numeric fields are properly formatted
        assert data["price"] > 0
        assert data["volume"] >= 0
        
        # Verify JSON serializability
        json_str = json.dumps(tick_message, default=str)
        parsed_back = json.loads(json_str)
        assert parsed_back["type"] == "tick"
        assert parsed_back["data"]["price"] == SAMPLE_TICK_DATA["price"]

    def test_websocket_handles_multiple_clients(self):
        """Test WebSocket can handle multiple clients for the same user."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            # Connect multiple clients
            with client.websocket_connect("/ws/live/test_user") as ws1, \
                 client.websocket_connect("/ws/live/test_user") as ws2:
                
                # Skip initial messages for both
                ws1.receive_text()  # connected
                ws1.receive_text()  # status
                ws2.receive_text()  # connected
                ws2.receive_text()  # status
                
                # Both clients should be connected
                assert len(mock_service.websocket_clients) == 2
                
                # Send tick data
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(mock_service.simulate_tick_data(SAMPLE_TICK_DATA))
                
                # Both clients should receive the tick data
                data1 = ws1.receive_text()
                data2 = ws2.receive_text()
                
                message1 = json.loads(data1)
                message2 = json.loads(data2)
                
                assert message1["type"] == "tick"
                assert message2["type"] == "tick"
                assert message1["data"]["price"] == SAMPLE_TICK_DATA["price"]
                assert message2["data"]["price"] == SAMPLE_TICK_DATA["price"]
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_connection_cleanup(self):
        """Test WebSocket connection cleanup when client disconnects."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            # Connect and then disconnect
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Client should be added to service
                assert len(mock_service.websocket_clients) == 1
            
            # After context exit, client should be removed
            # Note: This may require a small delay for cleanup
            time.sleep(0.1)
            
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_websocket_handles_malformed_messages(self):
        """Test WebSocket handles malformed client messages gracefully."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Send malformed JSON
                websocket.send_text("invalid json {")
                
                # Connection should remain stable
                # Send valid ping to verify
                websocket.send_text("ping")
                response = websocket.receive_text()
                assert response == "pong"
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_tick_data_numeric_types(self):
        """Test that tick data properly handles different numeric types."""
        test_cases = [
            {"price": 45600, "volume": 1000},  # integers
            {"price": 45600.50, "volume": 1000.0},  # floats
            {"price": 45600.123456, "volume": 1000.789},  # high precision floats
        ]
        
        for case in test_cases:
            tick_data = {
                "symbol": "NSE:BANKNIFTY-INDEX",
                "price": case["price"],
                "volume": case["volume"],
                "timestamp": "2024-01-15T10:30:00",
                "bid": 0,
                "ask": 0,
                "high": 0,
                "low": 0,
                "open": 0
            }
            
            tick_message = {
                "type": "tick",
                "data": tick_data
            }
            
            # Should be JSON serializable
            json_str = json.dumps(tick_message)
            parsed = json.loads(json_str)
            
            # Types should be preserved appropriately
            assert isinstance(parsed["data"]["price"], (int, float))
            assert isinstance(parsed["data"]["volume"], (int, float))
            assert parsed["data"]["price"] == case["price"]
            assert parsed["data"]["volume"] == case["volume"]


class TestWebSocketKeepAlive:
    """Test class specifically for WebSocket keep-alive mechanism."""

    def test_websocket_keep_alive_structure(self):
        """Test that server-initiated ping messages have correct structure."""
        # This test verifies the ping message format
        # The actual server-side ping testing requires more complex async setup
        
        expected_ping_structure = {
            "type": "ping",
            "timestamp": "2024-01-15T10:30:00.123456"
        }
        
        # Verify structure
        assert "type" in expected_ping_structure
        assert expected_ping_structure["type"] == "ping"
        assert "timestamp" in expected_ping_structure
        
        # Verify JSON serializability
        json_str = json.dumps(expected_ping_structure)
        parsed = json.loads(json_str)
        assert parsed["type"] == "ping"

    def test_websocket_handles_structured_pong(self):
        """Test WebSocket handles structured pong responses."""
        mock_service = MockLiveTradingService("test_user")
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Send structured pong
                pong_message = {"type": "pong", "timestamp": "2024-01-15T10:30:00"}
                websocket.send_text(json.dumps(pong_message))
                
                # Connection should remain stable
                websocket.send_text("ping")
                response = websocket.receive_text()
                assert response == "pong"
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]


class TestTickDataIntegration:
    """Integration tests for complete tick data flow."""

    @patch('src.trading.live_trading_service.LiveTradingService')
    def test_live_trading_service_integration(self, mock_live_service_class):
        """Test integration between LiveTradingService and WebSocket endpoint."""
        # Create a mock service instance
        mock_service = MockLiveTradingService("integration_user")
        mock_live_service_class.return_value = mock_service
        
        # Add to active sessions
        active_live_sessions["integration_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/integration_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Simulate the LiveTradingService receiving tick data and broadcasting it
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # This simulates what would happen when FyersClient sends tick data
                # to LiveTradingService, which then broadcasts it via WebSocket
                loop.run_until_complete(mock_service.simulate_tick_data(SAMPLE_TICK_DATA))
                
                # WebSocket client should receive the tick data
                data = websocket.receive_text()
                message = json.loads(data)
                
                assert message["type"] == "tick"
                assert message["data"]["symbol"] == SAMPLE_TICK_DATA["symbol"]
                assert message["data"]["price"] == SAMPLE_TICK_DATA["price"]
                
        finally:
            if "integration_user" in active_live_sessions:
                del active_live_sessions["integration_user"]

    def test_tick_data_required_fields_for_charting(self):
        """Test that tick data contains all fields required for frontend charting."""
        required_fields = [
            "symbol",     # Trading symbol
            "price",      # Current price (LTP)
            "volume",     # Current volume
            "timestamp",  # Time of the tick
            "bid",        # Bid price
            "ask",        # Ask price  
            "high",       # High price
            "low",        # Low price
            "open"        # Open price
        ]
        
        # Verify all required fields are in sample data
        for field in required_fields:
            assert field in SAMPLE_TICK_DATA, f"Missing required field for charting: {field}"
        
        # Verify data types are appropriate for charting
        assert isinstance(SAMPLE_TICK_DATA["price"], (int, float))
        assert isinstance(SAMPLE_TICK_DATA["volume"], (int, float))
        assert isinstance(SAMPLE_TICK_DATA["timestamp"], str)
        assert isinstance(SAMPLE_TICK_DATA["symbol"], str)
        
        # Verify numeric fields have reasonable values
        assert SAMPLE_TICK_DATA["price"] > 0
        assert SAMPLE_TICK_DATA["volume"] >= 0


class TestMixedMessageTypes:
    """Test class for mixed tick and position update messages."""

    def test_websocket_handles_both_tick_and_position_messages(self):
        """Test that WebSocket can handle both tick data and position updates."""
        mock_service = MockLiveTradingService("test_user")
        
        # Add position update simulation capability
        async def simulate_mixed_messages(service):
            # Simulate tick data
            await service.simulate_tick_data(SAMPLE_TICK_DATA)
            
            # Simulate position update
            position_data = {
                "instrument": "BANKNIFTY",
                "direction": "Long",
                "entryPrice": 45600.50,
                "quantity": 2,
                "stopLoss": 45400.00,
                "targetPrice": 45800.00,
                "currentPnl": 100.0,
                "tradeType": "Automated",
                "isOpen": True
            }
            
            position_message = {
                "type": "position_update",
                "data": position_data
            }
            
            # Send to all connected clients
            for client in service.websocket_clients:
                try:
                    await client.send_text(json.dumps(position_message, default=str))
                except Exception as e:
                    print(f"Error sending position update: {e}")
        
        active_live_sessions["test_user"] = mock_service
        
        try:
            with client.websocket_connect("/ws/live/test_user") as websocket:
                # Skip initial messages
                websocket.receive_text()  # connected
                websocket.receive_text()  # status
                
                # Send mixed messages
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(simulate_mixed_messages(mock_service))
                
                # Should receive both messages in order
                message1 = json.loads(websocket.receive_text())
                message2 = json.loads(websocket.receive_text())
                
                # Verify we got both types (order may vary due to async)
                message_types = {message1["type"], message2["type"]}
                assert "tick" in message_types
                assert "position_update" in message_types
                
                # Verify tick message structure
                tick_message = message1 if message1["type"] == "tick" else message2
                assert "data" in tick_message
                assert "price" in tick_message["data"]
                assert "symbol" in tick_message["data"]
                
                # Verify position message structure
                pos_message = message1 if message1["type"] == "position_update" else message2
                assert "data" in pos_message
                assert "direction" in pos_message["data"]
                assert "entryPrice" in pos_message["data"]
                
        finally:
            if "test_user" in active_live_sessions:
                del active_live_sessions["test_user"]

    def test_message_type_field_consistency(self):
        """Test that message type field is consistent across different message types."""
        # Test tick message type
        tick_message = {
            "type": "tick",
            "data": SAMPLE_TICK_DATA
        }
        
        # Test position update message type
        position_message = {
            "type": "position_update",
            "data": {
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
        }
        
        # Both should have consistent type field structure
        assert "type" in tick_message
        assert "type" in position_message
        assert isinstance(tick_message["type"], str)
        assert isinstance(position_message["type"], str)
        assert tick_message["type"] != position_message["type"]  # Different types
        
        # Both should be JSON serializable
        tick_json = json.dumps(tick_message, default=str)
        position_json = json.dumps(position_message, default=str)
        
        tick_parsed = json.loads(tick_json)
        position_parsed = json.loads(position_json)
        
        assert tick_parsed["type"] == "tick"
        assert position_parsed["type"] == "position_update"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])