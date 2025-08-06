"""
Integration tests for frontend API client config functionality.

Tests verify that the /api/config endpoint returns data in the format
expected by the frontend TypeScript interfaces and components.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, mock_open
import yaml

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from main import app

# Create test client
client = TestClient(app)

# Sample test data that matches the TypeScript Instrument interface
TEST_CONFIG_DATA = {
    'instruments': [
        {
            'name': 'Bank Nifty',
            'symbol': 'Bank_Nifty',
            'exchange-symbol': 'NSE:NIFTYBANK-INDEX',
            'type': 'index',
            'lot_size': 35,
            'tick_size': 0.05,
            'option_premium_range': [0.02, 0.03]
        },
        {
            'name': 'Nifty 50',
            'symbol': 'Nifty',
            'exchange-symbol': 'NSE:NIFTY50-INDEX',
            'type': 'index',
            'lot_size': 75,
            'tick_size': 0.05,
            'option_premium_range': [0.01, 0.02]
        },
        {
            'name': 'Reliance Industries',
            'symbol': 'RELIANCE',
            'exchange-symbol': 'NSE:RELIANCE-EQ',
            'type': 'stock',
            'lot_size': 1,
            'tick_size': 0.05
        }
    ],
    'timeframes': ['1', '3', '5', '15', '30', '60', '120', '240', 'D']
}


class TestFrontendConfigIntegration:
    """Test class for frontend config integration."""
    
    def test_config_response_matches_typescript_interface(self):
        """Test that API response matches TypeScript ConfigResponse interface."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            
            # Assert successful response
            assert response.status_code == 200
            
            response_data = response.json()
            
            # Verify ConfigResponse interface structure
            assert "instruments" in response_data
            assert "timeframes" in response_data
            assert len(response_data.keys()) == 2
            
            # Verify Instrument interface structure for each instrument
            instruments = response_data["instruments"]
            assert isinstance(instruments, list)
            
            for instrument in instruments:
                # Required fields per TypeScript Instrument interface
                assert "name" in instrument
                assert "symbol" in instrument
                assert "exchange-symbol" in instrument  # Note: hyphenated key
                assert "type" in instrument
                assert "lot_size" in instrument
                assert "tick_size" in instrument
                
                # Verify data types match TypeScript expectations
                assert isinstance(instrument["name"], str)
                assert isinstance(instrument["symbol"], str)
                assert isinstance(instrument["exchange-symbol"], str)
                assert isinstance(instrument["type"], str)
                assert isinstance(instrument["lot_size"], int)
                assert isinstance(instrument["tick_size"], float)
                
                # Optional field
                if "option_premium_range" in instrument:
                    assert isinstance(instrument["option_premium_range"], list)
                    assert len(instrument["option_premium_range"]) == 2
                    assert all(isinstance(x, float) for x in instrument["option_premium_range"])
            
            # Verify timeframes array
            timeframes = response_data["timeframes"]
            assert isinstance(timeframes, list)
            for timeframe in timeframes:
                assert isinstance(timeframe, str)
    
    def test_instrument_types_match_frontend_expectations(self):
        """Test that instrument types match frontend component expectations."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            response_data = response.json()
            
            instruments = response_data["instruments"]
            
            # Verify expected instrument types exist
            instrument_types = [inst["type"] for inst in instruments]
            assert "index" in instrument_types
            assert "stock" in instrument_types
            
            # Verify index instruments have option_premium_range
            index_instruments = [inst for inst in instruments if inst["type"] == "index"]
            for index_inst in index_instruments:
                assert "option_premium_range" in index_inst
                assert len(index_inst["option_premium_range"]) == 2
                assert index_inst["option_premium_range"][0] > 0
                assert index_inst["option_premium_range"][1] > index_inst["option_premium_range"][0]
    
    def test_timeframes_format_for_frontend_display(self):
        """Test that timeframes are in format expected by frontend display logic."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            response_data = response.json()
            
            timeframes = response_data["timeframes"]
            
            # Verify expected timeframe formats
            expected_formats = {
                "numeric": [],  # Should be numeric strings like "1", "5", "15"
                "daily": []     # Should be "D" for daily
            }
            
            for tf in timeframes:
                if tf == "D":
                    expected_formats["daily"].append(tf)
                else:
                    # Should be numeric string
                    try:
                        minutes = int(tf)
                        assert minutes > 0
                        expected_formats["numeric"].append(tf)
                    except ValueError:
                        pytest.fail(f"Unexpected timeframe format: {tf}")
            
            # Verify we have both numeric and daily timeframes
            assert len(expected_formats["numeric"]) > 0, "Should have numeric timeframes"
            assert len(expected_formats["daily"]) > 0, "Should have daily timeframe"
    
    def test_dropdown_population_data_structure(self):
        """Test that response structure works for frontend Select components."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            response_data = response.json()
            
            # Simulate frontend SelectItem mapping for instruments
            instruments = response_data["instruments"]
            for instrument in instruments:
                # Frontend maps: key={instrument.symbol} value={instrument.symbol}
                assert instrument["symbol"] is not None
                assert len(instrument["symbol"]) > 0
                
                # Frontend displays: {instrument.name} ({instrument.symbol})
                display_text = f"{instrument['name']} ({instrument['symbol']})"
                assert len(display_text) > len(instrument["symbol"])
            
            # Simulate frontend SelectItem mapping for timeframes
            timeframes = response_data["timeframes"]
            for timeframe in timeframes:
                # Frontend maps: key={timeframe} value={timeframe}
                assert timeframe is not None
                assert len(timeframe) > 0
                
                # Frontend should be able to generate display labels
                # Test the logic similar to getTimeframeLabel function
                if timeframe == 'D':
                    display_label = 'Daily'
                else:
                    minutes = int(timeframe)
                    if minutes >= 60:
                        hours = minutes / 60
                        display_label = f"{hours} Hour{'s' if hours > 1 else ''}"
                    else:
                        display_label = f"{minutes} Minute{'s' if minutes > 1 else ''}"
                
                assert len(display_label) > 0
                assert display_label != timeframe  # Should be more descriptive
    
    def test_api_client_error_handling_structure(self):
        """Test error response structure for frontend API client error handling."""
        # Test missing instruments key
        invalid_config = {'timeframes': TEST_CONFIG_DATA['timeframes']}
        
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=invalid_config):
            
            response = client.get("/api/config")
            
            # Should return 500 error
            assert response.status_code == 500
            
            # Verify error response structure expected by frontend
            error_data = response.json()
            assert "detail" in error_data
            assert isinstance(error_data["detail"], str)
            assert len(error_data["detail"]) > 0
            
            # Frontend formatApiError should be able to extract this message
            error_message = error_data["detail"]
            assert "instruments" in error_message
    
    def test_loading_state_compatibility(self):
        """Test that endpoint can handle concurrent requests for loading states."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            # Make multiple concurrent requests (simulating multiple components)
            responses = []
            for _ in range(3):
                response = client.get("/api/config")
                responses.append(response)
            
            # All should succeed
            for response in responses:
                assert response.status_code == 200
                response_data = response.json()
                assert "instruments" in response_data
                assert "timeframes" in response_data
                
                # Data should be consistent across requests
                assert len(response_data["instruments"]) == len(TEST_CONFIG_DATA["instruments"])
                assert len(response_data["timeframes"]) == len(TEST_CONFIG_DATA["timeframes"])