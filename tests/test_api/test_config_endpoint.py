"""
Unit tests for the /api/config endpoint.

Tests verify that the endpoint correctly serves instrument and timeframe data
from the config/instruments.yaml file with proper error handling.
"""

import pytest
import tempfile
import os
import yaml
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, mock_open

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from main import app

# Create test client
client = TestClient(app)

# Sample test data that matches the structure of instruments.yaml
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
        }
    ],
    'timeframes': ['1', '3', '5', '15', '30', '60', '120', '240', 'D']
}

EXPECTED_RESPONSE = {
    'instruments': TEST_CONFIG_DATA['instruments'],
    'timeframes': TEST_CONFIG_DATA['timeframes']
}


class TestConfigEndpoint:
    """Test class for the /api/config endpoint."""
    
    def test_get_config_success(self):
        """Test successful GET request to /api/config returns 200 and correct data."""
        # Mock the yaml.safe_load to return our test data
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 200
            
            # Assert response structure
            response_data = response.json()
            assert "instruments" in response_data
            assert "timeframes" in response_data
            
            # Assert response content matches expected data
            assert response_data == EXPECTED_RESPONSE
            
            # Assert instruments structure
            assert isinstance(response_data["instruments"], list)
            assert len(response_data["instruments"]) == len(TEST_CONFIG_DATA["instruments"])
            
            # Verify first instrument structure
            first_instrument = response_data["instruments"][0]
            expected_first = TEST_CONFIG_DATA["instruments"][0]
            assert first_instrument["name"] == expected_first["name"]
            assert first_instrument["symbol"] == expected_first["symbol"]
            assert first_instrument["type"] == expected_first["type"]
            
            # Assert timeframes structure
            assert isinstance(response_data["timeframes"], list)
            assert response_data["timeframes"] == TEST_CONFIG_DATA["timeframes"]
    
    def test_get_config_file_not_found(self):
        """Test that missing config file returns 500 with appropriate error message."""
        # Mock file not found error
        with patch('builtins.open', side_effect=FileNotFoundError("File not found")):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 500
            
            # Assert error message
            response_data = response.json()
            assert "detail" in response_data
            assert "Configuration file not found" in response_data["detail"]
    
    def test_get_config_yaml_parse_error(self):
        """Test that malformed YAML file returns 500 with appropriate error message."""
        # Mock YAML parse error
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML syntax")):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 500
            
            # Assert error message
            response_data = response.json()
            assert "detail" in response_data
            assert "Invalid YAML configuration" in response_data["detail"]
    
    def test_get_config_missing_instruments_key(self):
        """Test that missing 'instruments' key in config returns 500."""
        # Config data missing instruments key
        invalid_config = {
            'timeframes': TEST_CONFIG_DATA['timeframes']
        }
        
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=invalid_config):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 500
            
            # Assert error message
            response_data = response.json()
            assert "detail" in response_data
            assert "instruments" in response_data["detail"]
            assert "missing from config file" in response_data["detail"]
    
    def test_get_config_missing_timeframes_key(self):
        """Test that missing 'timeframes' key in config returns 500."""
        # Config data missing timeframes key
        invalid_config = {
            'instruments': TEST_CONFIG_DATA['instruments']
        }
        
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=invalid_config):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 500
            
            # Assert error message
            response_data = response.json()
            assert "detail" in response_data
            assert "timeframes" in response_data["detail"]
            assert "missing from config file" in response_data["detail"]
    
    def test_get_config_general_exception(self):
        """Test that unexpected errors return 500 with generic error message."""
        # Mock unexpected exception
        with patch('builtins.open', side_effect=Exception("Unexpected error")):
            
            response = client.get("/api/config")
            
            # Assert response status code
            assert response.status_code == 500
            
            # Assert error message
            response_data = response.json()
            assert "detail" in response_data
            assert "Failed to read configuration" in response_data["detail"]
    
    def test_config_endpoint_response_structure(self):
        """Test that the response structure exactly matches expected format."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            response_data = response.json()
            
            # Verify response has exactly the expected keys
            assert set(response_data.keys()) == {"instruments", "timeframes"}
            
            # Verify instruments is a list with correct structure
            instruments = response_data["instruments"]
            assert isinstance(instruments, list)
            if instruments:  # If there are instruments
                for instrument in instruments:
                    assert isinstance(instrument, dict)
                    # Verify required fields exist (based on test data)
                    required_fields = {"name", "symbol", "exchange-symbol", "type", "lot_size", "tick_size"}
                    assert required_fields.issubset(set(instrument.keys()))
            
            # Verify timeframes is a list of strings
            timeframes = response_data["timeframes"]
            assert isinstance(timeframes, list)
            for timeframe in timeframes:
                assert isinstance(timeframe, str)


class TestConfigEndpointIntegration:
    """Integration tests using the actual config file."""
    
    def test_real_config_file_access(self):
        """Test that the endpoint can read the actual config/instruments.yaml file."""
        # This test uses the real file - only run if it exists and is valid
        config_path = Path(__file__).parent.parent.parent / "config" / "instruments.yaml"
        
        if config_path.exists():
            response = client.get("/api/config")
            
            # Should succeed with real file
            assert response.status_code == 200
            
            response_data = response.json()
            assert "instruments" in response_data
            assert "timeframes" in response_data
            
            # Verify data types
            assert isinstance(response_data["instruments"], list)
            assert isinstance(response_data["timeframes"], list)
        else:
            pytest.skip("Config file not found - skipping integration test")