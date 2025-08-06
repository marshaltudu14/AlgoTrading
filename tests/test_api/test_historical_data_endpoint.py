"""
Unit tests for the /api/historical-data endpoint.

Tests verify the endpoint correctly fetches historical candlestick data from 
RealtimeDataLoader with proper authentication, query parameter validation,
and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from pathlib import Path

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from main import app

# Create test client
client = TestClient(app)

# Sample test data for historical candles
SAMPLE_HISTORICAL_DATA = pd.DataFrame({
    'datetime': pd.date_range('2024-01-01', periods=5, freq='5min'),
    'open': [100.0, 101.0, 102.0, 103.0, 104.0],
    'high': [101.0, 102.0, 103.0, 104.0, 105.0],
    'low': [99.0, 100.0, 101.0, 102.0, 103.0],
    'close': [101.0, 102.0, 103.0, 104.0, 105.0],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Set datetime as index to match expected data structure
SAMPLE_HISTORICAL_DATA.set_index('datetime', inplace=True)

EXPECTED_RESPONSE_FORMAT = [
    {
        "time": "2024-01-01T00:00:00",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 101.0,
        "volume": 1000.0
    },
    {
        "time": "2024-01-01T00:05:00",
        "open": 101.0,
        "high": 102.0,
        "low": 100.0,
        "close": 102.0,
        "volume": 1100.0
    }
]

# Mock JWT token for authentication tests
VALID_JWT_PAYLOAD = {
    "user_id": "test_user",
    "access_token": "test_token",
    "app_id": "test_app"
}


class TestHistoricalDataEndpoint:
    """Test class for the /api/historical-data endpoint."""

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_success(self, mock_verify_jwt, mock_data_loader_class):
        """Test successful GET request returns 200 and correctly formatted historical data."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader mock
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = SAMPLE_HISTORICAL_DATA
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert response status
        assert response.status_code == 200
        
        # Assert response structure
        response_data = response.json()
        assert isinstance(response_data, list)
        assert len(response_data) == 5  # Same as sample data
        
        # Assert each candle has required fields
        for candle in response_data:
            assert "time" in candle
            assert "open" in candle
            assert "high" in candle
            assert "low" in candle
            assert "close" in candle
            assert "volume" in candle
            
            # Assert data types
            assert isinstance(candle["time"], str)
            assert isinstance(candle["open"], float)
            assert isinstance(candle["high"], float)
            assert isinstance(candle["low"], float)
            assert isinstance(candle["close"], float)
            assert isinstance(candle["volume"], float)
        
        # Verify data loader was called with correct parameters
        mock_data_loader.fetch_and_process_data.assert_called_once_with(
            symbol="BANKNIFTY",
            timeframe="5"
        )

    def test_get_historical_data_missing_auth_token(self):
        """Test that missing authentication token returns 401 Unauthorized."""
        # Clear any existing cookies
        client.cookies.clear()
        
        # Make request without auth cookie
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert unauthorized response
        assert response.status_code == 401
        assert "detail" in response.json()
        assert "Authentication required" in response.json()["detail"]

    @patch('main.verify_jwt_token')
    def test_get_historical_data_invalid_jwt_token(self, mock_verify_jwt):
        """Test that invalid JWT token returns 401 Unauthorized."""
        from fastapi import HTTPException
        
        # Setup JWT verification to fail
        mock_verify_jwt.side_effect = HTTPException(status_code=401, detail="Invalid token")
        
        # Set invalid auth cookie and make request
        client.cookies.set("auth_token", "invalid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert unauthorized response
        assert response.status_code == 401
        assert "detail" in response.json()

    @patch('main.verify_jwt_token')
    def test_get_historical_data_missing_instrument_parameter(self, mock_verify_jwt):
        """Test that missing instrument parameter returns 422 Unprocessable Entity."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Set auth cookie and make request without instrument
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?timeframe=5")
        
        # Assert unprocessable entity response
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data
        # FastAPI will return field validation error for missing required parameter

    @patch('main.verify_jwt_token')
    def test_get_historical_data_missing_timeframe_parameter(self, mock_verify_jwt):
        """Test that missing timeframe parameter returns 422 Unprocessable Entity."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Set auth cookie and make request without timeframe
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY")
        
        # Assert unprocessable entity response
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data

    @patch('main.verify_jwt_token')  
    def test_get_historical_data_empty_instrument_parameter(self, mock_verify_jwt):
        """Test that empty instrument parameter returns 422."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Set auth cookie and make request with empty instrument
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=&timeframe=5")
        
        # Assert validation error response
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data
        assert "Instrument parameter is required" in response_data["detail"]

    @patch('main.verify_jwt_token')
    def test_get_historical_data_empty_timeframe_parameter(self, mock_verify_jwt):
        """Test that empty timeframe parameter returns 422."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Set auth cookie and make request with empty timeframe
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=")
        
        # Assert validation error response
        assert response.status_code == 422
        response_data = response.json()
        assert "detail" in response_data
        assert "Timeframe parameter is required" in response_data["detail"]

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_loader_returns_none(self, mock_verify_jwt, mock_data_loader_class):
        """Test that data loader returning None results in 500 Internal Server Error."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader to return None
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = None
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert internal server error
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Failed to fetch historical data" in response_data["detail"]

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_loader_returns_empty_dataframe(self, mock_verify_jwt, mock_data_loader_class):
        """Test that data loader returning empty DataFrame results in 500 Internal Server Error."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader to return empty DataFrame
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = pd.DataFrame()
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert internal server error
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Failed to fetch historical data" in response_data["detail"]

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_loader_raises_exception(self, mock_verify_jwt, mock_data_loader_class):
        """Test that data loader raising an exception results in 500 Internal Server Error."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader to raise exception
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.side_effect = Exception("Data loading failed")
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert internal server error
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Internal server error while fetching historical data" in response_data["detail"]

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_missing_ohlcv_columns(self, mock_verify_jwt, mock_data_loader_class):
        """Test that DataFrame missing OHLCV columns results in 500 Internal Server Error."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Create DataFrame without OHLCV columns
        invalid_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='5min'),
            'invalid_column': [1, 2, 3]
        })
        invalid_data.set_index('datetime', inplace=True)
        
        # Setup data loader to return invalid data
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = invalid_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert internal server error
        assert response.status_code == 500
        response_data = response.json()
        assert "detail" in response_data
        assert "Historical data missing required OHLCV columns" in response_data["detail"]

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_partial_ohlcv_columns(self, mock_verify_jwt, mock_data_loader_class):
        """Test that DataFrame with partial OHLCV columns still works."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Create DataFrame with only some OHLCV columns
        partial_data = pd.DataFrame({
            'datetime': pd.date_range('2024-01-01', periods=3, freq='5min'),
            'open': [100.0, 101.0, 102.0],
            'close': [101.0, 102.0, 103.0],
            'volume': [1000, 1100, 1200]
        })
        partial_data.set_index('datetime', inplace=True)
        
        # Setup data loader to return partial data
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = partial_data
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Should succeed with available columns (missing ones default to 0)
        assert response.status_code == 200
        response_data = response.json()
        assert len(response_data) == 3
        
        # Verify missing columns are set to 0.0
        for candle in response_data:
            assert candle["open"] > 0  # Available column
            assert candle["close"] > 0  # Available column  
            assert candle["high"] == 0.0  # Missing column, defaults to 0
            assert candle["low"] == 0.0  # Missing column, defaults to 0
            assert candle["volume"] > 0  # Available column

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_get_historical_data_different_instruments_and_timeframes(self, mock_verify_jwt, mock_data_loader_class):
        """Test endpoint works with different instrument and timeframe combinations."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader mock
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = SAMPLE_HISTORICAL_DATA
        mock_data_loader_class.return_value = mock_data_loader
        
        test_cases = [
            ("NIFTY", "1"),
            ("BANKNIFTY", "15"),
            ("SENSEX", "30"),
            ("RELIANCE", "60")
        ]
        
        for instrument, timeframe in test_cases:
            # Set auth cookie and make request
            client.cookies.set("auth_token", "valid_test_token")
            response = client.get(f"/api/historical-data?instrument={instrument}&timeframe={timeframe}")
            
            # Assert success
            assert response.status_code == 200
            
            # Verify correct parameters were passed to data loader
            mock_data_loader.fetch_and_process_data.assert_called_with(
                symbol=instrument,
                timeframe=timeframe
            )


class TestHistoricalDataEndpointResponseFormat:
    """Test class specifically for response format validation."""

    @patch('main.RealtimeDataLoader')
    @patch('main.verify_jwt_token')
    def test_historical_data_response_format_structure(self, mock_verify_jwt, mock_data_loader_class):
        """Test that the response format exactly matches frontend requirements."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader mock with specific test data
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = SAMPLE_HISTORICAL_DATA
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert success
        assert response.status_code == 200
        
        response_data = response.json()
        
        # Verify it's a list of candle objects
        assert isinstance(response_data, list)
        assert len(response_data) > 0
        
        # Check first candle structure in detail
        first_candle = response_data[0]
        
        # Verify all required keys are present
        required_keys = ["time", "open", "high", "low", "close", "volume"]
        assert set(first_candle.keys()) == set(required_keys)
        
        # Verify data types
        assert isinstance(first_candle["time"], str)
        assert isinstance(first_candle["open"], (int, float))
        assert isinstance(first_candle["high"], (int, float))
        assert isinstance(first_candle["low"], (int, float))
        assert isinstance(first_candle["close"], (int, float))
        assert isinstance(first_candle["volume"], (int, float))
        
        # Verify time format is ISO format
        time_str = first_candle["time"]
        assert "T" in time_str  # ISO format contains T separator
        
        # Verify all numeric values are non-negative (for this test data)
        assert first_candle["open"] >= 0
        assert first_candle["high"] >= 0
        assert first_candle["low"] >= 0
        assert first_candle["close"] >= 0
        assert first_candle["volume"] >= 0

    @patch('main.RealtimeDataLoader')  
    @patch('main.verify_jwt_token')
    def test_historical_data_json_serializable(self, mock_verify_jwt, mock_data_loader_class):
        """Test that the response is properly JSON serializable."""
        # Setup authentication mock
        mock_verify_jwt.return_value = VALID_JWT_PAYLOAD
        
        # Setup data loader mock
        mock_data_loader = Mock()
        mock_data_loader.fetch_and_process_data.return_value = SAMPLE_HISTORICAL_DATA
        mock_data_loader_class.return_value = mock_data_loader
        
        # Set auth cookie and make request
        client.cookies.set("auth_token", "valid_test_token")
        response = client.get("/api/historical-data?instrument=BANKNIFTY&timeframe=5")
        
        # Assert success
        assert response.status_code == 200
        
        # Verify response can be parsed as JSON (this is implicit in response.json())
        response_data = response.json()
        
        # Test that response can be re-serialized as JSON
        import json
        try:
            json_str = json.dumps(response_data)
            assert isinstance(json_str, str)
            assert len(json_str) > 0
        except (TypeError, ValueError) as e:
            pytest.fail(f"Response data is not JSON serializable: {e}")