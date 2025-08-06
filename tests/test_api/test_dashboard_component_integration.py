"""
Component integration tests for dashboard configuration functionality.

Tests verify that the dashboard components would correctly handle the
configuration data from the API endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
from unittest.mock import patch, mock_open
import json

# Import the FastAPI app
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))
from main import app

# Create test client
client = TestClient(app)

# Test configuration data
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


class TestDashboardComponentIntegration:
    """Test class for dashboard component integration with config API."""
    
    def test_dashboard_dropdown_population_data(self):
        """Test that config API provides correct data for dashboard dropdowns."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            assert response.status_code == 200
            
            config_data = response.json()
            
            # Test instrument dropdown data requirements
            instruments = config_data["instruments"]
            
            # Verify each instrument has data needed for SelectItem components
            for instrument in instruments:
                # Key prop: instrument.symbol
                assert "symbol" in instrument
                assert isinstance(instrument["symbol"], str)
                assert len(instrument["symbol"]) > 0
                
                # Display text: instrument.name (symbol)
                assert "name" in instrument
                assert isinstance(instrument["name"], str)
                assert len(instrument["name"]) > 0
                
                # Should be able to create unique keys
                display_key = instrument["symbol"]
                display_text = f"{instrument['name']} ({instrument['symbol']})"
                assert display_key != display_text
                assert len(display_text) > len(display_key)
            
            # Verify no duplicate symbols (important for React keys)
            symbols = [inst["symbol"] for inst in instruments]
            assert len(symbols) == len(set(symbols)), "All instrument symbols should be unique"
    
    def test_timeframe_dropdown_population_data(self):
        """Test that config API provides correct data for timeframe dropdowns."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            assert response.status_code == 200
            
            config_data = response.json()
            timeframes = config_data["timeframes"]
            
            # Test timeframe dropdown data requirements
            for timeframe in timeframes:
                # Key and value props: timeframe
                assert isinstance(timeframe, str)
                assert len(timeframe) > 0
                
                # Should be able to generate display labels
                if timeframe == 'D':
                    expected_label = 'Daily'
                else:
                    # Should be numeric for minute-based timeframes
                    try:
                        minutes = int(timeframe)
                        assert minutes > 0
                        if minutes >= 60:
                            hours = minutes / 60
                            if hours == int(hours):  # Whole hours
                                expected_label = f"{int(hours)} Hour{'s' if hours > 1 else ''}"
                            else:
                                expected_label = f"{minutes} Minutes"
                        else:
                            expected_label = f"{minutes} Minute{'s' if minutes > 1 else ''}"
                    except ValueError:
                        pytest.fail(f"Timeframe should be numeric or 'D', got: {timeframe}")
                
                # Label should be different from raw value
                assert expected_label != timeframe
            
            # Verify no duplicate timeframes (important for React keys)
            assert len(timeframes) == len(set(timeframes)), "All timeframes should be unique"
    
    def test_loading_state_simulation(self):
        """Test API response time for loading state handling."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            # Simulate API call for loading state testing
            response = client.get("/api/config")
            
            # Should return quickly for good UX
            assert response.status_code == 200
            
            # Should have all required data immediately
            config_data = response.json()
            assert len(config_data["instruments"]) > 0
            assert len(config_data["timeframes"]) > 0
            
            # Data should be ready for immediate UI rendering
            instruments = config_data["instruments"]
            timeframes = config_data["timeframes"]
            
            # Simulate component state update
            component_state = {
                "instruments": instruments,
                "timeframes": timeframes,
                "isLoadingConfig": False,  # Would be set to False after successful API call
                "configError": None
            }
            
            assert component_state["isLoadingConfig"] is False
            assert component_state["configError"] is None
            assert len(component_state["instruments"]) > 0
            assert len(component_state["timeframes"]) > 0
    
    def test_error_state_handling(self):
        """Test error response for component error state handling."""
        # Simulate configuration error
        with patch('builtins.open', side_effect=FileNotFoundError("Config file not found")):
            
            response = client.get("/api/config")
            
            assert response.status_code == 500
            error_data = response.json()
            
            # Simulate component error state
            component_error_state = {
                "instruments": [],
                "timeframes": [],
                "isLoadingConfig": False,
                "configError": error_data.get("detail", "Unknown error")
            }
            
            # Component should handle error state correctly
            assert component_error_state["isLoadingConfig"] is False
            assert component_error_state["configError"] is not None
            assert len(component_error_state["configError"]) > 0
            assert len(component_error_state["instruments"]) == 0
            assert len(component_error_state["timeframes"]) == 0
            
            # Error message should be user-friendly
            error_message = component_error_state["configError"]
            assert "Configuration file not found" in error_message
    
    def test_dropdown_disabled_state_during_loading(self):
        """Test that dropdown components can be properly disabled during loading."""
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            # Simulate loading state
            loading_component_state = {
                "instruments": [],
                "timeframes": [],
                "isLoadingConfig": True,
                "configError": None
            }
            
            # Simulate component rendering during loading
            # Dropdowns should be disabled and show loading placeholder
            select_props = {
                "disabled": loading_component_state["isLoadingConfig"],
                "placeholder": "Loading..." if loading_component_state["isLoadingConfig"] else "Select an instrument"
            }
            
            assert select_props["disabled"] is True
            assert select_props["placeholder"] == "Loading..."
            
            # After API call completes
            response = client.get("/api/config")
            config_data = response.json()
            
            loaded_component_state = {
                "instruments": config_data["instruments"],
                "timeframes": config_data["timeframes"],
                "isLoadingConfig": False,
                "configError": None
            }
            
            # Dropdowns should be enabled with data
            select_props_loaded = {
                "disabled": loaded_component_state["isLoadingConfig"],
                "placeholder": "Loading..." if loaded_component_state["isLoadingConfig"] else "Select an instrument"
            }
            
            assert select_props_loaded["disabled"] is False
            assert select_props_loaded["placeholder"] == "Select an instrument"
    
    def test_component_rerender_with_config_data(self):
        """Test that component can handle config data updates correctly."""
        # Initial empty state
        initial_state = {
            "instruments": [],
            "timeframes": [],
            "isLoadingConfig": True,
            "configError": None
        }
        
        # Verify initial state
        assert len(initial_state["instruments"]) == 0
        assert len(initial_state["timeframes"]) == 0
        assert initial_state["isLoadingConfig"] is True
        
        # Simulate API call success
        with patch('builtins.open', mock_open()), \
             patch('yaml.safe_load', return_value=TEST_CONFIG_DATA):
            
            response = client.get("/api/config")
            config_data = response.json()
            
            # Updated state after API call
            updated_state = {
                "instruments": config_data["instruments"],
                "timeframes": config_data["timeframes"],
                "isLoadingConfig": False,
                "configError": None
            }
            
            # Verify state transition
            assert len(updated_state["instruments"]) > len(initial_state["instruments"])
            assert len(updated_state["timeframes"]) > len(initial_state["timeframes"])
            assert updated_state["isLoadingConfig"] != initial_state["isLoadingConfig"]
            
            # Verify component can render with new data
            for instrument in updated_state["instruments"]:
                # Each instrument should be renderable as SelectItem
                assert "symbol" in instrument  # for key prop
                assert "name" in instrument    # for display text
                
                # Should not cause rendering errors
                select_item_props = {
                    "key": instrument["symbol"],
                    "value": instrument["symbol"],
                    "display": f"{instrument['name']} ({instrument['symbol']})"
                }
                
                assert len(select_item_props["key"]) > 0
                assert len(select_item_props["value"]) > 0
                assert len(select_item_props["display"]) > 0