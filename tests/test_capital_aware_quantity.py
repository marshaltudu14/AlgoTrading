"""
Test suite for capital-aware quantity selection.
"""

import pytest
import os
from src.utils.capital_aware_quantity import CapitalAwareQuantitySelector, adjust_quantity_for_capital, get_affordable_quantities
from src.utils.instrument_loader import load_instruments

class TestCapitalAwareQuantitySelector:
    """Test suite for CapitalAwareQuantitySelector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.selector = CapitalAwareQuantitySelector()
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        self.instruments = load_instruments(yaml_path)
        self.bank_nifty = self.instruments["Bank_Nifty"]
        self.reliance = self.instruments["RELIANCE"]
    
    def test_options_quantity_adjustment_sufficient_capital(self):
        """Test quantity adjustment for options with sufficient capital."""
        # High capital, should not adjust
        predicted_quantity = 3.0
        available_capital = 100000
        current_price = 45000
        proxy_premium = 200
        
        adjusted = self.selector.adjust_quantity_for_capital(
            predicted_quantity, available_capital, current_price, self.bank_nifty, proxy_premium
        )
        
        assert adjusted == 3  # Should not be adjusted
    
    def test_options_quantity_adjustment_insufficient_capital(self):
        """Test quantity adjustment for options with insufficient capital."""
        # Low capital, should adjust down
        predicted_quantity = 5.0
        available_capital = 10000
        current_price = 45000
        proxy_premium = 1000  # Expensive premium
        
        adjusted = self.selector.adjust_quantity_for_capital(
            predicted_quantity, available_capital, current_price, self.bank_nifty, proxy_premium
        )
        
        # With ₹10,000 capital and ₹1,000 premium per lot (25 lot size)
        # Cost per lot = 1000 * 25 = ₹25,000
        # Should adjust to 0 (not affordable)
        assert adjusted == 0
    
    def test_options_quantity_adjustment_partial_capital(self):
        """Test quantity adjustment for options with partial capital."""
        predicted_quantity = 4.0
        available_capital = 50000
        current_price = 45000
        proxy_premium = 400  # Moderate premium
        
        adjusted = self.selector.adjust_quantity_for_capital(
            predicted_quantity, available_capital, current_price, self.bank_nifty, proxy_premium
        )
        
        # With ₹50,000 capital and ₹400 premium per lot (25 lot size)
        # Cost per lot = 400 * 25 = ₹10,000
        # Max affordable = (50000 - 25) / 10000 = 4.9 → 4 lots
        # Should not adjust (4 lots is affordable)
        assert adjusted == 4
    
    def test_stocks_quantity_adjustment_sufficient_capital(self):
        """Test quantity adjustment for stocks with sufficient capital."""
        predicted_quantity = 2.0
        available_capital = 100000
        current_price = 2500
        
        adjusted = self.selector.adjust_quantity_for_capital(
            predicted_quantity, available_capital, current_price, self.reliance
        )
        
        assert adjusted == 2  # Should not be adjusted
    
    def test_stocks_quantity_adjustment_insufficient_capital(self):
        """Test quantity adjustment for stocks with insufficient capital."""
        predicted_quantity = 5.0
        available_capital = 10000
        current_price = 3000  # Expensive stock
        
        adjusted = self.selector.adjust_quantity_for_capital(
            predicted_quantity, available_capital, current_price, self.reliance
        )
        
        # With ₹10,000 capital and ₹3,000 per share (lot size 1)
        # Cost per lot = 3000 * 1 = ₹3,000
        # Max affordable = (10000 - 25) / 3000 = 3.3 → 3 lots
        assert adjusted == 3
    
    def test_calculate_trade_cost_options(self):
        """Test trade cost calculation for options."""
        quantity = 2
        current_price = 45000
        proxy_premium = 300
        
        cost = self.selector.calculate_trade_cost(
            quantity, current_price, self.bank_nifty, proxy_premium
        )
        
        expected_cost = (300 * 2 * 25) + 25  # (premium * quantity * lot_size) + brokerage
        assert cost == expected_cost
    
    def test_calculate_trade_cost_stocks(self):
        """Test trade cost calculation for stocks."""
        quantity = 10
        current_price = 2500
        
        cost = self.selector.calculate_trade_cost(
            quantity, current_price, self.reliance
        )
        
        expected_cost = (2500 * 10 * 1) + 25  # (price * quantity * lot_size) + brokerage
        assert cost == expected_cost
    
    def test_get_affordable_quantities_options(self):
        """Test getting affordable quantities for options."""
        available_capital = 50000
        current_price = 45000
        proxy_premium = 400
        
        affordable = self.selector.get_affordable_quantities(
            available_capital, current_price, self.bank_nifty, proxy_premium
        )
        
        # With ₹50,000 capital and ₹400 premium per lot (25 lot size)
        # Cost per lot = 400 * 25 = ₹10,000
        # 1 lot = ₹10,025, 2 lots = ₹20,025, 3 lots = ₹30,025, 4 lots = ₹40,025, 5 lots = ₹50,025
        # Should be affordable: [1, 2, 3, 4] (5 lots exceeds capital)
        expected = [1, 2, 3, 4]
        assert affordable == expected
    
    def test_get_affordable_quantities_stocks(self):
        """Test getting affordable quantities for stocks."""
        available_capital = 20000
        current_price = 2000
        
        affordable = self.selector.get_affordable_quantities(
            available_capital, current_price, self.reliance
        )
        
        # With ₹20,000 capital and ₹2,000 per share (lot size 1)
        # 1 lot = ₹2,025, 2 lots = ₹4,025, 3 lots = ₹6,025, 4 lots = ₹8,025, 5 lots = ₹10,025
        # All should be affordable
        expected = [1, 2, 3, 4, 5]
        assert affordable == expected
    
    def test_get_capital_utilization_ratio(self):
        """Test capital utilization ratio calculation."""
        quantity = 2
        available_capital = 50000
        current_price = 45000
        proxy_premium = 500
        
        ratio = self.selector.get_capital_utilization_ratio(
            quantity, available_capital, current_price, self.bank_nifty, proxy_premium
        )
        
        # Cost = (500 * 2 * 25) + 25 = ₹25,025
        # Ratio = 25025 / 50000 = 0.5005
        expected_ratio = 25025 / 50000
        assert abs(ratio - expected_ratio) < 0.001
    
    def test_convenience_functions(self):
        """Test convenience functions."""
        # Test adjust_quantity_for_capital
        adjusted = adjust_quantity_for_capital(
            3.0, 50000, 45000, self.bank_nifty, 300
        )
        assert isinstance(adjusted, int)
        assert 1 <= adjusted <= 5
        
        # Test get_affordable_quantities
        affordable = get_affordable_quantities(
            50000, 45000, self.bank_nifty, 300
        )
        assert isinstance(affordable, list)
        assert all(isinstance(q, int) for q in affordable)
        assert all(1 <= q <= 5 for q in affordable)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Zero capital
        adjusted = self.selector.adjust_quantity_for_capital(
            3.0, 0, 45000, self.bank_nifty, 300
        )
        assert adjusted == 0
        
        # Negative predicted quantity
        adjusted = self.selector.adjust_quantity_for_capital(
            -1.0, 50000, 45000, self.bank_nifty, 300
        )
        assert adjusted >= 1  # Should be clamped to minimum 1
        
        # Very high predicted quantity
        adjusted = self.selector.adjust_quantity_for_capital(
            100.0, 50000, 45000, self.bank_nifty, 300
        )
        assert adjusted <= 5  # Should be clamped to maximum 5
        
        # Very low capital
        affordable = self.selector.get_affordable_quantities(
            100, 45000, self.bank_nifty, 1000
        )
        assert affordable == []  # Nothing affordable
    
    def test_premium_estimation(self):
        """Test automatic premium estimation for options."""
        # Test without providing proxy_premium
        adjusted = self.selector.adjust_quantity_for_capital(
            2.0, 50000, 45000, self.bank_nifty  # No proxy_premium provided
        )
        
        # Should use estimated premium (1.5% of underlying)
        # Estimated premium = 45000 * 0.015 = 675
        # Cost per lot = 675 * 25 = ₹16,875
        # Should be affordable
        assert adjusted >= 1
