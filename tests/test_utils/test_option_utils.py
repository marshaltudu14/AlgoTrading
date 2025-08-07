import unittest
from datetime import datetime, timedelta

# Add project root to path to allow imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.option_utils import (
    get_nearest_itm_strike,
    get_nearest_expiry,
    map_underlying_to_option_price
)

class TestOptionUtils(unittest.TestCase):

    def test_get_nearest_itm_strike(self):
        strikes = [45000, 45100, 45200, 45300, 45400]
        
        # Test Call options
        self.assertEqual(get_nearest_itm_strike(45250, strikes, 'CE'), 45200)
        self.assertEqual(get_nearest_itm_strike(45200, strikes, 'CE'), 45200)
        self.assertIsNone(get_nearest_itm_strike(44900, strikes, 'CE'))

        # Test Put options
        self.assertEqual(get_nearest_itm_strike(45250, strikes, 'PE'), 45300)
        self.assertEqual(get_nearest_itm_strike(45300, strikes, 'PE'), 45300)
        self.assertIsNone(get_nearest_itm_strike(45500, strikes, 'PE'))

        # Test edge cases
        self.assertIsNone(get_nearest_itm_strike(45250, [], 'CE'))
        self.assertIsNone(get_nearest_itm_strike(45250, strikes, 'INVALID'))

    def test_get_nearest_expiry(self):
        today = datetime.now().date()
        expiries = [
            (today + timedelta(days=1)).strftime("%Y-%m-%d"),
            (today + timedelta(days=7)).strftime("%Y-%m-%d"),
            (today + timedelta(days=30)).strftime("%Y-%m-%d")
        ]
        
        self.assertEqual(get_nearest_expiry(expiries), expiries[0])
        self.assertEqual(get_nearest_expiry(expiries[1:]), expiries[1])
        self.assertIsNone(get_nearest_expiry([]))
        self.assertIsNone(get_nearest_expiry(["2020-01-01"])) # Past expiry

    def test_map_underlying_to_option_price(self):
        # Test Call option
        self.assertAlmostEqual(map_underlying_to_option_price(45100, 45000, 150, 'CE'), 250.0)
        self.assertAlmostEqual(map_underlying_to_option_price(44900, 45000, 150, 'CE'), 50.0)

        # Test Put option
        self.assertAlmostEqual(map_underlying_to_option_price(45100, 45000, 150, 'PE'), 50.0)
        self.assertAlmostEqual(map_underlying_to_option_price(44900, 45000, 150, 'PE'), 250.0)

        # Test price not going below zero
        self.assertAlmostEqual(map_underlying_to_option_price(44800, 45000, 150, 'CE'), 0.0)

if __name__ == '__main__':
    unittest.main()