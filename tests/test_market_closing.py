#!/usr/bin/env python3
"""
Test suite for market closing functionality in TerminationManager.
"""

import sys
import os
import pandas as pd
from datetime import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.env.termination_manager import TerminationManager
from src.env.trading_mode import TradingMode

def test_market_closing_functionality():
    """Test market closing functionality."""
    print("Testing Market Closing Functionality...")
    
    # Create termination manager for LIVE mode
    tm = TerminationManager(TradingMode.LIVE)
    
    # Test cases for times that should trigger market closing (after 3:15 PM)
    closing_times = [
        pd.Timestamp('2024-03-20 15:15:00'),  # 3:15 PM - should close
        pd.Timestamp('2024-03-20 15:25:00'),  # 3:25 PM - should close
        pd.Timestamp('2024-03-20 16:00:00'),  # 4:00 PM - should close
    ]
    
    # Test cases for times that should NOT trigger market closing (before 3:15 PM)
    normal_times = [
        pd.Timestamp('2024-03-20 09:15:00'),  # 9:15 AM - should not close
        pd.Timestamp('2024-03-20 12:00:00'),  # 12:00 PM - should not close
        pd.Timestamp('2024-03-20 15:00:00'),  # 3:00 PM - should not close
        pd.Timestamp('2024-03-20 15:14:59'),  # 3:14:59 PM - should not close
    ]
    
    print("Testing times that should trigger market closing (after 3:15 PM):")
    for test_time in closing_times:
        should_close = tm._is_market_closed(test_time)
        print(f"  {test_time.time()} -> {should_close} (expected: True)")
        assert should_close, f"Expected {test_time.time()} to trigger closing"
    
    print("\nTesting times that should NOT trigger market closing (before 3:15 PM):")
    for test_time in normal_times:
        should_close = tm._is_market_closed(test_time)
        print(f"  {test_time.time()} -> {should_close} (expected: False)")
        assert not should_close, f"Expected {test_time.time()} to NOT trigger closing"
    
    print("\nTesting check_termination_conditions with market closing time:")
    
    # Test with LIVE mode and closing time - should terminate
    done, reason = tm.check_termination_conditions(
        current_step=100,
        data_length=1000,
        current_capital=100000.0,
        current_datetime=pd.Timestamp('2024-03-20 15:25:00')
    )
    
    print(f"  LIVE mode + closing time -> done: {done}, reason: {reason}")
    assert done, "Should terminate in LIVE mode when market is closed"
    assert "market_closed" in reason, "Should have market closed reason"
    
    # Test with TRAINING mode and closing time - should NOT terminate
    tm_training = TerminationManager(TradingMode.TRAINING)
    done, reason = tm_training.check_termination_conditions(
        current_step=100,
        data_length=1000,
        current_capital=100000.0,
        current_datetime=pd.Timestamp('2024-03-20 15:25:00')
    )
    
    print(f"  TRAINING mode + closing time -> done: {done}, reason: {reason}")
    assert not done, "Should NOT terminate in TRAINING mode even with closing time"
    
    print("\nAll market closing tests passed!")
    return True

def test_edge_cases():
    """Test edge cases for market closing functionality."""
    print("\nTesting Edge Cases...")
    
    tm = TerminationManager(TradingMode.LIVE)
    
    # Test with None datetime
    done, reason = tm.check_termination_conditions(
        current_step=100,
        data_length=1000,
        current_capital=100000.0,
        current_datetime=None
    )
    
    print(f"  None datetime -> done: {done}, reason: {reason}")
    assert not done, "Should not terminate with None datetime"
    
    # Test with non-Timestamp datetime
    done, reason = tm.check_termination_conditions(
        current_step=100,
        data_length=1000,
        current_capital=100000.0,
        current_datetime="not a timestamp"
    )
    
    print(f"  Non-Timestamp datetime -> done: {done}, reason: {reason}")
    assert not done, "Should not terminate with non-Timestamp datetime"
    
    print("Edge case tests passed!")
    return True

def main():
    """Main test function."""
    print("=" * 60)
    print("MARKET CLOSING FUNCTIONALITY TESTS")
    print("=" * 60)
    
    try:
        test_market_closing_functionality()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)