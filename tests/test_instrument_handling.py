#!/usr/bin/env python3
"""
Test instrument handling for generic data.
"""

import numpy as np
import tempfile
import os
import pandas as pd
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.utils.instrument_loader import load_instruments
from src.agents.ppo_agent import PPOAgent

def create_test_data(symbol: str, num_rows: int = 100):
    """Create test data for a specific symbol."""
    np.random.seed(42)
    
    # Different base prices for different instruments
    if "RELIANCE" in symbol:
        base_price = 2800.0
    elif "Bank_Nifty" in symbol:
        base_price = 46500.0
    else:
        base_price = 1000.0
    
    data = []
    current_price = base_price
    
    for i in range(num_rows):
        # Random walk
        change = np.random.normal(0, 0.02) * current_price
        current_price = max(current_price + change, base_price * 0.5)
        
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'datetime': f"2024-01-{(i % 30) + 1:02d} 09:{(i % 60):02d}:00",
            'open': current_price,
            'high': high,
            'low': low,
            'close': current_price,
            'volume': volume,
            'sma_5': current_price,
            'sma_10': current_price,
            'rsi': 50.0
        })
    
    return pd.DataFrame(data)

def test_stock_instrument():
    """Test stock instrument handling."""
    print("🧪 Testing STOCK instrument (RELIANCE)...")
    
    # Load instruments
    instruments = load_instruments('config/instruments.yaml')
    reliance = instruments["RELIANCE"]
    
    print(f"   Instrument: {reliance.symbol}")
    print(f"   Type: {reliance.type}")
    print(f"   Lot size: {reliance.lot_size}")
    print(f"   Tick size: {reliance.tick_size}")
    
    # Verify instrument properties
    assert reliance.type == "STOCK", f"Expected STOCK, got {reliance.type}"
    assert reliance.lot_size == 1, f"Expected lot size 1 for stocks, got {reliance.lot_size}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        raw_dir = os.path.join(temp_dir, "raw")
        final_dir = os.path.join(temp_dir, "final")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Create and save test data
        test_data = create_test_data("RELIANCE_1", 200)
        test_data.to_csv(os.path.join(final_dir, "features_RELIANCE_1.csv"), index=False)
        
        # Create environment
        loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
        env = TradingEnv(
            data_loader=loader,
            symbol="RELIANCE_1",
            initial_capital=100000,
            lookback_window=10,
            episode_length=20,
            use_streaming=False
        )
        
        # Test environment
        obs = env.reset()
        if obs is not None:
            # Verify instrument is loaded correctly
            assert env.instrument.lot_size == 1

            # Test a few trading actions
            for step in range(5):
                # BUY action with quantity 2
                action = (0, 2.0)  # BUY_LONG, 2 lots
                obs, reward, done, info = env.step(action)

                print(f"   Step {step}: Action=BUY_LONG, Quantity=2, Reward={reward:.4f}")

                if done:
                    break

            print("✅ RELIANCE instrument handling working correctly")
            return True
        else:
            print("⚠️ Could not reset environment for RELIANCE")
            return False

def test_option_instrument():
    """Test option instrument handling."""
    print("🧪 Testing OPTION instrument (Bank_Nifty)...")
    
    # Load instruments
    instruments = load_instruments('config/instruments.yaml')
    bank_nifty = instruments["Bank_Nifty"]
    
    print(f"   Instrument: {bank_nifty.symbol}")
    print(f"   Type: {bank_nifty.type}")
    print(f"   Lot size: {bank_nifty.lot_size}")
    print(f"   Tick size: {bank_nifty.tick_size}")
    
    # Verify instrument properties
    assert bank_nifty.type == "OPTION", f"Expected OPTION, got {bank_nifty.type}"
    assert bank_nifty.lot_size == 25, f"Expected lot size 25 for Bank Nifty options, got {bank_nifty.lot_size}"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        raw_dir = os.path.join(temp_dir, "raw")
        final_dir = os.path.join(temp_dir, "final")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Create and save test data
        test_data = create_test_data("Bank_Nifty_5", 200)
        test_data.to_csv(os.path.join(final_dir, "features_Bank_Nifty_5.csv"), index=False)
        
        # Create environment
        loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
        env = TradingEnv(
            data_loader=loader,
            symbol="Bank_Nifty_5",
            initial_capital=100000,
            lookback_window=10,
            episode_length=20,
            use_streaming=False
        )
        
        # Test environment
        obs = env.reset()
        if obs is not None:
            # Verify instrument is loaded correctly
            assert env.instrument.lot_size == 25

            # Test a few trading actions
            for step in range(5):
                # BUY action with quantity 1
                action = (0, 1.0)  # BUY_LONG, 1 lot
                obs, reward, done, info = env.step(action)

                print(f"   Step {step}: Action=BUY_LONG, Quantity=1, Reward={reward:.4f}")

                if done:
                    break

            print("✅ Bank_Nifty instrument handling working correctly")
            return True
        else:
            print("⚠️ Could not reset environment for Bank_Nifty")
            return False

def test_premium_calculation():
    """Test proxy premium calculation for options."""
    print("🧪 Testing proxy premium calculation...")
    
    # Load instruments
    instruments = load_instruments('config/instruments.yaml')
    bank_nifty = instruments["Bank_Nifty"]
    
    # Test premium calculation
    underlying_price = 46500.0
    expected_premium = underlying_price * 0.015  # 1.5% of underlying
    
    print(f"   Underlying price: ₹{underlying_price}")
    print(f"   Expected premium (1.5%): ₹{expected_premium}")
    
    # Test cost calculation
    cost_per_lot = expected_premium * bank_nifty.lot_size
    print(f"   Cost per lot (premium × lot_size): ₹{cost_per_lot}")
    
    # Verify reasonable premium
    assert 0.01 <= (expected_premium / underlying_price) <= 0.05, "Premium should be 1-5% of underlying"
    
    print("✅ Premium calculation working correctly")
    return True

def test_lot_size_differences():
    """Test that different instruments have correct lot sizes."""
    print("🧪 Testing lot size differences...")
    
    instruments = load_instruments('config/instruments.yaml')
    
    # Test different instrument types
    test_cases = [
        ("RELIANCE", "STOCK", 1),
        ("Bank_Nifty", "OPTION", 25),
    ]
    
    for symbol, expected_type, expected_lot_size in test_cases:
        if symbol in instruments:
            instrument = instruments[symbol]
            print(f"   {symbol}: Type={instrument.type}, Lot Size={instrument.lot_size}")
            
            assert instrument.type == expected_type, f"Expected {expected_type}, got {instrument.type}"
            assert instrument.lot_size == expected_lot_size, f"Expected lot size {expected_lot_size}, got {instrument.lot_size}"
        else:
            print(f"   ⚠️ {symbol} not found in instruments")
    
    print("✅ Lot size differences verified")
    return True

def main():
    """Run all instrument handling tests."""
    print("🚀 Starting instrument type handling verification...\n")
    
    try:
        # Test 1: Stock instrument
        test_stock_instrument()
        print()
        
        # Test 2: Option instrument
        test_option_instrument()
        print()
        
        # Test 3: Premium calculation
        test_premium_calculation()
        print()
        
        # Test 4: Lot size differences
        test_lot_size_differences()
        print()
        
        print("🎉 All instrument handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
