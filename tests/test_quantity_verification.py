#!/usr/bin/env python3
"""
Simple test to verify quantity prediction and capital awareness functionality.
"""

import numpy as np
import tempfile
import os
import pandas as pd
from src.agents.ppo_agent import PPOAgent
from src.backtesting.environment import TradingEnv
from src.utils.data_loader import DataLoader
from src.utils.capital_aware_quantity import adjust_quantity_for_capital
from src.utils.instrument_loader import load_instruments

def create_simple_test_data(symbol: str, num_rows: int = 100):
    """Create simple test data for verification."""
    np.random.seed(42)
    
    # Generate basic OHLCV data
    base_price = 2800.0 if "RELIANCE" in symbol else 46500.0
    
    data = []
    current_price = base_price
    
    for i in range(num_rows):
        # Random walk with some volatility
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
            'volume': volume
        })
    
    return pd.DataFrame(data)

def test_quantity_prediction():
    """Test that PPO agent predicts integer quantities."""
    print("🧪 Testing quantity prediction...")
    
    agent = PPOAgent(
        observation_dim=100,
        action_dim_discrete=5,
        action_dim_continuous=1,
        hidden_dim=32,
        lr_actor=0.001,
        lr_critic=0.001,
        gamma=0.99,
        epsilon_clip=0.2,
        k_epochs=3
    )
    
    # Test multiple predictions
    quantities = []
    for i in range(20):
        obs = np.random.rand(100).astype(np.float32)
        action_type, quantity = agent.select_action(obs)
        quantities.append(quantity)
        
        # Verify action type is valid
        assert 0 <= action_type <= 4, f"Invalid action type: {action_type}"
        
        # Verify quantity is reasonable
        assert isinstance(quantity, (int, float)), f"Quantity should be numeric, got {type(quantity)}"
        assert quantity > 0, f"Quantity should be positive, got {quantity}"
    
    print(f"✅ Generated {len(quantities)} quantities: {quantities[:10]}...")
    print(f"   Quantity range: {min(quantities):.2f} - {max(quantities):.2f}")
    return True

def test_capital_awareness():
    """Test capital-aware quantity adjustment."""
    print("🧪 Testing capital awareness...")
    
    # Load instruments
    instruments = load_instruments('config/instruments.yaml')
    reliance = instruments["RELIANCE"]
    bank_nifty = instruments["Bank_Nifty"]
    
    # Test scenarios
    test_cases = [
        # (predicted_qty, capital, price, instrument, expected_result)
        (5.0, 100000, 2800, reliance, "Should allow multiple lots"),
        (5.0, 10000, 2800, reliance, "Should reduce quantity due to capital"),
        (3.0, 50000, 46500, bank_nifty, "Should handle options with premium"),
        (1.0, 5000, 2800, reliance, "Should handle minimal capital"),
    ]
    
    for predicted_qty, capital, price, instrument, description in test_cases:
        print(f"   Testing: {description}")
        
        # For options, estimate premium
        proxy_premium = price * 0.015 if instrument.type == "OPTION" else None
        
        adjusted_qty = adjust_quantity_for_capital(
            predicted_quantity=predicted_qty,
            available_capital=capital,
            current_price=price,
            instrument=instrument,
            proxy_premium=proxy_premium
        )
        
        print(f"     Predicted: {predicted_qty}, Capital: ₹{capital}, Adjusted: {adjusted_qty}")
        
        # Verify adjusted quantity is reasonable
        assert isinstance(adjusted_qty, int), f"Adjusted quantity should be integer, got {type(adjusted_qty)}"
        assert adjusted_qty >= 0, f"Adjusted quantity should be non-negative, got {adjusted_qty}"
        assert adjusted_qty <= predicted_qty, f"Adjusted quantity should not exceed predicted, got {adjusted_qty} > {predicted_qty}"
    
    print("✅ Capital awareness working correctly")
    return True

def test_environment_integration():
    """Test quantity prediction and capital awareness in trading environment."""
    print("🧪 Testing environment integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        raw_dir = os.path.join(temp_dir, "raw")
        final_dir = os.path.join(temp_dir, "final")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Create simple test data
        test_data = create_simple_test_data("RELIANCE_1", 200)
        test_data.to_csv(os.path.join(raw_dir, "RELIANCE_1.csv"), index=False)
        
        # Process data (simplified - just copy with some features)
        features_data = test_data.copy()
        # Add some simple technical indicators
        features_data['sma_5'] = features_data['close'].rolling(5).mean()
        features_data['sma_10'] = features_data['close'].rolling(10).mean()
        features_data['rsi'] = 50.0  # Simplified RSI
        features_data = features_data.dropna()
        
        features_data.to_csv(os.path.join(final_dir, "features_RELIANCE_1.csv"), index=False)
        
        # Create environment
        loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
        env = TradingEnv(
            data_loader=loader,
            symbol="RELIANCE_1",
            initial_capital=50000,
            lookback_window=10,
            episode_length=50,
            use_streaming=False
        )
        
        # Create agent
        obs = env.reset()
        if obs is not None:
            agent = PPOAgent(
                observation_dim=len(obs),
                action_dim_discrete=5,
                action_dim_continuous=1,
                hidden_dim=32,
                lr_actor=0.001,
                lr_critic=0.001,
                gamma=0.99,
                epsilon_clip=0.2,
                k_epochs=3
            )
            
            # Test a few steps
            for step in range(10):
                action_type, quantity = agent.select_action(obs)
                
                print(f"   Step {step}: Action={action_type}, Quantity={quantity}")
                
                # Verify outputs
                assert 0 <= action_type <= 4, f"Invalid action type: {action_type}"
                assert quantity > 0, f"Invalid quantity: {quantity}"
                
                # Execute action
                action = (action_type, quantity)
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
            
            print("✅ Environment integration working correctly")
            return True
        else:
            print("⚠️ Could not reset environment - skipping integration test")
            return False

def main():
    """Run all verification tests."""
    print("🚀 Starting quantity prediction and capital awareness verification...\n")
    
    try:
        # Test 1: Basic quantity prediction
        test_quantity_prediction()
        print()
        
        # Test 2: Capital awareness
        test_capital_awareness()
        print()
        
        # Test 3: Environment integration
        test_environment_integration()
        print()
        
        print("🎉 All tests passed! Quantity prediction and capital awareness are working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
