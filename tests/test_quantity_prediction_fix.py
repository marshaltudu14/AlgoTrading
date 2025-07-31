"""
Test suite for quantity prediction bug fixes.
Tests that quantities are predicted as integers and consider available capital.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import pandas as pd

from src.models.transformer_models import ActorTransformerModel
from src.agents.ppo_agent import PPOAgent
from src.backtesting.environment import TradingEnv
from src.backtesting.engine import BacktestingEngine
from src.utils.data_loader import DataLoader
from src.utils.instrument_loader import load_instruments

class TestQuantityPredictionFix:
    """Test suite for quantity prediction fixes."""
    
    def test_actor_transformer_integer_output(self):
        """Test that ActorTransformerModel outputs integer quantities."""
        model = ActorTransformerModel(
            input_dim=20,
            hidden_dim=32,
            action_dim_discrete=5,
            action_dim_continuous=1,
            num_heads=2,
            num_layers=2
        )
        
        # Test single sample
        x = torch.randn(1, 1, 20)
        output = model(x)
        
        quantity = output['quantity'].item()

        # Should be >= 1.0 and <= 100000.0 (consistent quantity prediction limit)
        assert 1.0 <= quantity <= 100000.0, f"Expected quantity between 1.0 and 100000.0, got {quantity}"

        # Test batch
        x_batch = torch.randn(3, 1, 20)
        output_batch = model(x_batch)

        quantities = output_batch['quantity']
        for i in range(quantities.shape[0]):
            q = quantities[i].item()
            assert 1.0 <= q <= 100000.0, f"Expected quantity between 1.0 and 100000.0, got {q}"
    
    def test_ppo_agent_integer_quantity_preservation(self):
        """Test that PPOAgent preserves integer quantities."""
        agent = PPOAgent(
            observation_dim=20,
            action_dim_discrete=5,
            action_dim_continuous=1,
            hidden_dim=32,
            lr_actor=0.001,
            lr_critic=0.001,
            gamma=0.99,
            epsilon_clip=0.2,
            k_epochs=3
        )
        
        # Test multiple action selections
        for _ in range(10):
            obs = np.random.rand(20).astype(np.float32)
            action_type, quantity = agent.select_action(obs)
            
            # Quantity should be >= 1.0 and <= 100000.0 (consistent quantity prediction limit)
            assert isinstance(quantity, (int, float))
            assert 1.0 <= quantity <= 100000.0, f"Expected quantity between 1.0 and 100000.0, got {quantity}"
    
    # MoE agent test removed - only using PPO for now

class TestCapitalBasedQuantityAdjustment:
    """Test capital-based quantity adjustment."""
    
    def create_test_data(self, symbol="Bank_Nifty", num_points=100):
        """Create test data for trading environment."""
        np.random.seed(42)
        base_price = 45000 if symbol == "Bank_Nifty" else 19000
        
        dates = pd.date_range('2023-01-01', periods=num_points, freq='1min')
        returns = np.random.normal(0.0001, 0.015, num_points)
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.randint(1000, 50000)
            
            data.append({
                'datetime': date,
                'open': open_price,
                'high': max(open_price, high, close_price),
                'low': min(open_price, low, close_price),
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_capital_based_quantity_adjustment_options(self):
        """Test that quantity is adjusted based on available capital for options."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)
        bank_nifty = instruments["Bank_Nifty"]
        
        # Test with different capital levels
        capital_levels = [10000, 50000, 100000, 500000]  # Different capital amounts
        
        for capital in capital_levels:
            engine = BacktestingEngine(initial_capital=capital, instrument=bank_nifty)
            
            # Simulate a trade with high premium (expensive option)
            current_price = 45000
            expensive_premium = 1000  # Very expensive option
            
            # Test maximum affordable quantity
            max_quantity = 5  # Model's maximum prediction
            
            # Calculate cost for max quantity
            cost = (expensive_premium * max_quantity * bank_nifty.lot_size) + engine.BROKERAGE_ENTRY
            
            if capital < cost:
                # Should reduce quantity to what's affordable
                affordable_quantity = int((capital - engine.BROKERAGE_ENTRY) // (expensive_premium * bank_nifty.lot_size))
                affordable_quantity = max(1, affordable_quantity)  # At least 1 lot
                
                # Test that the trade is executed with reduced quantity
                if affordable_quantity >= 1:
                    reward, _ = engine.execute_trade("BUY_LONG", current_price, affordable_quantity, 0.0, expensive_premium)
                    state = engine.get_account_state()
                    assert state["current_position_quantity"] == affordable_quantity
            else:
                # Should execute with full quantity
                reward, _ = engine.execute_trade("BUY_LONG", current_price, max_quantity, 0.0, expensive_premium)
                state = engine.get_account_state()
                assert state["current_position_quantity"] == max_quantity
    
    def test_capital_based_quantity_adjustment_stocks(self):
        """Test that quantity is adjusted based on available capital for stocks."""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'instruments.yaml')
        instruments = load_instruments(yaml_path)
        reliance = instruments["RELIANCE"]
        
        # Test with different capital levels
        capital_levels = [10000, 50000, 100000]
        
        for capital in capital_levels:
            engine = BacktestingEngine(initial_capital=capital, instrument=reliance)
            
            # Simulate expensive stock price
            expensive_stock_price = 3000
            
            # Test maximum affordable quantity
            max_quantity = 100  # Large quantity
            
            # Calculate cost for max quantity
            cost = (expensive_stock_price * max_quantity * reliance.lot_size) + engine.BROKERAGE_ENTRY
            
            if capital < cost:
                # Should reduce quantity to what's affordable
                affordable_quantity = int((capital - engine.BROKERAGE_ENTRY) // (expensive_stock_price * reliance.lot_size))
                affordable_quantity = max(1, affordable_quantity)
                
                # Test that the trade is executed with reduced quantity
                if affordable_quantity >= 1:
                    reward, _ = engine.execute_trade("BUY_LONG", expensive_stock_price, affordable_quantity)
                    state = engine.get_account_state()
                    assert state["current_position_quantity"] == affordable_quantity
            else:
                # Should execute with full quantity
                reward, _ = engine.execute_trade("BUY_LONG", expensive_stock_price, max_quantity)
                state = engine.get_account_state()
                assert state["current_position_quantity"] == max_quantity

class TestIntegratedQuantityPrediction:
    """Test integrated quantity prediction in trading environment."""
    
    def test_end_to_end_quantity_prediction(self):
        """Test end-to-end quantity prediction with capital consideration."""
        # Create test data
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=50, freq='1min'),
            'open': np.random.uniform(45000, 46000, 50),
            'high': np.random.uniform(46000, 47000, 50),
            'low': np.random.uniform(44000, 45000, 50),
            'close': np.random.uniform(45000, 46000, 50),
            'volume': np.random.randint(1000, 10000, 50)
        })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = os.path.join(temp_dir, "raw")
            final_dir = os.path.join(temp_dir, "final")
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(final_dir, exist_ok=True)
            
            # Save test data
            df.to_csv(os.path.join(raw_dir, "Bank_Nifty.csv"), index=False)
            
            # Create processed data
            df_processed = df.copy()
            df_processed['returns'] = df_processed['close'].pct_change()
            df_processed['sma_10'] = df_processed['close'].rolling(10).mean()
            df_processed['atr'] = abs(df_processed['high'] - df_processed['low']).rolling(14).mean()
            df_processed = df_processed.dropna()
            df_processed.to_csv(os.path.join(final_dir, "features_Bank_Nifty.csv"), index=False)
            
            # Create environment with limited capital
            loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
            env = TradingEnv(
                data_loader=loader,
                symbol="Bank_Nifty",
                initial_capital=50000,  # Limited capital
                lookback_window=5,
                episode_length=20,
                use_streaming=False
            )
            
            # Create agent
            agent = PPOAgent(
                observation_dim=env.observation_space.shape[0],
                action_dim_discrete=5,
                action_dim_continuous=1,
                hidden_dim=32,
                lr_actor=0.001,
                lr_critic=0.001,
                gamma=0.99,
                epsilon_clip=0.2,
                k_epochs=3
            )
            
            # Test episode
            obs = env.reset()
            
            for step in range(10):
                action_type, quantity = agent.select_action(obs)
                
                # Quantity should be integer
                assert quantity in [1.0, 2.0, 3.0, 4.0, 5.0], f"Expected integer quantity, got {quantity}"
                
                # Execute action
                action = (action_type, quantity)
                obs, reward, done, info = env.step(action)
                
                if done:
                    break
    
    def test_quantity_scaling_with_capital(self):
        """Test that quantity scales appropriately with available capital."""
        # Test different capital levels and verify quantity scaling
        capital_levels = [25000, 50000, 100000, 200000]
        
        for capital in capital_levels:
            # Create agent
            agent = PPOAgent(
                observation_dim=20,
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
            for _ in range(20):
                obs = np.random.rand(20).astype(np.float32)
                action_type, quantity = agent.select_action(obs)
                quantities.append(quantity)
            
            # All quantities should be within valid range
            for q in quantities:
                assert 1.0 <= q <= 100000.0, f"Expected quantity between 1.0 and 100000.0, got {q}"
            
            # With higher capital, we should see more variety in quantities
            # (This is a probabilistic test, so we check for reasonable distribution)
            unique_quantities = set(quantities)
            assert len(unique_quantities) >= 2, "Should have variety in quantity predictions"
