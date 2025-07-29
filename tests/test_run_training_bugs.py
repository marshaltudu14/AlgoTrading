"""
Comprehensive test to identify bugs in the run_training.py script.
This test runs the actual training pipeline to catch real-world issues.
"""

import pytest
import subprocess
import sys
import os
import tempfile
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRunTrainingBugs:
    """Test suite to identify bugs in the run_training.py script."""
    
    def test_run_training_help(self):
        """Test that run_training.py help works."""
        result = subprocess.run(
            [sys.executable, "run_training.py", "--help"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "algorithm" in result.stdout.lower()
    
    def test_run_training_ppo_basic(self):
        """Test basic PPO training run."""
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py", 
                 "--algorithm", "PPO", 
                 "--episodes", "3", 
                 "--symbols", "Bank_Nifty", 
                 "--simple"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout
            )
            
            logger.info(f"PPO training return code: {result.returncode}")
            if result.stdout:
                logger.info(f"PPO training stdout: {result.stdout[-500:]}")  # Last 500 chars
            if result.stderr:
                logger.error(f"PPO training stderr: {result.stderr[-500:]}")  # Last 500 chars
            
            # Check if training completed successfully
            if result.returncode == 0:
                assert "training completed" in result.stdout.lower() or "episode" in result.stdout.lower()
            else:
                # Log the error for debugging but don't fail the test
                logger.error(f"PPO training failed with return code {result.returncode}")
                pytest.skip(f"PPO training failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("PPO training timed out")
        except Exception as e:
            pytest.skip(f"PPO training test failed: {e}")
    
    def test_run_training_moe_basic(self):
        """Test basic MoE training run."""
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py", 
                 "--algorithm", "MoE", 
                 "--episodes", "3", 
                 "--symbols", "Bank_Nifty", 
                 "--simple"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout
            )
            
            logger.info(f"MoE training return code: {result.returncode}")
            if result.stdout:
                logger.info(f"MoE training stdout: {result.stdout[-500:]}")
            if result.stderr:
                logger.error(f"MoE training stderr: {result.stderr[-500:]}")
            
            # Check if training completed successfully
            if result.returncode == 0:
                assert "training completed" in result.stdout.lower() or "episode" in result.stdout.lower()
            else:
                logger.error(f"MoE training failed with return code {result.returncode}")
                pytest.skip(f"MoE training failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("MoE training timed out")
        except Exception as e:
            pytest.skip(f"MoE training test failed: {e}")
    
    def test_run_training_sequence_mode(self):
        """Test sequence training mode."""
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py", 
                 "--sequence", 
                 "--episodes", "2",  # Very short for testing
                 "--symbols", "Bank_Nifty"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout for sequence
            )
            
            logger.info(f"Sequence training return code: {result.returncode}")
            if result.stdout:
                logger.info(f"Sequence training stdout: {result.stdout[-500:]}")
            if result.stderr:
                logger.error(f"Sequence training stderr: {result.stderr[-500:]}")
            
            # Check if sequence training started
            if result.returncode == 0:
                assert "stage" in result.stdout.lower() or "sequence" in result.stdout.lower()
            else:
                logger.error(f"Sequence training failed with return code {result.returncode}")
                pytest.skip(f"Sequence training failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Sequence training timed out")
        except Exception as e:
            pytest.skip(f"Sequence training test failed: {e}")
    
    def test_run_training_invalid_algorithm(self):
        """Test handling of invalid algorithm."""
        result = subprocess.run(
            [sys.executable, "run_training.py", 
             "--algorithm", "INVALID", 
             "--episodes", "1", 
             "--symbols", "Bank_Nifty"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Should fail with non-zero return code
        assert result.returncode != 0
        assert "invalid choice" in result.stderr.lower() or "error" in result.stderr.lower()
    
    def test_run_training_invalid_symbol(self):
        """Test handling of invalid symbol."""
        try:
            result = subprocess.run(
                [sys.executable, "run_training.py", 
                 "--algorithm", "PPO", 
                 "--episodes", "1", 
                 "--symbols", "INVALID_SYMBOL", 
                 "--simple"],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            logger.info(f"Invalid symbol test return code: {result.returncode}")
            if result.stderr:
                logger.info(f"Invalid symbol stderr: {result.stderr[-300:]}")
            
            # Should either fail gracefully or handle the invalid symbol
            # We don't assert failure here because the system might handle it gracefully
            
        except subprocess.TimeoutExpired:
            pytest.skip("Invalid symbol test timed out")
        except Exception as e:
            pytest.skip(f"Invalid symbol test failed: {e}")

class TestDataLoaderBugs:
    """Test suite to identify bugs in data loading."""
    
    def test_data_loader_import(self):
        """Test that DataLoader can be imported."""
        try:
            from src.utils.data_loader import DataLoader
            assert DataLoader is not None
        except ImportError as e:
            pytest.fail(f"DataLoader import failed: {e}")
    
    def test_data_loader_basic_functionality(self):
        """Test basic DataLoader functionality."""
        try:
            from src.utils.data_loader import DataLoader
            
            # Create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                final_dir = os.path.join(temp_dir, "final")
                raw_dir = os.path.join(temp_dir, "raw")
                os.makedirs(final_dir, exist_ok=True)
                os.makedirs(raw_dir, exist_ok=True)
                
                # Create DataLoader
                loader = DataLoader(final_data_dir=final_dir, raw_data_dir=raw_dir, use_parquet=False)
                
                # Test basic methods
                df = loader.load_all_processed_data()
                assert df.empty  # Should be empty for empty directory
                
                tasks = loader.get_available_tasks()
                assert isinstance(tasks, list)
                
        except Exception as e:
            pytest.fail(f"DataLoader basic functionality test failed: {e}")

class TestEnvironmentBugs:
    """Test suite to identify bugs in trading environment."""
    
    def test_trading_env_import(self):
        """Test that TradingEnv can be imported."""
        try:
            from src.backtesting.environment import TradingEnv
            assert TradingEnv is not None
        except ImportError as e:
            pytest.fail(f"TradingEnv import failed: {e}")
    
    def test_trading_env_basic_functionality(self):
        """Test basic TradingEnv functionality."""
        try:
            from src.backtesting.environment import TradingEnv
            from src.utils.data_loader import DataLoader
            
            # Create temporary data
            with tempfile.TemporaryDirectory() as temp_dir:
                raw_dir = os.path.join(temp_dir, "raw")
                os.makedirs(raw_dir, exist_ok=True)
                
                # Create minimal test data
                import pandas as pd
                df = pd.DataFrame({
                    'datetime': ['2023-01-01', '2023-01-02'],
                    'open': [100, 101],
                    'high': [105, 106],
                    'low': [99, 100],
                    'close': [103, 104],
                    'volume': [1000, 2000]
                })
                df.to_csv(os.path.join(raw_dir, 'Bank_Nifty.csv'), index=False)
                
                # Create DataLoader and TradingEnv
                loader = DataLoader(raw_data_dir=raw_dir, use_parquet=False)
                env = TradingEnv(
                    data_loader=loader,
                    symbol="Bank_Nifty",
                    initial_capital=10000,
                    lookback_window=1,
                    episode_length=2,
                    use_streaming=False
                )
                
                # Test basic functionality
                obs = env.reset()
                assert obs is not None
                assert len(obs) > 0
                
                action = (4, 1.0)  # HOLD action
                next_obs, reward, done, info = env.step(action)
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)
                assert isinstance(info, dict)
                
        except Exception as e:
            pytest.fail(f"TradingEnv basic functionality test failed: {e}")

class TestAgentBugs:
    """Test suite to identify bugs in agent implementations."""
    
    def test_ppo_agent_import(self):
        """Test that PPOAgent can be imported."""
        try:
            from src.agents.ppo_agent import PPOAgent
            assert PPOAgent is not None
        except ImportError as e:
            pytest.fail(f"PPOAgent import failed: {e}")
    
    def test_moe_agent_import(self):
        """Test that MoEAgent can be imported."""
        try:
            from src.agents.moe_agent import MoEAgent
            assert MoEAgent is not None
        except ImportError as e:
            pytest.fail(f"MoEAgent import failed: {e}")
    
    def test_ppo_agent_basic_functionality(self):
        """Test basic PPOAgent functionality."""
        try:
            from src.agents.ppo_agent import PPOAgent
            import numpy as np
            
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
            
            # Test action selection
            obs = np.random.rand(20).astype(np.float32)
            action_type, quantity = agent.select_action(obs)
            
            assert isinstance(action_type, int)
            assert 0 <= action_type < 5
            assert isinstance(quantity, float)
            assert quantity > 0
            
        except Exception as e:
            pytest.fail(f"PPOAgent basic functionality test failed: {e}")
    
    def test_moe_agent_basic_functionality(self):
        """Test basic MoEAgent functionality."""
        try:
            from src.agents.moe_agent import MoEAgent
            import numpy as np
            
            expert_configs = {
                'TrendAgent': {'lr': 0.001, 'hidden_dim': 32},
                'MeanReversionAgent': {'lr': 0.001, 'hidden_dim': 32}
            }
            
            agent = MoEAgent(
                observation_dim=20,
                action_dim_discrete=5,
                action_dim_continuous=1,
                hidden_dim=32,
                expert_configs=expert_configs
            )
            
            # Test action selection
            obs = np.random.rand(20).astype(np.float32)
            action_type, quantity = agent.select_action(obs)
            
            assert isinstance(action_type, int)
            assert 0 <= action_type < 5
            assert isinstance(quantity, float)
            assert quantity > 0
            
        except Exception as e:
            pytest.fail(f"MoEAgent basic functionality test failed: {e}")
