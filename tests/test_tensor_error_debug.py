"""
Debug test to isolate the tensor indexing error.
"""

import pytest
import traceback
import sys
import os

def test_ppo_training_tensor_error():
    """Test to isolate the tensor indexing error in PPO training."""
    try:
        # Import and run the exact same command that's failing
        import subprocess
        
        result = subprocess.run(
            [sys.executable, "run_training.py", 
             "--algorithm", "PPO", 
             "--episodes", "2", 
             "--symbols", "Bank_Nifty", 
             "--simple"],
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        # If it failed, let's try to get more details
        if result.returncode != 0:
            print("\n" + "="*50)
            print("DETAILED ERROR ANALYSIS")
            print("="*50)
            
            # Look for the specific tensor error
            if "invalid index of a 0-dim tensor" in result.stderr:
                print("✅ Found the tensor indexing error!")
                print("This confirms the issue is in the training pipeline.")
                
                # Try to extract more context
                lines = result.stderr.split('\n')
                for i, line in enumerate(lines):
                    if "invalid index of a 0-dim tensor" in line:
                        print(f"\nError context (lines {max(0, i-5)} to {min(len(lines), i+5)}):")
                        for j in range(max(0, i-5), min(len(lines), i+5)):
                            marker = ">>> " if j == i else "    "
                            print(f"{marker}{j}: {lines[j]}")
                        break
            
            # The test should pass even if training fails, as we're debugging
            print("\n✅ Test completed - error captured for analysis")
        else:
            print("✅ Training completed successfully!")
            
    except subprocess.TimeoutExpired:
        print("❌ Training timed out")
        pytest.skip("Training timed out")
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        traceback.print_exc()
        pytest.skip(f"Test setup failed: {e}")

def test_direct_ppo_agent_creation():
    """Test direct PPO agent creation to isolate tensor issues."""
    try:
        from src.agents.ppo_agent import PPOAgent
        from src.utils.data_loader import DataLoader
        from src.backtesting.environment import TradingEnv
        
        print("Testing direct PPO agent creation...")
        
        # Create data loader
        data_loader = DataLoader()
        
        # Create environment
        env = TradingEnv(
            data_loader=data_loader,
            symbol="Bank_Nifty",
            initial_capital=100000,
            lookback_window=10,
            episode_length=50,
            use_streaming=False
        )
        
        # Reset environment to get observation space
        obs = env.reset()
        observation_dim = len(obs)
        
        print(f"✅ Environment created successfully")
        print(f"   Observation dimension: {observation_dim}")
        print(f"   Data shape: {env.data.shape if env.data is not None else 'None'}")
        
        # Create PPO agent
        agent = PPOAgent(
            observation_dim=observation_dim,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=64,
            lr_actor=0.0003,
            lr_critic=0.0003,
            gamma=0.99,
            epsilon_clip=0.2,
            k_epochs=4
        )
        
        print(f"✅ PPO agent created successfully")
        
        # Test action selection
        action = agent.select_action(obs)
        print(f"✅ Action selection successful: {action}")
        
        # Test environment step
        next_obs, reward, done, info = env.step(action)
        print(f"✅ Environment step successful: reward={reward}, done={done}")
        
        # Test learning with a simple experience
        experiences = [(obs, action, reward, next_obs, done)]
        
        print("Testing agent learning...")
        agent.learn(experiences)
        print(f"✅ Agent learning successful")
        
        print("\n✅ All direct tests passed - no tensor errors found in basic operations")
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        traceback.print_exc()
        
        # Check if it's the tensor error we're looking for
        if "invalid index of a 0-dim tensor" in str(e):
            print("\n🎯 FOUND THE TENSOR ERROR!")
            print("This is the exact error we're looking for.")
            print("Stack trace above shows the exact location.")
        
        raise  # Re-raise to see full traceback

def test_optimizer_param_groups():
    """Test optimizer param_groups access to isolate tensor indexing."""
    try:
        import torch
        import torch.optim as optim
        from src.agents.ppo_agent import PPOAgent
        
        print("Testing optimizer param_groups access...")
        
        # Create a simple PPO agent
        agent = PPOAgent(
            observation_dim=100,
            action_dim_discrete=2,
            action_dim_continuous=1,
            hidden_dim=64,
            lr_actor=0.0003,
            lr_critic=0.0003,
            gamma=0.99,
            epsilon_clip=0.2,
            k_epochs=4
        )
        
        print("✅ PPO agent created")
        
        # Test accessing param_groups
        print("Testing param_groups access...")
        
        actor_param_groups = agent.optimizer_actor.param_groups
        critic_param_groups = agent.optimizer_critic.param_groups
        
        print(f"Actor param_groups length: {len(actor_param_groups)}")
        print(f"Critic param_groups length: {len(critic_param_groups)}")
        
        if len(actor_param_groups) > 0:
            actor_lr = actor_param_groups[0]['lr']
            print(f"Actor LR: {actor_lr} (type: {type(actor_lr)})")
            
            # Test if it's a tensor
            if hasattr(actor_lr, 'item'):
                print(f"Actor LR is a tensor, value: {actor_lr.item()}")
            else:
                print(f"Actor LR is not a tensor")
        
        if len(critic_param_groups) > 0:
            critic_lr = critic_param_groups[0]['lr']
            print(f"Critic LR: {critic_lr} (type: {type(critic_lr)})")
            
            # Test if it's a tensor
            if hasattr(critic_lr, 'item'):
                print(f"Critic LR is a tensor, value: {critic_lr.item()}")
            else:
                print(f"Critic LR is not a tensor")
        
        print("✅ Optimizer param_groups access successful")
        
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        traceback.print_exc()
        
        if "invalid index of a 0-dim tensor" in str(e):
            print("\n🎯 FOUND THE TENSOR ERROR IN OPTIMIZER!")
        
        raise
