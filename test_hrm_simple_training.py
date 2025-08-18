#!/usr/bin/env python3
"""
Simple HRM training test to verify the model works with training loops.
This is a minimal test to catch any basic integration issues.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
from src.utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_market_data(batch_size=8, num_steps=20, feature_dim=256):
    """Create synthetic market data for testing"""
    # Generate realistic-looking market data
    data = []
    
    for _ in range(num_steps):
        # Base market features (OHLCV + technical indicators)
        market_step = torch.randn(batch_size, feature_dim)
        
        # Normalize to reasonable ranges
        market_step = torch.clamp(market_step, -3, 3)
        
        data.append(market_step)
    
    return torch.stack(data, dim=1)  # [batch_size, num_steps, feature_dim]

def create_sample_training_targets(batch_size=8, num_steps=20):
    """Create sample training targets"""
    targets = {
        'actions': torch.randint(0, 5, (batch_size, num_steps)),  # Discrete actions
        'quantities': torch.rand(batch_size, num_steps) * 100 + 1,  # Quantities 1-101
        'rewards': torch.randn(batch_size, num_steps) * 0.1,  # Small rewards
        'values': torch.randn(batch_size, num_steps) * 10,  # Value estimates
        'dones': torch.zeros(batch_size, num_steps, dtype=torch.bool)
    }
    
    # Mark last step as done for each episode
    targets['dones'][:, -1] = True
    
    return targets

def test_hrm_basic_training():
    """Test basic HRM training functionality"""
    logger.info("🧪 Starting HRM Basic Training Test")
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # Ensure model config exists
        if 'model' not in config:
            config['model'] = {}
        config['model'].update({
            'observation_dim': 256,
            'action_dim_discrete': 5,
            'action_dim_continuous': 1
        })
        
        # Create smaller config for testing
        if 'hierarchical_reasoning_model' not in config:
            config['hierarchical_reasoning_model'] = {}
        
        # Override with smaller testing config
        config['hierarchical_reasoning_model'].update({
            'h_module': {
                'hidden_dim': 128,
                'num_layers': 2,
                'n_heads': 4,
                'ff_dim': 256,
                'dropout': 0.1
            },
            'l_module': {
                'hidden_dim': 64,
                'num_layers': 2,
                'n_heads': 4,
                'ff_dim': 128,
                'dropout': 0.1
            },
            'input_embedding': {
                'input_dim': 256,
                'embedding_dim': 128,
                'dropout': 0.1
            },
            'hierarchical': {
                'N_cycles': 2,
                'T_timesteps': 3
            },
            'embeddings': {
                'instrument_dim': 16,
                'timeframe_dim': 8,
                'max_instruments': 10,
                'max_timeframes': 5
            },
            'output_heads': {
                'action_dim': 5,
                'quantity_min': 1.0,
                'quantity_max': 100.0,
                'value_estimation': True,
                'q_learning_prep': True
            }
        })
        
        logger.info("✅ Configuration loaded successfully")
        
        # Initialize HRM model
        model = HierarchicalReasoningModel(config)
        model.train()
        
        param_count = model.count_parameters()
        logger.info(f"✅ HRM model initialized with {param_count:,} parameters")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        logger.info("✅ Optimizer created")
        
        # Create sample data
        batch_size = 4
        num_steps = 10
        
        market_data = create_sample_market_data(batch_size, num_steps)
        targets = create_sample_training_targets(batch_size, num_steps)
        
        logger.info(f"✅ Sample data created: {market_data.shape}")
        
        # Test training loop
        total_loss = 0.0
        losses = []
        
        logger.info("🚀 Starting training loop...")
        
        for step in range(num_steps):
            # Get data for this step
            x = market_data[:, step, :]  # [batch_size, feature_dim]
            target_actions = targets['actions'][:, step]
            target_values = targets['values'][:, step]
            
            # Forward pass
            outputs, states = model.forward(x)
            
            # Compute losses
            policy_loss = nn.CrossEntropyLoss()(outputs['action_type'], target_actions)
            value_loss = nn.MSELoss()(outputs['value'], target_values)
            quantity_loss = nn.MSELoss()(outputs['quantity'], targets['quantities'][:, step])
            
            # Combined loss
            loss = policy_loss + value_loss + 0.1 * quantity_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            
            # Track metrics
            step_loss = loss.item()
            total_loss += step_loss
            losses.append(step_loss)
            
            if step % 5 == 0:
                logger.info(f"  Step {step:2d}: Loss = {step_loss:.4f}")
        
        avg_loss = total_loss / num_steps
        logger.info(f"✅ Training completed. Average loss: {avg_loss:.4f}")
        
        # Test inference mode
        model.eval()
        with torch.no_grad():
            test_x = market_data[0, 0, :].unsqueeze(0)  # Single sample
            test_outputs, _ = model.forward(test_x)
            
            action_probs = torch.softmax(test_outputs['action_type'], dim=-1)
            best_action = torch.argmax(action_probs, dim=-1).item()
            quantity = test_outputs['quantity'].item()
            value = test_outputs['value'].item()
            
            logger.info(f"✅ Inference test:")
            logger.info(f"    Best action: {best_action}")
            logger.info(f"    Quantity: {quantity:.2f}")
            logger.info(f"    Value estimate: {value:.2f}")
        
        # Test save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_model(temp_path)
            logger.info(f"✅ Model saved to {temp_path}")
            
            # Load model
            model2 = HierarchicalReasoningModel(config)
            model2.load_model(temp_path)
            logger.info("✅ Model loaded successfully")
            
            # Test loaded model
            model2.eval()
            with torch.no_grad():
                test_outputs2, _ = model2.forward(test_x)
                logger.info("✅ Loaded model inference works")
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Test convergence diagnostics
        logger.info("🔍 Testing convergence diagnostics...")
        diagnostics = model.get_convergence_diagnostics(test_x)
        
        logger.info(f"    Cycles converged: {diagnostics['convergence_metrics']['cycles_converged']}/2")
        logger.info(f"    Final H-norm: {diagnostics['convergence_metrics']['final_h_norm']:.6f}")
        logger.info(f"    Action entropy: {diagnostics['output_statistics']['action_entropy']:.6f}")
        
        logger.info("🎉 ALL TESTS PASSED! HRM training integration is working correctly.")
        return True
        
    except Exception as e:
        logger.error(f"❌ HRM training test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_hrm_performance_benchmark():
    """Quick performance benchmark"""
    logger.info("🏃 Running HRM performance benchmark...")
    
    try:
        config = {
            'model': {
                'observation_dim': 256,
                'action_dim_discrete': 5
            },
            'hierarchical_reasoning_model': {
                'h_module': {'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 128},
                'l_module': {'hidden_dim': 32, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 64},
                'input_embedding': {'input_dim': 256, 'embedding_dim': 64},
                'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                'embeddings': {'instrument_dim': 8, 'timeframe_dim': 4, 'max_instruments': 5, 'max_timeframes': 3},
                'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 100.0, 'value_estimation': True, 'q_learning_prep': True}
            }
        }
        
        model = HierarchicalReasoningModel(config)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        import time
        
        # Warm up
        for _ in range(3):
            x = torch.randn(4, 256)
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Benchmark
        batch_size = 8
        num_steps = 20
        
        start_time = time.time()
        
        for _ in range(num_steps):
            x = torch.randn(batch_size, 256)
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean() + outputs['value'].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        total_time = end_time - start_time
        time_per_step = total_time / num_steps
        samples_per_second = (batch_size * num_steps) / total_time
        
        logger.info(f"⚡ Performance Results:")
        logger.info(f"    Total time: {total_time:.3f}s")
        logger.info(f"    Time per step: {time_per_step:.3f}s")
        logger.info(f"    Samples/sec: {samples_per_second:.1f}")
        
        if time_per_step < 0.5 and samples_per_second > 20:
            logger.info("✅ Performance benchmark PASSED")
            return True
        else:
            logger.warning("⚠️  Performance below expected thresholds")
            return False
            
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Run all HRM training tests"""
    logger.info("=" * 60)
    logger.info("🧠 HRM TRAINING INTEGRATION TEST SUITE")
    logger.info("=" * 60)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Basic training functionality
    if test_hrm_basic_training():
        tests_passed += 1
    
    # Test 2: Performance benchmark
    if test_hrm_performance_benchmark():
        tests_passed += 1
    
    logger.info("=" * 60)
    logger.info(f"📊 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED! HRM is ready for production training.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())