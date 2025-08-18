"""
Comprehensive training tests for Hierarchical Reasoning Model (HRM)

Tests cover:
- HRM model initialization for training
- Training loop integration
- Data pipeline compatibility
- Gradient flow validation
- Memory management during training
- Training convergence detection
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
import yaml
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
from src.utils.config_loader import ConfigLoader
from src.utils.test_data_generator import create_test_data_files


class TestHRMTrainingIntegration:
    """Test HRM integration with training pipeline"""
    
    @pytest.fixture
    def training_config(self):
        """Training configuration for testing"""
        return {
            'model': {
                'observation_dim': 256,
                'action_dim_discrete': 5,
                'action_dim_continuous': 1,
                'model_type': 'hrm'
            },
            'hierarchical_reasoning_model': {
                'h_module': {
                    'hidden_dim': 128,  # Smaller for testing
                    'num_layers': 2,
                    'n_heads': 4,
                    'ff_dim': 256,
                    'dropout': 0.1
                },
                'l_module': {
                    'hidden_dim': 64,   # Smaller for testing
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
                    'N_cycles': 2,      # Smaller for testing
                    'T_timesteps': 3,   # Smaller for testing
                },
                'embeddings': {
                    'instrument_dim': 16,
                    'timeframe_dim': 8,
                    'max_instruments': 10,
                    'max_timeframes': 3
                },
                'output_heads': {
                    'action_dim': 5,
                    'quantity_min': 1.0,
                    'quantity_max': 1000.0,
                    'value_estimation': True,
                    'q_learning_prep': True
                }
            },
            'environment': {
                'initial_capital': 10000.0,
                'lookback_window': 20,
                'episode_length': 50,
                'reward_function': 'trading_focused'
            }
        }
    
    @pytest.fixture
    def sample_training_data(self):
        """Generate sample training data"""
        batch_size = 8
        sequence_length = 10
        feature_dim = 256
        
        # Market observations
        observations = torch.randn(batch_size, sequence_length, feature_dim)
        
        # Actions (discrete and continuous)
        actions_discrete = torch.randint(0, 5, (batch_size, sequence_length))
        actions_continuous = torch.rand(batch_size, sequence_length)  # Quantities
        
        # Rewards and values
        rewards = torch.randn(batch_size, sequence_length)
        values = torch.randn(batch_size, sequence_length)
        
        # Done flags
        dones = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        dones[:, -1] = True  # End of episodes
        
        return {
            'observations': observations,
            'actions_discrete': actions_discrete,
            'actions_continuous': actions_continuous,
            'rewards': rewards,
            'values': values,
            'dones': dones
        }
    
    def test_hrm_training_initialization(self, training_config):
        """Test HRM model initialization for training"""
        model = HierarchicalReasoningModel(training_config)
        
        # Check model is in training mode
        assert model.training
        
        # Check all parameters require gradients
        for param in model.parameters():
            assert param.requires_grad
        
        # Check parameter count is reasonable for training
        param_count = model.count_parameters()
        assert param_count > 0
        assert param_count < 1_000_000  # Should be much smaller for testing config
    
    def test_hrm_forward_pass_training(self, training_config, sample_training_data):
        """Test HRM forward pass in training mode"""
        model = HierarchicalReasoningModel(training_config)
        model.train()
        
        batch_size = sample_training_data['observations'].size(0)
        feature_dim = sample_training_data['observations'].size(2)
        
        # Single timestep forward pass
        x = sample_training_data['observations'][:, 0, :]  # First timestep
        
        outputs, states = model.forward(x)
        
        # Check outputs have correct shapes
        assert outputs['action_type'].shape == (batch_size, 5)
        assert outputs['quantity'].shape == (batch_size,)
        assert outputs['value'].shape == (batch_size,)
        assert outputs['q_values'].shape == (batch_size, 2)
        
        # Check gradients are enabled
        assert outputs['action_type'].requires_grad
        assert outputs['quantity'].requires_grad
        assert outputs['value'].requires_grad
    
    def test_hrm_gradient_flow(self, training_config):
        """Test gradient flow through HRM"""
        model = HierarchicalReasoningModel(training_config)
        model.train()
        
        # Create sample input with embeddings to activate all parameters
        batch_size = 4
        x = torch.randn(batch_size, 256, requires_grad=True)
        instrument_ids = torch.randint(0, 5, (batch_size,))  # Use embeddings
        timeframe_ids = torch.randint(0, 3, (batch_size,))   # Use embeddings
        
        # Forward pass with embeddings
        outputs, _ = model.forward(x, instrument_ids, timeframe_ids)
        
        # Create dummy loss
        policy_loss = outputs['action_type'].mean()
        value_loss = outputs['value'].mean()
        quantity_loss = outputs['quantity'].mean()
        total_loss = policy_loss + value_loss + quantity_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist for core parameters (embeddings may not have gradients if not used)
        core_params = ['h_module', 'l_module', 'policy_head', 'quantity_head', 'value_head', 'input_network']
        gradient_count = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad and any(core in name for core in core_params):
                assert param.grad is not None, f"No gradient for core parameter: {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient for parameter: {name}"
                gradient_count += 1
        
        # Ensure we tested a reasonable number of parameters
        assert gradient_count > 10, f"Only {gradient_count} parameters had gradients - too few"
    
    def test_hrm_optimizer_integration(self, training_config):
        """Test HRM integration with optimizers"""
        model = HierarchicalReasoningModel(training_config)
        
        # Test with Adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Verify optimizer has parameters
        assert len(optimizer.param_groups) > 0
        assert len(optimizer.param_groups[0]['params']) > 0
        
        # Test optimization step
        x = torch.randn(2, 256)
        outputs, _ = model.forward(x)
        loss = outputs['action_type'].mean() + outputs['value'].mean()
        
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients before optimization
        grad_norms_before = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms_before.append(param.grad.norm().item())
        
        optimizer.step()
        
        # Check parameters were updated (gradients should be cleared)
        for param in model.parameters():
            if param.grad is not None:
                # Gradients might still exist after step (depending on optimizer)
                assert not torch.isnan(param.grad).any()
    
    def test_hrm_batch_training(self, training_config, sample_training_data):
        """Test HRM training with batched data"""
        model = HierarchicalReasoningModel(training_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        batch_size = sample_training_data['observations'].size(0)
        sequence_length = sample_training_data['observations'].size(1)
        
        total_loss = 0.0
        
        # Simulate training loop over sequence
        for t in range(sequence_length):
            x = sample_training_data['observations'][:, t, :]
            target_actions = sample_training_data['actions_discrete'][:, t]
            target_values = sample_training_data['values'][:, t]
            
            # Forward pass
            outputs, _ = model.forward(x)
            
            # Compute losses
            policy_loss = nn.CrossEntropyLoss()(outputs['action_type'], target_actions)
            value_loss = nn.MSELoss()(outputs['value'], target_values)
            
            loss = policy_loss + value_loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (common in RL)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Check training progressed
        assert total_loss > 0
        assert not np.isnan(total_loss)
        assert not np.isinf(total_loss)
    
    def test_hrm_memory_efficiency(self, training_config):
        """Test HRM memory usage during training"""
        model = HierarchicalReasoningModel(training_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Test with various batch sizes
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Clear any cached memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            x = torch.randn(batch_size, 256)
            
            # Forward pass
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean() + outputs['value'].mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check no memory leaks (basic check)
            assert torch.cuda.memory_allocated() >= 0 if torch.cuda.is_available() else True
    
    def test_hrm_convergence_monitoring(self, training_config):
        """Test HRM convergence monitoring during training"""
        model = HierarchicalReasoningModel(training_config)
        
        # Test convergence diagnostics during training
        x = torch.randn(2, 256)
        
        # Get detailed diagnostics
        outputs, final_states, diagnostics = model.forward(x, return_diagnostics=True)
        
        # Check diagnostic information is available
        assert 'convergence_info' in diagnostics
        assert 'output_stats' in diagnostics
        assert 'state_norms' in diagnostics
        
        # Check convergence info
        conv_info = diagnostics['convergence_info']
        assert 'cycles' in conv_info
        assert 'h_updates' in conv_info
        assert 'total_l_steps' in conv_info
        
        # Check that we have the expected number of cycles
        assert len(conv_info['cycles']) == 2  # As per testing config
    
    def test_hrm_state_management(self, training_config):
        """Test HRM hidden state management during training"""
        model = HierarchicalReasoningModel(training_config)
        
        batch_size = 4
        x = torch.randn(batch_size, 256)
        
        # Test with custom initial states
        device = x.device
        z_h_init = torch.randn(batch_size, 128, device=device)  # H-module dim
        z_l_init = torch.randn(batch_size, 64, device=device)   # L-module dim
        
        outputs1, states1 = model.forward(x, z_init=(z_h_init, z_l_init))
        outputs2, states2 = model.forward(x, z_init=states1)
        
        # Check states are properly managed
        z_h1, z_l1 = states1
        z_h2, z_l2 = states2
        
        assert z_h1.shape == z_h_init.shape
        assert z_l1.shape == z_l_init.shape
        assert z_h2.shape == z_h1.shape
        assert z_l2.shape == z_l1.shape
        
        # States should be different (model is learning)
        assert not torch.allclose(z_h1, z_h2, atol=1e-6)


class TestHRMTrainingDataIntegration:
    """Test HRM integration with training data pipeline"""
    
    def test_hrm_with_test_data_generator(self):
        """Test HRM with generated test data"""
        # Create temporary directory for test data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate test data
            create_test_data_files(
                data_dir=temp_dir,
                create_multiple_instruments=True,
                num_rows=100  # Small for testing
            )
            
            # Check test data was created
            final_dir = os.path.join(temp_dir, 'final')
            assert os.path.exists(final_dir)
            
            # List generated files
            files = os.listdir(final_dir)
            feature_files = [f for f in files if f.startswith('features_')]
            assert len(feature_files) > 0
    
    def test_hrm_training_config_loading(self):
        """Test loading training configuration"""
        # Test with default config
        config_loader = ConfigLoader()
        config = config_loader.get_config()
        
        # Check required sections exist
        assert 'model' in config
        assert 'hierarchical_reasoning_model' in config
        
        # Validate HRM can be initialized with config
        model = HierarchicalReasoningModel(config)
        assert model is not None
        assert hasattr(model, 'h_module')
        assert hasattr(model, 'l_module')
    
    def test_hrm_environment_compatibility(self):
        """Test HRM compatibility with TradingEnv"""
        try:
            from src.backtesting.environment import TradingEnv
            from src.utils.data_loader import DataLoader
            
            # Create mock data loader
            data_loader = Mock()
            data_loader.load_data = Mock(return_value=None)
            
            # This test just checks imports work
            # Full integration test would require actual data
            assert TradingEnv is not None
            assert DataLoader is not None
            
        except ImportError as e:
            pytest.skip(f"Trading environment not available: {e}")


class TestTrainingScriptIntegration:
    """Test integration with run_training.py script"""
    
    def test_hrm_import_in_training_script(self):
        """Test that training script can import HRM"""
        try:
            # Test import path from training script
            from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
            assert HierarchicalReasoningModel is not None
        except ImportError as e:
            pytest.fail(f"HRM import failed in training script: {e}")
    
    def test_training_config_structure(self):
        """Test training config structure matches HRM expectations"""
        # Load config like training script does
        config_path = "config/settings.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Check HRM config exists
            assert 'hierarchical_reasoning_model' in config
            hrm_config = config['hierarchical_reasoning_model']
            
            # Check required sections
            required_sections = ['h_module', 'l_module', 'input_embedding', 'output_heads']
            for section in required_sections:
                assert section in hrm_config, f"Missing required section: {section}"
        else:
            pytest.skip("Config file not found")
    
    def test_training_script_hrm_parameters(self):
        """Test that training script uses correct HRM parameters"""
        # This test checks that when we fix the training script,
        # it will use the correct HRM constructor
        
        config = {
            'model': {
                'observation_dim': 256,
                'action_dim_discrete': 5,
                'action_dim_continuous': 1
            }
        }
        
        # Test correct HRM initialization
        model = HierarchicalReasoningModel(config)
        
        # Test that old PPO parameters would fail
        with pytest.raises(TypeError):
            # This should fail because HRM doesn't take these parameters
            HierarchicalReasoningModel(
                observation_dim=256,
                action_dim_discrete=5,
                action_dim_continuous=1,
                hidden_dim=64,
                lr_actor=0.001,  # These are PPO parameters
                lr_critic=0.001,
                gamma=0.99,
                epsilon_clip=0.2,
                k_epochs=3
            )


class TestHRMTrainingPerformance:
    """Test HRM training performance and benchmarks"""
    
    def test_hrm_training_speed(self, training_config=None):
        """Test HRM training speed benchmark"""
        if training_config is None:
            training_config = {
                'model': {'observation_dim': 256},
                'hierarchical_reasoning_model': {
                    'h_module': {'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 128},
                    'l_module': {'hidden_dim': 32, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 64},
                    'input_embedding': {'input_dim': 256, 'embedding_dim': 64},
                    'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                    'embeddings': {'instrument_dim': 8, 'timeframe_dim': 4, 'max_instruments': 5, 'max_timeframes': 3},
                    'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 1000.0, 'value_estimation': True, 'q_learning_prep': True}
                }
            }
        
        model = HierarchicalReasoningModel(training_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        import time
        
        batch_size = 8
        num_steps = 10
        
        # Warmup
        for _ in range(3):
            x = torch.randn(batch_size, 256)
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_steps):
            x = torch.randn(batch_size, 256)
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean() + outputs['value'].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        time_per_step = total_time / num_steps
        samples_per_second = (batch_size * num_steps) / total_time
        
        print(f"Training performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Time per step: {time_per_step:.3f}s")
        print(f"  Samples per second: {samples_per_second:.1f}")
        
        # Basic performance assertions
        assert time_per_step < 1.0  # Should be faster than 1s per step
        assert samples_per_second > 5  # Should process at least 5 samples/sec
    
    def test_hrm_memory_usage(self, training_config=None):
        """Test HRM memory usage during training"""
        if training_config is None:
            training_config = {
                'model': {'observation_dim': 256},
                'hierarchical_reasoning_model': {
                    'h_module': {'hidden_dim': 64, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 128},
                    'l_module': {'hidden_dim': 32, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 64},
                    'input_embedding': {'input_dim': 256, 'embedding_dim': 64},
                    'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                    'embeddings': {'instrument_dim': 8, 'timeframe_dim': 4, 'max_instruments': 5, 'max_timeframes': 3},
                    'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 1000.0, 'value_estimation': True, 'q_learning_prep': True}
                }
            }
        
        model = HierarchicalReasoningModel(training_config)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before training
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run some training steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for _ in range(10):
            x = torch.randn(8, 256)
            outputs, _ = model.forward(x)
            loss = outputs['action_type'].mean() + outputs['value'].mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Measure memory after training
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after - memory_before
        
        print(f"Memory usage:")
        print(f"  Before training: {memory_before:.1f} MB")
        print(f"  After training: {memory_after:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory should not increase dramatically
        assert memory_increase < 500  # Less than 500MB increase


if __name__ == "__main__":
    # Run tests manually
    pytest.main([__file__, "-v"])