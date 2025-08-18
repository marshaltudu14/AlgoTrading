"""
Comprehensive unit tests for Hierarchical Reasoning Model (HRM)

Tests cover all HRM components including:
- Dual-module architecture (H-module and L-module)
- Hierarchical convergence mechanism
- Embedding systems
- Output heads
- Error handling and integration
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

# Import HRM components
from src.models.hierarchical_reasoning_model import (
    HierarchicalReasoningModel,
    InputEmbeddingNetwork,
    HighLevelModule,
    LowLevelModule,
    PolicyHead,
    QuantityHead,
    ValueHead,
    QHead,
    InstrumentEmbedding,
    TimeframeEmbedding,
    RMSNorm,
    GLU,
    TransformerBlock
)


class TestRMSNorm:
    """Test Root Mean Square Layer Normalization"""
    
    def test_rmsnorm_initialization(self):
        """Test RMSNorm initialization"""
        norm = RMSNorm(256)
        assert norm.scale.shape == (256,)
        assert torch.allclose(norm.scale, torch.ones(256))
    
    def test_rmsnorm_forward(self):
        """Test RMSNorm forward pass"""
        batch_size, seq_len, dim = 2, 10, 256
        norm = RMSNorm(dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = norm(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestGLU:
    """Test Gated Linear Unit"""
    
    def test_glu_initialization(self):
        """Test GLU initialization"""
        glu = GLU(256, 512)
        assert glu.gate.in_features == 256
        assert glu.gate.out_features == 512
        assert glu.linear.in_features == 256
        assert glu.linear.out_features == 512
    
    def test_glu_forward(self):
        """Test GLU forward pass"""
        batch_size, input_dim, hidden_dim = 2, 256, 512
        glu = GLU(input_dim, hidden_dim)
        x = torch.randn(batch_size, input_dim)
        
        output = glu(x)
        assert output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(output).any()


class TestTransformerBlock:
    """Test Enhanced Transformer Block"""
    
    def test_transformer_block_initialization(self):
        """Test TransformerBlock initialization"""
        dim, n_heads, ff_dim = 256, 8, 1024
        block = TransformerBlock(dim, n_heads, ff_dim)
        
        assert block.dim == dim
        assert block.n_heads == n_heads
        assert block.head_dim == dim // n_heads
    
    def test_transformer_block_forward(self):
        """Test TransformerBlock forward pass"""
        batch_size, seq_len, dim = 2, 10, 256
        n_heads, ff_dim = 8, 1024
        
        block = TransformerBlock(dim, n_heads, ff_dim)
        x = torch.randn(batch_size, seq_len, dim)
        
        output = block(x)
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestEmbeddings:
    """Test Embedding Components"""
    
    def test_instrument_embedding(self):
        """Test InstrumentEmbedding"""
        vocab_size, embedding_dim = 1000, 64
        emb = InstrumentEmbedding(vocab_size, embedding_dim)
        
        batch_size = 4
        instrument_ids = torch.randint(0, vocab_size, (batch_size,))
        
        output = emb(instrument_ids)
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.isnan(output).any()
    
    def test_timeframe_embedding(self):
        """Test TimeframeEmbedding"""
        vocab_size, embedding_dim = 10, 32
        emb = TimeframeEmbedding(vocab_size, embedding_dim)
        
        batch_size = 4
        timeframe_ids = torch.randint(0, vocab_size, (batch_size,))
        
        output = emb(timeframe_ids)
        assert output.shape == (batch_size, embedding_dim)
        assert not torch.isnan(output).any()


class TestInputEmbeddingNetwork:
    """Test Input Embedding Network"""
    
    def test_initialization(self):
        """Test InputEmbeddingNetwork initialization"""
        config = {'input_dim': 256, 'embedding_dim': 512, 'dropout': 0.1}
        network = InputEmbeddingNetwork(config)
        
        assert network.input_dim == 256
        assert network.embedding_dim == 512
    
    def test_forward(self):
        """Test InputEmbeddingNetwork forward pass"""
        config = {'input_dim': 256, 'embedding_dim': 512, 'dropout': 0.1}
        network = InputEmbeddingNetwork(config)
        
        batch_size = 4
        x = torch.randn(batch_size, 256)
        
        output = network(x)
        assert output.shape == (batch_size, 512)
        assert not torch.isnan(output).any()


class TestOutputHeads:
    """Test Output Head Components"""
    
    def test_policy_head(self):
        """Test PolicyHead"""
        input_dim, action_dim = 512, 5
        head = PolicyHead(input_dim, action_dim)
        
        batch_size = 4
        z_h = torch.randn(batch_size, input_dim)
        
        output = head(z_h)
        assert output.shape == (batch_size, action_dim)
        assert not torch.isnan(output).any()
    
    def test_quantity_head(self):
        """Test QuantityHead"""
        input_dim = 512
        quantity_min, quantity_max = 1.0, 10000.0
        head = QuantityHead(input_dim, quantity_min, quantity_max)
        
        batch_size = 4
        z_h = torch.randn(batch_size, input_dim)
        
        output = head(z_h)
        assert output.shape == (batch_size,)
        assert torch.all(output >= quantity_min)
        assert torch.all(output <= quantity_max)
        assert not torch.isnan(output).any()
    
    def test_value_head(self):
        """Test ValueHead"""
        input_dim = 512
        head = ValueHead(input_dim)
        
        batch_size = 4
        z_h = torch.randn(batch_size, input_dim)
        
        output = head(z_h)
        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
    
    def test_q_head(self):
        """Test QHead"""
        input_dim = 512
        head = QHead(input_dim)
        
        batch_size = 4
        z_h = torch.randn(batch_size, input_dim)
        
        output = head(z_h)
        assert output.shape == (batch_size, 2)  # halt, continue
        assert torch.all(output >= 0) and torch.all(output <= 1)  # sigmoid output
        assert not torch.isnan(output).any()


class TestModules:
    """Test H-module and L-module"""
    
    def test_high_level_module(self):
        """Test HighLevelModule (H-module)"""
        config = {
            'hidden_dim': 512,
            'num_layers': 2,
            'n_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1
        }
        h_module = HighLevelModule(config)
        
        batch_size = 4
        z_h_prev = torch.randn(batch_size, 512)
        z_l_converged = torch.randn(batch_size, 512)
        
        z_h_new = h_module(z_h_prev, z_l_converged)
        assert z_h_new.shape == (batch_size, 512)
        assert not torch.isnan(z_h_new).any()
        assert not torch.isinf(z_h_new).any()
    
    def test_low_level_module(self):
        """Test LowLevelModule (L-module)"""
        config = {
            'hidden_dim': 256,
            'num_layers': 2,
            'n_heads': 8,
            'ff_dim': 512,
            'dropout': 0.1
        }
        l_module = LowLevelModule(config)
        
        batch_size = 4
        z_l_prev = torch.randn(batch_size, 256)
        z_h = torch.randn(batch_size, 256)
        x_embedded = torch.randn(batch_size, 256)
        
        z_l_new = l_module(z_l_prev, z_h, x_embedded)
        assert z_l_new.shape == (batch_size, 256)
        assert not torch.isnan(z_l_new).any()
        assert not torch.isinf(z_l_new).any()


class TestHierarchicalReasoningModel:
    """Comprehensive tests for the complete HRM model"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'model': {
                'observation_dim': 256,
                'action_dim_discrete': 5
            },
            'hierarchical_reasoning_model': {
                'h_module': {
                    'hidden_dim': 256,
                    'num_layers': 2,
                    'n_heads': 4,
                    'ff_dim': 512,
                    'dropout': 0.1
                },
                'l_module': {
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'n_heads': 4,
                    'ff_dim': 256,
                    'dropout': 0.1
                },
                'input_embedding': {
                    'input_dim': 256,
                    'embedding_dim': 256,
                    'dropout': 0.1
                },
                'hierarchical': {
                    'N_cycles': 2,
                    'T_timesteps': 3,
                    'convergence_threshold': 1e-6,
                    'max_convergence_steps': 100
                },
                'embeddings': {
                    'instrument_dim': 32,
                    'timeframe_dim': 16,
                    'max_instruments': 100,
                    'max_timeframes': 5
                },
                'output_heads': {
                    'action_dim': 5,
                    'quantity_min': 1.0,
                    'quantity_max': 1000.0,
                    'value_estimation': True,
                    'q_learning_prep': True
                }
            }
        }
    
    def test_hrm_initialization(self, sample_config):
        """Test HRM initialization"""
        model = HierarchicalReasoningModel(sample_config)
        
        assert model.N == 2  # N_cycles
        assert model.T == 3  # T_timesteps
        assert hasattr(model, 'h_module')
        assert hasattr(model, 'l_module')
        assert hasattr(model, 'policy_head')
        assert hasattr(model, 'quantity_head')
        assert hasattr(model, 'value_head')
        assert hasattr(model, 'q_head')
    
    def test_hrm_forward_pass(self, sample_config):
        """Test HRM forward pass"""
        model = HierarchicalReasoningModel(sample_config)
        
        batch_size = 2
        x = torch.randn(batch_size, 256)
        
        outputs, final_states = model.forward(x)
        
        # Check outputs
        assert 'action_type' in outputs
        assert 'quantity' in outputs
        assert 'value' in outputs
        assert 'q_values' in outputs
        
        assert outputs['action_type'].shape == (batch_size, 5)
        assert outputs['quantity'].shape == (batch_size,)
        assert outputs['value'].shape == (batch_size,)
        assert outputs['q_values'].shape == (batch_size, 2)
        
        # Check final states
        z_h, z_l = final_states
        assert z_h.shape == (batch_size, 256)
        assert z_l.shape == (batch_size, 128)
        
        # Check for NaN/Inf values
        for key, value in outputs.items():
            assert not torch.isnan(value).any(), f"NaN found in {key}"
            assert not torch.isinf(value).any(), f"Inf found in {key}"
    
    def test_hrm_forward_with_embeddings(self, sample_config):
        """Test HRM forward pass with instrument and timeframe embeddings"""
        model = HierarchicalReasoningModel(sample_config)
        
        batch_size = 2
        x = torch.randn(batch_size, 256)
        instrument_ids = torch.randint(0, 100, (batch_size,))
        timeframe_ids = torch.randint(0, 5, (batch_size,))
        
        outputs, final_states = model.forward(x, instrument_ids, timeframe_ids)
        
        assert 'action_type' in outputs
        assert 'quantity' in outputs
        assert outputs['action_type'].shape == (batch_size, 5)
        assert outputs['quantity'].shape == (batch_size,)
    
    def test_hrm_act_method(self, sample_config):
        """Test HRM act method"""
        model = HierarchicalReasoningModel(sample_config)
        
        observation = torch.randn(256)
        action_type, quantity = model.act(observation)
        
        assert isinstance(action_type, int)
        assert 0 <= action_type < 5
        assert isinstance(quantity, float)
        assert 1.0 <= quantity <= 1000.0
    
    def test_hrm_act_with_embeddings(self, sample_config):
        """Test HRM act method with embeddings"""
        model = HierarchicalReasoningModel(sample_config)
        
        observation = torch.randn(256)
        action_type, quantity = model.act(observation, instrument_id=5, timeframe_id=2)
        
        assert isinstance(action_type, int)
        assert 0 <= action_type < 5
        assert isinstance(quantity, float)
        assert 1.0 <= quantity <= 1000.0
    
    def test_hrm_parameter_count(self, sample_config):
        """Test HRM parameter counting"""
        model = HierarchicalReasoningModel(sample_config)
        
        param_count = model.count_parameters()
        assert isinstance(param_count, int)
        assert param_count > 0
        
        # Verify it's within reasonable bounds for 27M parameter target
        assert param_count < 50_000_000  # Should be well under 50M
    
    def test_hrm_state_initialization(self, sample_config):
        """Test HRM state initialization"""
        model = HierarchicalReasoningModel(sample_config)
        
        batch_size = 4
        device = torch.device('cpu')
        
        z_h, z_l = model.initialize_states(batch_size, device)
        
        assert z_h.shape == (batch_size, 256)
        assert z_l.shape == (batch_size, 128)
        assert z_h.device == device
        assert z_l.device == device
        
        # Check truncated normal initialization (values should be within [-2, 2])
        assert torch.all(z_h >= -2.0) and torch.all(z_h <= 2.0)
        assert torch.all(z_l >= -2.0) and torch.all(z_l <= 2.0)
    
    def test_hrm_l_module_reset(self, sample_config):
        """Test L-module reset mechanism"""
        model = HierarchicalReasoningModel(sample_config)
        
        batch_size = 2
        z_l = torch.randn(batch_size, 128)
        z_l_original = z_l.clone()
        
        z_l_reset = model.reset_l_module(z_l)
        
        assert z_l_reset.shape == z_l.shape
        # Should be different from original (some reset occurred)
        assert not torch.allclose(z_l_reset, z_l_original)
        assert not torch.isnan(z_l_reset).any()
    
    def test_hrm_save_load_model(self, sample_config):
        """Test HRM model saving and loading"""
        model = HierarchicalReasoningModel(sample_config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save model
            model.save_model(temp_path)
            assert os.path.exists(temp_path)
            
            # Create new model and load
            model2 = HierarchicalReasoningModel(sample_config)
            model2.load_model(temp_path)
            
            # Test that loaded model works (produces valid outputs)
            x = torch.randn(1, 256)
            outputs, _ = model2.forward(x)
            
            # Verify output structure and validity
            assert 'action_type' in outputs
            assert 'quantity' in outputs
            assert outputs['action_type'].shape == (1, 5)
            assert outputs['quantity'].shape == (1,)
            assert not torch.isnan(outputs['action_type']).any()
            assert not torch.isnan(outputs['quantity']).any()
            
            # Test act method works
            action_type, quantity = model2.act(x.squeeze())
            assert isinstance(action_type, int)
            assert 0 <= action_type < 5
            assert isinstance(quantity, float)
            assert quantity >= 1.0
        
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestHRMErrorHandling:
    """Test HRM error handling and fallback mechanisms"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'model': {'observation_dim': 256, 'action_dim_discrete': 5},
            'hierarchical_reasoning_model': {
                'h_module': {'hidden_dim': 256, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 512, 'dropout': 0.1},
                'l_module': {'hidden_dim': 128, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 256, 'dropout': 0.1},
                'input_embedding': {'input_dim': 256, 'embedding_dim': 256, 'dropout': 0.1},
                'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                'embeddings': {'instrument_dim': 32, 'timeframe_dim': 16, 'max_instruments': 100, 'max_timeframes': 5},
                'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 1000.0, 'value_estimation': True, 'q_learning_prep': True}
            }
        }
    
    def test_hrm_invalid_input_handling(self, sample_config):
        """Test HRM handling of invalid inputs"""
        model = HierarchicalReasoningModel(sample_config)
        
        # Test with None input
        action_type, quantity = model.act(None)
        assert action_type == 2  # HOLD action
        assert quantity == 1.0  # minimum quantity
        
        # Test with empty tensor
        empty_tensor = torch.empty(0)
        action_type, quantity = model.act(empty_tensor)
        assert action_type == 2
        assert quantity == 1.0
        
        # Test with wrong-shaped input
        wrong_shape = torch.randn(256, 256, 256)  # 3D instead of 1D/2D
        action_type, quantity = model.act(wrong_shape)
        assert isinstance(action_type, int)
        assert isinstance(quantity, float)
    
    def test_hrm_invalid_embedding_ids(self, sample_config):
        """Test HRM handling of invalid embedding IDs"""
        model = HierarchicalReasoningModel(sample_config)
        
        observation = torch.randn(256)
        
        # Test with out-of-bounds instrument ID
        action_type, quantity = model.act(observation, instrument_id=999999)
        assert isinstance(action_type, int)
        assert isinstance(quantity, float)
        
        # Test with negative timeframe ID
        action_type, quantity = model.act(observation, timeframe_id=-5)
        assert isinstance(action_type, int)
        assert isinstance(quantity, float)
    
    def test_hrm_dimension_mismatch_handling(self, sample_config):
        """Test HRM handling of input dimension mismatches"""
        model = HierarchicalReasoningModel(sample_config)
        
        # Test with smaller input dimension
        small_input = torch.randn(128)  # Expected 256
        outputs, _ = model.forward(small_input.unsqueeze(0))
        assert 'action_type' in outputs
        assert 'quantity' in outputs
        
        # Test with larger input dimension
        large_input = torch.randn(512)  # Expected 256
        outputs, _ = model.forward(large_input.unsqueeze(0))
        assert 'action_type' in outputs
        assert 'quantity' in outputs


class TestHRMDiagnostics:
    """Test HRM diagnostic and debugging functionality"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'model': {'observation_dim': 256, 'action_dim_discrete': 5},
            'hierarchical_reasoning_model': {
                'h_module': {'hidden_dim': 256, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 512, 'dropout': 0.1},
                'l_module': {'hidden_dim': 128, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 256, 'dropout': 0.1},
                'input_embedding': {'input_dim': 256, 'embedding_dim': 256, 'dropout': 0.1},
                'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                'embeddings': {'instrument_dim': 32, 'timeframe_dim': 16, 'max_instruments': 100, 'max_timeframes': 5},
                'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 1000.0, 'value_estimation': True, 'q_learning_prep': True}
            }
        }
    
    def test_convergence_diagnostics(self, sample_config):
        """Test convergence diagnostics functionality"""
        model = HierarchicalReasoningModel(sample_config)
        
        x = torch.randn(1, 256)
        diagnostics = model.get_convergence_diagnostics(x)
        
        # Check diagnostic structure
        assert 'cycles' in diagnostics
        assert 'h_module_states' in diagnostics
        assert 'l_module_states' in diagnostics
        assert 'convergence_metrics' in diagnostics
        assert 'output_statistics' in diagnostics
        assert 'parameter_statistics' in diagnostics
        
        # Check cycles information
        assert len(diagnostics['cycles']) == 2  # N_cycles = 2
        
        # Check state tracking
        assert len(diagnostics['h_module_states']) >= 2  # Initial + updates
        assert len(diagnostics['l_module_states']) >= 6  # Initial + 2 cycles * 3 timesteps
        
        # Check metrics
        metrics = diagnostics['convergence_metrics']
        assert 'cycles_converged' in metrics
        assert 'total_cycles' in metrics
        assert 'final_h_norm' in metrics
        assert 'final_l_norm' in metrics
        
        # Check parameter statistics
        params = diagnostics['parameter_statistics']
        assert 'total_parameters' in params
        assert 'h_module_parameters' in params
        assert 'l_module_parameters' in params
    
    def test_reasoning_pattern_analysis(self, sample_config):
        """Test reasoning pattern analysis"""
        model = HierarchicalReasoningModel(sample_config)
        
        # Create batch of market data
        batch_size = 20
        market_data_batch = torch.randn(batch_size, 256)
        
        analysis = model.analyze_reasoning_patterns(market_data_batch, num_samples=5)
        
        # Check analysis structure
        assert 'decision_consistency' in analysis
        assert 'reasoning_depth_usage' in analysis
        
        # Check decision consistency metrics
        consistency = analysis['decision_consistency']
        assert 'mean_confidence' in consistency
        assert 'confidence_std' in consistency
        assert 'high_confidence_ratio' in consistency
        
        # Check reasoning depth metrics
        depth = analysis['reasoning_depth_usage']
        assert 'mean_convergence_rate' in depth
        assert 'convergence_rate_std' in depth
        assert 'full_convergence_ratio' in depth


class TestHRMIntegration:
    """Test HRM integration with existing systems"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'model': {'observation_dim': 256, 'action_dim_discrete': 5},
            'hierarchical_reasoning_model': {
                'h_module': {'hidden_dim': 256, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 512, 'dropout': 0.1},
                'l_module': {'hidden_dim': 128, 'num_layers': 2, 'n_heads': 4, 'ff_dim': 256, 'dropout': 0.1},
                'input_embedding': {'input_dim': 256, 'embedding_dim': 256, 'dropout': 0.1},
                'hierarchical': {'N_cycles': 2, 'T_timesteps': 3},
                'embeddings': {'instrument_dim': 32, 'timeframe_dim': 16, 'max_instruments': 100, 'max_timeframes': 5},
                'output_heads': {'action_dim': 5, 'quantity_min': 1.0, 'quantity_max': 1000.0, 'value_estimation': True, 'q_learning_prep': True}
            }
        }
    
    def test_hrm_output_compatibility(self, sample_config):
        """Test HRM output compatibility with trading service expectations"""
        model = HierarchicalReasoningModel(sample_config)
        
        observation = torch.randn(256)
        action_type, quantity = model.act(observation)
        
        # Check action space compatibility
        assert isinstance(action_type, int)
        assert 0 <= action_type <= 4  # 5 discrete actions
        
        # Check quantity compatibility
        assert isinstance(quantity, float)
        assert quantity >= 1.0  # minimum quantity
        assert quantity <= 1000.0  # maximum quantity
    
    def test_hrm_batch_processing(self, sample_config):
        """Test HRM batch processing capability"""
        model = HierarchicalReasoningModel(sample_config)
        
        batch_sizes = [1, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 256)
            outputs, final_states = model.forward(x)
            
            # Check output shapes
            assert outputs['action_type'].shape == (batch_size, 5)
            assert outputs['quantity'].shape == (batch_size,)
            
            # Check state shapes
            z_h, z_l = final_states
            assert z_h.shape == (batch_size, 256)
            assert z_l.shape == (batch_size, 128)
    
    def test_hrm_device_compatibility(self, sample_config):
        """Test HRM device compatibility (CPU/GPU)"""
        model = HierarchicalReasoningModel(sample_config)
        
        # Test CPU
        x_cpu = torch.randn(2, 256)
        outputs_cpu, _ = model.forward(x_cpu)
        assert outputs_cpu['action_type'].device.type == 'cpu'
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = torch.randn(2, 256).cuda()
            outputs_gpu, _ = model_gpu.forward(x_gpu)
            assert outputs_gpu['action_type'].device.type == 'cuda'


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])