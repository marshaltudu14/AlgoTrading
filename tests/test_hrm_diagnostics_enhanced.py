"""
Enhanced HRM Diagnostics Tests
"""
import torch
import pytest
import numpy as np
from src.models.hierarchical_reasoning_model import HierarchicalReasoningModel
from src.models.hrm_diagnostics import (
    ParticipationRatioAnalyzer, 
    ConvergencePatternAnalyzer, 
    HRMDiagnosticSuite
)

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        'hierarchical_reasoning_model': {
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
                'input_dim': 64,
                'embedding_dim': 128,
                'dropout': 0.1
            },
            'hierarchical': {
                'N_cycles': 2,
                'T_timesteps': 3,
                'convergence_threshold': 1e-6
            },
            'embeddings': {
                'instrument_dim': 16,
                'timeframe_dim': 8,
                'max_instruments': 100,
                'max_timeframes': 5
            },
            'output_heads': {
                'action_dim': 5,
                'quantity_min': 1.0,
                'quantity_max': 1000.0,
                'value_estimation': True,
                'q_learning_enabled': True
            },
            'architecture': {
                'rms_norm_eps': 1e-8,
                'truncated_normal_std': 1.0,
                'truncated_normal_limit': 2.0
            }
        },
        'model': {
            'observation_dim': 64,
            'action_dim_discrete': 5
        }
    }

class TestEnhancedHRMDiagnostics:
    """Test enhanced HRM diagnostics functionality"""
    
    def test_enhanced_convergence_tracking(self, sample_config):
        """Test enhanced convergence tracking in HierarchicalConvergenceEngine"""
        model = HierarchicalReasoningModel(sample_config)
        
        x = torch.randn(2, 64)
        outputs, final_states, diagnostics = model.forward(x, return_diagnostics=True)
        
        # Check that we have enhanced convergence info
        conv_info = diagnostics['convergence_info']
        assert 'residuals' in conv_info
        assert 'l_states' in conv_info
        assert 'h_states' in conv_info
        assert len(conv_info['residuals']) > 0
        
        # Check that we have the expected number of states
        assert len(conv_info['h_states']) == conv_info['h_updates'] + 1
        assert len(conv_info['l_states']) == conv_info['h_updates'] + 1
    
    def test_get_convergence_diagnostics_method(self, sample_config):
        """Test the get_convergence_diagnostics method"""
        model = HierarchicalReasoningModel(sample_config)
        
        x = torch.randn(2, 64)
        conv_diagnostics = model.get_convergence_diagnostics(x)
        
        # Check that we have the expected structure
        assert 'cycles' in conv_diagnostics
        assert 'total_l_steps' in conv_diagnostics
        assert 'h_updates' in conv_diagnostics
        assert 'convergence_metrics' in conv_diagnostics
        
        # Check convergence metrics
        conv_metrics = conv_diagnostics['convergence_metrics']
        assert 'total_cycles' in conv_metrics
        assert 'total_l_steps' in conv_metrics
        assert 'h_updates' in conv_metrics
    
    def test_participation_ratio_analyzer_enhanced(self):
        """Test enhanced Participation Ratio Analyzer"""
        pr_analyzer = ParticipationRatioAnalyzer(min_samples=5)
        
        # Generate some sample trajectories
        for i in range(10):
            z_h = torch.randn(4, 128)  # 4 samples, 128 dim
            z_l = torch.randn(4, 64)   # 4 samples, 64 dim
            pr_analyzer.collect_trajectory(z_h, z_l)
        
        # Analyze dimensionality hierarchy
        pr_results = pr_analyzer.analyze_dimensionality_hierarchy()
        
        # Check that we have the expected results
        assert 'pr_h_module' in pr_results
        assert 'pr_l_module' in pr_results
        assert 'hierarchy_ratio' in pr_results
        assert 'brain_similarity_score' in pr_results
        
        # Reset and check that it works
        pr_analyzer.reset()
        assert len(pr_analyzer.trajectories_h) == 0
        assert len(pr_analyzer.trajectories_l) == 0
    
    def test_convergence_pattern_analyzer_enhanced(self):
        """Test enhanced Convergence Pattern Analyzer"""
        conv_analyzer = ConvergencePatternAnalyzer()
        
        # Generate some sample convergence data
        for cycle in range(3):
            for timestep in range(5):
                z_h_prev = torch.randn(4, 128)
                z_h_curr = torch.randn(4, 128)
                z_l_prev = torch.randn(4, 64)
                z_l_curr = torch.randn(4, 64)
                
                conv_analyzer.track_convergence_step(
                    z_h_prev, z_h_curr, z_l_prev, z_l_curr, cycle, timestep
                )
        
        # Analyze convergence patterns
        conv_results = conv_analyzer.analyze_convergence_patterns()
        
        # Check that we have the expected results
        assert 'residual_analysis' in conv_results
        assert 'pca_analysis' in conv_results
        assert 'hierarchical_pattern' in conv_results
        assert 'stability_metrics' in conv_results
        
        # Reset and check that it works
        conv_analyzer.reset()
        assert len(conv_analyzer.h_residuals) == 0
        assert len(conv_analyzer.l_residuals) == 0
    
    def test_hrm_diagnostic_suite_integration(self, sample_config):
        """Test HRM Diagnostic Suite integration with model"""
        model = HierarchicalReasoningModel(sample_config)
        diagnostic_suite = HRMDiagnosticSuite()
        
        # Check that the model has the required methods
        assert hasattr(model, 'get_convergence_diagnostics')
        
        # Test that we can call the method
        x = torch.randn(2, 64)
        conv_diagnostics = model.get_convergence_diagnostics(x)
        
        # Check structure
        assert 'cycles' in conv_diagnostics
        assert 'total_l_steps' in conv_diagnostics
        assert 'h_updates' in conv_diagnostics