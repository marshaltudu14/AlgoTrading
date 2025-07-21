"""
Unit tests for SelfModificationManager.

Tests the functionality of self-modification logic including performance
evaluation, adaptive changes, and various modification types.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock
from src.reasoning.self_modification import (
    SelfModificationManager, 
    PerformanceMetrics, 
    ModificationConfig, 
    ModificationType,
    check_performance_and_adapt
)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self):
        self.world_model = Mock()
        self.world_model.input_dim = 64
        self.world_model.action_dim = 3
        self.world_model.transformer = Mock()
        self.world_model.transformer.d_model = 128
        
        # Add some parameters for pruning tests
        self.world_model.named_parameters = Mock(return_value=[
            ('layer1.weight', torch.randn(10, 10, requires_grad=True)),
            ('layer2.weight', torch.randn(5, 5, requires_grad=True))
        ])
        
        self.state_embedder = Mock()
        self.state_embedder.named_parameters = Mock(return_value=[
            ('embedding.weight', torch.randn(8, 8, requires_grad=True))
        ])
        
        self.external_memory = Mock()
        self.external_memory.memories = [
            {'reward': 0.5, 'state': 'state1'},
            {'reward': 0.8, 'state': 'state2'},
            {'reward': 0.3, 'state': 'state3'},
            {'reward': 0.9, 'state': 'state4'}
        ]
        
        self.risk_aversion_factor = 1.0
        self.position_sizing_factor = 1.0
        self.optimizer = Mock()
        self.optimizer.param_groups = [{'lr': 1e-3}]


class TestPerformanceMetrics:
    """Test cases for PerformanceMetrics dataclass."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            profit_factor=1.5,
            sharpe_ratio=0.8,
            max_drawdown=-0.1,
            win_rate=0.6,
            total_return=0.15
        )
        
        assert metrics.profit_factor == 1.5
        assert metrics.sharpe_ratio == 0.8
        assert metrics.max_drawdown == -0.1
        assert metrics.win_rate == 0.6
        assert metrics.total_return == 0.15
    
    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics with default values."""
        metrics = PerformanceMetrics()
        
        assert metrics.profit_factor == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.num_trades == 0


class TestModificationConfig:
    """Test cases for ModificationConfig dataclass."""
    
    def test_modification_config_creation(self):
        """Test ModificationConfig creation."""
        config = ModificationConfig(
            min_profit_factor=1.5,
            max_drawdown_threshold=-0.2,
            enable_architecture_search=False
        )
        
        assert config.min_profit_factor == 1.5
        assert config.max_drawdown_threshold == -0.2
        assert config.enable_architecture_search is False
    
    def test_modification_config_defaults(self):
        """Test ModificationConfig with default values."""
        config = ModificationConfig()
        
        assert config.min_profit_factor == 1.2
        assert config.min_sharpe_ratio == 0.5
        assert config.max_drawdown_threshold == -0.15
        assert config.enable_architecture_search is True


class TestSelfModificationManager:
    """Test cases for SelfModificationManager class."""
    
    def test_self_modification_manager_initialization(self):
        """Test SelfModificationManager initialization."""
        config = ModificationConfig()
        nas_controller = Mock()
        
        manager = SelfModificationManager(config=config, nas_controller=nas_controller)
        
        assert manager.config == config
        assert manager.nas_controller == nas_controller
        assert len(manager.modification_history) == 0
        assert len(manager.performance_history) == 0
        assert manager.current_risk_factor == 1.0
        assert manager.current_learning_rate == 1e-3
    
    def test_check_performance_and_adapt_no_modifications(self):
        """Test performance check with good performance (no modifications needed)."""
        manager = SelfModificationManager()
        agent = MockAgent()
        
        # Good performance metrics
        metrics = PerformanceMetrics(
            profit_factor=2.0,
            sharpe_ratio=1.5,
            max_drawdown=-0.05,
            win_rate=0.7,
            consecutive_losses=1
        )
        
        modifications = manager.check_performance_and_adapt(agent, metrics, episode_number=5)
        
        assert ModificationType.NO_MODIFICATION in modifications
        assert len(manager.performance_history) == 1
    
    def test_check_performance_and_adapt_risk_adjustment(self):
        """Test performance check triggering risk adjustment."""
        config = ModificationConfig(max_drawdown_threshold=-0.1)
        manager = SelfModificationManager(config=config)
        agent = MockAgent()
        
        # Poor performance metrics (high drawdown)
        metrics = PerformanceMetrics(
            profit_factor=0.8,
            sharpe_ratio=0.2,
            max_drawdown=-0.2,  # Exceeds threshold
            win_rate=0.4,
            consecutive_losses=6  # Exceeds threshold
        )
        
        modifications = manager.check_performance_and_adapt(agent, metrics, episode_number=5)
        
        assert ModificationType.RISK_ADJUSTMENT in modifications
        assert manager.current_risk_factor < 1.0  # Risk should be reduced
        assert agent.risk_aversion_factor > 1.0  # Risk aversion should increase
    
    def test_check_performance_and_adapt_architecture_search(self):
        """Test performance check triggering architecture search."""
        nas_controller = Mock()
        nas_controller.initialize_population = Mock()
        nas_controller.evolve_population = Mock()
        nas_controller.get_best_architecture = Mock(return_value=[Mock()])
        nas_controller.population_size = 10
        
        config = ModificationConfig(
            min_profit_factor=1.5,
            min_episodes_before_search=5,
            architecture_search_cooldown=1
        )
        manager = SelfModificationManager(config=config, nas_controller=nas_controller)
        agent = MockAgent()
        
        # Poor performance metrics
        metrics = PerformanceMetrics(
            profit_factor=0.5,  # Below threshold
            sharpe_ratio=0.1,   # Below threshold
            max_drawdown=-0.3
        )
        
        modifications = manager.check_performance_and_adapt(agent, metrics, episode_number=10)
        
        assert ModificationType.ARCHITECTURE_SEARCH in modifications
        nas_controller.initialize_population.assert_called_once()
        nas_controller.evolve_population.assert_called()
        nas_controller.get_best_architecture.assert_called_once()
    
    def test_check_performance_and_adapt_synaptic_pruning(self):
        """Test performance check triggering synaptic pruning."""
        config = ModificationConfig(pruning_threshold=0.1)
        manager = SelfModificationManager(config=config)
        agent = MockAgent()
        
        # Set up agent parameters for pruning
        weight1 = torch.tensor([[0.5, 0.05], [0.8, 0.02]], requires_grad=True)  # Some small weights
        weight2 = torch.tensor([[0.3, 0.7], [0.01, 0.9]], requires_grad=True)
        
        agent.world_model.named_parameters = Mock(return_value=[
            ('layer1.weight', weight1),
            ('layer2.weight', weight2)
        ])
        agent.state_embedder.named_parameters = Mock(return_value=[])
        
        # Poor performance metrics
        metrics = PerformanceMetrics(
            profit_factor=0.5,  # Poor performance
            sharpe_ratio=-0.2
        )
        
        modifications = manager.check_performance_and_adapt(agent, metrics, episode_number=5)
        
        assert ModificationType.SYNAPTIC_PRUNING in modifications
        
        # Check that small weights were pruned (set to zero)
        assert weight1[0, 1].item() == 0.0  # 0.05 should be pruned
        assert weight1[1, 1].item() == 0.0  # 0.02 should be pruned
        assert weight2[1, 0].item() == 0.0  # 0.01 should be pruned
        
        # Check that large weights were preserved
        assert weight1[0, 0].item() == 0.5  # 0.5 should be preserved
        assert weight2[1, 1].item() == 0.9  # 0.9 should be preserved
    
    def test_check_performance_and_adapt_memory_consolidation(self):
        """Test performance check triggering memory consolidation."""
        config = ModificationConfig(memory_consolidation_threshold=3)
        manager = SelfModificationManager(config=config)
        agent = MockAgent()
        
        # Set up agent with many memories
        agent.external_memory.memories = [
            {'reward': 0.1}, {'reward': 0.9}, {'reward': 0.3}, 
            {'reward': 0.7}, {'reward': 0.2}, {'reward': 0.8}
        ]
        
        metrics = PerformanceMetrics(profit_factor=1.0)
        
        modifications = manager.check_performance_and_adapt(agent, metrics, episode_number=5)
        
        assert ModificationType.MEMORY_CONSOLIDATION in modifications
        
        # Check that memory was consolidated (should keep top 50%)
        assert len(agent.external_memory.memories) == 3
        
        # Check that highest reward memories were kept
        rewards = [m['reward'] for m in agent.external_memory.memories]
        assert 0.9 in rewards
        assert 0.8 in rewards
        assert 0.7 in rewards
    
    def test_check_performance_and_adapt_learning_rate_adaptation(self):
        """Test performance check triggering learning rate adaptation."""
        manager = SelfModificationManager()
        agent = MockAgent()
        
        # Add performance history with declining trend
        declining_metrics = [
            PerformanceMetrics(profit_factor=1.5),
            PerformanceMetrics(profit_factor=1.3),
            PerformanceMetrics(profit_factor=1.1),
            PerformanceMetrics(profit_factor=0.9),
            PerformanceMetrics(profit_factor=0.7)
        ]
        
        for i, metrics in enumerate(declining_metrics):
            manager.check_performance_and_adapt(agent, metrics, episode_number=i)
        
        # The last call should trigger learning rate adaptation
        modifications = manager.modification_history[-1]['modifications']
        assert 'learning_rate_adaptation' in modifications
        
        # Learning rate should be reduced due to declining performance
        assert manager.current_learning_rate < 1e-3
    
    def test_performance_metrics_from_dict(self):
        """Test creating PerformanceMetrics from dictionary."""
        manager = SelfModificationManager()
        agent = MockAgent()
        
        metrics_dict = {
            'profit_factor': 1.5,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.1,
            'win_rate': 0.6
        }
        
        modifications = manager.check_performance_and_adapt(agent, metrics_dict, episode_number=5)
        
        assert len(manager.performance_history) == 1
        assert manager.performance_history[0].profit_factor == 1.5
        assert manager.performance_history[0].sharpe_ratio == 0.8
    
    def test_get_modification_statistics(self):
        """Test getting modification statistics."""
        manager = SelfModificationManager()
        agent = MockAgent()
        
        # Trigger some modifications
        poor_metrics = PerformanceMetrics(
            profit_factor=0.5,
            max_drawdown=-0.3,
            consecutive_losses=6
        )
        
        manager.check_performance_and_adapt(agent, poor_metrics, episode_number=5)
        
        stats = manager.get_modification_statistics()
        
        assert 'total_modifications' in stats
        assert 'modification_counts' in stats
        assert 'current_risk_factor' in stats
        assert 'current_learning_rate' in stats
        assert stats['total_modifications'] > 0
    
    def test_reset_modification_state(self):
        """Test resetting modification state."""
        manager = SelfModificationManager()
        agent = MockAgent()
        
        # Add some history
        metrics = PerformanceMetrics(profit_factor=0.5)
        manager.check_performance_and_adapt(agent, metrics, episode_number=5)
        
        assert len(manager.performance_history) > 0
        assert len(manager.modification_history) > 0
        
        # Reset state
        manager.reset_modification_state()
        
        assert len(manager.performance_history) == 0
        assert len(manager.modification_history) == 0
        assert manager.current_risk_factor == 1.0
        assert manager.current_learning_rate == 1e-3
    
    def test_standalone_function(self):
        """Test the standalone check_performance_and_adapt function."""
        agent = MockAgent()
        metrics = PerformanceMetrics(profit_factor=1.5)
        config = ModificationConfig()
        
        modifications = check_performance_and_adapt(
            agent=agent,
            performance_metrics=metrics,
            config=config,
            episode_number=5
        )
        
        assert isinstance(modifications, list)
        assert all(isinstance(mod, ModificationType) for mod in modifications)


if __name__ == "__main__":
    pytest.main([__file__])
