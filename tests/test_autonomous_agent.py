"""
Unit tests for AutonomousAgent.

Tests the functionality of the autonomous agent including
integration with World Model and External Memory.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from src.agents.autonomous_agent import AutonomousAgent


class TestAutonomousAgent:
    """Test cases for AutonomousAgent class."""
    
    def test_autonomous_agent_initialization(self):
        """Test that AutonomousAgent initializes correctly."""
        observation_dim = 50
        action_dim = 3
        hidden_dim = 64
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            memory_size=100,
            memory_embedding_dim=32
        )
        
        assert agent.observation_dim == observation_dim
        assert agent.action_dim == action_dim
        assert agent.hidden_dim == hidden_dim
        assert agent.memory_embedding_dim == 32
        
        # Check that components are initialized
        assert agent.world_model is not None
        assert agent.external_memory is not None
        assert agent.state_embedder is not None
        assert agent.memory_aggregator is not None
        assert agent.market_classifier is not None
        assert agent.pattern_recognizer is not None
        
        # Check that world model has correct input dimension
        # obs_dim + memory_embedding_dim + regime_embedding_dim (8) + pattern_embedding_dim (16)
        expected_world_model_input = observation_dim + 32 + 8 + 16
        assert agent.world_model.input_dim == expected_world_model_input
        
        # Check memory configuration
        assert agent.external_memory.max_memories == 100
        assert agent.external_memory.embedding_dim == 32
    
    def test_autonomous_agent_act_basic(self):
        """Test basic action selection functionality."""
        observation_dim = 20
        action_dim = 4
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=50,
            memory_embedding_dim=16
        )
        
        # Test with numpy array
        market_state = np.random.randn(observation_dim)
        action = agent.act(market_state)
        
        assert isinstance(action, int)
        assert 0 <= action < action_dim
        
        # Test with torch tensor
        market_state_tensor = torch.randn(observation_dim)
        action_tensor = agent.act(market_state_tensor)
        
        assert isinstance(action_tensor, int)
        assert 0 <= action_tensor < action_dim
        
        # Test with batch dimension
        market_state_batch = torch.randn(1, observation_dim)
        action_batch = agent.act(market_state_batch)
        
        assert isinstance(action_batch, int)
        assert 0 <= action_batch < action_dim
    
    def test_autonomous_agent_act_with_memory(self):
        """Test action selection with stored memories."""
        observation_dim = 15
        action_dim = 3
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=10,
            memory_embedding_dim=16
        )
        
        # Store some experiences first
        for i in range(5):
            market_state = np.random.randn(observation_dim)
            action = i % action_dim
            reward = np.random.randn()
            next_state = np.random.randn(observation_dim)
            
            agent.learn_from_experience(
                market_state=market_state,
                action=action,
                reward=reward,
                next_market_state=next_state,
                done=False,
                importance=0.5 + i * 0.1
            )
        
        # Now test action selection with memories
        test_state = np.random.randn(observation_dim)
        action = agent.act(test_state)
        
        assert isinstance(action, int)
        assert 0 <= action < action_dim
        
        # Check that memories were used
        stats = agent.get_agent_statistics()
        assert stats['total_memories_used'] > 0
    
    def test_autonomous_agent_think_and_predict(self):
        """Test detailed thinking and prediction functionality."""
        observation_dim = 25
        action_dim = 3
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=64,
            prediction_horizon=3,
            memory_size=20,
            memory_embedding_dim=32
        )
        
        # Store some experiences
        for i in range(3):
            market_state = np.random.randn(observation_dim)
            agent.learn_from_experience(
                market_state=market_state,
                action=i % action_dim,
                reward=np.random.randn(),
                next_market_state=np.random.randn(observation_dim),
                done=False
            )
        
        # Test thinking and prediction
        test_state = np.random.randn(observation_dim)
        result = agent.think_and_predict(test_state)
        
        # Check output structure
        assert isinstance(result, dict)
        assert 'recommended_action' in result
        assert 'action_probabilities' in result
        assert 'predicted_market_state' in result
        assert 'predicted_volatility' in result
        assert 'market_regime_probs' in result
        assert 'value_estimate' in result
        assert 'confidence' in result
        assert 'retrieved_memories' in result
        assert 'memory_similarities' in result
        
        # Check output shapes and types
        assert isinstance(result['recommended_action'], int)
        assert 0 <= result['recommended_action'] < action_dim
        
        assert result['action_probabilities'].shape == (1, action_dim)
        assert result['predicted_market_state'].shape == (1, 3, 5)  # (batch, horizon, features)
        assert result['predicted_volatility'].shape == (1, 3)  # (batch, horizon)
        assert result['value_estimate'].shape == (1, 1)
        assert result['confidence'].shape == (1, 1)
        
        # Check that probabilities sum to 1
        assert np.allclose(result['action_probabilities'].sum(), 1.0, atol=1e-6)
        
        # Check that confidence is between 0 and 1
        assert 0 <= result['confidence'][0, 0] <= 1
    
    def test_autonomous_agent_learn_from_experience(self):
        """Test learning from experience functionality."""
        observation_dim = 10
        action_dim = 2
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=5,
            memory_embedding_dim=16
        )
        
        # Test storing experiences
        experiences = []
        for i in range(3):
            market_state = np.random.randn(observation_dim)
            action = i % action_dim
            reward = np.random.randn()
            next_state = np.random.randn(observation_dim)
            done = (i == 2)  # Last experience ends episode
            
            agent.learn_from_experience(
                market_state=market_state,
                action=action,
                reward=reward,
                next_market_state=next_state,
                done=done,
                importance=0.8
            )
            
            experiences.append((market_state, action, reward, next_state, done))
        
        # Check that memories were stored
        memory_stats = agent.external_memory.get_memory_statistics()
        assert memory_stats['total_memories'] == 3
        assert memory_stats['total_stored'] == 3
        
        # Test with torch tensors
        market_state_tensor = torch.randn(observation_dim)
        agent.learn_from_experience(
            market_state=market_state_tensor,
            action=1,
            reward=1.5,
            next_market_state=torch.randn(observation_dim),
            done=False
        )
        
        memory_stats = agent.external_memory.get_memory_statistics()
        assert memory_stats['total_memories'] == 4
    
    def test_autonomous_agent_memory_capacity(self):
        """Test memory capacity limits."""
        observation_dim = 8
        action_dim = 2
        memory_size = 3  # Small memory for testing
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=memory_size,
            memory_embedding_dim=16
        )
        
        # Store more experiences than memory capacity
        for i in range(5):
            market_state = np.random.randn(observation_dim)
            agent.learn_from_experience(
                market_state=market_state,
                action=i % action_dim,
                reward=i * 0.1,
                next_market_state=np.random.randn(observation_dim),
                done=False,
                importance=0.5 + i * 0.1  # Increasing importance
            )
        
        # Should only keep memory_size memories
        memory_stats = agent.external_memory.get_memory_statistics()
        assert memory_stats['total_memories'] == memory_size
        assert memory_stats['total_stored'] == 5  # Total stored should track all
    
    def test_autonomous_agent_statistics(self):
        """Test agent statistics functionality."""
        observation_dim = 12
        action_dim = 3
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=10,
            memory_embedding_dim=16
        )
        
        # Initial statistics
        stats = agent.get_agent_statistics()
        assert stats['total_actions'] == 0
        assert stats['total_memories_used'] == 0
        assert stats['avg_memories_per_action'] == 0
        assert 'memory_stats' in stats
        assert 'world_model_params' in stats
        assert 'state_embedder_params' in stats
        assert 'memory_aggregator_params' in stats
        
        # Perform some actions and store experiences
        for i in range(3):
            market_state = np.random.randn(observation_dim)
            
            # Store experience
            agent.learn_from_experience(
                market_state=market_state,
                action=i % action_dim,
                reward=np.random.randn(),
                next_market_state=np.random.randn(observation_dim),
                done=False
            )
            
            # Take action
            agent.act(market_state)
        
        # Check updated statistics
        stats = agent.get_agent_statistics()
        assert stats['total_actions'] == 3
        assert stats['memory_stats']['total_memories'] == 3
        assert stats['world_model_params'] > 0
        assert stats['state_embedder_params'] > 0
        assert stats['memory_aggregator_params'] > 0
    
    def test_autonomous_agent_save_and_load(self):
        """Test saving and loading agent functionality."""
        observation_dim = 15
        action_dim = 2
        
        # Create and configure agent
        agent1 = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=5,
            memory_embedding_dim=16
        )
        
        # Store some experiences and take actions
        for i in range(3):
            market_state = np.random.randn(observation_dim)
            agent1.learn_from_experience(
                market_state=market_state,
                action=i % action_dim,
                reward=i * 0.5,
                next_market_state=np.random.randn(observation_dim),
                done=False
            )
            agent1.act(market_state)
        
        # Save agent
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            agent1.save_agent(tmp_path)
            
            # Create new agent and load
            agent2 = AutonomousAgent(
                observation_dim=observation_dim,
                action_dim=action_dim,
                hidden_dim=32,
                memory_size=5,
                memory_embedding_dim=16
            )
            
            agent2.load_agent(tmp_path)
            
            # Check that memories were loaded
            stats1 = agent1.get_agent_statistics()
            stats2 = agent2.get_agent_statistics()
            
            assert stats2['memory_stats']['total_memories'] == stats1['memory_stats']['total_memories']
            
            # Test that loaded agent can still act
            test_state = np.random.randn(observation_dim)
            action = agent2.act(test_state)
            assert isinstance(action, int)
            assert 0 <= action < action_dim
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            memory_path = tmp_path.replace('.pt', '_memory.pkl')
            if os.path.exists(memory_path):
                os.unlink(memory_path)
    
    def test_autonomous_agent_base_agent_compatibility(self):
        """Test compatibility with BaseAgent interface."""
        observation_dim = 10
        action_dim = 3
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=10,
            memory_embedding_dim=16
        )
        
        # Test BaseAgent interface method
        observation = np.random.randn(observation_dim)
        action = agent.select_action(observation)
        
        assert isinstance(action, int)
        assert 0 <= action < action_dim
    
    def test_autonomous_agent_temperature_exploration(self):
        """Test temperature-based exploration."""
        observation_dim = 8
        action_dim = 3
        
        agent = AutonomousAgent(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_dim=32,
            memory_size=5,
            memory_embedding_dim=16
        )
        
        market_state = np.random.randn(observation_dim)
        
        # Test with different temperatures
        agent.temperature = 0.1  # Low temperature (more deterministic)
        actions_low_temp = [agent.act(market_state) for _ in range(10)]
        
        agent.temperature = 2.0  # High temperature (more random)
        actions_high_temp = [agent.act(market_state) for _ in range(10)]
        
        # High temperature should generally produce more diverse actions
        # (though this is probabilistic, so we just check it doesn't crash)
        assert all(isinstance(a, int) and 0 <= a < action_dim for a in actions_low_temp)
        assert all(isinstance(a, int) and 0 <= a < action_dim for a in actions_high_temp)


if __name__ == "__main__":
    pytest.main([__file__])
