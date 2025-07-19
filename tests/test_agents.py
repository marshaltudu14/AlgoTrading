import pytest
import torch
from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.trend_agent import TrendAgent
from src.agents.mean_reversion_agent import MeanReversionAgent
from src.agents.volatility_agent import VolatilityAgent
from src.agents.consolidation_agent import ConsolidationAgent
from src.agents.moe_agent import GatingNetwork, MoEAgent
import abc
import numpy as np
from typing import Tuple, List

class ConcreteAgent(BaseAgent):
    def select_action(self, observation: np.ndarray) -> int:
        return 0

    def learn(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        pass

    def adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        return self

    def save_model(self, path: str) -> None:
        pass

    def load_model(self, path: str) -> None:
        pass

def test_base_agent_abstract_methods():
    with pytest.raises(TypeError):
        # This should raise a TypeError because not all abstract methods are implemented
        class IncompleteAgent(BaseAgent):
            def select_action(self, observation: np.ndarray) -> int:
                return 0

        IncompleteAgent()

    # This should not raise an error because all abstract methods are implemented
    agent = ConcreteAgent()
    assert isinstance(agent, BaseAgent)

def test_ppo_agent_initialization():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.99
    epsilon_clip = 0.2
    k_epochs = 3

    agent = PPOAgent(observation_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, epsilon_clip, k_epochs)

    assert isinstance(agent, PPOAgent)
    assert isinstance(agent, BaseAgent)
    assert agent.actor is not None
    assert agent.critic is not None
    assert agent.policy_old is not None

def test_ppo_agent_select_action():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.99
    epsilon_clip = 0.2
    k_epochs = 3

    agent = PPOAgent(observation_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, epsilon_clip, k_epochs)
    
    # Create a dummy observation
    observation = np.random.rand(observation_dim).astype(np.float32)
    
    action = agent.select_action(observation)
    
    assert isinstance(action, int)
    assert 0 <= action < action_dim

def test_ppo_agent_learn_placeholder():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    lr_actor = 0.001
    lr_critic = 0.001
    gamma = 0.99
    epsilon_clip = 0.2
    k_epochs = 3

    agent = PPOAgent(observation_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, epsilon_clip, k_epochs)
    
    # Create dummy experiences
    experiences = [
        (np.random.rand(observation_dim).astype(np.float32), 0, 0.1, np.random.rand(observation_dim).astype(np.float32), False),
        (np.random.rand(observation_dim).astype(np.float32), 1, -0.5, np.random.rand(observation_dim).astype(np.float32), True)
    ]
    
    # The learn method is a placeholder, so we just check if it runs without error
    try:
        agent.learn(experiences)
    except Exception as e:
        pytest.fail(f"learn method raised an unexpected exception: {e}")

@pytest.mark.parametrize(
    "AgentClass", [TrendAgent, MeanReversionAgent, VolatilityAgent, ConsolidationAgent]
)
def test_specialized_agent_initialization(AgentClass):
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64

    agent = AgentClass(observation_dim, action_dim, hidden_dim)

    assert isinstance(agent, AgentClass)
    assert isinstance(agent, BaseAgent)
    assert agent.actor is not None
    assert agent.critic is not None

@pytest.mark.parametrize(
    "AgentClass", [TrendAgent, MeanReversionAgent, VolatilityAgent, ConsolidationAgent]
)
def test_specialized_agent_select_action(AgentClass):
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64

    agent = AgentClass(observation_dim, action_dim, hidden_dim)
    observation = np.random.rand(observation_dim).astype(np.float32)

    action = agent.select_action(observation)

    assert isinstance(action, int)
    assert 0 <= action < action_dim

@pytest.mark.parametrize(
    "AgentClass", [TrendAgent, MeanReversionAgent, VolatilityAgent, ConsolidationAgent]
)
def test_specialized_agent_learn_placeholder(AgentClass):
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64

    agent = AgentClass(observation_dim, action_dim, hidden_dim)
    experiences = [
        (np.random.rand(observation_dim).astype(np.float32), 0, 0.1, np.random.rand(observation_dim).astype(np.float32), False),
        (np.random.rand(observation_dim).astype(np.float32), 1, -0.5, np.random.rand(observation_dim).astype(np.float32), True)
    ]

    try:
        agent.learn(experiences)
    except Exception as e:
        pytest.fail(f"learn method raised an unexpected exception: {e}")

def test_gating_network_output_shape():
    input_dim = 20
    num_experts = 4
    hidden_dim = 10

    gating_network = GatingNetwork(input_dim, num_experts, hidden_dim)
    
    # Create a dummy input tensor
    batch_size = 2
    market_features = torch.randn(batch_size, input_dim)
    
    output = gating_network(market_features)
    
    assert output.shape == (batch_size, num_experts)
    # Check if the output sums to approximately 1 along the expert dimension (due to softmax)
    assert torch.allclose(output.sum(dim=-1), torch.ones(batch_size))

def test_moe_agent_initialization():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    expert_configs = {
        "TrendAgent": {},
        "MeanReversionAgent": {},
        "VolatilityAgent": {},
        "ConsolidationAgent": {}
    }

    moe_agent = MoEAgent(observation_dim, action_dim, hidden_dim, expert_configs)

    assert isinstance(moe_agent, MoEAgent)
    assert isinstance(moe_agent, BaseAgent)
    assert moe_agent.gating_network is not None
    assert len(moe_agent.experts) == len(expert_configs)
    for expert in moe_agent.experts:
        assert isinstance(expert, BaseAgent)

def test_moe_agent_select_action():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    expert_configs = {
        "TrendAgent": {},
        "MeanReversionAgent": {},
        "VolatilityAgent": {},
        "ConsolidationAgent": {}
    }

    moe_agent = MoEAgent(observation_dim, action_dim, hidden_dim, expert_configs)
    observation = np.random.rand(observation_dim).astype(np.float32)

    action = moe_agent.select_action(observation)

    assert isinstance(action, int)
    assert 0 <= action < action_dim

def test_moe_agent_learn_placeholder():
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    expert_configs = {
        "TrendAgent": {},
        "MeanReversionAgent": {},
        "VolatilityAgent": {},
        "ConsolidationAgent": {}
    }

    moe_agent = MoEAgent(observation_dim, action_dim, hidden_dim, expert_configs)
    experiences = [
        (np.random.rand(observation_dim).astype(np.float32), 0, 0.1, np.random.rand(observation_dim).astype(np.float32), False),
        (np.random.rand(observation_dim).astype(np.float32), 1, -0.5, np.random.rand(observation_dim).astype(np.float32), True)
    ]

    try:
        moe_agent.learn(experiences)
    except Exception as e:
        pytest.fail(f"learn method raised an unexpected exception: {e}")

@pytest.mark.parametrize(
    "AgentClass", [PPOAgent, TrendAgent, MeanReversionAgent, VolatilityAgent, ConsolidationAgent, MoEAgent]
)
def test_agent_adapt_method(AgentClass):
    observation_dim = 10
    action_dim = 5
    hidden_dim = 64
    num_gradient_steps = 1

    if AgentClass == PPOAgent:
        agent = AgentClass(observation_dim, action_dim, hidden_dim, 0.001, 0.001, 0.99, 0.2, 3)
    elif AgentClass == MoEAgent:
        expert_configs = {
            "TrendAgent": {},
            "MeanReversionAgent": {},
            "VolatilityAgent": {},
            "ConsolidationAgent": {}
        }
        agent = AgentClass(observation_dim, action_dim, hidden_dim, expert_configs)
    else:
        agent = AgentClass(observation_dim, action_dim, hidden_dim)

    observation = np.random.rand(observation_dim).astype(np.float32)
    action = 0
    reward = 1.0
    next_observation = np.random.rand(observation_dim).astype(np.float32)
    done = False

    adapted_agent = agent.adapt(observation, action, reward, next_observation, done, num_gradient_steps)

    assert isinstance(adapted_agent, BaseAgent)
    # Further assertions could be added here to check if parameters are indeed adapted