import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict

from src.agents.base_agent import BaseAgent
from src.models.lstm_model import LSTMModel, ActorLSTMModel

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(market_features))
        return F.softmax(self.fc2(x), dim=-1)

class MoEAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int, expert_configs: Dict):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.expert_configs = expert_configs
        print(f"MoEAgent __init__: observation_dim={observation_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}, num_experts={len(expert_configs)}")

        self.gating_network = GatingNetwork(observation_dim, len(expert_configs), hidden_dim)
        self.experts = []
        for expert_type, config in expert_configs.items():
            if expert_type == "TrendAgent":
                from src.agents.trend_agent import TrendAgent
                self.experts.append(TrendAgent(observation_dim, action_dim, hidden_dim))
            elif expert_type == "MeanReversionAgent":
                from src.agents.mean_reversion_agent import MeanReversionAgent
                self.experts.append(MeanReversionAgent(observation_dim, action_dim, hidden_dim))
            elif expert_type == "VolatilityAgent":
                from src.agents.volatility_agent import VolatilityAgent
                self.experts.append(VolatilityAgent(observation_dim, action_dim, hidden_dim))
            elif expert_type == "ConsolidationAgent":
                from src.agents.consolidation_agent import ConsolidationAgent
                self.experts.append(ConsolidationAgent(observation_dim, action_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

    def select_action(self, observation: np.ndarray) -> int:
        # For simplicity, assuming market_features are the entire observation for now
        market_features = torch.FloatTensor(observation).unsqueeze(0)
        expert_weights = self.gating_network(market_features).squeeze(0)

        # Get actions from all experts
        expert_actions = []
        for expert in self.experts:
            expert_actions.append(expert.select_action(observation))
        
        # Combine actions based on weights (e.g., weighted average of one-hot encoded actions)
        # This is a simplified approach. A more robust approach would involve weighted probabilities.
        weighted_actions = torch.zeros(len(expert_actions))
        for i, action in enumerate(expert_actions):
            weighted_actions[i] = expert_weights[i] * action # This is not correct for discrete actions

        # A better way for discrete actions: choose the action of the highest weighted expert
        selected_expert_idx = torch.argmax(expert_weights).item()
        final_action = expert_actions[selected_expert_idx]

        return final_action

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        # Placeholder for learning logic for both gating network and experts
        pass

    def adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        # Create adapted copies of gating network and experts
        print(f"MoEAgent adapt: observation_dim={self.observation_dim}, action_dim={self.action_dim}, hidden_dim={self.hidden_dim}, expert_configs={self.expert_configs}")
        adapted_gating_network = GatingNetwork(self.gating_network.fc1.in_features, self.gating_network.fc2.out_features, self.gating_network.fc1.out_features)
        adapted_gating_network.load_state_dict(self.gating_network.state_dict())

        adapted_experts = []
        for expert in self.experts:
            # Assuming experts also have an adapt method
            adapted_experts.append(expert.adapt(observation, action, reward, next_observation, done, num_gradient_steps))

        # For simplicity, we'll return a new MoEAgent instance with adapted components
        adapted_moe_agent = MoEAgent(self.observation_dim, self.action_dim, self.hidden_dim, self.expert_configs)
        adapted_moe_agent.gating_network.load_state_dict(adapted_gating_network.state_dict())
        adapted_moe_agent.experts = adapted_experts
        return adapted_moe_agent

    def save_model(self, path: str) -> None:
        # Save gating network and all experts
        torch.save({
            'gating_network_state_dict': self.gating_network.state_dict(),
            'experts_state_dicts': [expert.actor.state_dict() for expert in self.experts]
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.gating_network.load_state_dict(checkpoint['gating_network_state_dict'])
        for i, expert_state_dict in enumerate(checkpoint['experts_state_dicts']):
            self.experts[i].actor.load_state_dict(expert_state_dict)