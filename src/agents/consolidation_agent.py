from src.agents.base_agent import BaseAgent
from src.models.lstm_model import LSTMModel, ActorLSTMModel
import numpy as np
import torch
from typing import Tuple, List

class ConsolidationAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        self.observation_dim = observation_dim
        self.actor = ActorLSTMModel(observation_dim, hidden_dim, action_dim)
        self.critic = LSTMModel(observation_dim, hidden_dim, 1)

    def select_action(self, observation: np.ndarray) -> int:
        state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        # Placeholder for learning logic
        pass

    def adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        # For simplicity, we'll return a new instance with the same parameters for now.
        # In a real MAML implementation, this would involve creating a temporary, differentiable copy
        # of the agent's parameters and performing gradient steps on them.
        adapted_agent = ConsolidationAgent(self.actor.input_dim, self.actor.linear.out_features, self.actor.hidden_dim)
        adapted_agent.actor.load_state_dict(self.actor.state_dict())
        adapted_agent.critic.load_state_dict(self.critic.state_dict())
        return adapted_agent

    def save_model(self, path: str) -> None:
        torch.save(self.actor.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.actor.load_state_dict(torch.load(path))
