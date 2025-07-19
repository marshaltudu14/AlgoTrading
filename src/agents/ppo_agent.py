import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List

from src.agents.base_agent import BaseAgent
from src.models.lstm_model import LSTMModel, ActorLSTMModel

class PPOAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int,
                 lr_actor: float, lr_critic: float, gamma: float, epsilon_clip: float, k_epochs: int):
        super(PPOAgent, self).__init__()

        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor = ActorLSTMModel(observation_dim, hidden_dim, action_dim)
        self.critic = LSTMModel(observation_dim, hidden_dim, 1)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = ActorLSTMModel(observation_dim, hidden_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, observation: np.ndarray) -> int:
        state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
        action_probs = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        # This learn method will be used for the outer loop meta-update in MAML
        # For inner loop adaptation, the 'adapt' method will be used.
        pass

    def adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        # Create a temporary copy of the agent's policy and value network parameters
        adapted_actor = ActorLSTMModel(self.observation_dim, self.hidden_dim, self.action_dim)
        adapted_actor.load_state_dict(self.actor.state_dict())
        adapted_critic = LSTMModel(self.observation_dim, self.hidden_dim, 1)
        adapted_critic.load_state_dict(self.critic.state_dict())

        adapted_optimizer_actor = optim.Adam(adapted_actor.parameters(), lr=self.optimizer_actor.param_groups[0]['lr'])
        adapted_optimizer_critic = optim.Adam(adapted_critic.parameters(), lr=self.optimizer_critic.param_groups[0]['lr'])

        # Perform num_gradient_steps of a standard RL update on these temporary parameters
        for _ in range(num_gradient_steps):
            # For simplicity, we'll use a single experience for adaptation here.
            # In a real MAML implementation, this would be a batch of experiences from the task.
            state, action, reward, next_state, done = observation, action, reward, next_observation, done

            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            action_tensor = torch.LongTensor([action])
            reward_tensor = torch.FloatTensor([reward])
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
            done_tensor = torch.FloatTensor([done])

            # Calculate advantage and target value (simplified for placeholder)
            value = adapted_critic(state_tensor)
            next_value = adapted_critic(next_state_tensor)
            target_value = reward_tensor + self.gamma * next_value * (1 - done_tensor)
            advantage = target_value - value

            # Actor loss
            old_action_probs = self.policy_old(state_tensor).detach()
            old_dist = Categorical(old_action_probs)
            old_log_prob = old_dist.log_prob(action_tensor)

            action_probs = adapted_actor(state_tensor)
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action_tensor)

            ratio = torch.exp(log_prob - old_log_prob)
            surr1 = ratio * advantage.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantage.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = self.MseLoss(value, target_value.detach())

            # Update adapted networks
            adapted_optimizer_actor.zero_grad()
            actor_loss.backward()
            adapted_optimizer_actor.step()

            adapted_optimizer_critic.zero_grad()
            critic_loss.backward()
            adapted_optimizer_critic.step()

        # Return a new BaseAgent instance with the adapted parameters
        # For simplicity, we'll return a new PPOAgent instance with adapted weights
        adapted_agent = PPOAgent(self.observation_dim, self.action_dim, self.hidden_dim, self.optimizer_actor.param_groups[0]['lr'], self.optimizer_critic.param_groups[0]['lr'], self.gamma, self.epsilon_clip, self.k_epochs)
        adapted_agent.actor.load_state_dict(adapted_actor.state_dict())
        adapted_agent.critic.load_state_dict(adapted_critic.state_dict())
        adapted_agent.policy_old.load_state_dict(adapted_actor.state_dict())
        return adapted_agent

    def save_model(self, path: str) -> None:
        torch.save(self.policy_old.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.policy_old.load_state_dict(torch.load(path))
        self.actor.load_state_dict(torch.load(path))