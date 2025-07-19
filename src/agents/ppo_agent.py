import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Tuple, List

from src.agents.base_agent import BaseAgent
from src.models.lstm_model import LSTMModel

class PPOAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 64,
                 lr_actor: float = 0.0003, lr_critic: float = 0.001, gamma: float = 0.99,
                 epsilon_clip: float = 0.2, k_epochs: int = 10):
        
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs

        # Actor network (policy)
        self.actor = LSTMModel(input_dim=observation_dim, hidden_dim=hidden_dim, output_dim=action_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic network (value function)
        self.critic = LSTMModel(input_dim=observation_dim, hidden_dim=hidden_dim, output_dim=1) # Output is a single value
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.mse_loss = nn.MSELoss()

        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []

    def select_action(self, observation: np.ndarray) -> int:
        state = torch.FloatTensor(observation).unsqueeze(0) # Add batch dimension
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        # Clear previous experiences if not already cleared
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

        # Populate experience buffer from provided experiences
        for state, action, reward, next_state, done in experiences:
            self.states.append(torch.FloatTensor(state).unsqueeze(0))
            self.actions.append(torch.tensor([action]))
            # log_prob needs to be calculated from the current policy, not stored from old policy
            # This is a simplification for now, will be properly handled in PPO update
            self.rewards.append(reward)
            self.dones.append(done)

        # Convert lists to tensors
        old_states = torch.cat(self.states).detach()
        old_actions = torch.cat(self.actions).detach()
        old_log_probs = torch.cat(self.log_probs).detach()
        rewards = torch.FloatTensor(self.rewards)
        dones = torch.FloatTensor(self.dones)

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Optimize policy for K epochs
        for _ in range(self.k_epochs):
            # Evaluate old actions and values
            values = self.critic(old_states).squeeze(1)
            action_probs = self.actor(old_states)
            dist = Categorical(logits=action_probs)
            new_log_probs = dist.log_prob(old_actions.squeeze(1))

            # Calculate advantages
            advantages = discounted_rewards - values.detach()

            # PPO ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            loss_actor = -torch.min(surrogate1, surrogate2).mean()

            # Value loss
            loss_critic = self.mse_loss(values, discounted_rewards)

            # Update actor
            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

        # Clear experience buffer after learning
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()

    def save_model(self, path: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
