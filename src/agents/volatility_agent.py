from src.agents.base_agent import BaseAgent
from src.models.transformer_models import ActorTransformerModel, CriticTransformerModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Tuple, List

class VolatilityAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim_discrete: int, action_dim_continuous: int, hidden_dim: int,
                 lr_actor: float = 0.001, lr_critic: float = 0.001,
                 gamma: float = 0.99, epsilon_clip: float = 0.2, k_epochs: int = 3):
        self.observation_dim = observation_dim
        self.action_dim_discrete = action_dim_discrete
        self.action_dim_continuous = action_dim_continuous
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs

        self.actor = ActorTransformerModel(observation_dim, hidden_dim, action_dim_discrete, action_dim_continuous)
        self.critic = CriticTransformerModel(observation_dim, hidden_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = ActorTransformerModel(observation_dim, hidden_dim, action_dim_discrete, action_dim_continuous)

        self.MseLoss = nn.MSELoss()

    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
        action_outputs = self.policy_old(state)
        action_probs = action_outputs['action_type']
        quantity_pred = action_outputs['quantity']

        dist = Categorical(action_probs)
        action_type = dist.sample()

        quantity = torch.clamp(quantity_pred, min=0.01).item()

        return action_type.item(), quantity

    def learn(self, experiences: List[Tuple[np.ndarray, Tuple[int, float], float, np.ndarray, bool]]) -> None:
        """
        Implement PPO learning algorithm for volatility-based strategies.
        This method processes a batch of experiences to update the agent's policy.
        """
        if not experiences:
            return

        # Convert experiences to tensors
        states = torch.FloatTensor([exp[0] for exp in experiences]).unsqueeze(1)  # Add sequence dimension
        action_types = torch.LongTensor([exp[1][0] for exp in experiences])
        quantities = torch.FloatTensor([exp[1][1] for exp in experiences])
        rewards = torch.FloatTensor([exp[2] for exp in experiences])
        next_states = torch.FloatTensor([exp[3] for exp in experiences]).unsqueeze(1)
        dones = torch.FloatTensor([exp[4] for exp in experiences])

        # Calculate discounted rewards and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        # Calculate returns using GAE (Generalized Advantage Estimation)
        returns = []
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else next_values[i]
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae  # lambda = 0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old action probabilities and quantity predictions
        old_action_outputs = self.policy_old(states)
        old_action_probs = old_action_outputs['action_type'].detach()
        old_quantity_preds = old_action_outputs['quantity'].detach()
        old_action_probs = old_action_outputs['action_type']
        old_quantity_preds = old_action_outputs['quantity']

        old_dist_discrete = Categorical(old_action_probs)
        old_log_probs_discrete = old_dist_discrete.log_prob(action_types).detach()

        quantity_std = 0.1 # This might need to be learned or tuned
        old_dist_continuous = Normal(old_quantity_preds, quantity_std)
        old_log_probs_continuous = old_dist_continuous.log_prob(quantities).detach()

        # Combine log probabilities (assuming independence)
        old_log_probs = old_log_probs_discrete + old_log_probs_continuous

        # PPO update for k_epochs
        for _ in range(self.k_epochs):
            # Current action probabilities and quantity predictions
            action_outputs = self.actor(states)
            action_probs = action_outputs['action_type']
            quantity_preds = action_outputs['quantity']

            dist_discrete = Categorical(action_probs)
            log_probs_discrete = dist_discrete.log_prob(action_types)
            entropy_discrete = dist_discrete.entropy().mean()

            dist_continuous = Normal(quantity_preds, quantity_std)
            log_probs_continuous = dist_continuous.log_prob(quantities)

            # Combine log probabilities
            log_probs = log_probs_discrete + log_probs_continuous

            # Calculate ratio and surrogate losses
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_discrete  # entropy bonus

            # Critic loss
            current_values = self.critic(states).squeeze()
            critic_loss = self.MseLoss(current_values, returns)

            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)  # gradient clipping
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)  # gradient clipping
            self.optimizer_critic.step()

        # Update old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

    def adapt(self, observation: np.ndarray, action_tuple: Tuple[int, float], reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        """
        MAML adaptation for volatility-based strategies.
        Creates a temporary copy and performs gradient steps for fast adaptation.
        """
        # Create adapted copies of networks
        adapted_actor = ActorTransformerModel(self.observation_dim, self.hidden_dim, self.action_dim_discrete, self.action_dim_continuous)
        adapted_actor.load_state_dict(self.actor.state_dict())
        adapted_critic = CriticTransformerModel(self.observation_dim, self.hidden_dim)
        adapted_critic.load_state_dict(self.critic.state_dict())

        adapted_optimizer_actor = optim.Adam(adapted_actor.parameters(), lr=self.optimizer_actor.param_groups[0]['lr'])
        adapted_optimizer_critic = optim.Adam(adapted_critic.parameters(), lr=self.optimizer_critic.param_groups[0]['lr'])

        # Perform adaptation steps
        for _ in range(num_gradient_steps):
            # Convert to tensors
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
            action_type_tensor = torch.LongTensor([action_tuple[0]])
            quantity_tensor = torch.FloatTensor([action_tuple[1]])
            reward_tensor = torch.FloatTensor([reward])
            next_state_tensor = torch.FloatTensor(next_observation).unsqueeze(0).unsqueeze(0)
            done_tensor = torch.FloatTensor([done])

            # Calculate advantage and target value
            value = adapted_critic(state_tensor)
            next_value = adapted_critic(next_state_tensor)
            target_value = reward_tensor + self.gamma * next_value * (1 - done_tensor)
            advantage = target_value - value

            # Actor loss (PPO-style)
            old_action_outputs = self.policy_old(state_tensor).detach()
            old_action_probs = old_action_outputs['action_type']
            old_quantity_preds = old_action_outputs['quantity']

            old_dist_discrete = Categorical(old_action_probs)
            old_log_prob_discrete = old_dist_discrete.log_prob(action_type_tensor)

            quantity_std = 0.1 # This might need to be learned or tuned
            old_dist_continuous = Normal(old_quantity_preds, quantity_std)
            old_log_prob_continuous = old_dist_continuous.log_prob(quantity_tensor)

            old_log_prob = old_log_prob_discrete + old_log_prob_continuous

            action_outputs = adapted_actor(state_tensor)
            action_probs = action_outputs['action_type']
            quantity_preds = action_outputs['quantity']

            dist_discrete = Categorical(action_probs)
            log_prob_discrete = dist_discrete.log_prob(action_type_tensor)

            dist_continuous = Normal(quantity_preds, quantity_std)
            log_prob_continuous = dist_continuous.log_prob(quantity_tensor)

            log_prob = log_prob_discrete + log_prob_continuous

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

        # Return adapted agent
        adapted_agent = VolatilityAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim,
                                      self.optimizer_actor.param_groups[0]['lr'],
                                      self.optimizer_critic.param_groups[0]['lr'],
                                      self.gamma, self.epsilon_clip, self.k_epochs)
        adapted_agent.actor.load_state_dict(adapted_actor.state_dict())
        adapted_agent.critic.load_state_dict(adapted_critic.state_dict())
        adapted_agent.policy_old.load_state_dict(adapted_actor.state_dict())
        return adapted_agent

    def save_model(self, path: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_old_state_dict': self.policy_old.state_dict(),
            'optimizer_actor_state_dict': self.optimizer_actor.state_dict(),
            'optimizer_critic_state_dict': self.optimizer_critic.state_dict(),
            'observation_dim': self.observation_dim,
            'action_dim_discrete': self.action_dim_discrete,
            'action_dim_continuous': self.action_dim_continuous,
            'hidden_dim': self.hidden_dim
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.observation_dim = checkpoint['observation_dim']
        self.action_dim_discrete = checkpoint['action_dim_discrete']
        self.action_dim_continuous = checkpoint['action_dim_continuous']
        self.hidden_dim = checkpoint['hidden_dim']

        # Re-initialize models with loaded dimensions
        self.actor = ActorTransformerModel(self.observation_dim, self.hidden_dim, self.action_dim_discrete, self.action_dim_continuous)
        self.critic = CriticTransformerModel(self.observation_dim, self.hidden_dim)
        self.policy_old = ActorTransformerModel(self.observation_dim, self.hidden_dim, self.action_dim_discrete, self.action_dim_continuous)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_old.load_state_dict(checkpoint['policy_old_state_dict'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
