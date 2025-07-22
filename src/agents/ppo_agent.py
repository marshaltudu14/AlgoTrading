import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, List

from src.agents.base_agent import BaseAgent
from src.models.transformer_models import ActorTransformerModel, CriticTransformerModel
from src.utils.hardware_optimizer import get_hardware_optimizer, optimize_for_device, to_device

class PPOAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim_discrete: int, action_dim_continuous: int, hidden_dim: int,
                 lr_actor: float, lr_critic: float, gamma: float, epsilon_clip: float, k_epochs: int):
        super(PPOAgent, self).__init__()

        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.k_epochs = k_epochs
        self.observation_dim = observation_dim
        self.action_dim_discrete = action_dim_discrete
        self.action_dim_continuous = action_dim_continuous
        self.hidden_dim = hidden_dim

        # Actor outputs both discrete action probabilities and continuous quantity
        self.actor = ActorTransformerModel(observation_dim, hidden_dim, action_dim_discrete, action_dim_continuous)
        self.critic = CriticTransformerModel(observation_dim, hidden_dim)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = ActorTransformerModel(observation_dim, hidden_dim, action_dim_discrete, action_dim_continuous)
        self.policy_old.load_state_dict(self.actor.state_dict())

        # Optimize for hardware
        self.hardware_optimizer = get_hardware_optimizer()
        self.actor = optimize_for_device(self.actor)
        self.critic = optimize_for_device(self.critic)
        self.policy_old = optimize_for_device(self.policy_old)

        self.MseLoss = nn.MSELoss()

    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
        state = to_device(state)
        
        # Get both discrete action probabilities and continuous quantity from the actor
        action_outputs = self.policy_old(state)
        action_probs = action_outputs['action_type']
        quantity_pred = action_outputs['quantity']

        # Sample discrete action
        dist = Categorical(action_probs)
        action_type = dist.sample()

        # For continuous quantity, we can use the predicted value directly or sample from a distribution
        # For now, let's use the predicted value directly, ensuring it's positive
        quantity = torch.clamp(quantity_pred, min=0.01).item() # Ensure quantity is at least 0.01

        return action_type.item(), quantity

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        """
        Implement PPO learning algorithm.
        This method processes a batch of experiences to update the agent's policy.
        """
        if not experiences:
            return

        # Limit buffer size to prevent memory issues
        max_buffer_size = 500
        if len(experiences) > max_buffer_size:
            experiences = experiences[-max_buffer_size:]  # Keep only recent experiences
            print(f"Limited experience buffer to {max_buffer_size} most recent experiences")



        # Convert experiences to tensors
        states = []
        action_types = []
        quantities = []
        rewards = []
        next_states = []
        dones = []

        for state, action_tuple, reward, next_state, done in experiences:
            states.append(state)
            action_types.append(action_tuple[0])
            quantities.append(action_tuple[1])
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Convert to tensors with proper shapes
        try:
            # Ensure all states are numpy arrays and have consistent shapes
            state_arrays = []
            expected_length = None

            for i, state in enumerate(states):
                state_array = np.array(state, dtype=np.float32)
                if len(state_array.shape) != 1:
                    print(f"Warning: State {i} has unexpected shape {state_array.shape}, skipping learning")
                    return

                # Set expected length from first state
                if expected_length is None:
                    expected_length = len(state_array)

                # Ensure all states have the same length as the first one
                if len(state_array) != expected_length:
                    if len(state_array) < expected_length:
                        # Pad with zeros
                        state_array = np.pad(state_array, (0, expected_length - len(state_array)), 'constant')
                    else:
                        # Truncate
                        state_array = state_array[:expected_length]

                state_arrays.append(state_array)

            # Verify all states have the same shape
            if not all(len(state) == expected_length for state in state_arrays):
                print(f"Error: Failed to normalize state lengths, skipping learning")
                return

            # Convert to numpy array and then to tensor
            states_array = np.array(state_arrays, dtype=np.float32)
            states = torch.FloatTensor(states_array).unsqueeze(1)  # Add sequence dimension

            # Same for next_states
            next_state_arrays = []
            for i, next_state in enumerate(next_states):
                next_state_array = np.array(next_state, dtype=np.float32)
                if len(next_state_array.shape) != 1:
                    print(f"Warning: Next state {i} has unexpected shape {next_state_array.shape}, skipping learning")
                    return
                next_state_arrays.append(next_state_array)

            # Check if all next_states have the same length
            next_state_lengths = [len(next_state) for next_state in next_state_arrays]
            if len(set(next_state_lengths)) > 1:
                print(f"Warning: Inconsistent next_state lengths: {set(next_state_lengths)}, using most common length")
                # Use the most common length
                from collections import Counter
                most_common_length = Counter(next_state_lengths).most_common(1)[0][0]

                # Pad or truncate to make them consistent
                fixed_next_states = []
                for next_state_array in next_state_arrays:
                    if len(next_state_array) < most_common_length:
                        fixed_next_state = np.pad(next_state_array, (0, most_common_length - len(next_state_array)), 'constant')
                    else:
                        fixed_next_state = next_state_array[:most_common_length]
                    fixed_next_states.append(fixed_next_state)
                next_state_arrays = fixed_next_states

            # Convert to numpy array and then to tensor
            next_states_array = np.array(next_state_arrays, dtype=np.float32)
            next_states = torch.FloatTensor(next_states_array).unsqueeze(1)

            # Convert other arrays to tensors
            action_types = torch.LongTensor(action_types)
            quantities = torch.FloatTensor(quantities)
            rewards = torch.FloatTensor(rewards)
            dones = torch.FloatTensor(dones)

        except Exception as e:
            print(f"Error converting experiences to tensors: {e}")
            print(f"Number of experiences: {len(experiences)}")
            print(f"Skipping this learning step to avoid crash")
            return  # Skip this learning step



        # Move to device
        states = to_device(states)
        action_types = to_device(action_types)
        quantities = to_device(quantities)
        rewards = to_device(rewards)
        next_states = to_device(next_states)
        dones = to_device(dones)

        # Calculate returns and advantages
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        returns = []
        advantages = []
        gae = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[i] * (1 - dones[i])
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value - values[i]
            gae = delta + self.gamma * 0.95 * gae * (1 - dones[i])  # GAE-lambda
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        advantages = to_device(advantages)
        returns = to_device(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get old action probabilities and quantity predictions
        old_action_outputs = self.policy_old(states)
        old_action_probs = old_action_outputs['action_type']
        old_quantity_preds = old_action_outputs['quantity']

        old_dist_discrete = Categorical(old_action_probs)
        old_log_probs_discrete = old_dist_discrete.log_prob(action_types).detach()

        # For continuous quantity, assume a Gaussian distribution
        # We need to define a standard deviation for the Normal distribution
        # For simplicity, let's assume a fixed std for now, or learn it
        # Here, we'll use a small fixed std for demonstration
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
            # No entropy for continuous action directly from Normal distribution, it's part of the log_prob

            # Combine log probabilities
            log_probs = log_probs_discrete + log_probs_continuous

            # Calculate ratio and surrogate losses
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_discrete  # entropy bonus for discrete action

            # Critic loss
            current_values = self.critic(states).squeeze()
            critic_loss = self.MseLoss(current_values, returns)

            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

        # Update old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

    def adapt(self, observation: np.ndarray, action_tuple: Tuple[int, float], reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        # Create a temporary copy of the agent's policy and value network parameters
        adapted_actor = ActorTransformerModel(self.observation_dim, self.hidden_dim, self.action_dim_discrete, self.action_dim_continuous)
        adapted_actor.load_state_dict(self.actor.state_dict())
        adapted_critic = CriticTransformerModel(self.observation_dim, self.hidden_dim)
        adapted_critic.load_state_dict(self.critic.state_dict())

        adapted_optimizer_actor = optim.Adam(adapted_actor.parameters(), lr=self.optimizer_actor.param_groups[0]['lr'])
        adapted_optimizer_critic = optim.Adam(adapted_critic.parameters(), lr=self.optimizer_critic.param_groups[0]['lr'])

        # Perform num_gradient_steps of a standard RL update on these temporary parameters
        for _ in range(num_gradient_steps):
            # For simplicity, we'll use a single experience for adaptation here.
            # In a real MAML implementation, this would be a batch of experiences from the task.
            state, action_type, quantity, reward, next_state, done = observation, action_tuple[0], action_tuple[1], reward, next_observation, done

            # Convert to tensors
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
            action_type_tensor = torch.LongTensor([action_type])
            quantity_tensor = torch.FloatTensor([quantity])
            reward_tensor = torch.FloatTensor([reward])
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
            done_tensor = torch.FloatTensor([done])

            # Calculate advantage and target value (simplified for placeholder)
            value = adapted_critic(state_tensor)
            next_value = adapted_critic(next_state_tensor)
            target_value = reward_tensor + self.gamma * next_value * (1 - done_tensor)
            advantage = target_value - value

            # Actor loss
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

        # Return a new BaseAgent instance with the adapted parameters
        # For simplicity, we'll return a new PPOAgent instance with adapted weights
        adapted_agent = PPOAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim, self.optimizer_actor.param_groups[0]['lr'], self.optimizer_critic.param_groups[0]['lr'], self.gamma, self.epsilon_clip, self.k_epochs)
        adapted_agent.actor.load_state_dict(adapted_actor.state_dict())
        adapted_agent.critic.load_state_dict(adapted_critic.state_dict())
        adapted_agent.policy_old.load_state_dict(adapted_actor.state_dict())
        return adapted_agent

    def save_model(self, path: str) -> None:
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict()
        }, path)

    def load_model(self, path: str) -> None:
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.policy_old.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])