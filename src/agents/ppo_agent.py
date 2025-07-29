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
        # Validate observation
        if not np.isfinite(observation).all():
            print("Warning: Invalid observation detected, using default action")
            return 4, 1.0  # Default to HOLD action with quantity 1.0

        state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
        state = to_device(state)

        try:
            # Get both discrete action probabilities and continuous quantity from the actor
            action_outputs = self.policy_old(state)
            action_probs = action_outputs['action_type']
            quantity_pred = action_outputs['quantity']

            # Check for NaN values
            if torch.isnan(action_probs).any() or torch.isnan(quantity_pred).any():
                print("Warning: NaN detected in action outputs, using default action")
                return 4, 1.0  # Default to HOLD action

            # Sample discrete action
            dist = Categorical(action_probs)
            action_type = dist.sample()

            # For continuous quantity, use the predicted value directly, ensuring it's positive
            quantity = torch.clamp(quantity_pred, min=0.01, max=10.0).item()

            return action_type.item(), quantity

        except Exception as e:
            print(f"Error in select_action: {e}, using default action")
            return 4, 1.0  # Default to HOLD action

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
                # Handle different input types
                if isinstance(state, (list, tuple)):
                    state_array = np.array(state, dtype=np.float32)
                elif isinstance(state, np.ndarray):
                    state_array = state.astype(np.float32)
                else:
                    print(f"Warning: State {i} has unexpected type {type(state)}, skipping learning")
                    return

                # Ensure it's a 1D array
                if len(state_array.shape) > 1:
                    state_array = state_array.flatten()
                elif len(state_array.shape) == 0:
                    state_array = np.array([state_array], dtype=np.float32)

                # Check for invalid values
                if not np.isfinite(state_array).all():
                    print(f"Warning: State {i} contains invalid values (inf/nan), skipping learning")
                    return

                # Set expected length from first valid state
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

            # Verify we have valid states
            if not state_arrays:
                print("Error: No valid states found, skipping learning")
                return

            # Verify all states have the same shape
            if not all(len(state) == expected_length for state in state_arrays):
                print(f"Error: Failed to normalize state lengths, skipping learning")
                return

            # Convert to numpy array and then to tensor with better error handling
            try:
                states_array = np.stack(state_arrays, axis=0).astype(np.float32)
                states = torch.FloatTensor(states_array).unsqueeze(1)  # Add sequence dimension
            except ValueError as e:
                print(f"Error creating states tensor: {e}")
                print(f"State array shapes: {[s.shape for s in state_arrays]}")
                return

            # Same for next_states
            next_state_arrays = []
            for i, next_state in enumerate(next_states):
                # Handle different input types
                if isinstance(next_state, (list, tuple)):
                    next_state_array = np.array(next_state, dtype=np.float32)
                elif isinstance(next_state, np.ndarray):
                    next_state_array = next_state.astype(np.float32)
                else:
                    print(f"Warning: Next state {i} has unexpected type {type(next_state)}, skipping learning")
                    return

                # Ensure it's a 1D array
                if len(next_state_array.shape) > 1:
                    next_state_array = next_state_array.flatten()
                elif len(next_state_array.shape) == 0:
                    next_state_array = np.array([next_state_array], dtype=np.float32)

                # Check for invalid values
                if not np.isfinite(next_state_array).all():
                    print(f"Warning: Next state {i} contains invalid values (inf/nan), skipping learning")
                    return

                # Ensure same length as states
                if len(next_state_array) != expected_length:
                    if len(next_state_array) < expected_length:
                        next_state_array = np.pad(next_state_array, (0, expected_length - len(next_state_array)), 'constant')
                    else:
                        next_state_array = next_state_array[:expected_length]

                next_state_arrays.append(next_state_array)

            # Convert to numpy array and then to tensor with better error handling
            try:
                next_states_array = np.stack(next_state_arrays, axis=0).astype(np.float32)
                next_states = torch.FloatTensor(next_states_array).unsqueeze(1)
            except ValueError as e:
                print(f"Error creating next_states tensor: {e}")
                print(f"Next state array shapes: {[s.shape for s in next_state_arrays]}")
                return

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

        # Calculate returns and advantages with numerical stability
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()

        # Ensure values and next_values are at least 1-dimensional for indexing
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if next_values.dim() == 0:
            next_values = next_values.unsqueeze(0)

        # Check for NaN in critic outputs
        if torch.isnan(values).any() or torch.isnan(next_values).any():
            print("Warning: NaN detected in critic values, skipping learning step")
            return

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

            # Clamp GAE to prevent extreme values
            gae = torch.clamp(gae, min=-10.0, max=10.0)

            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages for better training stability
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

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

            # Check for NaN values in action probabilities
            if torch.isnan(action_probs).any():
                print("Warning: NaN detected in action probabilities, skipping learning step")
                return

            dist_discrete = Categorical(action_probs)
            log_probs_discrete = dist_discrete.log_prob(action_types)
            entropy_discrete = dist_discrete.entropy().mean()

            # Clamp quantity predictions to prevent extreme values
            quantity_preds = torch.clamp(quantity_preds, min=0.01, max=10.0)
            dist_continuous = Normal(quantity_preds, quantity_std)
            log_probs_continuous = dist_continuous.log_prob(quantities)

            # Combine log probabilities
            log_probs = log_probs_discrete + log_probs_continuous

            # Check for NaN values in log probabilities
            if torch.isnan(log_probs).any() or torch.isnan(old_log_probs).any():
                print("Warning: NaN detected in log probabilities, skipping learning step")
                return

            # Calculate ratio and surrogate losses
            ratio = torch.exp(log_probs - old_log_probs)
            # Clamp ratio to prevent extreme values
            ratio = torch.clamp(ratio, min=0.1, max=10.0)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy_discrete

            # Critic loss
            current_values = self.critic(states).squeeze()
            critic_loss = self.MseLoss(current_values, returns)

            # Check for NaN values in losses
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                print("Warning: NaN detected in losses, skipping learning step")
                return

            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.optimizer_critic.step()

        # Update old policy
        self.policy_old.load_state_dict(self.actor.state_dict())

    def adapt(self, observation: np.ndarray, action_tuple: Tuple[int, float], reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        # Create a temporary copy of the agent's policy and value network parameters
        adapted_actor = ActorTransformerModel(self.observation_dim, self.hidden_dim, self.action_dim_discrete, self.action_dim_continuous)
        adapted_actor.load_state_dict(self.actor.state_dict())
        adapted_critic = CriticTransformerModel(self.observation_dim, self.hidden_dim)
        adapted_critic.load_state_dict(self.critic.state_dict())

        # Safely extract learning rates
        actor_lr = self.optimizer_actor.param_groups[0]['lr']
        critic_lr = self.optimizer_critic.param_groups[0]['lr']

        # Handle tensor learning rates
        if hasattr(actor_lr, 'item'):
            actor_lr = actor_lr.item()
        if hasattr(critic_lr, 'item'):
            critic_lr = critic_lr.item()

        adapted_optimizer_actor = optim.Adam(adapted_actor.parameters(), lr=actor_lr)
        adapted_optimizer_critic = optim.Adam(adapted_critic.parameters(), lr=critic_lr)

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
        # Safely extract learning rates for new agent
        actor_lr = self.optimizer_actor.param_groups[0]['lr']
        critic_lr = self.optimizer_critic.param_groups[0]['lr']

        # Handle tensor learning rates
        if hasattr(actor_lr, 'item'):
            actor_lr = actor_lr.item()
        if hasattr(critic_lr, 'item'):
            critic_lr = critic_lr.item()

        adapted_agent = PPOAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim, actor_lr, critic_lr, self.gamma, self.epsilon_clip, self.k_epochs)
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