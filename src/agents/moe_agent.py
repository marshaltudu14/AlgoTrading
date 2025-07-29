import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict
import logging

from src.agents.base_agent import BaseAgent
from src.models.transformer_models import TransformerModel, ActorTransformerModel

logger = logging.getLogger(__name__)

def safe_tensor_to_scalar(tensor: torch.Tensor, default_value: float = 0.0) -> float:
    """
    Safely convert a tensor to a scalar value, handling various tensor dimensions.

    Args:
        tensor: Input tensor to convert
        default_value: Default value to return if conversion fails

    Returns:
        Scalar float value
    """
    try:
        if tensor is None:
            return default_value

        # Handle different tensor types and dimensions
        if isinstance(tensor, (int, float)):
            return float(tensor)

        if not isinstance(tensor, torch.Tensor):
            return float(tensor)

        # Handle 0-dimensional tensors
        if tensor.dim() == 0:
            return tensor.item()

        # Handle 1-dimensional tensors
        if tensor.dim() == 1:
            if tensor.numel() == 1:
                return tensor.item()
            else:
                # Take the first element if multiple elements
                return tensor[0].item()

        # Handle multi-dimensional tensors
        flattened = tensor.flatten()
        if flattened.numel() > 0:
            return flattened[0].item()
        else:
            return default_value

    except Exception as e:
        logger.warning(f"Failed to convert tensor to scalar: {e}. Using default value {default_value}")
        return default_value

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

        # Initialize weights with small values for numerical stability
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.01)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        # Check for NaN/Inf in input
        if torch.isnan(market_features).any() or torch.isinf(market_features).any():
            market_features = torch.zeros_like(market_features)

        x = self.fc1(market_features)

        # Check for NaN/Inf after first layer
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)

        x = F.relu(x)
        x = self.fc2(x)

        # Check for NaN/Inf before softmax
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.zeros_like(x)

        # Clamp logits to prevent extreme values
        x = torch.clamp(x, min=-10, max=10)

        # Apply softmax with numerical stability
        output = F.softmax(x, dim=-1)

        # Final check for NaN and use uniform distribution as fallback
        if torch.isnan(output).any():
            output = torch.ones_like(output) / output.size(-1)

        return output

class MoEAgent(BaseAgent):
    def __init__(self, observation_dim: int, action_dim_discrete: int, action_dim_continuous: int, hidden_dim: int, expert_configs: Dict, lr_gating: float = 0.001):
        self.observation_dim = observation_dim
        self.action_dim_discrete = action_dim_discrete
        self.action_dim_continuous = action_dim_continuous
        self.hidden_dim = hidden_dim
        self.expert_configs = expert_configs
        print(f"MoEAgent __init__: observation_dim={observation_dim}, action_dim_discrete={action_dim_discrete}, action_dim_continuous={action_dim_continuous}, hidden_dim={hidden_dim}, num_experts={len(expert_configs)}")

        self.gating_network = GatingNetwork(observation_dim, len(expert_configs), hidden_dim)
        self.gating_optimizer = optim.Adam(self.gating_network.parameters(), lr=lr_gating)

        self.experts = []
        for expert_type, config in expert_configs.items():
            if expert_type == "TrendAgent":
                from src.agents.trend_agent import TrendAgent
                self.experts.append(TrendAgent(observation_dim, action_dim_discrete, action_dim_continuous, hidden_dim))
            elif expert_type == "MeanReversionAgent":
                from src.agents.mean_reversion_agent import MeanReversionAgent
                self.experts.append(MeanReversionAgent(observation_dim, action_dim_discrete, action_dim_continuous, hidden_dim))
            elif expert_type == "VolatilityAgent":
                from src.agents.volatility_agent import VolatilityAgent
                self.experts.append(VolatilityAgent(observation_dim, action_dim_discrete, action_dim_continuous, hidden_dim))
            elif expert_type == "ConsolidationAgent":
                from src.agents.consolidation_agent import ConsolidationAgent
                self.experts.append(ConsolidationAgent(observation_dim, action_dim_discrete, action_dim_continuous, hidden_dim))
            else:
                raise ValueError(f"Unknown expert type: {expert_type}")

    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Select action using weighted average of expert action probabilities and quantities.
        This implements soft routing for better ensemble behavior.
        """
        # Get gating network weights
        market_features = torch.FloatTensor(observation).unsqueeze(0)
        expert_weights = self.gating_network(market_features).squeeze(0)

        # Check for NaN/Inf in expert weights
        if torch.isnan(expert_weights).any() or torch.isinf(expert_weights).any():
            logger.warning("NaN/Inf detected in expert weights, using uniform distribution")
            expert_weights = torch.ones_like(expert_weights) / expert_weights.size(0)

        # Get action probabilities and quantities from all experts
        expert_action_probs = []
        expert_quantities = []
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)

        for expert in self.experts:
            # Get action probabilities and quantities from each expert's current policy
            # CRITICAL: Use expert.actor (not policy_old) to preserve gradients for training
            action_outputs = expert.actor(state_tensor)
            action_probs = action_outputs['action_type'].squeeze(0)
            quantity_pred = action_outputs['quantity'].squeeze(0)

            # Add stability checks for each expert's output
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                logger.warning(f"NaN/Inf detected in expert action_probs, using uniform distribution")
                action_probs = torch.ones_like(action_probs, requires_grad=True) / action_probs.size(0)

            # Ensure probabilities are positive and sum to 1 (for both NaN and normal cases)
            action_probs = torch.clamp(action_probs, min=1e-8)
            action_probs = action_probs / action_probs.sum()

            # Clamp quantity to reasonable range (for both NaN and normal cases)
            quantity_pred = torch.clamp(quantity_pred, min=1.0, max=5.0)

            # Always append the processed values
            expert_action_probs.append(action_probs)
            expert_quantities.append(quantity_pred)

        # Stack expert probabilities and quantities
        expert_probs_tensor = torch.stack(expert_action_probs)  # Shape: [num_experts, action_dim_discrete]
        expert_quantities_tensor = torch.stack(expert_quantities) # Shape: [num_experts, action_dim_continuous]

        expert_weights_expanded_discrete = expert_weights.unsqueeze(1)  # Shape: [num_experts, 1]
        expert_weights_expanded_continuous = expert_weights.unsqueeze(1) # Shape: [num_experts, 1]

        # Weighted average of action probabilities and quantities
        weighted_action_probs = torch.sum(expert_probs_tensor * expert_weights_expanded_discrete, dim=0)
        weighted_quantity = torch.sum(expert_quantities_tensor * expert_weights_expanded_continuous, dim=0)

        # Add stability checks for NaN/Inf values
        if torch.isnan(weighted_action_probs).any() or torch.isinf(weighted_action_probs).any():
            logger.warning("NaN/Inf detected in weighted_action_probs, using uniform distribution")
            weighted_action_probs = torch.ones_like(weighted_action_probs) / weighted_action_probs.size(0)

        # Ensure probabilities are positive and sum to 1
        weighted_action_probs = torch.clamp(weighted_action_probs, min=1e-8)
        weighted_action_probs = weighted_action_probs / weighted_action_probs.sum()

        # Sample discrete action from the weighted probability distribution
        from torch.distributions import Categorical
        dist = Categorical(weighted_action_probs)
        action_sample = dist.sample()

        # Use safe tensor conversion for action type
        final_action_type = int(safe_tensor_to_scalar(action_sample, default_value=4))  # Default to HOLD

        # For continuous quantity, use the weighted average, ensuring it's positive
        quantity_tensor = torch.clamp(weighted_quantity, min=0.01)
        final_quantity = safe_tensor_to_scalar(quantity_tensor, default_value=1.0)

        # Ensure final_quantity is positive and reasonable
        final_quantity = max(0.01, min(final_quantity, 10.0))

        return final_action_type, final_quantity

    def learn(self, experiences: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> None:
        """
        Orchestrate learning for both the GatingNetwork and individual experts.
        This implements a joint optimization strategy for the entire MoE system.
        """
        if not experiences:
            return

        # First, let each expert learn from the experiences
        for expert in self.experts:
            expert.learn(experiences)

        # Now train the gating network based on expert performance
        self._train_gating_network(experiences)

    def _train_gating_network(self, experiences: List[Tuple[np.ndarray, Tuple[int, float], float, np.ndarray, bool]]) -> None:
        """
        Train the gating network to better select experts based on their performance.
        Uses a reward-weighted loss to encourage the gating network to assign higher
        weights to experts that would have performed better on the given experiences.
        """
        states = torch.FloatTensor([exp[0] for exp in experiences])
        # actions = torch.LongTensor([exp[1] for exp in experiences]) # No longer a single action
        rewards = torch.FloatTensor([exp[2] for exp in experiences])

        # Get gating network weights for all states
        gating_weights = self.gating_network(states)  # Shape: [batch_size, num_experts]

        # Calculate expert performance on these experiences
        expert_performances = []
        for i, expert in enumerate(self.experts):
            # Estimate how well this expert would have performed
            expert_performance = self._estimate_expert_performance(expert, experiences)
            expert_performances.append(expert_performance)

        expert_performances = torch.stack(expert_performances).T  # Shape: [batch_size, num_experts]

        # Normalize expert performances to create target weights
        target_weights = F.softmax(expert_performances, dim=1)

        # Gating network loss: KL divergence between current weights and target weights
        gating_loss = F.kl_div(
            F.log_softmax(gating_weights, dim=1),
            target_weights,
            reduction='batchmean'
        )

        # Update gating network
        self.gating_optimizer.zero_grad()
        gating_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gating_network.parameters(), 0.5)
        self.gating_optimizer.step()

    def _estimate_expert_performance(self, expert, experiences) -> torch.Tensor:
        """
        Estimate how well an expert would perform on the given experiences.
        This is a simplified approach using the expert's value function.
        """
        performances = []
        for state, action, reward, next_state, done in experiences:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)

            # Use the expert's critic to estimate state value
            with torch.no_grad():
                critic_output = expert.critic(state_tensor)
                # Use safe tensor conversion
                state_value = safe_tensor_to_scalar(critic_output, default_value=0.0)
                # Combine actual reward with estimated value for performance metric
                performance = reward + 0.5 * state_value  # Simple heuristic
                performances.append(performance)

        return torch.FloatTensor(performances)

    def adapt(self, observation: np.ndarray, action_tuple: Tuple[int, float], reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        """
        Orchestrate MAML adaptation for both the GatingNetwork and individual experts.
        This creates adapted copies and performs gradient steps for fast adaptation.
        """
        print(f"MoEAgent adapt: observation_dim={self.observation_dim}, action_dim_discrete={self.action_dim_discrete}, action_dim_continuous={self.action_dim_continuous}, hidden_dim={self.hidden_dim}, expert_configs={self.expert_configs}")

        # Create adapted copy of gating network
        adapted_gating_network = GatingNetwork(
            self.gating_network.fc1.in_features,
            self.gating_network.fc2.out_features,
            self.gating_network.fc1.out_features
        )
        adapted_gating_network.load_state_dict(self.gating_network.state_dict())
        adapted_gating_optimizer = optim.Adam(adapted_gating_network.parameters(), lr=self.gating_optimizer.param_groups[0]['lr'])

        # Adapt experts using their individual adapt methods
        adapted_experts = []
        for expert in self.experts:
            adapted_expert = expert.adapt(observation, action_tuple, reward, next_observation, done, num_gradient_steps)
            adapted_experts.append(adapted_expert)

        # Adapt gating network based on the experience
        self._adapt_gating_network(adapted_gating_network, adapted_gating_optimizer,
                                 observation, action_tuple, reward, next_observation, done,
                                 adapted_experts, num_gradient_steps)

        # Create adapted MoE agent
        adapted_moe_agent = MoEAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim, self.expert_configs)
        adapted_moe_agent.gating_network.load_state_dict(adapted_gating_network.state_dict())
        adapted_moe_agent.experts = adapted_experts
        return adapted_moe_agent

    def _adapt_gating_network(self, adapted_gating_network, adapted_optimizer,
                            observation, action_tuple, reward, next_observation, done,
                            adapted_experts, num_gradient_steps):
        """
        Adapt the gating network based on expert performance on the given experience.
        """
        for _ in range(num_gradient_steps):
            state_tensor = torch.FloatTensor(observation).unsqueeze(0)

            # Get current gating weights
            gating_weights = adapted_gating_network(state_tensor).squeeze(0)

            # Estimate how well each adapted expert would perform
            expert_performances = []
            for expert in adapted_experts:
                # Use expert's critic to estimate performance
                state_input = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    critic_output = expert.critic(state_input)
                    # Use safe tensor conversion
                    expert_value = safe_tensor_to_scalar(critic_output, default_value=0.0)
                    # Combine reward with expert's value estimate
                    performance = reward + 0.5 * expert_value
                    expert_performances.append(performance)

            expert_performances = torch.FloatTensor(expert_performances)

            # Create target weights based on performance (softmax of performances)
            target_weights = F.softmax(expert_performances, dim=0)

            # Gating loss: encourage weights to match expert performance
            gating_loss = F.mse_loss(gating_weights, target_weights)

            # Update adapted gating network
            adapted_optimizer.zero_grad()
            gating_loss.backward()
            adapted_optimizer.step()

    def save_model(self, path: str) -> None:
        """Save gating network, optimizer, and all experts."""
        expert_states = []
        for expert in self.experts:
            expert_states.append({
                'actor_state_dict': expert.actor.state_dict(),
                'critic_state_dict': expert.critic.state_dict(),
                'policy_old_state_dict': expert.policy_old.state_dict(),
                'optimizer_actor_state_dict': expert.optimizer_actor.state_dict(),
                'optimizer_critic_state_dict': expert.optimizer_critic.state_dict(),
            })

        torch.save({
            'gating_network_state_dict': self.gating_network.state_dict(),
            'gating_optimizer_state_dict': self.gating_optimizer.state_dict(),
            'experts_state_dicts': expert_states,
            'expert_configs': self.expert_configs,
            'observation_dim': self.observation_dim,
            'action_dim_discrete': self.action_dim_discrete,
            'action_dim_continuous': self.action_dim_continuous,
            'hidden_dim': self.hidden_dim
        }, path)

    def load_model(self, path: str) -> None:
        """Load gating network, optimizer, and all experts with dimension mismatch handling."""
        try:
            checkpoint = torch.load(path, map_location='cpu')

            # Check for dimension mismatch
            saved_obs_dim = checkpoint.get('observation_dim', None)
            if saved_obs_dim is not None and saved_obs_dim != self.observation_dim:
                logger.warning(f"Observation dimension mismatch: saved={saved_obs_dim}, current={self.observation_dim}")
                logger.info("Skipping model loading due to dimension mismatch - using fresh initialization")
                return
            elif saved_obs_dim is None:
                logger.warning("No observation_dim found in checkpoint, attempting to detect mismatch from model structure")
                # Try to detect dimension mismatch from gating network structure
                try:
                    gating_state = checkpoint.get('gating_network_state_dict', {})
                    if gating_state:
                        # Check the first layer input dimension
                        first_layer_key = next((k for k in gating_state.keys() if 'weight' in k and 'layers.0' in k), None)
                        if first_layer_key:
                            first_layer_weight = gating_state[first_layer_key]
                            saved_input_dim = first_layer_weight.shape[1]  # Input dimension
                            if saved_input_dim != self.observation_dim:
                                logger.warning(f"Detected dimension mismatch from model structure: saved={saved_input_dim}, current={self.observation_dim}")
                                logger.info("Skipping model loading due to detected dimension mismatch")
                                return
                except Exception as e:
                    logger.warning(f"Could not detect dimension mismatch from model structure: {e}")
                    logger.info("Proceeding with model loading (may cause errors)")

            # Load gating network and optimizer
            try:
                self.gating_network.load_state_dict(checkpoint['gating_network_state_dict'])
                self.gating_optimizer.load_state_dict(checkpoint['gating_optimizer_state_dict'])
                logger.info("Loaded gating network and optimizer")
            except Exception as e:
                logger.warning(f"Failed to load gating network/optimizer: {e}")
                logger.info("Using fresh gating network initialization")

            # Update dimensions from checkpoint (only if they match)
            self.action_dim_discrete = checkpoint.get('action_dim_discrete', self.action_dim_discrete)
            self.action_dim_continuous = checkpoint.get('action_dim_continuous', self.action_dim_continuous)
            self.hidden_dim = checkpoint.get('hidden_dim', self.hidden_dim)
            self.expert_configs = checkpoint.get('expert_configs', self.expert_configs)

            # Clear existing experts and re-create them with correct dimensions
            self.experts = []
            for expert_type, config in self.expert_configs.items():
                if expert_type == "TrendAgent":
                    from src.agents.trend_agent import TrendAgent
                    self.experts.append(TrendAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim))
                elif expert_type == "MeanReversionAgent":
                    from src.agents.mean_reversion_agent import MeanReversionAgent
                    self.experts.append(MeanReversionAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim))
                elif expert_type == "VolatilityAgent":
                    from src.agents.volatility_agent import VolatilityAgent
                    self.experts.append(VolatilityAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim))
                elif expert_type == "ConsolidationAgent":
                    from src.agents.consolidation_agent import ConsolidationAgent
                    self.experts.append(ConsolidationAgent(self.observation_dim, self.action_dim_discrete, self.action_dim_continuous, self.hidden_dim))
                else:
                    raise ValueError(f"Unknown expert type: {expert_type}")

            # Load expert states with dimension-aware error handling
            if 'experts_state_dicts' in checkpoint:
                for i, expert_state in enumerate(checkpoint['experts_state_dicts']):
                    if i < len(self.experts):
                        try:
                            # Check if expert state dimensions match current expert
                            actor_state = expert_state['actor_state_dict']
                            # Check the first layer weight to detect dimension mismatch
                            first_layer_key = next((k for k in actor_state.keys() if 'weight' in k and 'layers.0' in k), None)
                            if first_layer_key:
                                first_layer_weight = actor_state[first_layer_key]
                                saved_expert_dim = first_layer_weight.shape[1]  # Input dimension
                                if saved_expert_dim != self.observation_dim:
                                    logger.warning(f"Expert {i} dimension mismatch: saved={saved_expert_dim}, current={self.observation_dim}")
                                    logger.info(f"Expert {i} will use fresh initialization due to dimension mismatch")
                                    continue

                            # If dimensions match, load the state
                            self.experts[i].actor.load_state_dict(expert_state['actor_state_dict'])
                            self.experts[i].critic.load_state_dict(expert_state['critic_state_dict'])
                            self.experts[i].policy_old.load_state_dict(expert_state['policy_old_state_dict'])
                            self.experts[i].optimizer_actor.load_state_dict(expert_state['optimizer_actor_state_dict'])
                            self.experts[i].optimizer_critic.load_state_dict(expert_state['optimizer_critic_state_dict'])
                            logger.info(f"Loaded expert {i} state")
                        except Exception as e:
                            logger.warning(f"Failed to load expert {i} state: {e}")
                            logger.info(f"Expert {i} will use fresh initialization")

            logger.info(f"MoE model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load MoE model from {path}: {e}")
            logger.info("Using fresh model initialization")