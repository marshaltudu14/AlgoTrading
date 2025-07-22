import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict

from src.agents.base_agent import BaseAgent
from src.models.transformer_models import TransformerModel, ActorTransformerModel

class GatingNetwork(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_experts)

    def forward(self, market_features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(market_features))
        return F.softmax(self.fc2(x), dim=-1)

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

        # Get action probabilities and quantities from all experts
        expert_action_probs = []
        expert_quantities = []
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0)

        for expert in self.experts:
            # Get action probabilities and quantities from each expert's policy
            with torch.no_grad():
                action_outputs = expert.policy_old(state_tensor)
                action_probs = action_outputs['action_type'].squeeze(0)
                quantity_pred = action_outputs['quantity'].squeeze(0)
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

        # Sample discrete action from the weighted probability distribution
        from torch.distributions import Categorical
        dist = Categorical(weighted_action_probs)
        final_action_type = dist.sample().item()

        # For continuous quantity, use the weighted average, ensuring it's positive
        final_quantity = torch.clamp(weighted_quantity, min=0.01).item()

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
                state_value = expert.critic(state_tensor).item()
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
                    expert_value = expert.critic(state_input).item()
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
        """Load gating network, optimizer, and all experts."""
        checkpoint = torch.load(path)
        self.gating_network.load_state_dict(checkpoint['gating_network_state_dict'])
        self.gating_optimizer.load_state_dict(checkpoint['gating_optimizer_state_dict'])

        # Re-initialize experts with loaded dimensions
        self.observation_dim = checkpoint['observation_dim']
        self.action_dim_discrete = checkpoint['action_dim_discrete']
        self.action_dim_continuous = checkpoint['action_dim_continuous']
        self.hidden_dim = checkpoint['hidden_dim']
        self.expert_configs = checkpoint['expert_configs']

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

        for i, expert_state in enumerate(checkpoint['experts_state_dicts']):
            self.experts[i].actor.load_state_dict(expert_state['actor_state_dict'])
            self.experts[i].critic.load_state_dict(expert_state['critic_state_dict'])
            self.experts[i].policy_old.load_state_dict(expert_state['policy_old_state_dict'])
            self.experts[i].optimizer_actor.load_state_dict(expert_state['optimizer_actor_state_dict'])
            self.experts[i].optimizer_critic.load_state_dict(expert_state['optimizer_critic_state_dict'])