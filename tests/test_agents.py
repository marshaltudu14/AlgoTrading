import torch
from src.agents.ppo_agent import PPOAgent
from unittest.mock import Mock, patch

import unittest
import abc
import numpy as np
from src.agents.base_agent import BaseAgent
import torch
from src.agents.ppo_agent import PPOAgent
from unittest.mock import Mock, patch
import os

class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self.observation_dim = 50 * 5 + 4  # Example from TradingEnv
        self.action_dim = 5
        self.hidden_dim = 64
        self.ppo_agent = PPOAgent(self.observation_dim, self.action_dim, self.hidden_dim)

    def test_ppo_agent_initialization(self):
        self.assertIsInstance(self.ppo_agent.actor, LSTMModel)
        self.assertIsInstance(self.ppo_agent.critic, LSTMModel)
        self.assertIsInstance(self.ppo_agent.optimizer_actor, torch.optim.Adam)
        self.assertIsInstance(self.ppo_agent.optimizer_critic, torch.optim.Adam)
        self.assertEqual(self.ppo_agent.gamma, 0.99)
        self.assertEqual(self.ppo_agent.epsilon_clip, 0.2)
        self.assertEqual(self.ppo_agent.k_epochs, 10)

    @patch('src.agents.ppo_agent.LSTMModel')
    def test_select_action(self, MockLSTMModel):
        mock_actor_instance = Mock()
        mock_actor_instance.return_value = torch.randn(1, self.action_dim) # Mock action probabilities
        MockLSTMModel.return_value = mock_actor_instance
        
        # Re-initialize PPOAgent with mocked LSTMModel
        self.ppo_agent = PPOAgent(self.observation_dim, self.action_dim, self.hidden_dim)
        self.ppo_agent.actor = mock_actor_instance # Assign the mocked instance to actor

        observation = np.random.rand(self.observation_dim)
        action = self.ppo_agent.select_action(observation)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
        mock_actor_instance.assert_called_once()

    @patch('src.agents.ppo_agent.LSTMModel')
    def test_learn(self, MockLSTMModel):
        mock_actor_instance = Mock()
        mock_critic_instance = Mock()
        MockLSTMModel.side_effect = [mock_actor_instance, mock_critic_instance]

        # Re-initialize PPOAgent with mocked LSTMModel
        self.ppo_agent = PPOAgent(self.observation_dim, self.action_dim, self.hidden_dim)
        self.ppo_agent.actor = mock_actor_instance
        self.ppo_agent.critic = mock_critic_instance

        # Mock return values for actor and critic
        mock_actor_instance.return_value = torch.randn(1, self.action_dim)
        mock_critic_instance.return_value = torch.randn(1, 1)

        # Create dummy experiences
        experiences = []
        for _ in range(5):
            state = np.random.rand(self.observation_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.rand()
            next_state = np.random.rand(self.observation_dim)
            done = False
            experiences.append((state, action, reward, next_state, done))
        
        # Manually populate the agent's internal buffers for the learn method to work
        # In a real scenario, select_action would populate these.
        for state, action, reward, _, done in experiences:
            self.ppo_agent.states.append(torch.FloatTensor(state).unsqueeze(0))
            self.ppo_agent.actions.append(torch.tensor([action]))
            self.ppo_agent.log_probs.append(torch.randn(1)) # Dummy log_prob
            self.ppo_agent.rewards.append(reward)
            self.ppo_agent.dones.append(done)

        # Mock optimizers to track calls
        self.ppo_agent.optimizer_actor = Mock(spec=torch.optim.Adam)
        self.ppo_agent.optimizer_critic = Mock(spec=torch.optim.Adam)

        self.ppo_agent.learn(experiences)

        self.assertEqual(mock_actor_instance.call_count, self.ppo_agent.k_epochs + 1) # 1 for initial, k_epochs for learn
        self.assertEqual(mock_critic_instance.call_count, self.ppo_agent.k_epochs + 1) # 1 for initial, k_epochs for learn
        self.assertEqual(self.ppo_agent.optimizer_actor.zero_grad.call_count, self.ppo_agent.k_epochs)
        self.assertEqual(self.ppo_agent.optimizer_actor.step.call_count, self.ppo_agent.k_epochs)
        self.assertEqual(self.ppo_agent.optimizer_critic.zero_grad.call_count, self.ppo_agent.k_epochs)
        self.assertEqual(self.ppo_agent.optimizer_critic.step.call_count, self.ppo_agent.k_epochs)
        
        # Assert buffers are cleared
        self.assertEqual(len(self.ppo_agent.states), 0)
        self.assertEqual(len(self.ppo_agent.actions), 0)
        self.assertEqual(len(self.ppo_agent.log_probs), 0)
        self.assertEqual(len(self.ppo_agent.rewards), 0)
        self.assertEqual(len(self.ppo_agent.dones), 0)

    def test_save_load_model(self):
        path = "test_ppo_model.pth"
        self.ppo_agent.save_model(path)
        self.assertTrue(Path(path).exists())

        new_agent = PPOAgent(self.observation_dim, self.action_dim, self.hidden_dim)
        new_agent.load_model(path)

        # Check if state dicts are equal (simple check)
        for param_old, param_new in zip(self.ppo_agent.actor.parameters(), new_agent.actor.parameters()):
            self.assertTrue(torch.equal(param_old, param_new))
        for param_old, param_new in zip(self.ppo_agent.critic.parameters(), new_agent.critic.parameters()):
            self.assertTrue(torch.equal(param_old, param_new))
        
        # Clean up
        os.remove(path)

if __name__ == '__main__':
    unittest.main()