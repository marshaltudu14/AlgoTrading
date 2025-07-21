"""
Autonomous Trading Agent

This module provides the core AutonomousAgent class that integrates the
TransformerWorldModel and ExternalMemory to create a fully autonomous
trading agent capable of reasoning, learning, and making decisions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

from src.models.world_model import TransformerWorldModel
from src.memory.episodic_memory import ExternalMemory
from src.reasoning.market_classifier import MarketClassifier
from src.reasoning.pattern_recognizer import PatternRecognizer
from src.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AutonomousAgent(BaseAgent):
    """
    Autonomous Trading Agent that integrates World Model and External Memory.
    
    This agent represents the next evolution in trading AI, combining:
    - TransformerWorldModel for prediction and policy reasoning
    - ExternalMemory for learning from past experiences
    - Autonomous decision-making capabilities
    
    The agent follows a "Think" loop:
    1. Embed the current market state
    2. Retrieve relevant memories from past experiences
    3. Use the World Model to predict future states and determine actions
    4. Store significant events in memory for future learning
    """
    
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        prediction_horizon: int = 5,
        market_features: int = 5,
        memory_size: int = 10000,
        memory_embedding_dim: int = 256,
        world_model_config: Optional[Dict] = None,
        memory_config: Optional[Dict] = None,
        market_classifier_config: Optional[Dict] = None,
        pattern_recognizer_config: Optional[Dict] = None,
        hyperparameters: Optional[Dict] = None,
        device: str = 'cpu'
    ):
        """
        Initialize the Autonomous Agent.
        
        Args:
            observation_dim: Dimension of market observations
            action_dim: Number of possible trading actions
            hidden_dim: Hidden dimension for transformer models
            prediction_horizon: Number of future steps to predict
            market_features: Number of market features (OHLCV = 5)
            memory_size: Maximum number of memories to store
            memory_embedding_dim: Dimension of memory embeddings
            world_model_config: Optional config dict for world model
            memory_config: Optional config dict for external memory
            market_classifier_config: Optional config dict for market classifier
            pattern_recognizer_config: Optional config dict for pattern recognizer
            hyperparameters: Optional dict of agent hyperparameters
            device: Device to run models on ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.prediction_horizon = prediction_horizon
        self.market_features = market_features
        self.memory_embedding_dim = memory_embedding_dim
        self.device = device
        
        # Initialize World Model
        world_model_config = world_model_config or {}
        regime_embedding_dim = 8  # Dimension for market regime embedding
        pattern_embedding_dim = 16  # Dimension for pattern features embedding
        self.regime_embedding_dim = regime_embedding_dim
        self.pattern_embedding_dim = pattern_embedding_dim
        self.world_model = TransformerWorldModel(
            input_dim=observation_dim + memory_embedding_dim + regime_embedding_dim + pattern_embedding_dim,  # State + memory + regime + patterns
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            market_features=market_features,
            hidden_dim=hidden_dim,
            **world_model_config
        ).to(device)
        
        # Initialize External Memory
        memory_config = memory_config or {}
        self.external_memory = ExternalMemory(
            max_memories=memory_size,
            embedding_dim=memory_embedding_dim,
            **memory_config
        )

        # Initialize Market Classifier
        market_classifier_config = market_classifier_config or {}
        self.market_classifier = MarketClassifier(**market_classifier_config)

        # Initialize Pattern Recognizer
        pattern_recognizer_config = pattern_recognizer_config or {}
        self.pattern_recognizer = PatternRecognizer(device=device, **pattern_recognizer_config)

        # Initialize Hyperparameters
        default_hyperparameters = {
            'learning_rate': 1e-3,
            'batch_size': 32,
            'discount_factor': 0.99,
            'entropy_coefficient': 0.01,
            'value_loss_coefficient': 0.5,
            'gradient_clip_norm': 0.5,
            'exploration_noise': 0.1,
            'memory_consolidation_threshold': 0.8,
            'risk_tolerance': 0.5,
            'pattern_confidence_threshold': 0.6
        }

        self.hyperparameters = {**default_hyperparameters, **(hyperparameters or {})}

        # Store hyperparameter history for tracking evolution
        self.hyperparameter_history = [self.hyperparameters.copy()]
        
        # State embedding layer (to convert market state to memory embedding space)
        self.state_embedder = nn.Sequential(
            nn.Linear(observation_dim, memory_embedding_dim),
            nn.ReLU(),
            nn.Linear(memory_embedding_dim, memory_embedding_dim)
        ).to(device)
        
        # Memory context aggregator (to combine retrieved memories)
        self.memory_aggregator = nn.Sequential(
            nn.Linear(memory_embedding_dim, memory_embedding_dim),
            nn.ReLU(),
            nn.Linear(memory_embedding_dim, memory_embedding_dim)
        ).to(device)
        
        # Action selection temperature for exploration
        self.temperature = 1.0
        
        # Statistics
        self.total_actions = 0
        self.total_memories_used = 0
        
        logger.info(f"Initialized AutonomousAgent with {observation_dim}D observations, "
                   f"{action_dim} actions, {memory_size} memory capacity")
    
    def act(self, market_state: Union[np.ndarray, torch.Tensor]) -> int:
        """
        Perform the "Think" loop and return a trading action.
        
        This is the core method that implements the autonomous decision-making process:
        1. Embed the current market state
        2. Retrieve relevant memories from past experiences
        3. Feed state and memories into the World Model
        4. Return action from the policy head
        
        Args:
            market_state: Current market observation
            
        Returns:
            Selected trading action (integer)
        """
        # Convert to tensor if needed
        if isinstance(market_state, np.ndarray):
            market_state = torch.FloatTensor(market_state).to(self.device)
        elif not isinstance(market_state, torch.Tensor):
            market_state = torch.FloatTensor(market_state).to(self.device)
        
        # Ensure correct shape (add batch dimension if needed)
        if len(market_state.shape) == 1:
            market_state = market_state.unsqueeze(0)  # (1, observation_dim)
        
        with torch.no_grad():
            # Step 1: Classify the current market regime
            market_regime, regime_confidence = self.market_classifier.classify_market(
                market_state.cpu().numpy(), return_confidence=True
            )

            # Step 2: Embed the current market state
            state_embedding = self.state_embedder(market_state)  # (1, memory_embedding_dim)

            # Step 3: Retrieve relevant memories from External Memory
            memory_context = self._retrieve_and_aggregate_memories(state_embedding)

            # Step 4: Recognize chart patterns
            pattern_detection = self.pattern_recognizer.recognize_pattern(market_state.cpu().numpy())

            # Step 5: Create market regime and pattern embeddings
            regime_embedding = self._encode_market_regime(market_regime, regime_confidence)
            pattern_embedding = self._encode_pattern_features(pattern_detection)

            # Step 6: Combine state, memory context, regime, and patterns for World Model input
            combined_input = torch.cat([market_state, memory_context, regime_embedding, pattern_embedding], dim=-1)
            
            # Add sequence dimension for transformer (batch_size, seq_len=1, input_dim)
            combined_input = combined_input.unsqueeze(1)

            # Step 7: Feed into World Model and get action
            world_model_output = self.world_model(combined_input)
            action_probs = world_model_output['policy']['actions']  # (1, action_dim)
            
            # Apply temperature for exploration
            if self.temperature != 1.0:
                action_probs = torch.softmax(torch.log(action_probs + 1e-8) / self.temperature, dim=-1)
            
            # Sample action from probability distribution
            action = torch.multinomial(action_probs, 1).item()
        
        self.total_actions += 1
        
        return action

    def _encode_market_regime(self, market_regime, confidence: float) -> torch.Tensor:
        """
        Encode market regime into a tensor representation.

        Args:
            market_regime: MarketRegime enum value
            confidence: Confidence score for the classification

        Returns:
            Tensor representation of market regime
        """
        # Create one-hot encoding for regime type
        regime_mapping = {
            'trending': [1, 0, 0, 0],
            'ranging': [0, 1, 0, 0],
            'volatile': [0, 0, 1, 0],
            'consolidation': [0, 0, 0, 1]
        }

        regime_vector = regime_mapping.get(market_regime.value, [0, 0, 0, 1])  # Default to consolidation

        # Add confidence and additional features
        regime_features = regime_vector + [
            confidence,  # Classification confidence
            confidence * 2 - 1,  # Normalized confidence (-1 to 1)
            1.0 if market_regime.value in ['trending', 'volatile'] else 0.0,  # High activity flag
            1.0 if market_regime.value in ['ranging', 'consolidation'] else 0.0  # Low activity flag
        ]

        # Convert to tensor and add batch dimension
        regime_tensor = torch.tensor(regime_features, dtype=torch.float32, device=self.device)
        return regime_tensor.unsqueeze(0)  # (1, regime_embedding_dim)

    def _encode_pattern_features(self, pattern_detection) -> torch.Tensor:
        """
        Encode pattern detection into a tensor representation.

        Args:
            pattern_detection: PatternDetection object

        Returns:
            Tensor representation of pattern features
        """
        # Create pattern type one-hot encoding (simplified to major categories)
        pattern_categories = {
            'candlestick_bullish': ['hammer', 'engulfing_bullish', 'morning_star'],
            'candlestick_bearish': ['shooting_star', 'hanging_man', 'engulfing_bearish', 'evening_star'],
            'candlestick_neutral': ['doji'],
            'chart_bullish': ['double_bottom', 'inverse_head_and_shoulders', 'triangle_ascending'],
            'chart_bearish': ['double_top', 'head_and_shoulders', 'triangle_descending'],
            'chart_continuation': ['flag_bullish', 'flag_bearish', 'triangle_symmetrical'],
            'no_pattern': ['no_pattern']
        }

        # Initialize category vector
        category_vector = [0.0] * 7  # 7 categories

        # Find which category the pattern belongs to
        pattern_value = pattern_detection.pattern_type.value
        for i, (category, patterns) in enumerate(pattern_categories.items()):
            if pattern_value in patterns:
                category_vector[i] = 1.0
                break

        # Add pattern-specific features
        pattern_features = category_vector + [
            pattern_detection.confidence,  # Pattern confidence
            pattern_detection.strength if hasattr(pattern_detection, 'strength') else 0.0,  # Pattern strength
            1.0 if pattern_detection.direction == "bullish" else 0.0,  # Bullish flag
            1.0 if pattern_detection.direction == "bearish" else 0.0,  # Bearish flag
            1.0 if pattern_detection.direction == "neutral" else 0.0,  # Neutral flag
            float(pattern_detection.end_index - pattern_detection.start_index),  # Pattern duration
            pattern_detection.confidence * (1.0 if pattern_detection.direction == "bullish" else -1.0 if pattern_detection.direction == "bearish" else 0.0),  # Directional confidence
            min(pattern_detection.confidence * 2, 1.0),  # Amplified confidence
            1.0 if pattern_detection.confidence > 0.7 else 0.0  # High confidence flag
        ]

        # Ensure we have exactly pattern_embedding_dim features
        if len(pattern_features) > self.pattern_embedding_dim:
            pattern_features = pattern_features[:self.pattern_embedding_dim]
        elif len(pattern_features) < self.pattern_embedding_dim:
            pattern_features.extend([0.0] * (self.pattern_embedding_dim - len(pattern_features)))

        # Convert to tensor and add batch dimension
        pattern_tensor = torch.tensor(pattern_features, dtype=torch.float32, device=self.device)
        return pattern_tensor.unsqueeze(0)  # (1, pattern_embedding_dim)

    def think_and_predict(
        self,
        market_state: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Perform deep thinking and return detailed predictions and reasoning.
        
        This method provides more detailed output than act(), including:
        - Future market predictions
        - Action reasoning
        - Memory retrieval details
        - Confidence scores
        
        Args:
            market_state: Current market observation
            
        Returns:
            Dictionary containing detailed predictions and reasoning
        """
        # Convert to tensor if needed
        if isinstance(market_state, np.ndarray):
            market_state = torch.FloatTensor(market_state).to(self.device)
        elif not isinstance(market_state, torch.Tensor):
            market_state = torch.FloatTensor(market_state).to(self.device)
        
        if len(market_state.shape) == 1:
            market_state = market_state.unsqueeze(0)
        
        with torch.no_grad():
            # Classify the current market regime
            market_regime, regime_confidence = self.market_classifier.classify_market(
                market_state.cpu().numpy(), return_confidence=True
            )

            # Embed state and retrieve memories
            state_embedding = self.state_embedder(market_state)
            retrieved_memories = self.external_memory.retrieve(state_embedding.cpu().numpy())
            memory_context = self._retrieve_and_aggregate_memories(state_embedding)

            # Recognize chart patterns
            pattern_detection = self.pattern_recognizer.recognize_pattern(market_state.cpu().numpy())

            # Encode market regime and patterns
            regime_embedding = self._encode_market_regime(market_regime, regime_confidence)
            pattern_embedding = self._encode_pattern_features(pattern_detection)

            # Get World Model predictions
            combined_input = torch.cat([market_state, memory_context, regime_embedding, pattern_embedding], dim=-1).unsqueeze(1)
            world_model_output = self.world_model(combined_input, return_attention=True)
            
            # Extract detailed information
            predictions = world_model_output['predictions']
            policy = world_model_output['policy']
            confidence = world_model_output['confidence']
            
            # Get action recommendation
            action_probs = policy['actions']
            recommended_action = torch.argmax(action_probs, dim=-1).item()
            
            return {
                'recommended_action': recommended_action,
                'action_probabilities': action_probs.cpu().numpy(),
                'predicted_market_state': predictions['market_state'].cpu().numpy(),
                'predicted_volatility': predictions['volatility'].cpu().numpy(),
                'market_regime_probs': predictions.get('market_regime', torch.zeros(1, 4)).cpu().numpy(),
                'value_estimate': policy['value'].cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'retrieved_memories': len(retrieved_memories),
                'memory_similarities': [sim for _, sim in retrieved_memories],
                'attention_weights': world_model_output.get('attention_weights', []),
                'market_regime': market_regime.value,
                'regime_confidence': regime_confidence,
                'market_features': self.market_classifier.get_market_features(market_state.cpu().numpy()),
                'detected_pattern': pattern_detection.pattern_type.value,
                'pattern_confidence': pattern_detection.confidence,
                'pattern_direction': pattern_detection.direction,
                'pattern_features': self.pattern_recognizer.get_pattern_features(market_state.cpu().numpy())
            }
    
    def learn_from_experience(
        self,
        market_state: Union[np.ndarray, torch.Tensor],
        action: int,
        reward: float,
        next_market_state: Union[np.ndarray, torch.Tensor],
        done: bool,
        importance: float = 1.0
    ) -> None:
        """
        Store a significant experience in External Memory for future learning.
        
        Args:
            market_state: The market state when action was taken
            action: The action that was taken
            reward: The reward received
            next_market_state: The resulting market state
            done: Whether the episode ended
            importance: Importance score for this experience (0.0 to 1.0)
        """
        # Convert to tensor if needed
        if isinstance(market_state, np.ndarray):
            market_state = torch.FloatTensor(market_state).to(self.device)
        
        if len(market_state.shape) == 1:
            market_state = market_state.unsqueeze(0)
        
        with torch.no_grad():
            # Embed the market state
            state_embedding = self.state_embedder(market_state)
            
            # Create outcome information
            outcome = {
                'action': action,
                'reward': reward,
                'done': done,
                'profit': reward,  # Assuming reward represents profit
                'market_state': market_state.cpu().numpy(),
                'next_market_state': next_market_state if isinstance(next_market_state, np.ndarray) 
                                   else np.array(next_market_state)
            }
            
            # Store in external memory
            self.external_memory.store(
                event_embedding=state_embedding.cpu().numpy(),
                outcome=outcome,
                importance=importance,
                metadata={
                    'timestamp': self.total_actions,
                    'episode_done': done
                }
            )
        
        logger.debug(f"Stored experience with reward {reward:.3f} and importance {importance:.3f}")
    
    def _retrieve_and_aggregate_memories(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant memories and aggregate them into a context vector.
        
        Args:
            state_embedding: Current state embedding
            
        Returns:
            Aggregated memory context tensor
        """
        # Retrieve memories
        retrieved_memories = self.external_memory.retrieve(state_embedding.cpu().numpy())
        
        if not retrieved_memories:
            # No relevant memories found, return zero context
            return torch.zeros(1, self.memory_embedding_dim).to(self.device)
        
        # Extract memory embeddings and similarities
        memory_embeddings = []
        similarities = []
        
        for memory_event, similarity in retrieved_memories:
            memory_embeddings.append(memory_event.embedding)
            similarities.append(similarity)
        
        # Convert to tensors
        memory_embeddings = torch.FloatTensor(np.stack(memory_embeddings)).to(self.device)
        similarities = torch.FloatTensor(similarities).to(self.device)
        
        # Weighted aggregation based on similarity scores
        similarities = similarities.unsqueeze(-1)  # (num_memories, 1)
        weighted_memories = memory_embeddings * similarities  # (num_memories, embedding_dim)
        
        # Sum and normalize
        aggregated_memory = torch.sum(weighted_memories, dim=0, keepdim=True)  # (1, embedding_dim)
        
        # Apply memory aggregator network
        memory_context = self.memory_aggregator(aggregated_memory)
        
        self.total_memories_used += len(retrieved_memories)
        
        return memory_context
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent's performance and memory usage."""
        memory_stats = self.external_memory.get_memory_statistics()
        
        return {
            'total_actions': self.total_actions,
            'total_memories_used': self.total_memories_used,
            'avg_memories_per_action': self.total_memories_used / max(self.total_actions, 1),
            'memory_stats': memory_stats,
            'world_model_params': sum(p.numel() for p in self.world_model.parameters()),
            'state_embedder_params': sum(p.numel() for p in self.state_embedder.parameters()),
            'memory_aggregator_params': sum(p.numel() for p in self.memory_aggregator.parameters())
        }
    
    def save_agent(self, filepath: str) -> None:
        """Save the agent's models and memory to file."""
        torch.save({
            'world_model_state_dict': self.world_model.state_dict(),
            'state_embedder_state_dict': self.state_embedder.state_dict(),
            'memory_aggregator_state_dict': self.memory_aggregator.state_dict(),
            'config': {
                'observation_dim': self.observation_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'prediction_horizon': self.prediction_horizon,
                'market_features': self.market_features,
                'memory_embedding_dim': self.memory_embedding_dim
            },
            'statistics': self.get_agent_statistics()
        }, filepath)
        
        # Save memory separately
        memory_filepath = filepath.replace('.pt', '_memory.pkl')
        self.external_memory.save_memories(memory_filepath)
        
        logger.info(f"Saved AutonomousAgent to {filepath} and memory to {memory_filepath}")
    
    def load_agent(self, filepath: str) -> None:
        """Load the agent's models and memory from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.state_embedder.load_state_dict(checkpoint['state_embedder_state_dict'])
        self.memory_aggregator.load_state_dict(checkpoint['memory_aggregator_state_dict'])
        
        # Load memory
        memory_filepath = filepath.replace('.pt', '_memory.pkl')
        self.external_memory.load_memories(memory_filepath)
        
        logger.info(f"Loaded AutonomousAgent from {filepath}")
    
    def select_action(self, observation: np.ndarray) -> int:
        """BaseAgent interface compatibility."""
        return self.act(observation)

    def learn(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """
        Learn from a single experience tuple.

        Args:
            experience: Tuple of (observation, action, reward, next_observation, done)
        """
        observation, action, reward, next_observation, done = experience

        # Calculate importance based on reward magnitude
        importance = min(1.0, abs(reward) / 10.0)  # Normalize to 0-1 range

        # Store the experience in memory
        self.learn_from_experience(
            market_state=observation,
            action=action,
            reward=reward,
            next_market_state=next_observation,
            done=done,
            importance=importance
        )

    def adapt(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        num_gradient_steps: int
    ) -> 'AutonomousAgent':
        """
        Perform rapid adaptation based on a single experience.

        For the autonomous agent, adaptation means storing the experience
        with high importance and returning a copy of itself.

        Args:
            observation: Current observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode ended
            num_gradient_steps: Number of gradient steps (not used for memory-based adaptation)

        Returns:
            Adapted agent (copy of self with new memory)
        """
        # Store the adaptation experience with high importance
        self.learn_from_experience(
            market_state=observation,
            action=action,
            reward=reward,
            next_market_state=next_observation,
            done=done,
            importance=1.0  # High importance for adaptation experiences
        )

        # For autonomous agent, we return self since adaptation is memory-based
        # In a more sophisticated implementation, we might fine-tune the world model
        return self

    def load_model(self, path: str) -> None:
        """
        Load the agent's models and memory from file.

        Args:
            path: File path to load from
        """
        self.load_agent(path)

    def get_hyperparameters(self) -> Dict[str, float]:
        """Get current hyperparameters."""
        return self.hyperparameters.copy()

    def update_hyperparameters(self, new_hyperparameters: Dict[str, float]) -> None:
        """
        Update agent hyperparameters.

        Args:
            new_hyperparameters: Dictionary of new hyperparameter values
        """
        self.hyperparameters.update(new_hyperparameters)
        self.hyperparameter_history.append(self.hyperparameters.copy())

        # Apply hyperparameters to relevant components
        self._apply_hyperparameters()

    def _apply_hyperparameters(self) -> None:
        """Apply current hyperparameters to agent components."""
        # Update pattern recognizer confidence threshold
        if hasattr(self.pattern_recognizer, 'min_pattern_confidence'):
            self.pattern_recognizer.min_pattern_confidence = self.hyperparameters.get(
                'pattern_confidence_threshold', 0.6
            )

        # Update memory consolidation threshold
        if hasattr(self.external_memory, 'consolidation_threshold'):
            self.external_memory.consolidation_threshold = self.hyperparameters.get(
                'memory_consolidation_threshold', 0.8
            )

        # Note: Learning rate and other training hyperparameters would be applied
        # during training by the trainer/optimizer

    def get_hyperparameter_evolution(self) -> List[Dict[str, float]]:
        """Get the evolution history of hyperparameters."""
        return self.hyperparameter_history.copy()

    def get_hyperparameter_statistics(self) -> Dict[str, Any]:
        """Get statistics about hyperparameter evolution."""
        if len(self.hyperparameter_history) < 2:
            return {}

        stats = {}
        for param_name in self.hyperparameters.keys():
            values = [h[param_name] for h in self.hyperparameter_history]
            stats[param_name] = {
                'current': values[-1],
                'initial': values[0],
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'change': values[-1] - values[0],
                'relative_change': (values[-1] - values[0]) / (values[0] + 1e-8)
            }

        return stats
