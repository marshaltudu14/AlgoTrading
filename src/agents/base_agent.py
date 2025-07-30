import abc
import numpy as np
from typing import Tuple, List

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Takes a normalized observation from the environment and returns a discrete action and quantity.
        Returns: (action_type, quantity) where action_type is 0-4 and quantity is a float.
        """
        pass

    @abc.abstractmethod
    def learn(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """
        Processes a single or batch of experiences to update the agent's internal policy.
        """
        pass

    @abc.abstractmethod
    def adapt(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool, num_gradient_steps: int) -> 'BaseAgent':
        """
        Performs rapid, task-specific adaptations without permanently altering global meta-parameters.
        Returns a new BaseAgent instance (or a representation of the adapted parameters) that is differentiable
        with respect to the original meta-parameters.
        """
        pass

    def save_model(self, path: str) -> None:
        """
        Saves the agent's learned policy (e.g., neural network weights) to a specified file path.
        """
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """
        Loads a previously saved policy from a specified file path.
        """
        pass