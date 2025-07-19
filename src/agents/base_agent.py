import abc
import numpy as np
from typing import Tuple, List

class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, observation: np.ndarray) -> int:
        """Selects an action based on the current observation."""
        pass

    @abc.abstractmethod
    def learn(self, experience: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        """Updates the agent's internal policy based on a single experience or a batch of experiences."""
        pass

    @abc.abstractmethod
    def save_model(self, path: str) -> None:
        """Saves the agent's learned policy to a specified file path."""
        pass

    @abc.abstractmethod
    def load_model(self, path: str) -> None:
        """Loads a previously saved policy from a specified file path."""
        pass
