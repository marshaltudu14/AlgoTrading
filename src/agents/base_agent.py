"""
Base Agent Interface for Trading System
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Union, Tuple, Dict, Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Defines the interface that all agents must implement to work with the training system.
    """
    
    @abstractmethod
    def select_action(self, observation: Union[torch.Tensor, np.ndarray]) -> Union[int, Tuple[int, float]]:
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current market observation/state
            
        Returns:
            Action (single int) or tuple of (action_type, quantity)
        """
        pass
    
    
    def act(self, observation: Union[torch.Tensor, np.ndarray]) -> Tuple[int, float]:
        """
        Alternative interface used by live trading system.
        Default implementation calls select_action.
        
        Args:
            observation: Current market observation/state
            
        Returns:
            Tuple of (action_type, quantity)
        """
        result = self.select_action(observation)
        if isinstance(result, tuple):
            return result
        else:
            # Convert single action to (action, default_quantity)
            return result, 1.0
    
    def save_model(self, path: str) -> None:
        """
        Save model to file. Optional method.
        
        Args:
            path: File path to save model
        """
        pass
    
    def load_model(self, path: str) -> None:
        """
        Load model from file. Optional method.
        
        Args:
            path: File path to load model from
        """
        pass
    
    def update(self) -> None:
        """
        Update model parameters. Optional method used by some trainers.
        """
        pass