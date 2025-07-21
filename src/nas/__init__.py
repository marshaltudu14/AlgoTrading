"""
Neural Architecture Search (NAS) Framework

This module provides a framework for automatic neural architecture search,
allowing agents to discover and evolve their own model architectures.
"""

from .search_space import SearchSpace, LayerConfig
from .search_controller import NASController

__all__ = ['SearchSpace', 'LayerConfig', 'NASController']
