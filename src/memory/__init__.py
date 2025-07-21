"""
Memory module for autonomous trading agents.

This module provides external memory capabilities for storing and retrieving
significant past events, enabling agents to learn from specific historical experiences.
"""

from .episodic_memory import ExternalMemory

__all__ = ['ExternalMemory']
