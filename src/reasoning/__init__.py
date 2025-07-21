"""
Reasoning module for autonomous trading agents.

This module provides advanced reasoning capabilities including market classification,
pattern recognition, and self-modification logic.
"""

from .market_classifier import MarketClassifier
from .pattern_recognizer import PatternRecognizer
from .self_modification import SelfModificationManager

__all__ = ['MarketClassifier', 'PatternRecognizer', 'SelfModificationManager']
