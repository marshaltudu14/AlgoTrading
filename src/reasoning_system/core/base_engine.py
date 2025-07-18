#!/usr/bin/env python3
"""
Base Reasoning Engine
====================

Abstract base class for all reasoning engines in the system.
Provides common interface and utility methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseReasoningEngine(ABC):
    """
    Abstract base class for all reasoning engines.
    
    All reasoning engines must inherit from this class and implement
    the generate_reasoning method.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the base reasoning engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.engine_name = self.__class__.__name__
        self.logger = logging.getLogger(f"{__name__}.{self.engine_name}")
        
        # Initialize engine-specific configuration
        self._initialize_config()
    
    def _initialize_config(self):
        """Initialize engine-specific configuration. Override in subclasses."""
        pass
    
    @abstractmethod
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate reasoning text for the current data point.
        
        Args:
            current_data: Current row data with all features
            context: Historical context and analysis data
            
        Returns:
            Professional reasoning text for this engine's domain
        """
        pass
    
    def validate_input_data(self, current_data: pd.Series) -> bool:
        """
        Validate that required data is present for reasoning generation.
        
        Args:
            current_data: Current row data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = self.get_required_columns()
        
        for col in required_columns:
            if col not in current_data.index:
                self.logger.warning(f"Missing required column: {col}")
                return False
            
            if pd.isna(current_data[col]):
                self.logger.warning(f"NaN value in required column: {col}")
                return False
        
        return True
    
    def get_required_columns(self) -> list:
        """
        Get list of required columns for this engine.
        Override in subclasses to specify requirements.
        
        Returns:
            List of required column names
        """
        return []
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about this reasoning engine.
        
        Returns:
            Dictionary with engine information
        """
        return {
            'name': self.engine_name,
            'required_columns': self.get_required_columns(),
            'config': self.config
        }
    
    def _format_percentage(self, value: float, decimals: int = 1) -> str:
        """
        Format a decimal value as a percentage string.
        
        Args:
            value: Decimal value to format
            decimals: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        if pd.isna(value):
            return "unknown"
        return f"{value:.{decimals}f}%"
    
    def _format_relative_level(self, distance: float) -> str:
        """
        Format distance to support/resistance as relative description.
        
        Args:
            distance: Distance percentage
            
        Returns:
            Relative description string
        """
        if pd.isna(distance):
            return "neutral zone"
        
        if distance < 0.5:
            return "immediate proximity"
        elif distance < 1.0:
            return "close proximity"
        elif distance < 2.0:
            return "moderate distance"
        else:
            return "significant distance"
    
    def _get_strength_descriptor(self, value: float, thresholds: Dict[str, float]) -> str:
        """
        Get strength descriptor based on value and thresholds.
        
        Args:
            value: Value to evaluate
            thresholds: Dictionary with 'weak', 'moderate', 'strong' thresholds
            
        Returns:
            Strength descriptor string
        """
        if pd.isna(value):
            return "neutral"
        
        if value >= thresholds.get('strong', 0.7):
            return "strong"
        elif value >= thresholds.get('moderate', 0.4):
            return "moderate"
        else:
            return "weak"
    
    def _safe_get_value(self, data: pd.Series, column: str, default: Any = None) -> Any:
        """
        Safely get value from pandas Series with default fallback.
        
        Args:
            data: Pandas Series to get value from
            column: Column name to retrieve
            default: Default value if column missing or NaN
            
        Returns:
            Value from series or default
        """
        if column not in data.index:
            return default
        
        value = data[column]
        if pd.isna(value):
            return default
        
        return value
    
    def _log_reasoning_generation(self, reasoning_length: int, data_quality: str = "good"):
        """
        Log reasoning generation metrics.
        
        Args:
            reasoning_length: Length of generated reasoning text
            data_quality: Quality assessment of input data
        """
        self.logger.debug(
            f"{self.engine_name} generated {reasoning_length} character reasoning "
            f"with {data_quality} data quality"
        )
