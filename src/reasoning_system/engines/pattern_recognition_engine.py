#!/usr/bin/env python3
"""
Pattern Recognition Engine
=========================

Generates sophisticated pattern recognition reasoning using professional
trading language and relative price references.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


class PatternRecognitionEngine(BaseReasoningEngine):
    """
    Generates pattern recognition reasoning text using professional trading language.
    
    Analyzes candlestick patterns, chart patterns, and market structure while
    maintaining relative price references and incorporating historical context.
    """
    
    def _initialize_config(self):
        """Initialize pattern recognition specific configuration."""
        self.pattern_templates = self._load_pattern_templates()
        self.structure_templates = self._load_structure_templates()
        
        # Configuration for pattern analysis
        self.config.setdefault('support_resistance_threshold', 1.5)  # % distance threshold
        self.config.setdefault('pattern_confidence_threshold', 0.7)
        self.config.setdefault('use_historical_context', True)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for pattern recognition."""
        return [
            'open', 'high', 'low', 'close',
            'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate pattern recognition reasoning text.
        
        Args:
            current_data: Current row data with all features
            context: Historical context from HistoricalContextManager
            
        Returns:
            Professional pattern recognition reasoning text
        """
        if not self.validate_input_data(current_data):
            return self._get_fallback_reasoning(current_data, context)
        
        try:
            # Identify active patterns
            active_patterns = self._identify_active_patterns(current_data)
            
            # Analyze pattern context
            pattern_context = self._analyze_pattern_context(current_data, context)
            
            # Generate reasoning text
            reasoning = self._construct_pattern_reasoning(active_patterns, pattern_context, context)
            
            self._log_reasoning_generation(len(reasoning), "good")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in pattern recognition reasoning: {str(e)}")
            return self._get_fallback_reasoning(current_data, context)
    
    def _load_pattern_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load candlestick pattern templates with descriptions and implications."""
        return {
            'doji': {
                'description': "indecision pattern with equal open and close levels",
                'implications': ["market uncertainty", "potential reversal", "consolidation phase"],
                'strength': "neutral",
                'context_dependent': True
            },
            'hammer': {
                'description': "bullish reversal pattern with long lower shadow",
                'implications': ["rejection of lower prices", "buying interest", "potential upward momentum"],
                'strength': "bullish",
                'context_dependent': True
            },
            'bullish_engulfing': {
                'description': "strong bullish reversal pattern engulfing previous bearish candle",
                'implications': ["shift in sentiment", "institutional buying", "momentum reversal"],
                'strength': "bullish",
                'context_dependent': False
            },
            'bearish_engulfing': {
                'description': "strong bearish reversal pattern engulfing previous bullish candle",
                'implications': ["shift in sentiment", "institutional selling", "momentum reversal"],
                'strength': "bearish",
                'context_dependent': False
            }
        }
    
    def _load_structure_templates(self) -> Dict[str, str]:
        """Load market structure analysis templates."""
        return {
            'near_support': "positioned near key support zone with {strength} historical significance",
            'near_resistance': "developing at important resistance level showing {strength} rejection characteristics",
            'neutral_zone': "trading within established range boundaries with neutral structural implications",
            'breakout_zone': "approaching critical structural level with potential for significant movement",
            'confluence_zone': "located at confluence of multiple technical levels enhancing setup significance"
        }
    
    def _identify_active_patterns(self, current_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Identify currently active candlestick patterns with confidence scores.
        
        Args:
            current_data: Current row data
            
        Returns:
            List of active pattern dictionaries
        """
        active_patterns = []
        pattern_columns = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']
        
        for pattern in pattern_columns:
            if pattern in current_data.index and current_data[pattern] == 1:
                pattern_info = {
                    'name': pattern,
                    'template': self.pattern_templates[pattern],
                    'confidence': self._calculate_pattern_confidence(pattern, current_data)
                }
                active_patterns.append(pattern_info)
        
        # Sort by confidence (highest first)
        active_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        
        return active_patterns
    
    def _calculate_pattern_confidence(self, pattern_name: str, current_data: pd.Series) -> float:
        """
        Calculate confidence score for a pattern based on context.
        
        Args:
            pattern_name: Name of the pattern
            current_data: Current row data
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        base_confidence = 0.7  # Base confidence for pattern presence
        
        # Adjust based on body size (stronger patterns have larger bodies)
        if 'body_size' in current_data.index:
            body_size = self._safe_get_value(current_data, 'body_size', 0.5)
            if body_size > 1.0:  # Larger body increases confidence
                base_confidence += 0.1
            elif body_size < 0.3:  # Very small body decreases confidence
                base_confidence -= 0.1
        
        # Adjust based on volume (if available in future)
        # Volume confirmation would increase confidence
        
        # Adjust based on ATR (volatility context)
        if 'atr' in current_data.index:
            atr = self._safe_get_value(current_data, 'atr', 1.0)
            hl_range = self._safe_get_value(current_data, 'hl_range', 1.0)
            
            # Pattern in high volatility environment gets slight boost
            if hl_range > atr * 1.5:
                base_confidence += 0.05
        
        return min(1.0, max(0.3, base_confidence))  # Clamp between 0.3 and 1.0
    
    def _analyze_pattern_context(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structural and historical context around current patterns.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Pattern context analysis
        """
        analysis = {
            'structural_position': 'neutral_zone',
            'support_proximity': None,
            'resistance_proximity': None,
            'trend_alignment': 'neutral',
            'volatility_environment': 'normal',
            'historical_pattern_bias': 'neutral'
        }
        
        # Support/Resistance proximity analysis
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        threshold = self.config['support_resistance_threshold']
        
        if support_dist < threshold:
            analysis['structural_position'] = 'near_support'
            analysis['support_proximity'] = {
                'distance': support_dist,
                'strength': 'strong' if support_dist < 1.0 else 'moderate'
            }
        elif resistance_dist < threshold:
            analysis['structural_position'] = 'near_resistance'
            analysis['resistance_proximity'] = {
                'distance': resistance_dist,
                'strength': 'strong' if resistance_dist < 1.0 else 'moderate'
            }
        elif support_dist < threshold * 2 and resistance_dist < threshold * 2:
            analysis['structural_position'] = 'confluence_zone'
        
        # Trend alignment
        trend_direction = self._safe_get_value(current_data, 'trend_direction', 0)
        if trend_direction == 1:
            analysis['trend_alignment'] = 'bullish'
        elif trend_direction == -1:
            analysis['trend_alignment'] = 'bearish'
        
        # Volatility environment
        vol_analysis = context.get('volatility_analysis', {})
        analysis['volatility_environment'] = vol_analysis.get('level', 'normal')
        
        # Historical pattern bias
        pattern_freq = context.get('pattern_frequency', {})
        analysis['historical_pattern_bias'] = pattern_freq.get('pattern_bias', 'neutral')
        
        return analysis
    
    def _construct_pattern_reasoning(self, active_patterns: List[Dict], 
                                   pattern_context: Dict, context: Dict) -> str:
        """
        Construct the final pattern recognition reasoning text.
        
        Args:
            active_patterns: List of active pattern information
            pattern_context: Pattern context analysis
            context: Historical context
            
        Returns:
            Complete pattern reasoning text
        """
        if not active_patterns:
            return self._generate_no_pattern_reasoning(pattern_context, context)
        
        reasoning_parts = []
        
        # Primary pattern analysis
        primary_pattern = active_patterns[0]
        template = primary_pattern['template']
        confidence = primary_pattern['confidence']
        
        # Pattern identification with confidence
        confidence_desc = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "developing"
        reasoning_parts.append(
            f"Current candle formation displays {confidence_desc} {template['description']}"
        )
        
        # Structural context integration
        structural_pos = pattern_context['structural_position']
        if structural_pos in self.structure_templates:
            if structural_pos == 'near_support' and pattern_context['support_proximity']:
                strength = pattern_context['support_proximity']['strength']
                reasoning_parts.append(
                    self.structure_templates[structural_pos].format(strength=strength)
                )
            elif structural_pos == 'near_resistance' and pattern_context['resistance_proximity']:
                strength = pattern_context['resistance_proximity']['strength']
                reasoning_parts.append(
                    self.structure_templates[structural_pos].format(strength=strength)
                )
            else:
                reasoning_parts.append(self.structure_templates[structural_pos])
        
        # Historical context integration
        hist_bias = pattern_context['historical_pattern_bias']
        if hist_bias != 'neutral':
            reasoning_parts.append(
                f"Recent pattern analysis reveals {hist_bias} bias over the observation period"
            )
        
        # Pattern implications
        implications = template['implications']
        if len(implications) >= 2:
            reasoning_parts.append(
                f"suggesting {implications[0]} with potential for {implications[1]}"
            )
        
        # Multiple pattern confluence
        if len(active_patterns) > 1:
            secondary_patterns = [p['name'].replace('_', ' ') for p in active_patterns[1:]]
            reasoning_parts.append(
                f"Additional pattern confluence from {', '.join(secondary_patterns)} strengthens the technical setup"
            )
        
        # Trend alignment consideration
        trend_alignment = pattern_context['trend_alignment']
        pattern_strength = template['strength']
        
        if trend_alignment != 'neutral' and pattern_strength != 'neutral':
            if trend_alignment == pattern_strength:
                reasoning_parts.append(
                    f"Pattern aligns with prevailing {trend_alignment} trend enhancing probability of success"
                )
            else:
                reasoning_parts.append(
                    f"Pattern suggests potential reversal of current {trend_alignment} trend"
                )
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_no_pattern_reasoning(self, pattern_context: Dict, context: Dict) -> str:
        """
        Generate reasoning when no specific patterns are present.
        
        Args:
            pattern_context: Pattern context analysis
            context: Historical context
            
        Returns:
            No-pattern reasoning text
        """
        reasoning_parts = []
        
        reasoning_parts.append(
            "Current price action shows standard candle formation without distinct pattern characteristics"
        )
        
        # Focus on market structure
        structural_pos = pattern_context['structural_position']
        if structural_pos == 'near_support':
            reasoning_parts.append(
                "positioned near key support zone requiring attention to potential bounce scenarios"
            )
        elif structural_pos == 'near_resistance':
            reasoning_parts.append(
                "approaching significant resistance level with focus on breakout or rejection dynamics"
            )
        elif structural_pos == 'confluence_zone':
            reasoning_parts.append(
                "trading within confluence zone of multiple technical levels"
            )
        else:
            reasoning_parts.append(
                "trading within established range boundaries with neutral pattern implications"
            )
        
        # Market regime context
        regime = context.get('market_regime', 'consolidation')
        reasoning_parts.append(f"consistent with current {regime} market environment")
        
        # Historical context
        hist_bias = pattern_context.get('historical_pattern_bias', 'neutral')
        if hist_bias != 'neutral':
            reasoning_parts.append(
                f"while recent pattern history suggests underlying {hist_bias} bias"
            )
        
        return ". ".join(reasoning_parts) + "."
    
    def _get_fallback_reasoning(self, current_data: pd.Series, context: Dict) -> str:
        """
        Generate fallback reasoning when main analysis fails.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Fallback reasoning text
        """
        return (
            "Current market structure shows standard price action without distinct pattern characteristics. "
            "Price action remains within normal parameters consistent with prevailing market conditions. "
            "Technical setup requires additional confirmation for directional bias assessment."
        )
