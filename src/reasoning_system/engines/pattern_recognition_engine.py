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
            
            # Analyze pattern context with current data
            pattern_context = self._analyze_pattern_context(current_data, context)
            pattern_context['current_data'] = current_data  # Pass current data for analysis
            pattern_context['historical_data'] = context.get('historical_data')  # Pass historical data

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
        Identify currently active patterns using both candlestick flags and OHLC analysis.

        Args:
            current_data: Current row data

        Returns:
            List of active pattern dictionaries
        """
        active_patterns = []

        # Check candlestick patterns from boolean flags
        candlestick_patterns = self._identify_candlestick_patterns(current_data)
        active_patterns.extend(candlestick_patterns)

        # Analyze OHLC data for additional patterns
        ohlc_patterns = self._identify_ohlc_patterns(current_data)
        active_patterns.extend(ohlc_patterns)

        # Sort by confidence (highest first)
        active_patterns.sort(key=lambda x: x['confidence'], reverse=True)

        return active_patterns

    def _identify_candlestick_patterns(self, current_data: pd.Series) -> List[Dict[str, Any]]:
        """Identify candlestick patterns from boolean flags."""
        patterns = []
        pattern_columns = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']

        for pattern in pattern_columns:
            if pattern in current_data.index and current_data[pattern] == 1:
                pattern_info = {
                    'name': pattern,
                    'template': self.pattern_templates[pattern],
                    'confidence': self._calculate_pattern_confidence(pattern, current_data),
                    'type': 'candlestick'
                }
                patterns.append(pattern_info)

        return patterns

    def _identify_ohlc_patterns(self, current_data: pd.Series) -> List[Dict[str, Any]]:
        """Identify patterns using actual OHLC data analysis."""
        patterns = []

        # Get OHLC values
        open_price = self._safe_get_value(current_data, 'open', 0)
        high_price = self._safe_get_value(current_data, 'high', 0)
        low_price = self._safe_get_value(current_data, 'low', 0)
        close_price = self._safe_get_value(current_data, 'close', 0)

        if not all([open_price, high_price, low_price, close_price]):
            return patterns

        # Analyze price action patterns
        price_patterns = self._analyze_price_action_patterns(
            open_price, high_price, low_price, close_price, current_data
        )
        patterns.extend(price_patterns)

        return patterns

    def _analyze_price_action_patterns(self, open_price: float, high_price: float,
                                     low_price: float, close_price: float,
                                     current_data: pd.Series) -> List[Dict[str, Any]]:
        """Analyze price action to identify specific patterns."""
        patterns = []

        # Calculate key metrics
        body_size = abs(close_price - open_price)
        total_range = high_price - low_price
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price

        # Avoid division by zero
        if total_range == 0:
            return patterns

        # Pattern 1: Strong directional move
        if body_size / total_range > 0.7:  # Body is >70% of total range
            direction = "bullish" if close_price > open_price else "bearish"
            patterns.append({
                'name': f'strong_{direction}_move',
                'template': {
                    'description': f"strong {direction} directional move with dominant body formation",
                    'implications': [f"{direction} momentum", "directional conviction", "trend continuation potential"],
                    'strength': direction,
                    'context_dependent': False
                },
                'confidence': 0.8,
                'type': 'price_action'
            })

        # Pattern 2: Rejection pattern (long shadows)
        elif max(upper_shadow, lower_shadow) / total_range > 0.6:  # Long shadow >60% of range
            if upper_shadow > lower_shadow:
                patterns.append({
                    'name': 'upper_rejection',
                    'template': {
                        'description': "rejection pattern with significant upper shadow indicating selling pressure",
                        'implications': ["resistance rejection", "selling interest", "potential downward pressure"],
                        'strength': "bearish",
                        'context_dependent': True
                    },
                    'confidence': 0.7,
                    'type': 'price_action'
                })
            else:
                patterns.append({
                    'name': 'lower_rejection',
                    'template': {
                        'description': "rejection pattern with significant lower shadow indicating buying support",
                        'implications': ["support holding", "buying interest", "potential upward bounce"],
                        'strength': "bullish",
                        'context_dependent': True
                    },
                    'confidence': 0.7,
                    'type': 'price_action'
                })

        # Pattern 3: Consolidation/indecision
        elif body_size / total_range < 0.3:  # Small body <30% of range
            patterns.append({
                'name': 'consolidation_pattern',
                'template': {
                    'description': "consolidation pattern with small body indicating market indecision",
                    'implications': ["market uncertainty", "potential breakout setup", "range-bound behavior"],
                    'strength': "neutral",
                    'context_dependent': True
                },
                'confidence': 0.6,
                'type': 'price_action'
            })

        return patterns

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
        
        # Pattern identification with specific price data from pattern_context
        current_data = pattern_context.get('current_data')
        if current_data is not None:
            close = self._safe_get_value(current_data, 'close', 0)
            atr = self._safe_get_value(current_data, 'atr', 0)
            rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        else:
            close = atr = rsi = 0

        confidence_desc = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "developing"
        reasoning_parts.append(
            f"Current candle formation displays {confidence_desc} {template['description']}"
        )

        # Add comprehensive technical context with actual values
        if current_data is not None and close > 0:
            # Price movement analysis using ATR
            if atr > 0:
                open_price = self._safe_get_value(current_data, 'open', close)
                price_movement_ratio = abs(close - open_price) / atr
                movement_desc = self._get_movement_description(price_movement_ratio)
                reasoning_parts.append(f"Price movement at {price_movement_ratio:.1f}x ATR suggests {movement_desc} phase")

            # Support/resistance analysis with actual levels
            support_level = self._safe_get_value(current_data, 'support_level', 0)
            resistance_level = self._safe_get_value(current_data, 'resistance_level', 0)

            if support_level > 0:
                support_distance = abs(close - support_level)
                reasoning_parts.append(f"Positioned near key support zone with {support_distance:.1f} point distance")

            # RSI momentum analysis
            if rsi != 50:
                rsi_desc = self._get_detailed_rsi_description(rsi)
                reasoning_parts.append(f"RSI-14 at {rsi:.1f} {rsi_desc}")

            # Additional momentum indicators
            stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
            williams_r = self._safe_get_value(current_data, 'williams_r', -50)

            if stoch_k > 80 or stoch_k < 20:
                stoch_condition = "overbought" if stoch_k > 80 else "oversold"
                reasoning_parts.append(f"Stochastic at {stoch_k:.1f} indicates {stoch_condition} conditions")

            if williams_r > -20 or williams_r < -80:
                williams_condition = "overbought" if williams_r > -20 else "oversold"
                reasoning_parts.append(f"Williams %R at {williams_r:.1f} confirms {williams_condition} environment")

        # Structural context with reduced repetition
        structural_pos = pattern_context['structural_position']
        if structural_pos == 'near_support' and pattern_context['support_proximity']:
            distance = pattern_context['support_proximity']['distance']
            reasoning_parts.append(f"Positioned near key support zone with {distance:.1f} point distance")
        elif structural_pos == 'near_resistance' and pattern_context['resistance_proximity']:
            distance = pattern_context['resistance_proximity']['distance']
            reasoning_parts.append(f"Developing at resistance level with {distance:.1f} point distance")
        
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
        
        # Generate more specific analysis based on actual price action
        price_action_desc = self._analyze_current_price_action(pattern_context)
        reasoning_parts.append(price_action_desc)
        
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

    def _analyze_current_price_action(self, pattern_context: Dict) -> str:
        """Analyze current price action using actual data like a real trader."""
        current_data = pattern_context.get('current_data')
        if current_data is None:
            return "Price action analysis requires current market data"

        # Get actual OHLC values
        open_price = self._safe_get_value(current_data, 'open', 0)
        high_price = self._safe_get_value(current_data, 'high', 0)
        low_price = self._safe_get_value(current_data, 'low', 0)
        close_price = self._safe_get_value(current_data, 'close', 0)

        # Get actual indicator values using correct column names
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        atr = self._safe_get_value(current_data, 'atr', 0)
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)

        # Analyze like a real trader
        analysis_parts = []

        # Price position analysis
        if support_dist < 20:
            analysis_parts.append(f"Price at {close_price:.1f} is testing support levels with {support_dist:.1f} point distance")
        elif resistance_dist < 20:
            analysis_parts.append(f"Price at {close_price:.1f} is approaching resistance with {resistance_dist:.1f} point gap")
        else:
            analysis_parts.append(f"Price at {close_price:.1f} is trading in middle range between key levels")

        # RSI momentum analysis
        if rsi > 70:
            analysis_parts.append(f"RSI at {rsi:.1f} indicates overbought momentum suggesting potential pullback")
        elif rsi < 30:
            analysis_parts.append(f"RSI at {rsi:.1f} shows oversold conditions with potential for bounce")
        else:
            analysis_parts.append(f"RSI at {rsi:.1f} reflects balanced momentum without extreme readings")

        # Volatility context
        if atr > 0:
            body_size = abs(close_price - open_price)
            volatility_ratio = body_size / atr if atr > 0 else 0
            if volatility_ratio > 1.5:
                analysis_parts.append(f"Current candle shows {volatility_ratio:.1f}x normal volatility indicating strong directional move")
            elif volatility_ratio < 0.5:
                analysis_parts.append(f"Price movement at {volatility_ratio:.1f}x ATR suggests consolidation phase")

        # Add historical context if available
        historical_context = self._get_historical_context(current_data, pattern_context)
        if historical_context:
            analysis_parts.append(historical_context)

        return ". ".join(analysis_parts)

    def _get_historical_context(self, current_data, pattern_context: Dict) -> str:
        """Get historical context for current price action."""
        # Get historical data from context if available
        historical_data = pattern_context.get('historical_data')
        if historical_data is None or len(historical_data) < 10:
            return ""

        try:
            # Analyze last 10 candles for pattern context
            recent_closes = historical_data['close'].tail(10).values
            recent_rsi = historical_data['rsi_14'].tail(10).values
            current_close = self._safe_get_value(current_data, 'close', 0)
            current_rsi = self._safe_get_value(current_data, 'rsi_14', 50)

            # Compare current vs recent average
            avg_close = recent_closes.mean()
            avg_rsi = recent_rsi.mean()

            context_parts = []

            # Price trend context
            if current_close > avg_close * 1.01:
                price_change = ((current_close - avg_close) / avg_close) * 100
                context_parts.append(f"Price {price_change:.1f}% above 10-period average suggesting upward momentum")
            elif current_close < avg_close * 0.99:
                price_change = ((avg_close - current_close) / avg_close) * 100
                context_parts.append(f"Price {price_change:.1f}% below 10-period average indicating downward pressure")

            # RSI momentum context
            if current_rsi > avg_rsi + 5:
                context_parts.append(f"RSI strengthening from {avg_rsi:.1f} average to current {current_rsi:.1f}")
            elif current_rsi < avg_rsi - 5:
                context_parts.append(f"RSI weakening from {avg_rsi:.1f} average to current {current_rsi:.1f}")

            return ". ".join(context_parts) if context_parts else ""

        except Exception as e:
            return ""

    def _get_fallback_reasoning(self, current_data: pd.Series, context: Dict) -> str:
        """
        Generate fallback reasoning when main analysis fails.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Fallback reasoning text
        """
        # Get more specific fallback based on available data
        if hasattr(current_data, 'index') and 'close' in current_data.index:
            close_price = self._safe_get_value(current_data, 'close', 0)
            support_dist = self._safe_get_value(current_data, 'support_distance', 999)
            resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)

            if support_dist < resistance_dist and support_dist < 50:
                return (
                    "Current market structure shows price consolidation near support levels. "
                    "Price action suggests potential for bounce or breakdown depending on volume confirmation. "
                    "Technical setup favors defensive positioning until directional clarity emerges."
                )
            elif resistance_dist < support_dist and resistance_dist < 50:
                return (
                    "Current market structure shows price testing resistance levels. "
                    "Price action indicates potential for rejection or breakthrough based on momentum. "
                    "Technical setup requires confirmation of breakout sustainability."
                )

        return (
            "Current market structure shows balanced price action within established ranges. "
            "Price dynamics suggest consolidation phase with potential for directional resolution. "
            "Technical setup requires additional momentum confirmation for position initiation."
        )

    def _get_volatility_description(self, ratio: float) -> str:
        """Get volatility description based on ATR ratio."""
        if ratio < 0.5:
            return "consolidation"
        elif ratio < 1.0:
            return "normal movement"
        elif ratio < 2.0:
            return "elevated volatility"
        else:
            return "high volatility"

    def _get_rsi_description(self, rsi: float) -> str:
        """Get RSI description with market implications."""
        if rsi < 30:
            return "shows oversold conditions with potential for bounce"
        elif rsi < 45:
            return "reflects balanced momentum without extreme readings"
        elif rsi < 55:
            return "indicates neutral momentum conditions"
        elif rsi < 70:
            return "shows healthy upward momentum"
        else:
            return "indicates overbought conditions suggesting caution"

    def _safe_get_value(self, data, column: str, default_value):
        """Safely get value from pandas Series with fallback to default."""
        try:
            if hasattr(data, 'get'):
                return data.get(column, default_value)
            elif hasattr(data, column):
                return getattr(data, column)
            else:
                return default_value
        except (KeyError, AttributeError, TypeError):
            return default_value

    def _get_movement_description(self, ratio: float) -> str:
        """Get movement description based on ATR ratio."""
        if ratio > 2.0:
            return "high volatility"
        elif ratio > 1.0:
            return "elevated volatility"
        elif ratio > 0.5:
            return "normal movement"
        elif ratio > 0.2:
            return "consolidation"
        else:
            return "range"

    def _get_detailed_rsi_description(self, rsi: float) -> str:
        """Get detailed RSI condition description."""
        if rsi > 80:
            return "indicates extremely overbought conditions suggesting caution"
        elif rsi > 70:
            return "shows overbought momentum suggesting potential pullback"
        elif rsi > 60:
            return "demonstrates healthy upward momentum"
        elif rsi > 50:
            return "shows neutral momentum conditions"
        elif rsi > 40:
            return "reflects balanced momentum without extreme readings"
        elif rsi > 30:
            return "indicates oversold momentum with potential for bounce"
        elif rsi > 20:
            return "shows oversold conditions with potential reversal"
        else:
            return "demonstrates extremely oversold conditions suggesting bounce"
