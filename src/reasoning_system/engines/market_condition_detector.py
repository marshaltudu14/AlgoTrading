#!/usr/bin/env python3
"""
Market Condition Detection Engine
================================

Detects trending/ranging/volatile markets with historical duration and
generates confidence based on market conditions, not signals.

Key Features:
- Detects market regimes (trending, ranging, volatile, low volatility)
- Analyzes regime duration and stability
- Generates confidence scores based on market conditions
- Provides historical context for regime changes
- NO signal column usage (prevents data leakage)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    """Market condition data structure."""
    current_trend_direction: str  # "bullish", "bearish", "sideways"
    historical_trend_strength: str  # Based on past periods
    volatility_level: str  # "low", "medium", "high"
    momentum_strength: str  # "weak", "moderate", "strong"
    support_resistance_context: str
    regime_duration: int  # How long current regime has lasted
    regime_stability: float  # How stable the current regime is (0-1)


class MarketConditionDetector(BaseReasoningEngine):
    """
    Market condition detection engine that identifies market regimes
    and generates confidence based on market conditions, not signals.
    """
    
    def _initialize_config(self):
        """Initialize market condition detection configuration."""
        # Regime detection thresholds
        self.config.setdefault('trend_threshold', 0.6)
        self.config.setdefault('volatility_high_threshold', 0.03)
        self.config.setdefault('volatility_low_threshold', 0.015)
        self.config.setdefault('momentum_threshold', 0.1)
        
        # Regime stability thresholds
        self.config.setdefault('stability_threshold', 0.7)
        self.config.setdefault('regime_change_threshold', 0.3)
        
        # Historical analysis periods
        self.config.setdefault('regime_analysis_period', 50)
        self.config.setdefault('stability_analysis_period', 20)
        
        logger.info("MarketConditionDetector initialized with regime detection thresholds")
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for market condition detection."""
        return [
            'trend_strength', 'trend_direction', 'volatility_20', 'atr',
            'rsi_14', 'macd_histogram', 'bb_width', 'bb_position',
            'support_distance', 'resistance_distance', 'close', 'high', 'low'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate market condition reasoning without using signal column.
        
        Args:
            current_data: Current row data
            context: Historical context from HistoricalContextManager
            
        Returns:
            Market condition reasoning text
        """
        try:
            # Detect current market regime
            current_regime = self._detect_current_regime(current_data)
            
            # Analyze regime characteristics
            regime_analysis = self._analyze_regime_characteristics(current_data, context)
            
            # Assess regime stability and duration
            stability_analysis = self._assess_regime_stability(current_data, context)
            
            # Generate confidence score based on market conditions
            confidence_analysis = self._generate_market_confidence(
                current_regime, regime_analysis, stability_analysis
            )
            
            # Construct comprehensive reasoning
            reasoning = self._construct_condition_reasoning(
                current_regime, regime_analysis, stability_analysis, confidence_analysis
            )
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in market condition detection: {str(e)}")
            return self._get_fallback_reasoning()
    
    def _detect_current_regime(self, current_data: pd.Series) -> Dict[str, Any]:
        """Detect current market regime."""
        regime = {
            'primary_regime': 'transitional',
            'secondary_characteristics': [],
            'regime_strength': 0.5,
            'regime_clarity': 'moderate'
        }
        
        # Analyze trend characteristics
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        volatility = self._safe_get_value(current_data, 'volatility_20', 0.02)
        
        # Determine primary regime
        if abs(trend_strength) > self.config['trend_threshold']:
            if volatility > self.config['volatility_high_threshold']:
                regime['primary_regime'] = 'volatile_trending'
            else:
                regime['primary_regime'] = 'trending'
            regime['regime_strength'] = abs(trend_strength)
        elif volatility > self.config['volatility_high_threshold']:
            regime['primary_regime'] = 'volatile_ranging'
            regime['regime_strength'] = min(volatility / 0.03, 1.0)
        elif volatility < self.config['volatility_low_threshold']:
            regime['primary_regime'] = 'low_volatility'
            regime['regime_strength'] = max(1.0 - volatility / 0.015, 0.5)
        else:
            regime['primary_regime'] = 'ranging'
            regime['regime_strength'] = 0.6
        
        # Determine regime clarity
        if regime['regime_strength'] > 0.8:
            regime['regime_clarity'] = 'high'
        elif regime['regime_strength'] < 0.4:
            regime['regime_clarity'] = 'low'
        
        # Add secondary characteristics
        regime['secondary_characteristics'] = self._identify_secondary_characteristics(current_data)
        
        return regime
    
    def _identify_secondary_characteristics(self, current_data: pd.Series) -> List[str]:
        """Identify secondary market characteristics."""
        characteristics = []
        
        # Momentum characteristics
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)
        
        if rsi_14 > 70 or rsi_14 < 30:
            characteristics.append('momentum_extreme')
        
        if abs(macd_histogram) > 0.1:
            characteristics.append('strong_momentum')
        
        # Volatility characteristics
        bb_width = self._safe_get_value(current_data, 'bb_width', 0.04)
        if bb_width < 0.02:
            characteristics.append('volatility_squeeze')
        elif bb_width > 0.06:
            characteristics.append('volatility_expansion')
        
        # Support/Resistance characteristics
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        if support_dist < 1.0 or resistance_dist < 1.0:
            characteristics.append('near_key_level')
        
        return characteristics[:3]  # Limit to 3 characteristics
    
    def _analyze_regime_characteristics(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detailed regime characteristics."""
        analysis = {
            'trend_direction': self._determine_trend_direction(current_data),
            'trend_strength_category': self._categorize_trend_strength(current_data),
            'volatility_category': self._categorize_volatility(current_data),
            'momentum_category': self._categorize_momentum(current_data),
            'market_structure': self._analyze_market_structure(current_data),
            'regime_maturity': 'developing'
        }
        
        # Determine regime maturity (simplified without full historical data)
        trend_strength = abs(self._safe_get_value(current_data, 'trend_strength', 0))
        if trend_strength > 0.8:
            analysis['regime_maturity'] = 'mature'
        elif trend_strength < 0.3:
            analysis['regime_maturity'] = 'early'
        
        return analysis
    
    def _determine_trend_direction(self, current_data: pd.Series) -> str:
        """Determine trend direction."""
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        
        if trend_strength > 0.3:
            return 'bullish'
        elif trend_strength < -0.3:
            return 'bearish'
        else:
            return 'sideways'
    
    def _categorize_trend_strength(self, current_data: pd.Series) -> str:
        """Categorize trend strength."""
        trend_strength = abs(self._safe_get_value(current_data, 'trend_strength', 0))
        
        if trend_strength > 0.8:
            return 'very_strong'
        elif trend_strength > 0.6:
            return 'strong'
        elif trend_strength > 0.4:
            return 'moderate'
        elif trend_strength > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _categorize_volatility(self, current_data: pd.Series) -> str:
        """Categorize volatility level."""
        volatility = self._safe_get_value(current_data, 'volatility_20', 0.02)
        
        if volatility > 0.04:
            return 'very_high'
        elif volatility > 0.03:
            return 'high'
        elif volatility > 0.02:
            return 'moderate'
        elif volatility > 0.015:
            return 'low'
        else:
            return 'very_low'
    
    def _categorize_momentum(self, current_data: pd.Series) -> str:
        """Categorize momentum strength."""
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)
        
        # Combine RSI and MACD for momentum assessment
        rsi_momentum = abs(rsi_14 - 50) / 50  # Normalized RSI momentum
        macd_momentum = min(abs(macd_histogram) * 10, 1.0)  # Normalized MACD momentum
        
        combined_momentum = (rsi_momentum + macd_momentum) / 2
        
        if combined_momentum > 0.8:
            return 'very_strong'
        elif combined_momentum > 0.6:
            return 'strong'
        elif combined_momentum > 0.4:
            return 'moderate'
        elif combined_momentum > 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _analyze_market_structure(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze market structure characteristics."""
        structure = {
            'support_proximity': 'distant',
            'resistance_proximity': 'distant',
            'range_position': 'middle',
            'structure_clarity': 'moderate'
        }
        
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        # Analyze support proximity
        if support_dist < 0.5:
            structure['support_proximity'] = 'very_close'
        elif support_dist < 1.0:
            structure['support_proximity'] = 'close'
        elif support_dist < 2.0:
            structure['support_proximity'] = 'moderate'
        
        # Analyze resistance proximity
        if resistance_dist < 0.5:
            structure['resistance_proximity'] = 'very_close'
        elif resistance_dist < 1.0:
            structure['resistance_proximity'] = 'close'
        elif resistance_dist < 2.0:
            structure['resistance_proximity'] = 'moderate'
        
        # Determine range position
        if support_dist < resistance_dist:
            if support_dist < 1.0:
                structure['range_position'] = 'lower'
        else:
            if resistance_dist < 1.0:
                structure['range_position'] = 'upper'
        
        # Assess structure clarity
        if min(support_dist, resistance_dist) < 1.0:
            structure['structure_clarity'] = 'high'
        elif min(support_dist, resistance_dist) > 3.0:
            structure['structure_clarity'] = 'low'
        
        return structure

    def _assess_regime_stability(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess regime stability and duration."""
        stability_analysis = {
            'regime_duration': 0,
            'stability_score': 0.5,
            'regime_confidence': 'moderate',
            'change_probability': 'low',
            'stability_factors': []
        }

        # Simplified stability analysis (would need historical data for full analysis)
        trend_strength = abs(self._safe_get_value(current_data, 'trend_strength', 0))
        volatility = self._safe_get_value(current_data, 'volatility_20', 0.02)

        # Calculate stability score based on current conditions
        stability_factors = []

        # Trend consistency factor
        if trend_strength > 0.7:
            stability_factors.append(0.8)
            stability_analysis['stability_factors'].append('strong_trend_consistency')
        elif trend_strength > 0.4:
            stability_factors.append(0.6)
            stability_analysis['stability_factors'].append('moderate_trend_consistency')
        else:
            stability_factors.append(0.3)
            stability_analysis['stability_factors'].append('weak_trend_consistency')

        # Volatility stability factor
        if 0.015 < volatility < 0.03:  # Normal volatility range
            stability_factors.append(0.7)
            stability_analysis['stability_factors'].append('normal_volatility_regime')
        elif volatility < 0.015:
            stability_factors.append(0.5)  # Low vol can be unstable
            stability_analysis['stability_factors'].append('low_volatility_regime')
        else:
            stability_factors.append(0.4)  # High vol is less stable
            stability_analysis['stability_factors'].append('high_volatility_regime')

        # Momentum consistency factor
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        if 30 < rsi_14 < 70:  # Normal momentum range
            stability_factors.append(0.7)
            stability_analysis['stability_factors'].append('balanced_momentum')
        else:
            stability_factors.append(0.4)  # Extreme momentum less stable
            stability_analysis['stability_factors'].append('extreme_momentum')

        # Calculate overall stability score
        stability_analysis['stability_score'] = np.mean(stability_factors)

        # Determine regime confidence
        if stability_analysis['stability_score'] > 0.7:
            stability_analysis['regime_confidence'] = 'high'
            stability_analysis['change_probability'] = 'low'
        elif stability_analysis['stability_score'] > 0.5:
            stability_analysis['regime_confidence'] = 'moderate'
            stability_analysis['change_probability'] = 'moderate'
        else:
            stability_analysis['regime_confidence'] = 'low'
            stability_analysis['change_probability'] = 'high'

        return stability_analysis

    def _generate_market_confidence(self, current_regime: Dict[str, Any],
                                  regime_analysis: Dict[str, Any],
                                  stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence score based on market conditions (NOT signals)."""
        confidence_analysis = {
            'overall_confidence': 50,
            'confidence_factors': [],
            'risk_factors': [],
            'confidence_category': 'moderate'
        }

        confidence_components = []

        # Regime clarity component
        regime_strength = current_regime.get('regime_strength', 0.5)
        regime_clarity = current_regime.get('regime_clarity', 'moderate')

        if regime_clarity == 'high':
            confidence_components.append(85)
            confidence_analysis['confidence_factors'].append('clear_market_regime')
        elif regime_clarity == 'moderate':
            confidence_components.append(65)
            confidence_analysis['confidence_factors'].append('moderate_regime_clarity')
        else:
            confidence_components.append(40)
            confidence_analysis['risk_factors'].append('unclear_market_regime')

        # Stability component
        stability_score = stability_analysis.get('stability_score', 0.5)
        stability_confidence = int(stability_score * 100)
        confidence_components.append(stability_confidence)

        if stability_score > 0.7:
            confidence_analysis['confidence_factors'].append('stable_market_conditions')
        elif stability_score < 0.4:
            confidence_analysis['risk_factors'].append('unstable_market_conditions')

        # Trend strength component
        trend_strength_category = regime_analysis.get('trend_strength_category', 'moderate')
        if trend_strength_category in ['strong', 'very_strong']:
            confidence_components.append(80)
            confidence_analysis['confidence_factors'].append('strong_directional_bias')
        elif trend_strength_category in ['weak', 'very_weak']:
            confidence_components.append(45)
            confidence_analysis['risk_factors'].append('weak_directional_conviction')
        else:
            confidence_components.append(60)

        # Market structure component
        market_structure = regime_analysis.get('market_structure', {})
        structure_clarity = market_structure.get('structure_clarity', 'moderate')

        if structure_clarity == 'high':
            confidence_components.append(75)
            confidence_analysis['confidence_factors'].append('clear_support_resistance_levels')
        elif structure_clarity == 'low':
            confidence_components.append(50)
            confidence_analysis['risk_factors'].append('unclear_key_levels')
        else:
            confidence_components.append(60)

        # Calculate overall confidence
        confidence_analysis['overall_confidence'] = int(np.mean(confidence_components))

        # Determine confidence category
        if confidence_analysis['overall_confidence'] > 75:
            confidence_analysis['confidence_category'] = 'high'
        elif confidence_analysis['overall_confidence'] > 60:
            confidence_analysis['confidence_category'] = 'moderate'
        else:
            confidence_analysis['confidence_category'] = 'low'

        return confidence_analysis

    def _construct_condition_reasoning(self, current_regime: Dict[str, Any],
                                     regime_analysis: Dict[str, Any],
                                     stability_analysis: Dict[str, Any],
                                     confidence_analysis: Dict[str, Any]) -> str:
        """Construct comprehensive market condition reasoning."""
        reasoning_parts = []

        # Primary regime identification
        primary_regime = current_regime.get('primary_regime', 'transitional')
        regime_strength = current_regime.get('regime_strength', 0.5)

        regime_descriptions = {
            'trending': 'trending market environment',
            'volatile_trending': 'volatile trending market conditions',
            'ranging': 'range-bound market environment',
            'volatile_ranging': 'volatile range-bound conditions',
            'low_volatility': 'low volatility consolidation environment',
            'transitional': 'transitional market conditions'
        }

        regime_desc = regime_descriptions.get(primary_regime, 'current market conditions')
        strength_desc = 'strong' if regime_strength > 0.7 else 'moderate' if regime_strength > 0.4 else 'weak'

        reasoning_parts.append(f"Market analysis reveals {strength_desc} {regime_desc}")

        # Trend and momentum characteristics
        trend_direction = regime_analysis.get('trend_direction', 'sideways')
        trend_strength_cat = regime_analysis.get('trend_strength_category', 'moderate')
        momentum_category = regime_analysis.get('momentum_category', 'moderate')

        if trend_direction != 'sideways':
            reasoning_parts.append(
                f"with {trend_strength_cat} {trend_direction} trend and {momentum_category} momentum characteristics"
            )

        # Volatility context
        volatility_category = regime_analysis.get('volatility_category', 'moderate')
        if volatility_category in ['very_high', 'very_low']:
            reasoning_parts.append(f"Operating within {volatility_category.replace('_', ' ')} volatility regime")

        # Stability assessment
        regime_confidence = stability_analysis.get('regime_confidence', 'moderate')
        stability_factors = stability_analysis.get('stability_factors', [])

        if stability_factors:
            primary_factor = stability_factors[0].replace('_', ' ')
            reasoning_parts.append(f"Regime stability shows {regime_confidence} confidence based on {primary_factor}")

        # Market structure context
        market_structure = regime_analysis.get('market_structure', {})
        structure_clarity = market_structure.get('structure_clarity', 'moderate')

        if structure_clarity == 'high':
            reasoning_parts.append("with well-defined support and resistance structure")
        elif structure_clarity == 'low':
            reasoning_parts.append("though key structural levels remain unclear")

        return ". ".join(reasoning_parts) + "."

    def _get_fallback_reasoning(self) -> str:
        """Get fallback reasoning when analysis fails."""
        return ("Market condition analysis indicates transitional environment with moderate "
                "volatility characteristics. Current regime shows balanced conditions requiring "
                "adaptive approach with attention to emerging directional signals.")

    def _safe_get_value(self, data: pd.Series, column: str, default: Any = 0) -> Any:
        """Safely get value from pandas Series."""
        try:
            return data.get(column, default) if column in data.index else default
        except Exception:
            return default
