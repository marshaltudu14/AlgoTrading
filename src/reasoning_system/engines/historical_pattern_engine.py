#!/usr/bin/env python3
"""
Historical Pattern Analysis Engine
=================================

Analyzes past 20-50 candles (short-term) and 100-200 candles (medium-term)
for realistic trader-like pattern recognition and trend analysis.

Key Features:
- Realistic time horizons like actual traders use
- Pattern recognition across multiple timeframes
- Trend duration and strength analysis
- Historical context for current market conditions
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
class HistoricalContext:
    """Historical context data structure."""
    short_term_lookback: int  # 20-50 candles
    medium_term_lookback: int  # 100-200 candles
    trend_duration: int
    trend_strength: str  # "weak", "moderate", "strong"
    pattern_type: str  # "continuation", "reversal", "consolidation"
    volatility_trend: str  # "increasing", "decreasing", "stable"
    momentum_history: str  # "accelerating", "decelerating", "consistent"
    support_resistance_history: str
    volume_pattern: str  # "increasing", "decreasing", "diverging"


class HistoricalPatternEngine(BaseReasoningEngine):
    """
    Historical pattern analysis engine that analyzes market patterns
    across realistic trader timeframes without using signal column.
    """
    
    def _initialize_config(self):
        """Initialize historical pattern analysis configuration."""
        # Time horizon configuration
        self.config.setdefault('short_term_min', 20)
        self.config.setdefault('short_term_max', 50)
        self.config.setdefault('medium_term_min', 100)
        self.config.setdefault('medium_term_max', 200)
        
        # Analysis thresholds
        self.config.setdefault('trend_strength_threshold', 0.6)
        self.config.setdefault('volatility_threshold', 0.02)
        self.config.setdefault('momentum_threshold', 0.1)
        
        logger.info("HistoricalPatternEngine initialized with realistic time horizons")
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for historical pattern analysis."""
        return [
            'open', 'high', 'low', 'close',
            'trend_strength', 'volatility_20', 'rsi_14', 'macd_histogram',
            'support_distance', 'resistance_distance'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate historical pattern reasoning without using signal column.
        
        Args:
            current_data: Current row data
            context: Historical context from HistoricalContextManager
            
        Returns:
            Historical pattern reasoning text
        """
        try:
            # Get historical data from context
            historical_data = context.get('historical_data', pd.DataFrame())
            if historical_data.empty:
                return self._get_fallback_reasoning()
            
            # Analyze short-term patterns (20-50 candles)
            short_term_analysis = self._analyze_short_term_patterns(
                historical_data, current_data
            )
            
            # Analyze medium-term patterns (100-200 candles)
            medium_term_analysis = self._analyze_medium_term_patterns(
                historical_data, current_data
            )
            
            # Detect pattern type and characteristics
            pattern_analysis = self._detect_pattern_characteristics(
                historical_data, short_term_analysis, medium_term_analysis
            )
            
            # Generate comprehensive reasoning
            reasoning = self._construct_historical_reasoning(
                short_term_analysis, medium_term_analysis, pattern_analysis, current_data
            )
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in historical pattern analysis: {str(e)}")
            return self._get_fallback_reasoning()
    
    def _analyze_short_term_patterns(self, historical_data: pd.DataFrame,
                                   current_data: pd.Series) -> Dict[str, Any]:
        """Analyze short-term patterns with dynamic lookback period."""
        # Dynamic lookback based on data availability and market conditions
        max_lookback = min(50, len(historical_data))
        min_lookback = min(20, len(historical_data))

        # Use adaptive lookback based on volatility
        if len(historical_data) > 0:
            recent_volatility = historical_data['volatility_20'].tail(10).mean() if 'volatility_20' in historical_data.columns else 0.02
            if recent_volatility > 0.03:  # High volatility - use shorter period
                lookback = min_lookback
            elif recent_volatility < 0.015:  # Low volatility - use longer period
                lookback = max_lookback
            else:  # Normal volatility - use medium period
                lookback = min(30, max_lookback)
        else:
            lookback = min_lookback

        recent_data = historical_data.tail(lookback) if lookback > 0 else pd.DataFrame()

        if recent_data.empty:
            return self._get_default_short_term_analysis()

        analysis = {
            'lookback_period': lookback,
            'actual_data_points': len(recent_data),
            'trend_consistency': self._calculate_trend_consistency(recent_data),
            'momentum_pattern': self._analyze_momentum_pattern(recent_data),
            'volatility_trend': self._analyze_volatility_trend(recent_data),
            'support_resistance_tests': self._analyze_sr_tests(recent_data, current_data),
            'pattern_strength': 'moderate',
            'data_quality': 'good' if len(recent_data) >= min_lookback else 'limited'
        }

        # Determine pattern strength based on actual data analysis
        if analysis['trend_consistency'] > 0.7 and analysis['momentum_pattern']['strength'] > 0.6:
            analysis['pattern_strength'] = 'strong'
        elif analysis['trend_consistency'] < 0.3 or analysis['momentum_pattern']['strength'] < 0.3:
            analysis['pattern_strength'] = 'weak'

        return analysis
    
    def _analyze_medium_term_patterns(self, historical_data: pd.DataFrame,
                                    current_data: pd.Series) -> Dict[str, Any]:
        """Analyze medium-term patterns (100-200 candles)."""
        # Use last 150 candles for medium-term analysis (middle of 100-200 range)
        lookback = min(150, len(historical_data))
        extended_data = historical_data.tail(lookback) if lookback > 0 else pd.DataFrame()
        
        if extended_data.empty:
            return self._get_default_medium_term_analysis()
        
        analysis = {
            'lookback_period': lookback,
            'major_trend_direction': self._determine_major_trend(extended_data),
            'trend_duration': self._calculate_trend_duration(extended_data),
            'structural_levels': self._identify_structural_levels(extended_data),
            'regime_characteristics': self._analyze_market_regime(extended_data),
            'pattern_maturity': 'developing'
        }
        
        # Determine pattern maturity
        if analysis['trend_duration'] > 50:
            analysis['pattern_maturity'] = 'mature'
        elif analysis['trend_duration'] < 20:
            analysis['pattern_maturity'] = 'early'
        
        return analysis
    
    def _detect_pattern_characteristics(self, historical_data: pd.DataFrame,
                                      short_term: Dict[str, Any],
                                      medium_term: Dict[str, Any]) -> Dict[str, Any]:
        """Detect overall pattern characteristics."""
        pattern_analysis = {
            'primary_pattern': 'continuation',
            'confidence_level': 0.5,
            'time_horizon_alignment': 'neutral',
            'pattern_implications': []
        }
        
        # Determine primary pattern type
        short_trend = short_term.get('trend_consistency', 0.5)
        medium_trend = medium_term.get('major_trend_direction', 'neutral')
        
        if short_trend > 0.6 and medium_trend in ['bullish', 'bearish']:
            pattern_analysis['primary_pattern'] = 'continuation'
            pattern_analysis['confidence_level'] = 0.8
        elif short_trend < 0.3 and medium_trend != 'neutral':
            pattern_analysis['primary_pattern'] = 'reversal'
            pattern_analysis['confidence_level'] = 0.7
        else:
            pattern_analysis['primary_pattern'] = 'consolidation'
            pattern_analysis['confidence_level'] = 0.6
        
        # Analyze time horizon alignment
        if (short_term.get('pattern_strength') == 'strong' and 
            medium_term.get('pattern_maturity') == 'mature'):
            pattern_analysis['time_horizon_alignment'] = 'strong'
        elif (short_term.get('pattern_strength') == 'weak' or 
              medium_term.get('pattern_maturity') == 'early'):
            pattern_analysis['time_horizon_alignment'] = 'weak'
        
        # Generate pattern implications
        pattern_analysis['pattern_implications'] = self._generate_pattern_implications(
            pattern_analysis, short_term, medium_term
        )
        
        return pattern_analysis
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency over the period."""
        if len(data) < 5:
            return 0.5
        
        # Use close prices for trend analysis
        closes = data['close'].values
        
        # Calculate directional consistency
        price_changes = np.diff(closes)
        positive_changes = np.sum(price_changes > 0)
        total_changes = len(price_changes)
        
        if total_changes == 0:
            return 0.5
        
        # Consistency score (0-1)
        consistency = max(positive_changes, total_changes - positive_changes) / total_changes
        
        return consistency
    
    def _analyze_momentum_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum patterns in the data."""
        momentum_analysis = {
            'direction': 'neutral',
            'strength': 0.5,
            'acceleration': 'stable'
        }
        
        if 'rsi_14' in data.columns and len(data) >= 5:
            rsi_values = data['rsi_14'].dropna()
            if len(rsi_values) >= 5:
                recent_rsi = rsi_values.tail(5).mean()
                earlier_rsi = rsi_values.head(5).mean()
                
                if recent_rsi > 55:
                    momentum_analysis['direction'] = 'bullish'
                    momentum_analysis['strength'] = min((recent_rsi - 50) / 50, 1.0)
                elif recent_rsi < 45:
                    momentum_analysis['direction'] = 'bearish'
                    momentum_analysis['strength'] = min((50 - recent_rsi) / 50, 1.0)
                
                # Analyze acceleration
                if recent_rsi > earlier_rsi + 5:
                    momentum_analysis['acceleration'] = 'accelerating'
                elif recent_rsi < earlier_rsi - 5:
                    momentum_analysis['acceleration'] = 'decelerating'
        
        return momentum_analysis
    
    def _analyze_volatility_trend(self, data: pd.DataFrame) -> str:
        """Analyze volatility trend over the period."""
        if 'volatility_20' not in data.columns or len(data) < 10:
            return 'stable'
        
        volatility = data['volatility_20'].dropna()
        if len(volatility) < 10:
            return 'stable'
        
        recent_vol = volatility.tail(5).mean()
        earlier_vol = volatility.head(5).mean()
        
        if recent_vol > earlier_vol * 1.2:
            return 'increasing'
        elif recent_vol < earlier_vol * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_sr_tests(self, data: pd.DataFrame, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze support/resistance level tests."""
        sr_analysis = {
            'support_tests': 0,
            'resistance_tests': 0,
            'level_strength': 'moderate'
        }
        
        # Count approximate tests of current levels
        current_support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        current_resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        if 'support_distance' in data.columns:
            try:
                support_distances = data['support_distance'].dropna()
                # Count how many times price was near support (within 1% distance)
                sr_analysis['support_tests'] = len(support_distances[support_distances < 1.0])
            except Exception:
                sr_analysis['support_tests'] = 0

        if 'resistance_distance' in data.columns:
            try:
                resistance_distances = data['resistance_distance'].dropna()
                # Count how many times price was near resistance
                sr_analysis['resistance_tests'] = len(resistance_distances[resistance_distances < 1.0])
            except Exception:
                sr_analysis['resistance_tests'] = 0
        
        # Determine level strength
        total_tests = sr_analysis['support_tests'] + sr_analysis['resistance_tests']
        if total_tests >= 5:
            sr_analysis['level_strength'] = 'strong'
        elif total_tests <= 2:
            sr_analysis['level_strength'] = 'weak'
        
        return sr_analysis

    def _determine_major_trend(self, data: pd.DataFrame) -> str:
        """Determine major trend direction over medium-term period."""
        if len(data) < 20:
            return 'neutral'

        # Use trend_strength column if available
        if 'trend_strength' in data.columns:
            trend_values = data['trend_strength'].dropna()
            if len(trend_values) >= 10:
                avg_trend = trend_values.tail(20).mean()
                if avg_trend > 0.3:
                    return 'bullish'
                elif avg_trend < -0.3:
                    return 'bearish'

        # Fallback to price analysis
        closes = data['close'].values
        if len(closes) >= 20:
            early_avg = np.mean(closes[:10])
            recent_avg = np.mean(closes[-10:])

            change_pct = (recent_avg - early_avg) / early_avg
            if change_pct > 0.05:  # 5% increase
                return 'bullish'
            elif change_pct < -0.05:  # 5% decrease
                return 'bearish'

        return 'neutral'

    def _calculate_trend_duration(self, data: pd.DataFrame) -> int:
        """Calculate how long the current trend has lasted."""
        if len(data) < 10:
            return 0

        # Simple trend duration calculation
        # Count consecutive periods in same direction
        closes = data['close'].values
        if len(closes) < 10:
            return 0

        # Look for trend changes
        changes = np.diff(closes)
        current_direction = 1 if changes[-1] > 0 else -1

        duration = 1
        for i in range(len(changes) - 2, -1, -1):
            change_direction = 1 if changes[i] > 0 else -1
            if change_direction == current_direction:
                duration += 1
            else:
                break

        return min(duration, len(data))

    def _identify_structural_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify key structural support/resistance levels."""
        levels = {
            'major_support': None,
            'major_resistance': None,
            'level_count': 0
        }

        if len(data) < 20:
            return levels

        # Use high/low data to identify key levels
        highs = data['high'].values
        lows = data['low'].values

        # Find major resistance (highest high in period)
        levels['major_resistance'] = np.max(highs)

        # Find major support (lowest low in period)
        levels['major_support'] = np.min(lows)

        # Count significant levels (simplified)
        levels['level_count'] = 2  # At minimum, support and resistance

        return levels

    def _analyze_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regime characteristics."""
        regime = {
            'type': 'trending',
            'volatility_regime': 'normal',
            'momentum_regime': 'balanced'
        }

        if len(data) < 20:
            return regime

        # Analyze volatility regime
        if 'volatility_20' in data.columns:
            vol_data = data['volatility_20'].dropna()
            if len(vol_data) >= 10:
                avg_vol = vol_data.mean()
                if avg_vol > 0.03:
                    regime['volatility_regime'] = 'high'
                elif avg_vol < 0.015:
                    regime['volatility_regime'] = 'low'

        # Analyze momentum regime
        if 'rsi_14' in data.columns:
            rsi_data = data['rsi_14'].dropna()
            if len(rsi_data) >= 10:
                rsi_std = rsi_data.std()
                if rsi_std > 15:
                    regime['momentum_regime'] = 'volatile'
                elif rsi_std < 8:
                    regime['momentum_regime'] = 'stable'

        # Determine regime type
        trend_consistency = self._calculate_trend_consistency(data)
        if trend_consistency > 0.7:
            regime['type'] = 'trending'
        elif trend_consistency < 0.4:
            regime['type'] = 'ranging'
        else:
            regime['type'] = 'transitional'

        return regime

    def _generate_pattern_implications(self, pattern_analysis: Dict[str, Any],
                                     short_term: Dict[str, Any],
                                     medium_term: Dict[str, Any]) -> List[str]:
        """Generate pattern implications based on analysis."""
        implications = []

        pattern_type = pattern_analysis['primary_pattern']
        confidence = pattern_analysis['confidence_level']

        if pattern_type == 'continuation' and confidence > 0.7:
            implications.extend([
                "trend continuation likely",
                "momentum expected to persist",
                "directional bias maintained"
            ])
        elif pattern_type == 'reversal' and confidence > 0.7:
            implications.extend([
                "trend reversal potential",
                "momentum shift developing",
                "directional change possible"
            ])
        elif pattern_type == 'consolidation':
            implications.extend([
                "range-bound behavior expected",
                "breakout preparation phase",
                "directional uncertainty"
            ])

        # Add time-specific implications
        if short_term.get('pattern_strength') == 'strong':
            implications.append("near-term clarity present")

        if medium_term.get('pattern_maturity') == 'mature':
            implications.append("established pattern structure")

        return implications[:3]  # Limit to 3 implications

    def _construct_historical_reasoning(self, short_term: Dict[str, Any],
                                       medium_term: Dict[str, Any],
                                       pattern_analysis: Dict[str, Any],
                                       current_data: Optional[pd.Series] = None) -> str:
        """Construct comprehensive historical reasoning."""
        reasoning_parts = []

        # Short-term analysis
        short_period = short_term.get('lookback_period', 30)
        trend_consistency = short_term.get('trend_consistency', 0.5)

        reasoning_parts.append(
            f"Analysis of the past {short_period} candles reveals "
            f"{'strong' if trend_consistency > 0.7 else 'moderate' if trend_consistency > 0.4 else 'weak'} "
            f"trend consistency at {trend_consistency:.1f}"
        )

        # Medium-term context
        medium_period = medium_term.get('lookback_period', 150)
        major_trend = medium_term.get('major_trend_direction', 'neutral')
        trend_duration = medium_term.get('trend_duration', 0)

        if major_trend != 'neutral':
            reasoning_parts.append(
                f"The broader {medium_period}-candle context shows {major_trend} trend "
                f"that has persisted for {trend_duration} periods"
            )

        # Pattern characteristics
        pattern_type = pattern_analysis.get('primary_pattern', 'continuation')
        confidence = pattern_analysis.get('confidence_level', 0.5)

        reasoning_parts.append(
            f"Historical pattern analysis indicates {pattern_type} characteristics "
            f"with {confidence:.1f} confidence level"
        )

        # Pattern implications
        implications = pattern_analysis.get('pattern_implications', [])
        if implications:
            reasoning_parts.append(f"suggesting {implications[0]}")

        return ". ".join(reasoning_parts) + "."

    def _get_default_short_term_analysis(self) -> Dict[str, Any]:
        """Get default short-term analysis when data is insufficient."""
        return {
            'lookback_period': 0,
            'trend_consistency': 0.5,
            'momentum_pattern': {'direction': 'neutral', 'strength': 0.5, 'acceleration': 'stable'},
            'volatility_trend': 'stable',
            'support_resistance_tests': {'support_tests': 0, 'resistance_tests': 0, 'level_strength': 'moderate'},
            'pattern_strength': 'weak'
        }

    def _get_default_medium_term_analysis(self) -> Dict[str, Any]:
        """Get default medium-term analysis when data is insufficient."""
        return {
            'lookback_period': 0,
            'major_trend_direction': 'neutral',
            'trend_duration': 0,
            'structural_levels': {'major_support': None, 'major_resistance': None, 'level_count': 0},
            'regime_characteristics': {'type': 'transitional', 'volatility_regime': 'normal', 'momentum_regime': 'balanced'},
            'pattern_maturity': 'early'
        }

    def _get_fallback_reasoning(self) -> str:
        """Get fallback reasoning when analysis fails."""
        return ("Historical pattern analysis indicates limited data availability for comprehensive "
                "temporal assessment. Current market conditions suggest cautious approach with "
                "focus on immediate price action and key technical levels.")

    def _safe_get_value(self, data: pd.Series, column: str, default: Any = 0) -> Any:
        """Safely get value from pandas Series."""
        try:
            return data.get(column, default) if column in data.index else default
        except Exception:
            return default
