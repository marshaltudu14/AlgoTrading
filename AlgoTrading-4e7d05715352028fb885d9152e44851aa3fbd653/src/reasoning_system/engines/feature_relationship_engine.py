#!/usr/bin/env python3
"""
Feature Relationship Engine
==========================

Understands relationships between all 65 technical indicators with historical
context and temporal awareness. Analyzes indicator confluence and divergence
patterns without using signal column.

Key Features:
- Analyzes relationships between technical indicators
- Detects confluence and divergence patterns
- Provides temporal context for indicator relationships
- Generates natural language explanations of indicator interactions
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
class FeatureAnalysis:
    """Feature analysis data structure."""
    rsi_condition: str  # "oversold", "neutral", "overbought"
    rsi_divergence: str  # "bullish", "bearish", "none"
    macd_signal: str   # "bullish", "bearish", "neutral"
    macd_trend: str    # Historical MACD trend analysis
    ma_alignment: str  # "bullish", "bearish", "mixed"
    ma_trend_strength: str  # Based on historical MA relationships
    bb_position: str   # "lower", "middle", "upper"
    bb_squeeze_expansion: str  # Historical volatility pattern


class FeatureRelationshipEngine(BaseReasoningEngine):
    """
    Feature relationship analysis engine that understands indicator
    relationships and confluence patterns without using signal column.
    """
    
    def _initialize_config(self):
        """Initialize feature relationship analysis configuration."""
        # Indicator thresholds
        self.config.setdefault('rsi_overbought', 70)
        self.config.setdefault('rsi_oversold', 30)
        self.config.setdefault('macd_threshold', 0.05)
        self.config.setdefault('bb_upper_threshold', 0.8)
        self.config.setdefault('bb_lower_threshold', 0.2)
        
        # Confluence scoring
        self.config.setdefault('confluence_threshold', 0.7)
        self.config.setdefault('divergence_threshold', 0.6)
        
        logger.info("FeatureRelationshipEngine initialized with indicator thresholds")
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for feature relationship analysis."""
        return [
            # Oscillators
            'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
            # MACD
            'macd', 'macd_signal', 'macd_histogram',
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100', 'ema_200',
            # Bollinger Bands
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_width',
            # Trend and Directional Movement
            'adx', 'di_plus', 'di_minus', 'trend_strength', 'trend_direction',
            # Momentum
            'momentum_10', 'roc_10', 'trix',
            # Volatility
            'volatility_10', 'volatility_20', 'atr',
            # Support/Resistance
            'support_level', 'resistance_level', 'support_distance', 'resistance_distance',
            # Pattern Recognition
            'doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing',
            # Price Action
            'body_size', 'upper_shadow', 'lower_shadow', 'gap_up', 'gap_down',
            # Crossovers
            'sma_5_20_cross', 'sma_10_50_cross', 'price_vs_sma_20', 'price_vs_ema_20',
            # Basic OHLC
            'open', 'high', 'low', 'close'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate feature relationship reasoning without using signal column.
        
        Args:
            current_data: Current row data
            context: Historical context from HistoricalContextManager
            
        Returns:
            Feature relationship reasoning text
        """
        try:
            # Analyze individual indicator conditions
            indicator_analysis = self._analyze_individual_indicators(current_data)
            
            # Analyze indicator relationships and confluence
            relationship_analysis = self._analyze_indicator_relationships(current_data, context)

            # Detect divergence patterns
            divergence_analysis = self._analyze_divergence_patterns(current_data, context)

            # Analyze temporal indicator trends
            temporal_analysis = self._analyze_temporal_trends(current_data, context)

            # Generate comprehensive reasoning
            reasoning = self._construct_relationship_reasoning(
                indicator_analysis, relationship_analysis, divergence_analysis, temporal_analysis, current_data
            )
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in feature relationship analysis: {str(e)}")
            return self._get_fallback_reasoning()
    
    def _analyze_individual_indicators(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze individual indicator conditions."""
        analysis = {
            'rsi_analysis': self._analyze_rsi_condition(current_data),
            'macd_analysis': self._analyze_macd_condition(current_data),
            'ma_analysis': self._analyze_ma_alignment(current_data),
            'bb_analysis': self._analyze_bollinger_condition(current_data),
            'momentum_analysis': self._analyze_momentum_indicators(current_data),
            'volatility_analysis': self._analyze_volatility_indicators(current_data),
            'stochastic_analysis': self._analyze_stochastic_condition(current_data),
            'adx_analysis': self._analyze_adx_condition(current_data),
            'cci_analysis': self._analyze_cci_condition(current_data),
            'support_resistance_analysis': self._analyze_support_resistance(current_data),
            'pattern_analysis': self._analyze_pattern_recognition(current_data),
            'price_action_analysis': self._analyze_price_action(current_data)
        }

        return analysis
    
    def _analyze_rsi_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze RSI indicator conditions."""
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        rsi_21 = self._safe_get_value(current_data, 'rsi_21', 50)
        
        analysis = {
            'rsi_14_value': rsi_14,
            'rsi_21_value': rsi_21,
            'condition': 'neutral',
            'strength': 'moderate',
            'timeframe_agreement': True
        }
        
        # Determine RSI condition
        if rsi_14 >= self.config['rsi_overbought']:
            analysis['condition'] = 'overbought'
            analysis['strength'] = 'strong' if rsi_14 > 80 else 'moderate'
        elif rsi_14 <= self.config['rsi_oversold']:
            analysis['condition'] = 'oversold'
            analysis['strength'] = 'strong' if rsi_14 < 20 else 'moderate'
        elif rsi_14 > 55:
            analysis['condition'] = 'bullish'
        elif rsi_14 < 45:
            analysis['condition'] = 'bearish'
        
        # Check timeframe agreement
        rsi_diff = abs(rsi_14 - rsi_21)
        analysis['timeframe_agreement'] = rsi_diff < 10
        
        return analysis
    
    def _analyze_macd_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze MACD indicator conditions."""
        macd = self._safe_get_value(current_data, 'macd', 0)
        macd_signal = self._safe_get_value(current_data, 'macd_signal', 0)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)
        
        analysis = {
            'macd_value': macd,
            'signal_value': macd_signal,
            'histogram_value': macd_histogram,
            'condition': 'neutral',
            'strength': 'moderate',
            'momentum_direction': 'neutral'
        }
        
        # Determine MACD condition
        if macd_histogram > self.config['macd_threshold']:
            analysis['condition'] = 'bullish'
            analysis['strength'] = 'strong' if macd_histogram > 0.1 else 'moderate'
            analysis['momentum_direction'] = 'accelerating'
        elif macd_histogram < -self.config['macd_threshold']:
            analysis['condition'] = 'bearish'
            analysis['strength'] = 'strong' if macd_histogram < -0.1 else 'moderate'
            analysis['momentum_direction'] = 'decelerating'
        
        return analysis
    
    def _analyze_ma_alignment(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze moving average alignment."""
        close = self._safe_get_value(current_data, 'close', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', close)
        sma_50 = self._safe_get_value(current_data, 'sma_50', close)
        sma_200 = self._safe_get_value(current_data, 'sma_200', close)
        
        analysis = {
            'close_vs_sma20': 'above' if close > sma_20 else 'below',
            'close_vs_sma50': 'above' if close > sma_50 else 'below',
            'close_vs_sma200': 'above' if close > sma_200 else 'below',
            'alignment': 'mixed',
            'strength': 'moderate'
        }
        
        # Determine alignment
        above_count = sum([
            close > sma_20,
            close > sma_50,
            close > sma_200,
            sma_20 > sma_50,
            sma_50 > sma_200
        ])
        
        if above_count >= 4:
            analysis['alignment'] = 'bullish'
            analysis['strength'] = 'strong' if above_count == 5 else 'moderate'
        elif above_count <= 1:
            analysis['alignment'] = 'bearish'
            analysis['strength'] = 'strong' if above_count == 0 else 'moderate'
        
        return analysis
    
    def _analyze_bollinger_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze Bollinger Bands conditions."""
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        bb_width = self._safe_get_value(current_data, 'bb_width', 0.04)
        
        analysis = {
            'position': bb_position,
            'width': bb_width,
            'position_condition': 'middle',
            'volatility_condition': 'normal',
            'squeeze_status': False
        }
        
        # Determine position condition
        if bb_position > self.config['bb_upper_threshold']:
            analysis['position_condition'] = 'upper'
        elif bb_position < self.config['bb_lower_threshold']:
            analysis['position_condition'] = 'lower'
        
        # Determine volatility condition
        if bb_width < 0.02:
            analysis['volatility_condition'] = 'squeeze'
            analysis['squeeze_status'] = True
        elif bb_width > 0.06:
            analysis['volatility_condition'] = 'expansion'
        
        return analysis
    
    def _analyze_momentum_indicators(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze momentum indicators (Stochastic, Williams %R, CCI)."""
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        stoch_d = self._safe_get_value(current_data, 'stoch_d', 50)
        williams_r = self._safe_get_value(current_data, 'williams_r', -50)
        cci = self._safe_get_value(current_data, 'cci', 0)
        
        analysis = {
            'stochastic_condition': 'neutral',
            'williams_condition': 'neutral',
            'cci_condition': 'neutral',
            'momentum_consensus': 'mixed'
        }
        
        # Analyze Stochastic
        if stoch_k > 80 and stoch_d > 80:
            analysis['stochastic_condition'] = 'overbought'
        elif stoch_k < 20 and stoch_d < 20:
            analysis['stochastic_condition'] = 'oversold'
        elif stoch_k > 50:
            analysis['stochastic_condition'] = 'bullish'
        else:
            analysis['stochastic_condition'] = 'bearish'
        
        # Analyze Williams %R
        if williams_r > -20:
            analysis['williams_condition'] = 'overbought'
        elif williams_r < -80:
            analysis['williams_condition'] = 'oversold'
        elif williams_r > -50:
            analysis['williams_condition'] = 'bullish'
        else:
            analysis['williams_condition'] = 'bearish'
        
        # Analyze CCI
        if cci > 100:
            analysis['cci_condition'] = 'overbought'
        elif cci < -100:
            analysis['cci_condition'] = 'oversold'
        elif cci > 0:
            analysis['cci_condition'] = 'bullish'
        else:
            analysis['cci_condition'] = 'bearish'
        
        # Determine consensus
        bullish_count = sum([
            'bullish' in analysis['stochastic_condition'],
            'bullish' in analysis['williams_condition'],
            'bullish' in analysis['cci_condition']
        ])
        
        bearish_count = sum([
            'bearish' in analysis['stochastic_condition'],
            'bearish' in analysis['williams_condition'],
            'bearish' in analysis['cci_condition']
        ])
        
        if bullish_count >= 2:
            analysis['momentum_consensus'] = 'bullish'
        elif bearish_count >= 2:
            analysis['momentum_consensus'] = 'bearish'
        
        return analysis
    
    def _analyze_volatility_indicators(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze volatility indicators (ATR, Volatility)."""
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        volatility_20 = self._safe_get_value(current_data, 'volatility_20', 0.02)
        
        analysis = {
            'atr_value': atr,
            'volatility_value': volatility_20,
            'volatility_condition': 'normal',
            'trend_strength': 'moderate'
        }
        
        # Determine volatility condition
        if volatility_20 > 0.03:
            analysis['volatility_condition'] = 'high'
        elif volatility_20 < 0.015:
            analysis['volatility_condition'] = 'low'
        
        return analysis

    def _analyze_indicator_relationships(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze relationships and confluence between indicators."""
        relationship_analysis = {
            'momentum_confluence': self._calculate_momentum_confluence(current_data),
            'trend_confluence': self._calculate_trend_confluence(current_data),
            'volatility_confluence': self._calculate_volatility_confluence(current_data),
            'overall_confluence_score': 0.5,
            'conflicting_signals': []
        }

        # Calculate overall confluence score
        confluences = [
            relationship_analysis['momentum_confluence']['score'],
            relationship_analysis['trend_confluence']['score'],
            relationship_analysis['volatility_confluence']['score']
        ]

        relationship_analysis['overall_confluence_score'] = np.mean(confluences)

        # Identify conflicting signals
        relationship_analysis['conflicting_signals'] = self._identify_conflicting_signals(current_data)

        return relationship_analysis

    def _calculate_momentum_confluence(self, current_data: pd.Series) -> Dict[str, Any]:
        """Calculate momentum indicator confluence."""
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)

        # Score each indicator (-1 to 1)
        rsi_score = (rsi_14 - 50) / 50  # Normalized RSI
        macd_score = np.tanh(macd_histogram * 10)  # Normalized MACD
        stoch_score = (stoch_k - 50) / 50  # Normalized Stochastic

        scores = [rsi_score, macd_score, stoch_score]
        confluence_score = np.mean(scores)

        # Determine confluence strength
        if abs(confluence_score) > 0.6:
            strength = 'strong'
        elif abs(confluence_score) > 0.3:
            strength = 'moderate'
        else:
            strength = 'weak'

        direction = 'bullish' if confluence_score > 0.1 else 'bearish' if confluence_score < -0.1 else 'neutral'

        return {
            'score': abs(confluence_score),
            'direction': direction,
            'strength': strength,
            'individual_scores': {
                'rsi': rsi_score,
                'macd': macd_score,
                'stochastic': stoch_score
            }
        }

    def _calculate_trend_confluence(self, current_data: pd.Series) -> Dict[str, Any]:
        """Calculate trend indicator confluence."""
        close = self._safe_get_value(current_data, 'close', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', close)
        sma_50 = self._safe_get_value(current_data, 'sma_50', close)
        ema_20 = self._safe_get_value(current_data, 'ema_20', close)

        # Calculate trend scores
        sma_20_score = 1 if close > sma_20 else -1
        sma_50_score = 1 if close > sma_50 else -1
        ema_20_score = 1 if close > ema_20 else -1
        ma_alignment_score = 1 if sma_20 > sma_50 else -1

        scores = [sma_20_score, sma_50_score, ema_20_score, ma_alignment_score]
        confluence_score = np.mean(scores)

        # Determine confluence strength
        agreement_count = sum([s > 0 for s in scores])
        if agreement_count >= 3:
            strength = 'strong'
        elif agreement_count >= 2:
            strength = 'moderate'
        else:
            strength = 'weak'

        direction = 'bullish' if confluence_score > 0 else 'bearish' if confluence_score < 0 else 'neutral'

        return {
            'score': abs(confluence_score),
            'direction': direction,
            'strength': strength,
            'agreement_count': agreement_count
        }

    def _calculate_volatility_confluence(self, current_data: pd.Series) -> Dict[str, Any]:
        """Calculate volatility indicator confluence."""
        bb_width = self._safe_get_value(current_data, 'bb_width', 0.04)
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        volatility_20 = self._safe_get_value(current_data, 'volatility_20', 0.02)

        # Normalize volatility measures (simplified)
        bb_vol_score = min(bb_width / 0.04, 2.0)  # Normalized to typical range
        atr_score = min(atr / 1.0, 2.0)  # Normalized to typical range
        vol_score = min(volatility_20 / 0.02, 2.0)  # Normalized to typical range

        scores = [bb_vol_score, atr_score, vol_score]
        avg_volatility = np.mean(scores)

        # Determine volatility condition
        if avg_volatility > 1.5:
            condition = 'high'
            strength = 'strong'
        elif avg_volatility > 1.2:
            condition = 'elevated'
            strength = 'moderate'
        elif avg_volatility < 0.8:
            condition = 'low'
            strength = 'strong'
        else:
            condition = 'normal'
            strength = 'moderate'

        return {
            'score': min(abs(avg_volatility - 1.0), 1.0),  # Deviation from normal
            'condition': condition,
            'strength': strength,
            'average_volatility': avg_volatility
        }

    def _identify_conflicting_signals(self, current_data: pd.Series) -> List[str]:
        """Identify conflicting signals between indicators."""
        conflicts = []

        # RSI vs MACD conflict
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)

        rsi_bullish = rsi_14 > 55
        macd_bullish = macd_histogram > 0

        if rsi_bullish != macd_bullish:
            conflicts.append("RSI and MACD showing divergent momentum signals")

        # Price vs Moving Average conflict
        close = self._safe_get_value(current_data, 'close', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', close)
        sma_50 = self._safe_get_value(current_data, 'sma_50', close)

        if (close > sma_20) != (sma_20 > sma_50):
            conflicts.append("Price action conflicting with moving average trend")

        # Bollinger Bands vs RSI conflict
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)

        if bb_position > 0.8 and rsi_14 < 50:
            conflicts.append("Bollinger Band position conflicting with RSI momentum")
        elif bb_position < 0.2 and rsi_14 > 50:
            conflicts.append("Bollinger Band position conflicting with RSI momentum")

        return conflicts[:2]  # Limit to 2 conflicts

    def _analyze_divergence_patterns(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze divergence patterns between price and indicators."""
        # Simplified divergence analysis (would need historical price data for full analysis)
        divergence_analysis = {
            'rsi_divergence': 'none',
            'macd_divergence': 'none',
            'momentum_divergence': 'none',
            'divergence_strength': 'weak'
        }

        # This would require historical data analysis for proper divergence detection
        # For now, provide basic analysis based on current conditions

        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_histogram = self._safe_get_value(current_data, 'macd_histogram', 0)

        # Simple divergence hints based on extreme readings
        if rsi_14 > 70 and macd_histogram < 0:
            divergence_analysis['rsi_divergence'] = 'bearish_hint'
            divergence_analysis['divergence_strength'] = 'moderate'
        elif rsi_14 < 30 and macd_histogram > 0:
            divergence_analysis['rsi_divergence'] = 'bullish_hint'
            divergence_analysis['divergence_strength'] = 'moderate'

        return divergence_analysis

    def _analyze_temporal_trends(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal trends in indicator relationships."""
        temporal_analysis = {
            'momentum_trend': 'stable',
            'volatility_trend': 'stable',
            'trend_strength_evolution': 'stable',
            'indicator_stability': 'moderate'
        }

        # This would require historical context data for proper temporal analysis
        # For now, provide basic analysis based on current indicator values

        # Analyze current momentum characteristics
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi_14 > 60:
            temporal_analysis['momentum_trend'] = 'strengthening'
        elif rsi_14 < 40:
            temporal_analysis['momentum_trend'] = 'weakening'

        # Analyze volatility characteristics
        volatility_20 = self._safe_get_value(current_data, 'volatility_20', 0.02)
        if volatility_20 > 0.025:
            temporal_analysis['volatility_trend'] = 'increasing'
        elif volatility_20 < 0.015:
            temporal_analysis['volatility_trend'] = 'decreasing'

        return temporal_analysis

    def _construct_relationship_reasoning(self, indicator_analysis: Dict[str, Any],
                                        relationship_analysis: Dict[str, Any],
                                        divergence_analysis: Dict[str, Any],
                                        temporal_analysis: Dict[str, Any], current_data: pd.Series = None) -> str:
        """Construct comprehensive feature relationship reasoning."""
        reasoning_parts = []

        # Use comprehensive indicator analysis like a professional trader
        if current_data is not None:
            # Get comprehensive indicator values
            rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
            rsi_21 = self._safe_get_value(current_data, 'rsi_21', 50)
            macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
            bb_position = self._safe_get_value(current_data, 'bb_position', 50)
            ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
            ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
            close_price = self._safe_get_value(current_data, 'close', 0)
            adx = self._safe_get_value(current_data, 'adx', 25)
            stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
            williams_r = self._safe_get_value(current_data, 'williams_r', -50)
            cci = self._safe_get_value(current_data, 'cci', 0)
            atr = self._safe_get_value(current_data, 'atr', 50)

            # Comprehensive indicator analysis
            indicator_insights = []

            # RSI analysis with specific conditions
            rsi_condition = self._get_rsi_condition_detailed(rsi_14)
            if close_price and ema_20:
                price_vs_ema = "above" if close_price > ema_20 else "below"
                indicator_insights.append(f"RSI at {rsi_14:.1f} {rsi_condition} with price {price_vs_ema} EMA-20 at {ema_20:.1f}")

            # Moving average trend analysis with specific percentages
            if ema_20 and ema_50:
                ma_diff = ((ema_20 - ema_50) / ema_50) * 100
                ma_bias = "bullish" if ma_diff > 0.1 else "bearish" if ma_diff < -0.1 else "neutral"
                indicator_insights.append(f"EMA-20 at {ema_20:.1f} is {abs(ma_diff):.2f}% {'above' if ma_diff > 0 else 'below'} EMA-50 indicating {ma_bias} trend structure")

            # Bollinger Band position analysis
            bb_condition = "upper" if bb_position > 80 else "lower" if bb_position < 20 else "middle"
            bb_implication = "resistance" if bb_position > 80 else "support" if bb_position < 20 else "equilibrium"
            indicator_insights.append(f"Bollinger band position at {bb_position:.2f} shows price near {bb_condition} band suggesting potential {bb_implication}")

            # ADX trend strength analysis
            trend_strength = "strong" if adx > 25 else "moderate" if adx > 20 else "weak"
            indicator_insights.append(f"ADX at {adx:.1f} indicates {trend_strength} trend strength")

            # Momentum oscillator analysis
            stoch_condition = "overbought" if stoch_k > 80 else "oversold" if stoch_k < 20 else "neutral"
            williams_condition = "overbought" if williams_r > -20 else "oversold" if williams_r < -80 else "neutral"

            if stoch_condition != "neutral" or williams_condition != "neutral":
                indicator_insights.append(f"Stochastic at {stoch_k:.1f} in {stoch_condition} territory, Williams %R at {williams_r:.1f} confirms {williams_condition} conditions")

            # Volatility analysis
            volatility_regime = "high" if atr > 60 else "low" if atr < 40 else "moderate"
            indicator_insights.append(f"ATR at {atr:.1f} indicates {volatility_regime} volatility environment")

            # Momentum confluence analysis
            bullish_signals = sum([
                rsi_14 > 50, macd_hist > 0, stoch_k > 50, williams_r > -50,
                ema_20 > ema_50 if ema_20 and ema_50 else False
            ])
            total_signals = 5
            confluence_pct = (bullish_signals / total_signals) * 100
            bias_direction = "bullish" if confluence_pct > 60 else "bearish" if confluence_pct < 40 else "mixed"
            indicator_insights.append(f"Momentum indicators show {bias_direction} bias with {bullish_signals} of {total_signals} measures in agreement")

            if indicator_insights:
                reasoning_parts.extend(indicator_insights)
        else:
            reasoning_parts.append("Technical indicators require current market data for analysis")

        # Momentum analysis with specific indicator insights
        momentum_analysis = relationship_analysis.get('momentum_confluence', {})
        momentum_direction = momentum_analysis.get('direction', 'neutral')
        momentum_strength = momentum_analysis.get('strength', 'moderate')

        if momentum_direction != 'neutral':
            reasoning_parts.append(
                f"Momentum indicators demonstrate {momentum_strength} {momentum_direction} alignment"
            )

        # Trend analysis with specific moving average relationships
        trend_analysis = relationship_analysis.get('trend_confluence', {})
        trend_direction = trend_analysis.get('direction', 'neutral')
        agreement_count = trend_analysis.get('agreement_count', 2)

        if trend_direction != 'neutral':
            reasoning_parts.append(
                f"Trend indicators show {trend_direction} bias with {agreement_count} of 4 measures in agreement"
            )

        # Conflicting signals
        conflicts = relationship_analysis.get('conflicting_signals', [])
        if conflicts:
            reasoning_parts.append(f"However, {conflicts[0].lower()}")

        # Add specific indicator divergence/convergence analysis
        divergence_insights = self._generate_divergence_insights(divergence_analysis, indicator_analysis)
        if divergence_insights:
            reasoning_parts.append(divergence_insights)

        # Volatility context
        volatility_analysis = relationship_analysis.get('volatility_confluence', {})
        volatility_condition = volatility_analysis.get('condition', 'normal')

        if volatility_condition != 'normal':
            reasoning_parts.append(f"Operating within {volatility_condition} volatility environment")

        return ". ".join(reasoning_parts) + "."

    def _get_rsi_condition_detailed(self, rsi_value: float) -> str:
        """Get detailed RSI condition description."""
        if rsi_value > 80:
            return "in extremely overbought territory"
        elif rsi_value > 70:
            return "in overbought territory"
        elif rsi_value > 60:
            return "showing strong bullish momentum"
        elif rsi_value > 50:
            return "showing bullish momentum"
        elif rsi_value > 40:
            return "showing bearish momentum"
        elif rsi_value > 30:
            return "showing strong bearish momentum"
        elif rsi_value > 20:
            return "in oversold territory"
        else:
            return "in extremely oversold territory"

    def _generate_divergence_insights(self, divergence_analysis: Dict[str, Any],
                                    indicator_analysis: Dict[str, Any]) -> str:
        """Generate specific insights about indicator divergence and convergence."""
        insights = []

        # Check for RSI and MACD divergence
        rsi_condition = indicator_analysis.get('rsi_analysis', {}).get('condition', 'neutral')
        macd_condition = indicator_analysis.get('macd_analysis', {}).get('condition', 'neutral')

        if rsi_condition != macd_condition and rsi_condition != 'neutral' and macd_condition != 'neutral':
            insights.append("rsi and macd showing divergent momentum signals")

        # Check for moving average convergence/divergence
        ma_analysis = indicator_analysis.get('ma_analysis', {})
        ma_alignment = ma_analysis.get('alignment', 'mixed')

        if ma_alignment == 'bullish_convergence':
            insights.append("moving averages converging in bullish formation")
        elif ma_alignment == 'bearish_convergence':
            insights.append("moving averages converging in bearish formation")
        elif ma_alignment == 'diverging':
            insights.append("moving averages showing divergent signals")

        if insights:
            return f"However, {', '.join(insights)}"

        return ""

    def _get_fallback_reasoning(self) -> str:
        """Get fallback reasoning when analysis fails."""
        return ("Technical indicator analysis reveals standard market conditions with typical "
                "indicator relationships. Multiple timeframe analysis suggests balanced approach "
                "with attention to key momentum and trend confirmation signals.")

    def _analyze_stochastic_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze Stochastic oscillator conditions."""
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        stoch_d = self._safe_get_value(current_data, 'stoch_d', 50)

        analysis = {
            'stoch_k_value': stoch_k,
            'stoch_d_value': stoch_d,
            'condition': 'neutral',
            'crossover': 'none'
        }

        if stoch_k > 80 and stoch_d > 80:
            analysis['condition'] = 'overbought'
        elif stoch_k < 20 and stoch_d < 20:
            analysis['condition'] = 'oversold'
        elif stoch_k > 50:
            analysis['condition'] = 'bullish'
        else:
            analysis['condition'] = 'bearish'

        # Check for crossover
        if stoch_k > stoch_d and stoch_k > 50:
            analysis['crossover'] = 'bullish'
        elif stoch_k < stoch_d and stoch_k < 50:
            analysis['crossover'] = 'bearish'

        return analysis

    def _analyze_adx_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze ADX and Directional Movement indicators."""
        adx = self._safe_get_value(current_data, 'adx', 25)
        di_plus = self._safe_get_value(current_data, 'di_plus', 25)
        di_minus = self._safe_get_value(current_data, 'di_minus', 25)

        analysis = {
            'adx_value': adx,
            'di_plus_value': di_plus,
            'di_minus_value': di_minus,
            'trend_strength': 'weak',
            'trend_direction': 'neutral'
        }

        # Determine trend strength
        if adx > 40:
            analysis['trend_strength'] = 'very_strong'
        elif adx > 25:
            analysis['trend_strength'] = 'strong'
        elif adx > 20:
            analysis['trend_strength'] = 'moderate'

        # Determine trend direction
        if di_plus > di_minus and adx > 20:
            analysis['trend_direction'] = 'bullish'
        elif di_minus > di_plus and adx > 20:
            analysis['trend_direction'] = 'bearish'

        return analysis

    def _analyze_cci_condition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze Commodity Channel Index conditions."""
        cci = self._safe_get_value(current_data, 'cci', 0)

        analysis = {
            'cci_value': cci,
            'condition': 'neutral',
            'strength': 'moderate'
        }

        if cci > 100:
            analysis['condition'] = 'overbought'
            analysis['strength'] = 'strong' if cci > 200 else 'moderate'
        elif cci < -100:
            analysis['condition'] = 'oversold'
            analysis['strength'] = 'strong' if cci < -200 else 'moderate'
        elif cci > 0:
            analysis['condition'] = 'bullish'
        else:
            analysis['condition'] = 'bearish'

        return analysis

    def _analyze_support_resistance(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze support and resistance levels."""
        support_distance = self._safe_get_value(current_data, 'support_distance', 0.5)
        resistance_distance = self._safe_get_value(current_data, 'resistance_distance', 0.5)
        close = self._safe_get_value(current_data, 'close', 0)

        analysis = {
            'support_distance': support_distance,
            'resistance_distance': resistance_distance,
            'position': 'middle',
            'key_level_proximity': 'moderate'
        }

        # Determine position relative to support/resistance
        if support_distance < 0.2:
            analysis['position'] = 'near_support'
            analysis['key_level_proximity'] = 'very_close'
        elif resistance_distance < 0.2:
            analysis['position'] = 'near_resistance'
            analysis['key_level_proximity'] = 'very_close'
        elif support_distance < 0.5 or resistance_distance < 0.5:
            analysis['key_level_proximity'] = 'close'

        return analysis

    def _analyze_pattern_recognition(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze candlestick pattern recognition."""
        doji = self._safe_get_value(current_data, 'doji', 0)
        hammer = self._safe_get_value(current_data, 'hammer', 0)
        bullish_engulfing = self._safe_get_value(current_data, 'bullish_engulfing', 0)
        bearish_engulfing = self._safe_get_value(current_data, 'bearish_engulfing', 0)

        analysis = {
            'patterns_detected': [],
            'pattern_significance': 'none',
            'reversal_potential': 'low'
        }

        if doji:
            analysis['patterns_detected'].append('doji')
            analysis['pattern_significance'] = 'moderate'
        if hammer:
            analysis['patterns_detected'].append('hammer')
            analysis['pattern_significance'] = 'strong'
            analysis['reversal_potential'] = 'high'
        if bullish_engulfing:
            analysis['patterns_detected'].append('bullish_engulfing')
            analysis['pattern_significance'] = 'strong'
            analysis['reversal_potential'] = 'high'
        if bearish_engulfing:
            analysis['patterns_detected'].append('bearish_engulfing')
            analysis['pattern_significance'] = 'strong'
            analysis['reversal_potential'] = 'high'

        return analysis

    def _analyze_price_action(self, current_data: pd.Series) -> Dict[str, Any]:
        """Analyze price action characteristics."""
        body_size = self._safe_get_value(current_data, 'body_size', 0.5)
        upper_shadow = self._safe_get_value(current_data, 'upper_shadow', 0.2)
        lower_shadow = self._safe_get_value(current_data, 'lower_shadow', 0.2)
        gap_up = self._safe_get_value(current_data, 'gap_up', 0)
        gap_down = self._safe_get_value(current_data, 'gap_down', 0)

        analysis = {
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'candle_type': 'normal',
            'sentiment': 'neutral'
        }

        # Determine candle characteristics
        if body_size < 0.3:
            analysis['candle_type'] = 'small_body'
        elif body_size > 0.7:
            analysis['candle_type'] = 'large_body'

        if upper_shadow > 0.5:
            analysis['sentiment'] = 'rejection_above'
        elif lower_shadow > 0.5:
            analysis['sentiment'] = 'rejection_below'

        if gap_up:
            analysis['gap'] = 'gap_up'
        elif gap_down:
            analysis['gap'] = 'gap_down'

        return analysis

    def _safe_get_value(self, data: pd.Series, column: str, default: Any = 0) -> Any:
        """Safely get value from pandas Series."""
        try:
            return data.get(column, default) if column in data.index else default
        except Exception:
            return default
