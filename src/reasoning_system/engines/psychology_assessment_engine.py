#!/usr/bin/env python3
"""
Psychology Assessment Engine
===========================

Generates market psychology interpretation reasoning that analyzes participant
behavior, sentiment indicators, and crowd psychology from price action patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


class PsychologyAssessmentEngine(BaseReasoningEngine):
    """
    Generates market psychology assessment reasoning based on price action,
    momentum indicators, and behavioral patterns.
    """
    
    def _initialize_config(self):
        """Initialize psychology assessment specific configuration."""
        self.psychology_templates = self._load_psychology_templates()
        self.sentiment_indicators = self._load_sentiment_indicators()
        
        # Configuration for psychology analysis
        self.config.setdefault('fear_greed_thresholds', {
            'extreme_fear': 20, 'fear': 35, 'neutral': 65, 'greed': 80, 'extreme_greed': 95
        })
        self.config.setdefault('momentum_psychology_thresholds', {
            'panic': 15, 'fear': 30, 'neutral_low': 45, 'neutral_high': 55, 'euphoria': 85
        })
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for psychology assessment."""
        return [
            'rsi_14', 'williams_r', 'cci', 'price_change', 'hl_range',
            'body_size', 'upper_shadow', 'lower_shadow'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate market psychology assessment reasoning.
        
        Args:
            current_data: Current row data with all features
            context: Historical context from HistoricalContextManager
            
        Returns:
            Professional psychology assessment reasoning text
        """
        try:
            # Analyze current sentiment indicators
            sentiment_analysis = self._analyze_sentiment_indicators(current_data)

            # Analyze price action psychology
            price_action_psychology = self._analyze_price_action_psychology(current_data)
            
            # Analyze crowd behavior patterns
            crowd_behavior = self._analyze_crowd_behavior(current_data, context)
            
            # Analyze institutional vs retail behavior
            institutional_analysis = self._analyze_institutional_behavior(current_data, context)
            
            # Construct psychology reasoning
            reasoning = self._construct_psychology_reasoning(
                sentiment_analysis, price_action_psychology, crowd_behavior,
                institutional_analysis, context, current_data
            )
            
            self._log_reasoning_generation(len(reasoning), "good")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in psychology assessment reasoning: {str(e)}")
            return self._get_fallback_reasoning(current_data, context)
    
    def _load_psychology_templates(self) -> Dict[str, Dict[str, str]]:
        """Load psychology assessment templates for different market states."""
        return {
            'fear': {
                'description': "fear-driven behavior with defensive positioning",
                'characteristics': ["selling pressure", "risk aversion", "oversold conditions"],
                'implications': ["potential reversal", "value opportunities", "contrarian signals"]
            },
            'greed': {
                'description': "greed-driven behavior with aggressive positioning",
                'characteristics': ["buying pressure", "risk appetite", "overbought conditions"],
                'implications': ["potential reversal", "distribution phase", "caution warranted"]
            },
            'uncertainty': {
                'description': "uncertainty with mixed participant behavior",
                'characteristics': ["indecision", "range-bound trading", "low conviction"],
                'implications': ["awaiting catalysts", "consolidation phase", "breakout potential"]
            },
            'confidence': {
                'description': "confident participant behavior with clear directional bias",
                'characteristics': ["strong conviction", "trend following", "momentum building"],
                'implications': ["trend continuation", "institutional participation", "sustained movement"]
            },
            'panic': {
                'description': "panic-driven behavior with forced liquidation",
                'characteristics': ["capitulation", "volume spikes", "extreme oversold"],
                'implications': ["washout bottom", "reversal potential", "opportunity for contrarians"]
            },
            'euphoria': {
                'description': "euphoric behavior with excessive optimism",
                'characteristics': ["FOMO buying", "momentum chasing", "extreme overbought"],
                'implications': ["distribution warning", "reversal risk", "profit-taking zone"]
            }
        }
    
    def _load_sentiment_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Load sentiment indicator configurations and interpretations."""
        return {
            'rsi': {
                'extreme_oversold': 20, 'oversold': 30, 'neutral_low': 45,
                'neutral_high': 55, 'overbought': 70, 'extreme_overbought': 80
            },
            'williams_r': {
                'extreme_oversold': -90, 'oversold': -80, 'neutral_low': -60,
                'neutral_high': -40, 'overbought': -20, 'extreme_overbought': -10
            },
            'cci': {
                'extreme_oversold': -200, 'oversold': -100, 'neutral_low': -50,
                'neutral_high': 50, 'overbought': 100, 'extreme_overbought': 200
            }
        }
    
    def _analyze_sentiment_indicators(self, current_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze sentiment based on momentum oscillators.
        
        Args:
            current_data: Current row data
            
        Returns:
            Sentiment analysis based on indicators
        """
        sentiment = {
            'overall_sentiment': 'neutral',
            'fear_greed_level': 'neutral',
            'extreme_conditions': False,
            'sentiment_strength': 'moderate'
        }
        
        # RSI sentiment analysis
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        rsi_config = self.sentiment_indicators['rsi']
        
        rsi_sentiment = 'neutral'
        if rsi <= rsi_config['extreme_oversold']:
            rsi_sentiment = 'extreme_fear'
            sentiment['extreme_conditions'] = True
        elif rsi <= rsi_config['oversold']:
            rsi_sentiment = 'fear'
        elif rsi >= rsi_config['extreme_overbought']:
            rsi_sentiment = 'extreme_greed'
            sentiment['extreme_conditions'] = True
        elif rsi >= rsi_config['overbought']:
            rsi_sentiment = 'greed'
        elif rsi >= rsi_config['neutral_high']:
            rsi_sentiment = 'mild_greed'
        elif rsi <= rsi_config['neutral_low']:
            rsi_sentiment = 'mild_fear'
        
        # Williams %R sentiment analysis
        williams_r = self._safe_get_value(current_data, 'williams_r', -50)
        williams_config = self.sentiment_indicators['williams_r']
        
        williams_sentiment = 'neutral'
        if williams_r <= williams_config['extreme_oversold']:
            williams_sentiment = 'extreme_fear'
        elif williams_r <= williams_config['oversold']:
            williams_sentiment = 'fear'
        elif williams_r >= williams_config['extreme_overbought']:
            williams_sentiment = 'extreme_greed'
        elif williams_r >= williams_config['overbought']:
            williams_sentiment = 'greed'
        
        # CCI sentiment analysis
        cci = self._safe_get_value(current_data, 'cci', 0)
        cci_config = self.sentiment_indicators['cci']
        
        cci_sentiment = 'neutral'
        if cci <= cci_config['extreme_oversold']:
            cci_sentiment = 'extreme_fear'
        elif cci <= cci_config['oversold']:
            cci_sentiment = 'fear'
        elif cci >= cci_config['extreme_overbought']:
            cci_sentiment = 'extreme_greed'
        elif cci >= cci_config['overbought']:
            cci_sentiment = 'greed'
        
        # Aggregate sentiment
        sentiments = [rsi_sentiment, williams_sentiment, cci_sentiment]
        fear_count = sum(1 for s in sentiments if 'fear' in s)
        greed_count = sum(1 for s in sentiments if 'greed' in s)
        extreme_count = sum(1 for s in sentiments if 'extreme' in s)
        
        if extreme_count >= 2:
            sentiment['sentiment_strength'] = 'extreme'
        elif fear_count >= 2 or greed_count >= 2:
            sentiment['sentiment_strength'] = 'strong'
        
        if fear_count > greed_count:
            sentiment['overall_sentiment'] = 'fear'
            sentiment['fear_greed_level'] = 'fear' if fear_count >= 2 else 'mild_fear'
        elif greed_count > fear_count:
            sentiment['overall_sentiment'] = 'greed'
            sentiment['fear_greed_level'] = 'greed' if greed_count >= 2 else 'mild_greed'
        
        sentiment['indicator_details'] = {
            'rsi_sentiment': rsi_sentiment,
            'williams_sentiment': williams_sentiment,
            'cci_sentiment': cci_sentiment
        }
        
        return sentiment
    
    def _analyze_price_action_psychology(self, current_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze psychology from price action characteristics.
        
        Args:
            current_data: Current row data
            
        Returns:
            Price action psychology analysis
        """
        psychology = {
            'candle_psychology': 'neutral',
            'shadow_analysis': 'balanced',
            'volatility_psychology': 'normal',
            'price_rejection': False
        }
        
        # Candle body and shadow analysis
        body_size = self._safe_get_value(current_data, 'body_size', 0.5)
        upper_shadow = self._safe_get_value(current_data, 'upper_shadow', 0.2)
        lower_shadow = self._safe_get_value(current_data, 'lower_shadow', 0.2)
        
        # Body size psychology
        if body_size > 1.5:
            psychology['candle_psychology'] = 'strong_conviction'
        elif body_size < 0.3:
            psychology['candle_psychology'] = 'indecision'
        
        # Shadow analysis for rejection patterns
        if lower_shadow > body_size * 2:
            psychology['shadow_analysis'] = 'lower_rejection'
            psychology['price_rejection'] = True
        elif upper_shadow > body_size * 2:
            psychology['shadow_analysis'] = 'upper_rejection'
            psychology['price_rejection'] = True
        elif lower_shadow > upper_shadow * 2:
            psychology['shadow_analysis'] = 'buying_interest'
        elif upper_shadow > lower_shadow * 2:
            psychology['shadow_analysis'] = 'selling_pressure'
        
        # Volatility psychology
        hl_range = self._safe_get_value(current_data, 'hl_range', 1.0)
        if hl_range > 2.0:
            psychology['volatility_psychology'] = 'high_emotion'
        elif hl_range < 0.5:
            psychology['volatility_psychology'] = 'low_interest'
        
        return psychology
    
    def _analyze_crowd_behavior(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze crowd behavior patterns from historical context.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Crowd behavior analysis
        """
        crowd = {
            'crowd_sentiment': 'neutral',
            'herd_behavior': 'moderate',
            'contrarian_signals': False,
            'crowd_extremes': False
        }
        
        # Analyze recent signal bias for crowd behavior
        recent_signals = context.get('recent_signals', {})
        signal_bias = recent_signals.get('signal_bias', 'neutral')
        
        if signal_bias == 'bullish':
            crowd['crowd_sentiment'] = 'optimistic'
        elif signal_bias == 'bearish':
            crowd['crowd_sentiment'] = 'pessimistic'
        
        # Analyze pattern frequency for herd behavior
        pattern_freq = context.get('pattern_frequency', {})
        pattern_bias = pattern_freq.get('pattern_bias', 'neutral')
        total_patterns = pattern_freq.get('total_patterns', 0)
        
        if total_patterns > 10:  # High pattern frequency suggests herd behavior
            crowd['herd_behavior'] = 'high'
            if pattern_bias != 'neutral':
                crowd['crowd_sentiment'] = f"{pattern_bias}_herd"
        
        # Check for contrarian signals (extreme sentiment conditions)
        sentiment_analysis = self._analyze_sentiment_indicators(current_data)
        if sentiment_analysis['extreme_conditions']:
            crowd['contrarian_signals'] = True
            crowd['crowd_extremes'] = True
        
        return crowd
    
    def _analyze_institutional_behavior(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze institutional vs retail behavior patterns.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Institutional behavior analysis
        """
        institutional = {
            'smart_money_activity': 'neutral',
            'institutional_bias': 'neutral',
            'accumulation_distribution': 'neutral'
        }
        
        # Analyze trend persistence for institutional activity
        trend_analysis = context.get('trend_analysis', {})
        trend_persistence = trend_analysis.get('trend_persistence', 0.5)
        trend_strength = trend_analysis.get('strength', 'weak')
        
        if trend_persistence > 0.7 and trend_strength in ['moderate', 'strong']:
            institutional['smart_money_activity'] = 'active'
            institutional['institutional_bias'] = trend_analysis.get('direction', 'neutral')
        
        # Analyze support/resistance behavior
        sr_history = context.get('support_resistance_history', {})
        support_tests = sr_history.get('support_tests', 0)
        resistance_tests = sr_history.get('resistance_tests', 0)
        
        if support_tests > 3:
            institutional['accumulation_distribution'] = 'accumulation'
        elif resistance_tests > 3:
            institutional['accumulation_distribution'] = 'distribution'
        
        return institutional
    
    def _construct_psychology_reasoning(self, sentiment_analysis: Dict, price_action_psychology: Dict,
                                      crowd_behavior: Dict, institutional_analysis: Dict,
                                      context: Dict, current_data: pd.Series = None) -> str:
        """
        Construct comprehensive psychology assessment reasoning.
        
        Args:
            sentiment_analysis: Sentiment indicator analysis
            price_action_psychology: Price action psychology
            crowd_behavior: Crowd behavior analysis
            institutional_analysis: Institutional behavior analysis
            context: Historical context
            
        Returns:
            Complete psychology reasoning text
        """
        reasoning_parts = []
        
        # Primary sentiment assessment using specific price action analysis
        overall_sentiment = sentiment_analysis['overall_sentiment']
        sentiment_strength = sentiment_analysis['sentiment_strength']

        # Generate comprehensive sentiment analysis with specific indicator values
        sentiment_description = self._generate_comprehensive_sentiment_analysis(
            sentiment_analysis, price_action_psychology, crowd_behavior, current_data
        )
        reasoning_parts.append(sentiment_description)
        
        # Price action psychology integration with specific analysis
        price_action_insight = self._generate_price_action_psychology_insight(price_action_psychology)
        if price_action_insight:
            reasoning_parts.append(price_action_insight)

        
        # Crowd behavior analysis
        crowd_sentiment = crowd_behavior['crowd_sentiment']
        if crowd_behavior['contrarian_signals']:
            reasoning_parts.append(
                "Extreme sentiment conditions suggest potential contrarian opportunities as crowd positioning reaches unsustainable levels"
            )
        elif crowd_sentiment != 'neutral':
            reasoning_parts.append(f"Crowd behavior indicates {crowd_sentiment} bias in recent positioning")
        
        # Institutional behavior
        smart_money = institutional_analysis['smart_money_activity']
        if smart_money == 'active':
            institutional_bias = institutional_analysis['institutional_bias']
            reasoning_parts.append(
                f"Smart money activity suggests {institutional_bias} institutional positioning with sustained directional bias"
            )
        
        # Historical context integration
        if sentiment_strength == 'extreme':
            reasoning_parts.append(
                "Historical precedent suggests such extreme sentiment conditions often precede significant reversals"
            )
        
        return ". ".join(reasoning_parts) + "."

    def _generate_comprehensive_sentiment_analysis(self, sentiment_analysis: Dict[str, Any],
                                                 price_action_psychology: Dict[str, Any],
                                                 crowd_behavior: Dict[str, Any],
                                                 current_data: pd.Series) -> str:
        """Generate comprehensive sentiment analysis with specific indicator values."""

        # Get actual indicator values
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        williams_r = self._safe_get_value(current_data, 'williams_r', -50)
        cci = self._safe_get_value(current_data, 'cci', 0)
        atr = self._safe_get_value(current_data, 'atr', 50)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)

        sentiment_parts = []

        # RSI sentiment analysis
        rsi_sentiment = self._get_rsi_sentiment_detailed(rsi_14)
        sentiment_parts.append(f"RSI at {rsi_14:.1f} {rsi_sentiment}")

        # Williams %R analysis
        if williams_r > -20:
            williams_sentiment = "confirms overbought conditions suggesting caution"
        elif williams_r < -80:
            williams_sentiment = "indicates oversold levels suggesting potential reversal"
        else:
            williams_sentiment = "shows neutral sentiment with balanced buyer-seller dynamics"
        sentiment_parts.append(f"Williams %R at {williams_r:.1f} {williams_sentiment}")

        # Volatility sentiment
        if atr > 60:
            volatility_sentiment = f"Current volatility at {atr/50:.1f}x ATR shows heightened emotional trading"
            sentiment_parts.append(volatility_sentiment)
        elif atr < 30:
            volatility_sentiment = f"Low volatility at {atr/50:.1f}x ATR suggests calm market sentiment"
            sentiment_parts.append(volatility_sentiment)

        # Market psychology context
        psychology_context = self._get_market_psychology_context_detailed(sentiment_analysis, crowd_behavior)
        if psychology_context:
            sentiment_parts.append(psychology_context)

        return ". ".join(sentiment_parts) + "."

    def _generate_specific_sentiment_analysis(self, sentiment_analysis: Dict,
                                            price_action_psychology: Dict,
                                            crowd_behavior: Dict, current_data: pd.Series = None) -> str:
        """Generate specific sentiment analysis using actual market data like a real trader."""
        if current_data is None:
            return "Market sentiment analysis requires current data"

        # Get actual indicator values using correct column names
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        williams_r = self._safe_get_value(current_data, 'williams_r', -50)
        close_price = self._safe_get_value(current_data, 'close', 0)
        volume_sma = self._safe_get_value(current_data, 'volume_sma', 0)
        atr = self._safe_get_value(current_data, 'atr', 0)

        # Analyze sentiment using actual data like a trader would
        sentiment_parts = []

        # RSI-based sentiment analysis
        if rsi > 75:
            sentiment_parts.append(f"RSI at {rsi:.1f} indicates extreme bullish sentiment with potential for profit-taking")
        elif rsi > 60:
            sentiment_parts.append(f"RSI at {rsi:.1f} shows strong bullish sentiment with buyers in control")
        elif rsi < 25:
            sentiment_parts.append(f"RSI at {rsi:.1f} reflects extreme bearish sentiment with potential for oversold bounce")
        elif rsi < 40:
            sentiment_parts.append(f"RSI at {rsi:.1f} indicates bearish sentiment with sellers dominating")
        else:
            sentiment_parts.append(f"RSI at {rsi:.1f} suggests neutral sentiment with balanced buyer-seller dynamics")

        # Williams %R confirmation
        if williams_r > -20:
            sentiment_parts.append(f"Williams %R at {williams_r:.1f} confirms overbought conditions suggesting caution")
        elif williams_r < -80:
            sentiment_parts.append(f"Williams %R at {williams_r:.1f} confirms oversold levels indicating potential reversal")

        # Volume and volatility context for sentiment
        if atr > 0:
            # Calculate current volatility relative to ATR
            open_price = self._safe_get_value(current_data, 'open', close_price)
            current_range = abs(self._safe_get_value(current_data, 'high', close_price) -
                               self._safe_get_value(current_data, 'low', close_price))
            volatility_ratio = current_range / atr if atr > 0 else 1

            if volatility_ratio > 1.5:
                sentiment_parts.append(f"Current volatility at {volatility_ratio:.1f}x ATR indicates heightened emotional trading")
            elif volatility_ratio < 0.5:
                sentiment_parts.append(f"Low volatility at {volatility_ratio:.1f}x ATR suggests calm market sentiment")

        return ". ".join(sentiment_parts) if sentiment_parts else "Market sentiment analysis requires sufficient data"

    def _generate_price_action_psychology_insight(self, price_action_psychology: Dict) -> str:
        """Generate specific insights from price action psychology."""
        candle_psychology = price_action_psychology.get('candle_psychology', 'neutral')
        shadow_analysis = price_action_psychology.get('shadow_analysis', 'balanced')
        price_rejection = price_action_psychology.get('price_rejection', False)

        insights = []

        # Analyze conviction levels
        if candle_psychology == 'strong_conviction':
            insights.append("evidenced by strong conviction in current price action")
        elif candle_psychology == 'indecision':
            insights.append("reflected in indecisive price action with limited conviction")

        # Analyze rejection patterns for psychology
        if price_rejection:
            if shadow_analysis == 'lower_rejection':
                insights.append("with clear rejection of lower price levels indicating buying interest")
            elif shadow_analysis == 'upper_rejection':
                insights.append("showing rejection of higher prices suggesting selling pressure")

        # Analyze gaps for psychology (if available)
        gap_psychology = price_action_psychology.get('gap_psychology')
        if gap_psychology == 'gap_up':
            insights.append("with opening gap higher reflecting overnight optimism")
        elif gap_psychology == 'gap_down':
            insights.append("with opening gap lower indicating overnight pessimism")

        return ". ".join(insights) if insights else ""

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
            "Market participants demonstrate balanced sentiment with no clear emotional extremes. "
            "Current price action reflects normal trading behavior without significant fear or greed indicators. "
            "Crowd psychology appears neutral with standard risk appetite and positioning characteristics."
        )

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

    def _get_rsi_sentiment_detailed(self, rsi_value: float) -> str:
        """Get detailed RSI sentiment description."""
        if rsi_value > 80:
            return "shows extreme bullish sentiment with potential for profit-taking"
        elif rsi_value > 70:
            return "indicates strong bullish sentiment with buyers in control"
        elif rsi_value > 60:
            return "shows bullish sentiment with positive momentum"
        elif rsi_value > 40:
            return "suggests neutral sentiment with balanced buyer-seller dynamics"
        elif rsi_value > 30:
            return "indicates bearish sentiment with sellers dominating"
        elif rsi_value > 20:
            return "shows strong bearish sentiment with oversold conditions"
        else:
            return "demonstrates extreme bearish sentiment with potential for reversal"

    def _get_market_psychology_context_detailed(self, sentiment_analysis: Dict[str, Any],
                                              crowd_behavior: Dict[str, Any]) -> str:
        """Get detailed market psychology context."""
        context_parts = []

        # Add crowd behavior insights
        if crowd_behavior.get('contrarian_signals'):
            context_parts.append("Extreme sentiment conditions suggest potential contrarian opportunities as crowd positioning reaches unsustainable levels")

        # Add historical precedent for extreme conditions
        if sentiment_analysis.get('extreme_conditions'):
            context_parts.append("Historical precedent suggests such extreme sentiment conditions often precede significant reversals")

        return ". ".join(context_parts) if context_parts else ""
