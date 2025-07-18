#!/usr/bin/env python3
"""
Context Analysis Engine
======================

Generates market context analysis reasoning focusing on trend strength,
market regime, and technical confluence without cross-timeframe references.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


class ContextAnalysisEngine(BaseReasoningEngine):
    """
    Generates comprehensive context analysis reasoning for market conditions.
    
    Analyzes trend characteristics, market regime, technical confluence,
    and momentum patterns while maintaining timeframe independence.
    """
    
    def _initialize_config(self):
        """Initialize context analysis specific configuration."""
        self.context_templates = self._load_context_templates()
        self.confluence_templates = self._load_confluence_templates()
        
        # Configuration thresholds
        self.config.setdefault('trend_strength_thresholds', {
            'strong': 0.6, 'moderate': 0.3, 'weak': 0.0
        })
        self.config.setdefault('momentum_thresholds', {
            'overbought': 70, 'oversold': 30, 'neutral_high': 55, 'neutral_low': 45
        })
        self.config.setdefault('volatility_thresholds', {
            'high': 3.0, 'normal': 1.5, 'low': 0.8
        })
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for context analysis."""
        return [
            'trend_strength', 'trend_direction', 'rsi_14', 'macd_histogram',
            'volatility_20', 'sma_5_20_cross', 'sma_10_50_cross'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate comprehensive context analysis reasoning.
        
        Args:
            current_data: Current row data with all features
            context: Historical context from HistoricalContextManager
            
        Returns:
            Professional context analysis reasoning text
        """
        try:
            # Analyze current market structure
            market_structure = self._analyze_market_structure(current_data, context)
            
            # Analyze trend characteristics
            trend_analysis = self._analyze_trend_characteristics(current_data, context)
            
            # Analyze technical confluence
            technical_confluence = self._analyze_technical_confluence(current_data)
            
            # Analyze momentum environment
            momentum_analysis = self._analyze_momentum_environment(current_data, context)
            
            # Construct comprehensive reasoning
            reasoning = self._construct_context_reasoning(
                market_structure, trend_analysis, technical_confluence, momentum_analysis, context
            )
            
            self._log_reasoning_generation(len(reasoning), "good")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in context analysis reasoning: {str(e)}")
            return self._get_fallback_reasoning(current_data, context)
    
    def _load_context_templates(self) -> Dict[str, str]:
        """Load context analysis templates for different market regimes."""
        return {
            'trending': "strong directional movement with consistent momentum characteristics",
            'consolidation': "range-bound price action with defined boundaries and neutral momentum",
            'volatile': "heightened price volatility with rapid directional changes",
            'transitional': "evolving market structure with shifting momentum dynamics",
            'breakout': "emerging from consolidation with potential for sustained movement"
        }
    
    def _load_confluence_templates(self) -> Dict[str, str]:
        """Load technical confluence analysis templates."""
        return {
            'bullish_confluence': "Technical indicators demonstrate bullish confluence across multiple analysis layers",
            'bearish_confluence': "Technical indicators show bearish alignment with consistent directional bias",
            'mixed_signals': "Mixed technical signals requiring careful interpretation and selective positioning",
            'neutral_confluence': "Technical indicators present balanced readings with no clear directional bias",
            'divergent_signals': "Conflicting technical signals suggest caution and wait-and-see approach"
        }
    
    def _analyze_market_structure(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current market structure characteristics.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Market structure analysis
        """
        structure = {
            'regime': context.get('market_regime', 'consolidation'),
            'trend_strength': 'weak',
            'trend_consistency': 'low',
            'volatility_state': 'normal'
        }
        
        # Trend strength analysis
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0.0)
        thresholds = self.config['trend_strength_thresholds']
        structure['trend_strength'] = self._get_strength_descriptor(trend_strength, thresholds)
        
        # Trend consistency from historical context
        trend_analysis = context.get('trend_analysis', {})
        structure['trend_consistency'] = trend_analysis.get('consistency', 'low')
        
        # Volatility state
        volatility = self._safe_get_value(current_data, 'volatility_20', 1.5)
        vol_thresholds = self.config['volatility_thresholds']
        
        if volatility >= vol_thresholds['high']:
            structure['volatility_state'] = 'high'
        elif volatility <= vol_thresholds['low']:
            structure['volatility_state'] = 'low'
        else:
            structure['volatility_state'] = 'normal'
        
        return structure
    
    def _analyze_trend_characteristics(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze trend characteristics and directional bias.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Trend characteristics analysis
        """
        trend_data = context.get('trend_analysis', {})
        
        characteristics = {
            'direction': trend_data.get('direction', 'neutral'),
            'strength': trend_data.get('strength', 'weak'),
            'consistency': trend_data.get('consistency', 'low'),
            'current_alignment': True,
            'persistence': trend_data.get('trend_persistence', 0.5)
        }
        
        # Current trend confirmation
        current_direction = self._safe_get_value(current_data, 'trend_direction', 0)
        historical_direction = trend_data.get('direction', 'neutral')
        
        if current_direction == 1:
            current_bias = 'bullish'
        elif current_direction == -1:
            current_bias = 'bearish'
        else:
            current_bias = 'neutral'
        
        characteristics['current_alignment'] = (current_bias == historical_direction)
        characteristics['current_bias'] = current_bias
        
        return characteristics
    
    def _analyze_technical_confluence(self, current_data: pd.Series) -> Dict[str, Any]:
        """
        Analyze technical indicator confluence and agreement.
        
        Args:
            current_data: Current row data
            
        Returns:
            Technical confluence analysis
        """
        confluence = {
            'moving_average_alignment': 'neutral',
            'oscillator_agreement': 'mixed',
            'overall_confluence': 'neutral',
            'confluence_strength': 'weak'
        }
        
        # Moving average confluence analysis
        ma_signals = []
        sma_5_20 = self._safe_get_value(current_data, 'sma_5_20_cross', 0)
        sma_10_50 = self._safe_get_value(current_data, 'sma_10_50_cross', 0)
        
        if sma_5_20 != 0:
            ma_signals.append(sma_5_20)
        if sma_10_50 != 0:
            ma_signals.append(sma_10_50)
        
        if ma_signals:
            if all(signal == 1 for signal in ma_signals):
                confluence['moving_average_alignment'] = 'bullish'
            elif all(signal == -1 for signal in ma_signals):
                confluence['moving_average_alignment'] = 'bearish'
            elif len(set(ma_signals)) == 1:  # All same but not 1 or -1
                confluence['moving_average_alignment'] = 'neutral'
            else:
                confluence['moving_average_alignment'] = 'mixed'
        
        # Oscillator analysis
        oscillator_signals = []
        
        # RSI analysis
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        momentum_thresholds = self.config['momentum_thresholds']
        
        if rsi >= momentum_thresholds['neutral_high']:
            oscillator_signals.append('bullish')
        elif rsi <= momentum_thresholds['neutral_low']:
            oscillator_signals.append('bearish')
        else:
            oscillator_signals.append('neutral')
        
        # MACD analysis
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0:
            oscillator_signals.append('bullish')
        elif macd_hist < 0:
            oscillator_signals.append('bearish')
        else:
            oscillator_signals.append('neutral')
        
        # Determine oscillator agreement
        bullish_osc = oscillator_signals.count('bullish')
        bearish_osc = oscillator_signals.count('bearish')
        neutral_osc = oscillator_signals.count('neutral')
        
        if bullish_osc > bearish_osc and bullish_osc > neutral_osc:
            confluence['oscillator_agreement'] = 'bullish'
        elif bearish_osc > bullish_osc and bearish_osc > neutral_osc:
            confluence['oscillator_agreement'] = 'bearish'
        elif neutral_osc >= max(bullish_osc, bearish_osc):
            confluence['oscillator_agreement'] = 'neutral'
        else:
            confluence['oscillator_agreement'] = 'mixed'
        
        # Overall confluence assessment
        ma_align = confluence['moving_average_alignment']
        osc_agree = confluence['oscillator_agreement']
        
        if ma_align == osc_agree and ma_align in ['bullish', 'bearish']:
            confluence['overall_confluence'] = ma_align
            confluence['confluence_strength'] = 'strong'
        elif ma_align in ['bullish', 'bearish'] and osc_agree == 'neutral':
            confluence['overall_confluence'] = ma_align
            confluence['confluence_strength'] = 'moderate'
        elif osc_agree in ['bullish', 'bearish'] and ma_align == 'neutral':
            confluence['overall_confluence'] = osc_agree
            confluence['confluence_strength'] = 'moderate'
        else:
            confluence['overall_confluence'] = 'mixed'
            confluence['confluence_strength'] = 'weak'
        
        return confluence
    
    def _analyze_momentum_environment(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current momentum environment and characteristics.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Momentum environment analysis
        """
        momentum = {
            'rsi_state': 'neutral',
            'macd_state': 'neutral',
            'momentum_consistency': 'low',
            'momentum_extremes': False
        }
        
        # RSI state analysis
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        thresholds = self.config['momentum_thresholds']
        
        if rsi >= thresholds['overbought']:
            momentum['rsi_state'] = 'overbought'
            momentum['momentum_extremes'] = True
        elif rsi <= thresholds['oversold']:
            momentum['rsi_state'] = 'oversold'
            momentum['momentum_extremes'] = True
        elif rsi >= thresholds['neutral_high']:
            momentum['rsi_state'] = 'bullish'
        elif rsi <= thresholds['neutral_low']:
            momentum['rsi_state'] = 'bearish'
        
        # MACD state analysis
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0:
            momentum['macd_state'] = 'bullish'
        elif macd_hist < 0:
            momentum['macd_state'] = 'bearish'
        
        # Momentum consistency with historical context
        momentum_analysis = context.get('momentum_analysis', {})
        momentum['momentum_consistency'] = momentum_analysis.get('momentum_consistency', 'low')
        
        return momentum
    
    def _construct_context_reasoning(self, market_structure: Dict, trend_analysis: Dict,
                                   technical_confluence: Dict, momentum_analysis: Dict,
                                   context: Dict) -> str:
        """
        Construct comprehensive context analysis reasoning.
        
        Args:
            market_structure: Market structure analysis
            trend_analysis: Trend characteristics
            technical_confluence: Technical confluence analysis
            momentum_analysis: Momentum environment analysis
            context: Historical context
            
        Returns:
            Complete context reasoning text
        """
        reasoning_parts = []
        
        # Market regime and structure
        regime = market_structure['regime']
        regime_template = self.context_templates.get(regime, "current market conditions")
        reasoning_parts.append(f"Market structure reveals {regime_template}")
        
        # Trend analysis integration
        trend_dir = trend_analysis['direction']
        trend_str = trend_analysis['strength']
        trend_cons = trend_analysis['consistency']
        
        if trend_dir != 'neutral':
            reasoning_parts.append(
                f"with {trend_str} {trend_dir} trend showing {trend_cons} consistency over recent sessions"
            )
        else:
            reasoning_parts.append(
                f"characterized by neutral directional bias and {trend_str} momentum characteristics"
            )
        
        # Technical confluence assessment
        confluence_type = technical_confluence['overall_confluence']
        confluence_strength = technical_confluence['confluence_strength']
        
        if confluence_type in ['bullish', 'bearish']:
            reasoning_parts.append(
                f"Technical indicators demonstrate {confluence_strength} {confluence_type} confluence across analysis layers"
            )
        else:
            reasoning_parts.append(
                "Technical indicators present mixed signals requiring careful interpretation"
            )
        
        # Momentum environment
        if momentum_analysis['momentum_extremes']:
            rsi_state = momentum_analysis['rsi_state']
            reasoning_parts.append(
                f"Momentum indicators show {rsi_state} conditions suggesting potential reversal scenarios"
            )
        else:
            momentum_consistency = momentum_analysis['momentum_consistency']
            reasoning_parts.append(
                f"Momentum environment remains balanced with {momentum_consistency} directional consistency"
            )
        
        # Volatility context
        vol_analysis = context.get('volatility_analysis', {})
        vol_level = vol_analysis.get('level', 'normal')
        vol_trend = vol_analysis.get('trend', 'stable')
        
        reasoning_parts.append(
            f"Current volatility environment registers {vol_level} levels with {vol_trend} characteristics"
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
            "Market structure reveals consolidation characteristics with neutral directional bias. "
            "Technical indicators present mixed signals requiring careful interpretation. "
            "Current volatility environment remains within normal parameters for the timeframe."
        )
