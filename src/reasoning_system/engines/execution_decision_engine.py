#!/usr/bin/env python3
"""
Execution Decision Engine
========================

Generates execution decision reasoning that synthesizes all technical analysis
into coherent trading recommendations with risk-reward assessment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


class ExecutionDecisionEngine(BaseReasoningEngine):
    """
    Generates execution decision reasoning by synthesizing technical analysis,
    market context, and psychology into actionable trading recommendations.
    """
    
    def _initialize_config(self):
        """Initialize execution decision specific configuration."""
        self.decision_templates = self._load_decision_templates()
        self.risk_reward_templates = self._load_risk_reward_templates()
        
        # Configuration for decision making
        self.config.setdefault('confluence_threshold', 0.7)
        self.config.setdefault('risk_reward_minimum', 1.5)
        self.config.setdefault('confidence_threshold', 60)
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for execution decisions."""
        return [
            # Core signal and price data
            'signal', 'open', 'high', 'low', 'close',
            # Oscillators
            'rsi_14', 'rsi_21', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
            # MACD
            'macd', 'macd_signal', 'macd_histogram',
            # Moving Averages
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
            # Bollinger Bands
            'bb_position', 'bb_upper', 'bb_lower', 'bb_width',
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
            'body_size', 'upper_shadow', 'lower_shadow',
            # Crossovers
            'sma_5_20_cross', 'price_vs_sma_20', 'price_vs_ema_20'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate execution decision reasoning.
        
        Args:
            current_data: Current row data with all features
            context: Historical context from HistoricalContextManager
            
        Returns:
            Professional execution decision reasoning text
        """
        try:
            # Analyze technical setup quality
            setup_quality = self._analyze_setup_quality(current_data, context)
            
            # Analyze risk-reward characteristics
            risk_reward = self._analyze_risk_reward(current_data, context)
            
            # Determine execution recommendation
            execution_recommendation = self._determine_execution_recommendation(
                current_data, setup_quality, risk_reward, context
            )
            
            # Analyze timing considerations
            timing_analysis = self._analyze_timing_considerations(current_data, context)
            
            # Construct execution reasoning
            reasoning = self._construct_execution_reasoning(
                setup_quality, risk_reward, execution_recommendation, timing_analysis, context
            )
            
            self._log_reasoning_generation(len(reasoning), "good")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in execution decision reasoning: {str(e)}")
            return self._get_fallback_reasoning(current_data, context)
    
    def _load_decision_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load execution decision templates for different scenarios."""
        return {
            'aggressive_long': {
                'description': "aggressive long positioning",
                'conditions': ["strong bullish confluence", "favorable risk-reward", "trend alignment"],
                'approach': "immediate execution with defined risk parameters"
            },
            'aggressive_short': {
                'description': "aggressive short positioning", 
                'conditions': ["strong bearish confluence", "favorable risk-reward", "trend alignment"],
                'approach': "immediate execution with defined risk parameters"
            },
            'cautious_long': {
                'description': "cautious long positioning",
                'conditions': ["moderate bullish signals", "acceptable risk-reward", "mixed confluence"],
                'approach': "selective positioning with tight risk management"
            },
            'cautious_short': {
                'description': "cautious short positioning",
                'conditions': ["moderate bearish signals", "acceptable risk-reward", "mixed confluence"],
                'approach': "selective positioning with tight risk management"
            },
            'wait_and_see': {
                'description': "wait-and-see approach",
                'conditions': ["mixed signals", "poor risk-reward", "low confluence"],
                'approach': "patience until clearer setup develops"
            },
            'risk_management': {
                'description': "risk management focus",
                'conditions': ["existing positions", "changing conditions", "profit protection"],
                'approach': "position adjustment and risk parameter review"
            }
        }
    
    def _load_risk_reward_templates(self) -> Dict[str, str]:
        """Load risk-reward assessment templates - REMOVED from reasoning text."""
        return {
            'excellent': "",  # Remove from reasoning
            'good': "",       # Remove from reasoning
            'acceptable': "", # Remove from reasoning
            'poor': "",       # Remove from reasoning
            'undefined': ""   # Remove from reasoning
        }
    
    def _analyze_setup_quality(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of the current technical setup.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Setup quality analysis
        """
        quality = {
            'overall_quality': 'moderate',
            'confluence_score': 0.5,
            'trend_alignment': False,
            'pattern_strength': 'weak',
            'support_resistance_context': 'neutral'
        }
        
        # Calculate confluence score
        confluence_factors = []
        
        # Trend strength factor
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0.0)
        if trend_strength > 0.6:
            confluence_factors.append(0.3)
        elif trend_strength > 0.3:
            confluence_factors.append(0.15)
        
        # RSI momentum factor
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 30 <= rsi <= 70:  # Not in extreme territory
            confluence_factors.append(0.2)
        elif rsi > 70 or rsi < 30:  # Extreme territory
            confluence_factors.append(0.1)
        
        # MACD factor
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if abs(macd_hist) > 0:  # Clear MACD direction
            confluence_factors.append(0.2)
        
        # Support/resistance proximity
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)

        if support_dist < 1.0 or resistance_dist < 1.0:
            confluence_factors.append(0.3)
            quality['support_resistance_context'] = 'strong'
        elif support_dist < 2.0 or resistance_dist < 2.0:
            confluence_factors.append(0.15)
            quality['support_resistance_context'] = 'moderate'

        # ADX and Directional Movement factor
        adx = self._safe_get_value(current_data, 'adx', 25)
        di_plus = self._safe_get_value(current_data, 'di_plus', 25)
        di_minus = self._safe_get_value(current_data, 'di_minus', 25)

        if adx > 25 and abs(di_plus - di_minus) > 5:
            confluence_factors.append(0.2)  # Strong directional movement
        elif adx > 20:
            confluence_factors.append(0.1)  # Moderate trend strength

        # Stochastic oscillator factor
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        stoch_d = self._safe_get_value(current_data, 'stoch_d', 50)

        if 20 < stoch_k < 80 and 20 < stoch_d < 80:  # Not in extreme territory
            if (stoch_k > stoch_d and stoch_k > 50) or (stoch_k < stoch_d and stoch_k < 50):
                confluence_factors.append(0.15)  # Good momentum direction

        # Moving average alignment factor
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', close)
        ema_50 = self._safe_get_value(current_data, 'ema_50', close)

        if close > 0 and ema_20 > 0 and ema_50 > 0:
            if (close > ema_20 > ema_50) or (close < ema_20 < ema_50):
                confluence_factors.append(0.15)  # Good MA alignment
            elif close > ema_20 or close < ema_20:
                confluence_factors.append(0.1)  # Partial alignment

        # Bollinger Band position factor
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        bb_width = self._safe_get_value(current_data, 'bb_width', 0.04)

        if 0.2 < bb_position < 0.8 and bb_width > 0.02:
            confluence_factors.append(0.1)  # Good position, adequate volatility

        # Pattern recognition factor
        hammer = self._safe_get_value(current_data, 'hammer', 0)
        bullish_engulfing = self._safe_get_value(current_data, 'bullish_engulfing', 0)
        bearish_engulfing = self._safe_get_value(current_data, 'bearish_engulfing', 0)

        if hammer or bullish_engulfing or bearish_engulfing:
            confluence_factors.append(0.15)  # Strong reversal pattern
        elif self._safe_get_value(current_data, 'doji', 0):
            confluence_factors.append(0.05)  # Indecision pattern

        # Calculate final confluence score
        quality['confluence_score'] = min(sum(confluence_factors), 1.0)  # Cap at 1.0
        
        # Determine overall quality
        if quality['confluence_score'] >= 0.8:
            quality['overall_quality'] = 'excellent'
        elif quality['confluence_score'] >= 0.6:
            quality['overall_quality'] = 'good'
        elif quality['confluence_score'] >= 0.4:
            quality['overall_quality'] = 'moderate'
        else:
            quality['overall_quality'] = 'poor'
        
        # Trend alignment
        trend_direction = self._safe_get_value(current_data, 'trend_direction', 0)
        signal = self._safe_get_value(current_data, 'signal', 0)
        
        if (trend_direction == 1 and signal == 1) or (trend_direction == -1 and signal == 2):
            quality['trend_alignment'] = True
        
        return quality
    
    def _analyze_risk_reward(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze risk-reward characteristics of the current setup.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Risk-reward analysis
        """
        risk_reward = {
            'ratio': 1.0,
            'quality': 'undefined',
            'stop_distance': 'normal',
            'target_distance': 'normal',
            'volatility_adjusted': True
        }
        
        # Get ATR for volatility-adjusted calculations
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        
        # Analyze distances to key levels
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        # Estimate risk-reward based on signal and key levels with realistic bounds
        signal = self._safe_get_value(current_data, 'signal', 0)

        # Set minimum risk distance to prevent impossible ratios
        min_risk_distance = atr * 0.5  # Minimum 0.5 ATR risk
        max_risk_distance = atr * 3.0   # Maximum 3 ATR risk

        if signal == 1:  # Buy signal
            # Risk to support, reward to resistance
            raw_risk_distance = min(support_dist, atr * 1.5)
            raw_reward_distance = min(resistance_dist, atr * 3.0)

            # Ensure realistic risk distance bounds
            risk_distance = max(min_risk_distance, min(raw_risk_distance, max_risk_distance))
            reward_distance = max(atr * 0.5, raw_reward_distance)  # Minimum reward of 0.5 ATR

            # Calculate ratio with bounds checking
            calculated_ratio = reward_distance / risk_distance
            risk_reward['ratio'] = self._validate_risk_reward_ratio(calculated_ratio)

        elif signal == 2:  # Sell signal
            # Risk to resistance, reward to support
            raw_risk_distance = min(resistance_dist, atr * 1.5)
            raw_reward_distance = min(support_dist, atr * 3.0)

            # Ensure realistic risk distance bounds
            risk_distance = max(min_risk_distance, min(raw_risk_distance, max_risk_distance))
            reward_distance = max(atr * 0.5, raw_reward_distance)  # Minimum reward of 0.5 ATR

            # Calculate ratio with bounds checking
            calculated_ratio = reward_distance / risk_distance
            risk_reward['ratio'] = self._validate_risk_reward_ratio(calculated_ratio)
        else:
            # No clear signal - use conservative default ratio
            risk_reward['ratio'] = 1.5  # Conservative default

        # Classify risk-reward quality based on validated ratio
        ratio = risk_reward['ratio']
        if ratio >= 3.0:
            risk_reward['quality'] = 'excellent'
        elif ratio >= 2.0:
            risk_reward['quality'] = 'good'
        elif ratio >= 1.5:
            risk_reward['quality'] = 'acceptable'
        elif ratio >= 1.0:
            risk_reward['quality'] = 'poor'
        else:
            risk_reward['quality'] = 'poor'
        
        return risk_reward

    def _validate_risk_reward_ratio(self, ratio: float) -> float:
        """
        Validate and constrain risk-reward ratio to realistic trading bounds.

        Args:
            ratio: Calculated risk-reward ratio

        Returns:
            Validated ratio between 0.1 and 5.0
        """
        # Constrain to realistic trading range (0.1:1 to 5:1)
        if ratio < 0.1:
            return 0.1
        elif ratio > 5.0:
            return 5.0
        else:
            # Round to 1 decimal place for clean display
            return round(ratio, 1)

    def _determine_execution_recommendation(self, current_data: pd.Series, setup_quality: Dict,
                                          risk_reward: Dict, context: Dict) -> Dict[str, Any]:
        """
        Determine the execution recommendation based on analysis.
        
        Args:
            current_data: Current row data
            setup_quality: Setup quality analysis
            risk_reward: Risk-reward analysis
            context: Historical context
            
        Returns:
            Execution recommendation
        """
        recommendation = {
            'action': 'wait_and_see',
            'conviction': 'low',
            'position_size': 'small',
            'urgency': 'low'
        }
        
        signal = self._safe_get_value(current_data, 'signal', 0)
        quality = setup_quality['overall_quality']
        rr_quality = risk_reward['quality']
        confluence_score = setup_quality['confluence_score']
        
        # Determine action based on signal and quality
        if signal == 1:  # Buy signal
            if quality in ['excellent', 'good'] and rr_quality in ['excellent', 'good']:
                recommendation['action'] = 'aggressive_long'
                recommendation['conviction'] = 'high'
                recommendation['position_size'] = 'normal'
                recommendation['urgency'] = 'high'
            elif quality == 'moderate' and rr_quality in ['good', 'acceptable']:
                recommendation['action'] = 'cautious_long'
                recommendation['conviction'] = 'moderate'
                recommendation['position_size'] = 'small'
                recommendation['urgency'] = 'moderate'
        
        elif signal == 2:  # Sell signal
            if quality in ['excellent', 'good'] and rr_quality in ['excellent', 'good']:
                recommendation['action'] = 'aggressive_short'
                recommendation['conviction'] = 'high'
                recommendation['position_size'] = 'normal'
                recommendation['urgency'] = 'high'
            elif quality == 'moderate' and rr_quality in ['good', 'acceptable']:
                recommendation['action'] = 'cautious_short'
                recommendation['conviction'] = 'moderate'
                recommendation['position_size'] = 'small'
                recommendation['urgency'] = 'moderate'
        
        # Override with risk management if conditions warrant
        if confluence_score < 0.3 or rr_quality == 'poor':
            recommendation['action'] = 'wait_and_see'
            recommendation['conviction'] = 'low'
        
        return recommendation
    
    def _analyze_timing_considerations(self, current_data: pd.Series, context: Dict) -> Dict[str, Any]:
        """
        Analyze timing considerations for execution.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Timing analysis
        """
        timing = {
            'market_regime_timing': 'neutral',
            'volatility_timing': 'normal',
            'momentum_timing': 'neutral'
        }
        
        # Market regime timing
        regime = context.get('market_regime', 'consolidation')
        if regime == 'trending':
            timing['market_regime_timing'] = 'favorable'
        elif regime == 'volatile':
            timing['market_regime_timing'] = 'challenging'
        elif regime == 'consolidation':
            timing['market_regime_timing'] = 'patient'
        
        # Volatility timing
        vol_analysis = context.get('volatility_analysis', {})
        vol_level = vol_analysis.get('level', 'normal')
        
        if vol_level == 'low':
            timing['volatility_timing'] = 'breakout_potential'
        elif vol_level == 'high':
            timing['volatility_timing'] = 'caution_warranted'
        
        # Momentum timing
        momentum_analysis = context.get('momentum_analysis', {})
        momentum_consistency = momentum_analysis.get('momentum_consistency', 'low')
        
        if momentum_consistency == 'high':
            timing['momentum_timing'] = 'favorable'
        elif momentum_consistency == 'low':
            timing['momentum_timing'] = 'wait_for_clarity'
        
        return timing
    
    def _construct_execution_reasoning(self, setup_quality: Dict, risk_reward: Dict,
                                     execution_recommendation: Dict, timing_analysis: Dict,
                                     context: Dict) -> str:
        """
        Construct comprehensive execution decision reasoning.
        
        Args:
            setup_quality: Setup quality analysis
            risk_reward: Risk-reward analysis
            execution_recommendation: Execution recommendation
            timing_analysis: Timing considerations
            context: Historical context
            
        Returns:
            Complete execution reasoning text
        """
        reasoning_parts = []
        
        # Setup quality assessment
        quality = setup_quality['overall_quality']
        confluence_score = setup_quality['confluence_score']
        
        # Remove robotic language - this will be handled by decision templates

        # Skip risk-reward in reasoning as it's already in config
        
        # Execution recommendation
        action = execution_recommendation['action']
        conviction = execution_recommendation['conviction']
        
        # Remove robotic template language - decision reasoning handled elsewhere
        
        # Timing considerations
        regime_timing = timing_analysis['market_regime_timing']
        vol_timing = timing_analysis['volatility_timing']
        
        if regime_timing == 'favorable':
            reasoning_parts.append("Market regime timing appears favorable for position initiation")
        elif regime_timing == 'challenging':
            reasoning_parts.append("Current market regime suggests cautious approach to new positions")
        
        if vol_timing == 'breakout_potential':
            reasoning_parts.append("Low volatility environment suggests potential for significant movement")
        elif vol_timing == 'caution_warranted':
            reasoning_parts.append("Elevated volatility requires enhanced risk management protocols")
        
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
            "Current technical setup presents moderate quality with standard risk-reward characteristics. "
            "Analysis supports cautious approach with focus on risk management over aggressive positioning. "
            "Market conditions suggest patience until clearer directional signals develop."
        )
