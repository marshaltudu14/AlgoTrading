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
            'signal', 'atr', 'support_distance', 'resistance_distance',
            'trend_strength', 'rsi_14', 'macd_histogram'
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
        """Load risk-reward assessment templates."""
        return {
            'excellent': "exceptional risk-reward characteristics with high probability setup",
            'good': "favorable risk-reward profile supporting position initiation",
            'acceptable': "acceptable risk-reward parameters for selective positioning",
            'poor': "unfavorable risk-reward characteristics suggesting caution",
            'undefined': "unclear risk-reward parameters requiring additional analysis"
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
        
        # Calculate final confluence score
        quality['confluence_score'] = sum(confluence_factors)
        
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
        
        # Estimate risk-reward based on signal and key levels
        signal = self._safe_get_value(current_data, 'signal', 0)
        
        if signal == 1:  # Buy signal
            # Risk to support, reward to resistance
            risk_distance = min(support_dist, atr * 1.5)  # Use closer of support or 1.5 ATR
            reward_distance = min(resistance_dist, atr * 3.0)  # Target resistance or 3 ATR
            
            if risk_distance > 0:
                risk_reward['ratio'] = reward_distance / risk_distance
        
        elif signal == 2:  # Sell signal
            # Risk to resistance, reward to support
            risk_distance = min(resistance_dist, atr * 1.5)
            reward_distance = min(support_dist, atr * 3.0)
            
            if risk_distance > 0:
                risk_reward['ratio'] = reward_distance / risk_distance
        
        # Classify risk-reward quality
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
            risk_reward['quality'] = 'poor'  # Changed from 'unfavorable' to 'poor'
        
        return risk_reward
    
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
        
        reasoning_parts.append(
            f"Technical setup demonstrates {quality} quality with confluence score of {confluence_score:.1f}"
        )
        
        # Risk-reward assessment
        rr_quality = risk_reward['quality']
        rr_ratio = risk_reward['ratio']
        
        if rr_quality != 'undefined':
            rr_template = self.risk_reward_templates[rr_quality]
            reasoning_parts.append(f"presenting {rr_template} at {rr_ratio:.1f}:1 ratio")
        
        # Execution recommendation
        action = execution_recommendation['action']
        conviction = execution_recommendation['conviction']
        
        if action in self.decision_templates:
            template = self.decision_templates[action]
            reasoning_parts.append(
                f"Current analysis supports {template['description']} with {conviction} conviction"
            )
            reasoning_parts.append(f"recommending {template['approach']}")
        
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
