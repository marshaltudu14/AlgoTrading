#!/usr/bin/env python3
"""
Risk Assessment Engine
=====================

Generates comprehensive risk assessment reasoning that identifies potential risks,
scenario planning, and risk management considerations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from ..core.base_engine import BaseReasoningEngine

logger = logging.getLogger(__name__)


class RiskAssessmentEngine(BaseReasoningEngine):
    """
    Generates comprehensive risk assessment reasoning covering market risks,
    technical risks, and scenario-based risk analysis.
    """
    
    def _initialize_config(self):
        """Initialize risk assessment specific configuration."""
        self.risk_templates = self._load_risk_templates()
        self.scenario_templates = self._load_scenario_templates()
        
        # Configuration for risk assessment
        self.config.setdefault('volatility_risk_thresholds', {
            'low': 1.0, 'normal': 2.5, 'high': 4.0, 'extreme': 6.0
        })
        self.config.setdefault('drawdown_thresholds', {
            'low': 2.0, 'moderate': 5.0, 'high': 10.0, 'severe': 20.0
        })
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for risk assessment."""
        return [
            'atr', 'volatility_20', 'support_distance', 'resistance_distance',
            'trend_strength', 'rsi_14', 'signal'
        ]
    
    def generate_reasoning(self, current_data: pd.Series, context: Dict[str, Any]) -> str:
        """
        Generate comprehensive risk assessment reasoning.
        
        Args:
            current_data: Current row data with all features
            context: Historical context from HistoricalContextManager
            
        Returns:
            Professional risk assessment reasoning text
        """
        try:
            # Analyze volatility risks
            volatility_risks = self._analyze_volatility_risks(current_data, context)
            
            # Analyze technical risks
            technical_risks = self._analyze_technical_risks(current_data, context)
            
            # Analyze market structure risks
            structure_risks = self._analyze_structure_risks(current_data, context)
            
            # Analyze scenario-based risks
            scenario_risks = self._analyze_scenario_risks(current_data, context)
            
            # Generate alternative scenarios
            alternative_scenarios = self._generate_alternative_scenarios(current_data, context)
            
            # Construct risk assessment reasoning
            reasoning = self._construct_risk_reasoning(
                volatility_risks, technical_risks, structure_risks, 
                scenario_risks, alternative_scenarios, context
            )
            
            self._log_reasoning_generation(len(reasoning), "good")
            return reasoning
            
        except Exception as e:
            logger.error(f"Error in risk assessment reasoning: {str(e)}")
            return self._get_fallback_reasoning(current_data, context)
    
    def _load_risk_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load risk assessment templates for different risk types."""
        return {
            'volatility_risk': {
                'low': "manageable volatility environment with standard risk parameters",
                'normal': "normal volatility conditions requiring standard risk management",
                'high': "elevated volatility requiring enhanced risk management protocols",
                'extreme': "extreme volatility conditions demanding defensive positioning"
            },
            'technical_risk': {
                'low': "minimal technical risk with strong support levels",
                'moderate': "moderate technical risk requiring careful monitoring",
                'high': "significant technical risk with potential for adverse moves",
                'extreme': "extreme technical risk suggesting defensive approach"
            },
            'structure_risk': {
                'consolidation': "range-bound structure with breakout risk considerations",
                'trending': "trending structure with reversal risk monitoring",
                'volatile': "unstable structure requiring dynamic risk management",
                'transitional': "evolving structure with uncertainty risk factors"
            }
        }
    
    def _load_scenario_templates(self) -> Dict[str, str]:
        """Load scenario analysis templates."""
        return {
            'bull_scenario': "Bullish scenario assumes continued upward momentum with target achievement",
            'bear_scenario': "Bearish scenario considers downward pressure with support level tests",
            'neutral_scenario': "Neutral scenario expects range-bound behavior with limited directional movement",
            'breakout_scenario': "Breakout scenario anticipates significant movement beyond current ranges",
            'reversal_scenario': "Reversal scenario considers trend change with momentum shift"
        }
    
    def _analyze_volatility_risks(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze volatility-related risks.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Volatility risk analysis
        """
        vol_risks = {
            'current_volatility_level': 'normal',
            'volatility_trend': 'stable',
            'risk_level': 'moderate',
            'gap_risk': 'low',
            'whipsaw_risk': 'moderate'
        }
        
        # Current volatility assessment
        current_vol = self._safe_get_value(current_data, 'volatility_20', 2.0)
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        
        vol_thresholds = self.config['volatility_risk_thresholds']
        
        if current_vol >= vol_thresholds['extreme']:
            vol_risks['current_volatility_level'] = 'extreme'
            vol_risks['risk_level'] = 'high'
            vol_risks['whipsaw_risk'] = 'high'
        elif current_vol >= vol_thresholds['high']:
            vol_risks['current_volatility_level'] = 'high'
            vol_risks['risk_level'] = 'elevated'
            vol_risks['whipsaw_risk'] = 'elevated'
        elif current_vol <= vol_thresholds['low']:
            vol_risks['current_volatility_level'] = 'low'
            vol_risks['risk_level'] = 'low'
            vol_risks['gap_risk'] = 'elevated'  # Low vol can lead to gaps
        
        # Volatility trend from context
        vol_analysis = context.get('volatility_analysis', {})
        vol_risks['volatility_trend'] = vol_analysis.get('trend', 'stable')
        
        # Gap risk assessment based on recent price action
        hl_range = self._safe_get_value(current_data, 'hl_range', 1.0)
        if hl_range < atr * 0.5:  # Very tight range
            vol_risks['gap_risk'] = 'elevated'
        
        return vol_risks
    
    def _analyze_technical_risks(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze technical analysis related risks.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Technical risk analysis
        """
        tech_risks = {
            'trend_reversal_risk': 'low',
            'support_break_risk': 'moderate',
            'resistance_rejection_risk': 'moderate',
            'false_signal_risk': 'moderate',
            'momentum_divergence_risk': 'low'
        }
        
        # Trend reversal risk
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0.5)
        if trend_strength < 0.3:
            tech_risks['trend_reversal_risk'] = 'high'
        elif trend_strength < 0.5:
            tech_risks['trend_reversal_risk'] = 'moderate'
        
        # Support/resistance risks
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        
        if support_dist < 1.0:
            tech_risks['support_break_risk'] = 'elevated'
        elif support_dist < 2.0:
            tech_risks['support_break_risk'] = 'moderate'
        else:
            tech_risks['support_break_risk'] = 'low'
        
        if resistance_dist < 1.0:
            tech_risks['resistance_rejection_risk'] = 'elevated'
        elif resistance_dist < 2.0:
            tech_risks['resistance_rejection_risk'] = 'moderate'
        else:
            tech_risks['resistance_rejection_risk'] = 'low'
        
        # False signal risk based on market regime
        regime = context.get('market_regime', 'consolidation')
        if regime == 'volatile':
            tech_risks['false_signal_risk'] = 'high'
        elif regime == 'consolidation':
            tech_risks['false_signal_risk'] = 'elevated'
        
        # Momentum divergence risk
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi > 80 or rsi < 20:
            tech_risks['momentum_divergence_risk'] = 'elevated'
        
        return tech_risks
    
    def _analyze_structure_risks(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market structure related risks.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Market structure risk analysis
        """
        structure_risks = {
            'regime_change_risk': 'moderate',
            'liquidity_risk': 'low',
            'correlation_risk': 'moderate',
            'structural_break_risk': 'low'
        }
        
        # Regime change risk
        regime = context.get('market_regime', 'consolidation')
        if regime == 'transitional':
            structure_risks['regime_change_risk'] = 'high'
        elif regime == 'volatile':
            structure_risks['regime_change_risk'] = 'elevated'
        
        # Structural break risk based on support/resistance tests
        sr_history = context.get('support_resistance_history', {})
        support_tests = sr_history.get('support_tests', 0)
        resistance_tests = sr_history.get('resistance_tests', 0)
        
        if support_tests > 5 or resistance_tests > 5:
            structure_risks['structural_break_risk'] = 'elevated'
        
        # Liquidity risk based on volatility and range
        vol_level = context.get('volatility_analysis', {}).get('level', 'normal')
        if vol_level == 'extreme':
            structure_risks['liquidity_risk'] = 'elevated'
        
        return structure_risks
    
    def _analyze_scenario_risks(self, current_data: pd.Series, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze scenario-based risks and probabilities.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            Scenario risk analysis
        """
        scenario_risks = {
            'adverse_scenario_probability': 'moderate',
            'maximum_adverse_move': 'moderate',
            'time_decay_risk': 'low',
            'event_risk': 'low'
        }
        
        # Adverse scenario probability based on signal and confluence
        signal = self._safe_get_value(current_data, 'signal', 0)
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0.5)
        
        if signal != 0 and trend_strength < 0.3:
            scenario_risks['adverse_scenario_probability'] = 'high'
        elif signal != 0 and trend_strength < 0.5:
            scenario_risks['adverse_scenario_probability'] = 'elevated'
        
        # Maximum adverse move estimation
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        vol_level = context.get('volatility_analysis', {}).get('level', 'normal')
        
        if vol_level == 'high':
            scenario_risks['maximum_adverse_move'] = 'high'
        elif vol_level == 'low':
            scenario_risks['maximum_adverse_move'] = 'low'
        
        return scenario_risks
    
    def _generate_alternative_scenarios(self, current_data: pd.Series, context: Dict[str, Any]) -> List[str]:
        """
        Generate alternative scenario descriptions.
        
        Args:
            current_data: Current row data
            context: Historical context
            
        Returns:
            List of alternative scenario descriptions
        """
        scenarios = []
        
        # Current signal context
        signal = self._safe_get_value(current_data, 'signal', 0)
        trend_direction = self._safe_get_value(current_data, 'trend_direction', 0)
        
        # Trend continuation vs reversal scenarios
        if trend_direction == 1:
            scenarios.append("Trend continuation scenario assumes sustained bullish momentum with higher targets")
            scenarios.append("Trend reversal scenario considers bearish divergence with support level tests")
        elif trend_direction == -1:
            scenarios.append("Trend continuation scenario expects sustained bearish pressure with lower targets")
            scenarios.append("Trend reversal scenario anticipates bullish reversal with resistance level challenges")
        else:
            scenarios.append("Breakout scenario considers directional movement beyond current consolidation range")
            scenarios.append("Continuation scenario expects range-bound behavior with defined boundaries")
        
        # Volatility scenarios
        vol_level = context.get('volatility_analysis', {}).get('level', 'normal')
        if vol_level == 'low':
            scenarios.append("Volatility expansion scenario could trigger rapid price movement beyond current expectations")
        elif vol_level == 'high':
            scenarios.append("Volatility compression scenario may lead to consolidation with reduced directional bias")
        
        # Market regime scenarios
        regime = context.get('market_regime', 'consolidation')
        if regime == 'trending':
            scenarios.append("Regime shift scenario would require adjustment from momentum to mean-reversion strategies")
        elif regime == 'consolidation':
            scenarios.append("Regime change scenario could establish new trending phase with sustained directional movement")
        
        return scenarios
    
    def _construct_risk_reasoning(self, volatility_risks: Dict, technical_risks: Dict,
                                structure_risks: Dict, scenario_risks: Dict,
                                alternative_scenarios: List[str], context: Dict) -> str:
        """
        Construct comprehensive risk assessment reasoning.
        
        Args:
            volatility_risks: Volatility risk analysis
            technical_risks: Technical risk analysis
            structure_risks: Structure risk analysis
            scenario_risks: Scenario risk analysis
            alternative_scenarios: Alternative scenario descriptions
            context: Historical context
            
        Returns:
            Complete risk assessment reasoning text
        """
        reasoning_parts = []
        
        # Volatility risk assessment
        vol_level = volatility_risks['current_volatility_level']
        vol_risk_template = self.risk_templates['volatility_risk'][vol_level]
        reasoning_parts.append(f"Current risk environment shows {vol_risk_template}")
        
        # Technical risk factors
        support_risk = technical_risks['support_break_risk']
        resistance_risk = technical_risks['resistance_rejection_risk']
        
        if support_risk == 'elevated':
            reasoning_parts.append("with elevated support break risk requiring defensive stop placement")
        elif resistance_risk == 'elevated':
            reasoning_parts.append("showing elevated resistance rejection risk suggesting cautious target setting")
        
        # Market structure considerations
        regime = context.get('market_regime', 'consolidation')
        if regime in self.risk_templates['structure_risk']:
            structure_template = self.risk_templates['structure_risk'][regime]
            reasoning_parts.append(f"Market structure presents {structure_template}")
        
        # Scenario analysis
        adverse_prob = scenario_risks['adverse_scenario_probability']
        if adverse_prob in ['high', 'elevated']:
            reasoning_parts.append(f"Scenario analysis indicates {adverse_prob} probability of adverse outcomes")
        
        # Alternative scenarios
        if alternative_scenarios:
            primary_scenario = alternative_scenarios[0]
            reasoning_parts.append(f"Primary alternative scenario suggests {primary_scenario.lower()}")
            
            if len(alternative_scenarios) > 1:
                secondary_scenario = alternative_scenarios[1]
                reasoning_parts.append(f"while secondary consideration includes {secondary_scenario.lower()}")
        
        # Risk management recommendations
        vol_trend = volatility_risks['volatility_trend']
        if vol_trend == 'increasing':
            reasoning_parts.append("Increasing volatility trend recommends position size reduction and tighter stops")
        elif vol_trend == 'decreasing':
            reasoning_parts.append("Decreasing volatility suggests potential for range expansion requiring breakout preparation")
        
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
            "Current risk environment shows manageable volatility conditions with standard risk parameters. "
            "Technical risk factors remain within normal ranges requiring standard risk management protocols. "
            "Alternative scenarios include continuation of current market behavior or potential directional breakout on volume expansion."
        )
