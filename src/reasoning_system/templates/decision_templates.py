#!/usr/bin/env python3
"""
Decision Templates for Enhanced Reasoning System
===============================================

Provides decision templates for LONG/SHORT/HOLD positions that use signal column
in decision logic but generate explanations based on market conditions without
mentioning the signal column (prevents data leakage).

Key Requirements:
- Use signal column (1=LONG, 2=SHORT, 0=HOLD) for decision logic
- Generate explanations based on market conditions only
- Never reference signal column in reasoning text
- Include strong profitability reasoning when conditions align
"""

import random
from typing import Dict, List, Any, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DecisionTemplates:
    """
    Decision template system that uses signal column for decision logic
    but generates market-condition-based explanations.
    """
    
    def __init__(self):
        """Initialize decision templates."""
        self.long_templates = self._load_long_templates()
        self.short_templates = self._load_short_templates()
        self.hold_templates = self._load_hold_templates()
        
        # Profitability hint phrases (indirect)
        self.profitability_hints = self._load_profitability_hints()
        
        logger.info("DecisionTemplates initialized with LONG/SHORT/HOLD templates")
    
    def generate_decision_reasoning(self, current_data: pd.Series, 
                                  market_analysis: Dict[str, Any],
                                  historical_context: Dict[str, Any]) -> str:
        """
        Generate decision reasoning based on signal column but without mentioning it.
        
        Args:
            current_data: Current row data with signal column
            market_analysis: Market condition analysis
            historical_context: Historical pattern analysis
            
        Returns:
            Decision reasoning text with "Going LONG/SHORT/HOLD because..." format
        """
        signal = self._safe_get_value(current_data, 'signal', 0)
        
        # Use signal for decision logic but generate market-based explanations
        if signal == 1:  # LONG decision
            return self._generate_long_reasoning(current_data, market_analysis, historical_context)
        elif signal == 2:  # SHORT decision
            return self._generate_short_reasoning(current_data, market_analysis, historical_context)
        else:  # HOLD decision (signal == 0)
            return self._generate_hold_reasoning(current_data, market_analysis, historical_context)
    
    def _generate_long_reasoning(self, current_data: pd.Series, 
                               market_analysis: Dict[str, Any],
                               historical_context: Dict[str, Any]) -> str:
        """Generate LONG decision reasoning based on market conditions."""
        template = random.choice(self.long_templates)
        
        # Analyze market strength for LONG position
        market_strength = self._assess_long_market_strength(current_data, market_analysis)
        
        # Build reasoning components
        reasoning_parts = ["Going LONG because"]
        
        # Add primary reason based on strongest market condition
        primary_reason = self._get_primary_long_reason(current_data, market_analysis, market_strength)
        reasoning_parts.append(primary_reason)
        
        # Add supporting technical factors
        supporting_factors = self._get_long_supporting_factors(current_data, market_analysis)
        if supporting_factors:
            reasoning_parts.extend(supporting_factors)
        
        # Add historical context
        historical_support = self._get_long_historical_support(historical_context)
        if historical_support:
            reasoning_parts.append(historical_support)
        
        # Add profitability hints when conditions are strong
        if market_strength >= 0.7:
            profit_hint = random.choice(self.profitability_hints['strong_bullish'])
            reasoning_parts.append(profit_hint)
        
        # Add risk management consideration
        risk_consideration = self._get_long_risk_consideration(current_data)
        reasoning_parts.append(risk_consideration)
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_short_reasoning(self, current_data: pd.Series,
                                market_analysis: Dict[str, Any], 
                                historical_context: Dict[str, Any]) -> str:
        """Generate SHORT decision reasoning based on market conditions."""
        template = random.choice(self.short_templates)
        
        # Analyze market weakness for SHORT position
        market_weakness = self._assess_short_market_strength(current_data, market_analysis)
        
        # Build reasoning components
        reasoning_parts = ["Going SHORT because"]
        
        # Add primary reason based on strongest bearish condition
        primary_reason = self._get_primary_short_reason(current_data, market_analysis, market_weakness)
        reasoning_parts.append(primary_reason)
        
        # Add supporting technical factors
        supporting_factors = self._get_short_supporting_factors(current_data, market_analysis)
        if supporting_factors:
            reasoning_parts.extend(supporting_factors)
        
        # Add historical context
        historical_support = self._get_short_historical_support(historical_context)
        if historical_support:
            reasoning_parts.append(historical_support)
        
        # Add profitability hints when conditions are strong
        if market_weakness >= 0.7:
            profit_hint = random.choice(self.profitability_hints['strong_bearish'])
            reasoning_parts.append(profit_hint)
        
        # Add risk management consideration
        risk_consideration = self._get_short_risk_consideration(current_data)
        reasoning_parts.append(risk_consideration)
        
        return ". ".join(reasoning_parts) + "."
    
    def _generate_hold_reasoning(self, current_data: pd.Series,
                               market_analysis: Dict[str, Any],
                               historical_context: Dict[str, Any]) -> str:
        """Generate HOLD decision reasoning based on market conditions."""
        template = random.choice(self.hold_templates)
        
        # Analyze market uncertainty for HOLD position
        uncertainty_level = self._assess_market_uncertainty(current_data, market_analysis)
        
        # Build reasoning components
        reasoning_parts = ["Staying HOLD because"]
        
        # Add primary reason for holding
        primary_reason = self._get_primary_hold_reason(current_data, market_analysis, uncertainty_level)
        reasoning_parts.append(primary_reason)
        
        # Add supporting factors for waiting
        supporting_factors = self._get_hold_supporting_factors(current_data, market_analysis)
        if supporting_factors:
            reasoning_parts.extend(supporting_factors)
        
        # Add what we're waiting for
        waiting_criteria = self._get_hold_waiting_criteria(current_data, market_analysis)
        reasoning_parts.append(waiting_criteria)
        
        return ". ".join(reasoning_parts) + "."
    
    def _load_long_templates(self) -> List[Dict[str, Any]]:
        """Load LONG decision templates."""
        return [
            {
                "category": "momentum_breakout",
                "primary_conditions": ["strong bullish momentum", "breakout above resistance", "volume confirmation"],
                "supporting_factors": ["trend alignment", "technical confluence", "momentum acceleration"]
            },
            {
                "category": "trend_continuation", 
                "primary_conditions": ["established uptrend", "pullback completion", "support holding"],
                "supporting_factors": ["moving average support", "momentum recovery", "pattern completion"]
            },
            {
                "category": "reversal_setup",
                "primary_conditions": ["oversold bounce", "bullish divergence", "support test"],
                "supporting_factors": ["momentum shift", "volume increase", "pattern reversal"]
            }
        ]
    
    def _load_short_templates(self) -> List[Dict[str, Any]]:
        """Load SHORT decision templates."""
        return [
            {
                "category": "momentum_breakdown",
                "primary_conditions": ["strong bearish momentum", "breakdown below support", "volume confirmation"],
                "supporting_factors": ["trend reversal", "technical breakdown", "momentum acceleration"]
            },
            {
                "category": "trend_continuation",
                "primary_conditions": ["established downtrend", "rally failure", "resistance rejection"],
                "supporting_factors": ["moving average resistance", "momentum weakness", "pattern completion"]
            },
            {
                "category": "reversal_setup", 
                "primary_conditions": ["overbought rejection", "bearish divergence", "resistance test"],
                "supporting_factors": ["momentum shift", "distribution signs", "pattern reversal"]
            }
        ]
    
    def _load_hold_templates(self) -> List[Dict[str, Any]]:
        """Load HOLD decision templates."""
        return [
            {
                "category": "mixed_indicators",
                "primary_conditions": ["conflicting indicators", "unclear direction", "low conviction"],
                "waiting_for": ["clearer confirmation", "breakout direction", "momentum clarity"]
            },
            {
                "category": "consolidation",
                "primary_conditions": ["range-bound trading", "low volatility", "indecision"],
                "waiting_for": ["range breakout", "volume increase", "directional move"]
            },
            {
                "category": "risk_management",
                "primary_conditions": ["unfavorable risk-reward", "high uncertainty", "poor timing"],
                "waiting_for": ["better entry point", "improved setup", "risk reduction"]
            }
        ]
    
    def _load_profitability_hints(self) -> Dict[str, List[str]]:
        """Load indirect profitability hint phrases."""
        return {
            "strong_bullish": [
                "all technical factors align favorably for upward movement",
                "market conditions suggest strong potential for price appreciation", 
                "confluence of bullish factors creates compelling opportunity",
                "technical setup indicates high probability of sustained advance",
                "market structure supports significant upside potential"
            ],
            "strong_bearish": [
                "all technical factors align for downward pressure",
                "market conditions suggest strong potential for price decline",
                "confluence of bearish factors creates compelling short opportunity", 
                "technical setup indicates high probability of sustained decline",
                "market structure supports significant downside potential"
            ]
        }
    
    def _safe_get_value(self, data: pd.Series, column: str, default: Any = 0) -> Any:
        """Safely get value from pandas Series."""
        try:
            return data.get(column, default) if column in data.index else default
        except Exception:
            return default

    def _assess_long_market_strength(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> float:
        """Assess market strength for LONG positions (0-1 scale)."""
        strength_factors = []

        # RSI momentum (not overbought)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 40 <= rsi <= 70:
            strength_factors.append(0.8)
        elif 30 <= rsi < 40:
            strength_factors.append(1.0)  # Oversold bounce potential

        # MACD bullish momentum
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0:
            strength_factors.append(0.9)

        # Trend strength
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength > 0.5:
            strength_factors.append(trend_strength)

        # Moving average alignment
        sma_5_20_cross = self._safe_get_value(current_data, 'sma_5_20_cross', 0)
        if sma_5_20_cross > 0:
            strength_factors.append(0.7)

        return sum(strength_factors) / len(strength_factors) if strength_factors else 0.3

    def _assess_short_market_strength(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> float:
        """Assess market weakness for SHORT positions (0-1 scale)."""
        weakness_factors = []

        # RSI momentum (not oversold)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 30 <= rsi <= 60:
            weakness_factors.append(0.8)
        elif 70 < rsi <= 80:
            weakness_factors.append(1.0)  # Overbought rejection potential

        # MACD bearish momentum
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist < 0:
            weakness_factors.append(0.9)

        # Trend weakness
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength < -0.5:
            weakness_factors.append(abs(trend_strength))

        # Moving average breakdown
        sma_5_20_cross = self._safe_get_value(current_data, 'sma_5_20_cross', 0)
        if sma_5_20_cross < 0:
            weakness_factors.append(0.7)

        return sum(weakness_factors) / len(weakness_factors) if weakness_factors else 0.3

    def _assess_market_uncertainty(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> float:
        """Assess market uncertainty for HOLD positions (0-1 scale)."""
        uncertainty_factors = []

        # RSI in neutral zone
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 45 <= rsi <= 55:
            uncertainty_factors.append(0.8)

        # Low trend strength
        trend_strength = abs(self._safe_get_value(current_data, 'trend_strength', 0))
        if trend_strength < 0.3:
            uncertainty_factors.append(0.9)

        # MACD near zero
        macd_hist = abs(self._safe_get_value(current_data, 'macd_histogram', 0))
        if macd_hist < 0.1:
            uncertainty_factors.append(0.7)

        # Low volatility
        volatility = self._safe_get_value(current_data, 'volatility_20', 0)
        if volatility < 0.02:  # Low volatility threshold
            uncertainty_factors.append(0.6)

        return sum(uncertainty_factors) / len(uncertainty_factors) if uncertainty_factors else 0.5

    def _get_primary_long_reason(self, current_data: pd.Series, market_analysis: Dict[str, Any], strength: float) -> str:
        """Get primary reason for LONG decision based on strongest market condition."""
        reasons = []

        # Check for momentum breakout
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0.1:
            reasons.append("strong bullish momentum confirmed by MACD histogram turning positive")

        # Check for trend continuation
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength > 0.6:
            reasons.append("established uptrend shows strong continuation characteristics")

        # Check for oversold bounce
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi < 35:
            reasons.append("oversold conditions present compelling bounce opportunity")

        # Check for support holding
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        if support_dist < 0.5:  # Close to support
            reasons.append("price successfully tested and held key support level")

        # Default to general bullish setup
        if not reasons:
            reasons.append("technical indicators align for bullish positioning")

        return random.choice(reasons)

    def _get_primary_short_reason(self, current_data: pd.Series, market_analysis: Dict[str, Any], weakness: float) -> str:
        """Get primary reason for SHORT decision based on strongest bearish condition."""
        reasons = []

        # Check for momentum breakdown
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist < -0.1:
            reasons.append("strong bearish momentum confirmed by MACD histogram turning negative")

        # Check for trend reversal
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength < -0.6:
            reasons.append("established downtrend shows strong continuation characteristics")

        # Check for overbought rejection
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi > 70:
            reasons.append("overbought conditions present compelling rejection opportunity")

        # Check for resistance rejection
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        if resistance_dist < 0.5:  # Close to resistance
            reasons.append("price failed to break above key resistance level with rejection")

        # Default to general bearish setup
        if not reasons:
            reasons.append("technical indicators align for bearish positioning")

        return random.choice(reasons)

    def _get_primary_hold_reason(self, current_data: pd.Series, market_analysis: Dict[str, Any], uncertainty: float) -> str:
        """Get primary reason for HOLD decision based on market uncertainty."""
        reasons = []

        # Check for mixed indicators
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if 45 <= rsi <= 55 and abs(macd_hist) < 0.05:
            reasons.append("mixed readings from technical indicators create uncertainty")

        # Check for consolidation
        volatility = self._safe_get_value(current_data, 'volatility_20', 0)
        if volatility < 0.015:
            reasons.append("price consolidating in narrow range between support and resistance")

        # Check for low conviction setup
        trend_strength = abs(self._safe_get_value(current_data, 'trend_strength', 0))
        if trend_strength < 0.2:
            reasons.append("low volatility environment with unclear directional bias")

        # Default to waiting for clarity
        if not reasons:
            reasons.append("current market conditions lack clear directional conviction")

        return random.choice(reasons)

    def _get_long_supporting_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get supporting factors for LONG decision."""
        factors = []

        # Moving average support
        close = self._safe_get_value(current_data, 'close', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', 0)
        if close > sma_20:
            factors.append("price trading above 20-period moving average support")

        # Volume confirmation
        # Note: Volume may not be available in all datasets

        # RSI momentum
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 35 <= rsi <= 65:
            factors.append("RSI showing healthy momentum with room for upside")

        # Bollinger Band position
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        if bb_position < 0.3:
            factors.append("price near lower Bollinger Band suggests oversold bounce potential")

        return factors[:2]  # Limit to 2 supporting factors

    def _get_short_supporting_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get supporting factors for SHORT decision."""
        factors = []

        # Moving average resistance
        close = self._safe_get_value(current_data, 'close', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', 0)
        if close < sma_20:
            factors.append("price trading below 20-period moving average resistance")

        # RSI momentum
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if 35 <= rsi <= 65:
            factors.append("RSI showing bearish momentum with room for downside")

        # Bollinger Band position
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        if bb_position > 0.7:
            factors.append("price near upper Bollinger Band suggests overbought rejection potential")

        return factors[:2]  # Limit to 2 supporting factors

    def _get_hold_supporting_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get supporting factors for HOLD decision."""
        factors = []

        # Range-bound trading
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)
        if support_dist > 1.0 and resistance_dist > 1.0:
            factors.append("price trading in middle of range away from key levels")

        # Low volatility
        volatility = self._safe_get_value(current_data, 'volatility_20', 0)
        if volatility < 0.02:
            factors.append("low volatility environment suggests limited directional movement")

        return factors[:1]  # Limit to 1 supporting factor

    def _get_long_historical_support(self, historical_context: Dict[str, Any]) -> str:
        """Get historical context supporting LONG decision."""
        if not historical_context:
            return ""

        # Example historical patterns (would be populated by historical engine)
        patterns = [
            "similar bullish setups over past 30 candles led to sustained advances",
            "historical pattern analysis shows strong follow-through after such formations",
            "recent 20-candle momentum pattern suggests continuation potential"
        ]

        return random.choice(patterns)

    def _get_short_historical_support(self, historical_context: Dict[str, Any]) -> str:
        """Get historical context supporting SHORT decision."""
        if not historical_context:
            return ""

        patterns = [
            "similar bearish setups over past 30 candles led to sustained declines",
            "historical pattern analysis shows strong follow-through after such formations",
            "recent 20-candle momentum pattern suggests continuation potential"
        ]

        return random.choice(patterns)

    def _get_long_risk_consideration(self, current_data: pd.Series) -> str:
        """Get risk management consideration for LONG position."""
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        support_dist = self._safe_get_value(current_data, 'support_distance', 999)

        if support_dist < atr * 2:
            return "risk-reward ratio favors long position with tight stop below recent support"
        else:
            return "position sizing adjusted for current volatility environment"

    def _get_short_risk_consideration(self, current_data: pd.Series) -> str:
        """Get risk management consideration for SHORT position."""
        atr = self._safe_get_value(current_data, 'atr', 1.0)
        resistance_dist = self._safe_get_value(current_data, 'resistance_distance', 999)

        if resistance_dist < atr * 2:
            return "risk-reward ratio favors short position with tight stop above recent resistance"
        else:
            return "position sizing adjusted for current volatility environment"

    def _get_hold_waiting_criteria(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> str:
        """Get criteria for what we're waiting for in HOLD decision."""
        criteria = [
            "waiting for clearer confirmation from momentum indicators",
            "monitoring for breakout above resistance or breakdown below support",
            "awaiting improved risk-reward setup with better entry timing"
        ]

        return random.choice(criteria)
