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
        """Generate comprehensive LONG decision reasoning with multiple supporting factors."""

        # Get comprehensive technical analysis
        supporting_factors = self._get_comprehensive_long_factors(current_data, market_analysis)

        # Assess overall signal strength
        signal_strength = self._assess_signal_strength(current_data, 'long')

        # Build natural reasoning based on signal strength
        if signal_strength >= 0.7:
            # Strong technical alignment
            reasoning = f"Taking long position because {', '.join(supporting_factors[:5])}, and {supporting_factors[5] if len(supporting_factors) > 5 else 'setup offers compelling opportunity'}."
        elif signal_strength >= 0.4:
            # Mixed signals - honest assessment
            main_factors = supporting_factors[:3]
            reasoning = f"Taking long position because {', '.join(main_factors)}, providing favorable setup despite some mixed signals requiring monitoring."
        else:
            # Weak technical alignment - focus on setup quality
            key_factors = supporting_factors[:2]
            reasoning = f"Taking long position because setup offers good entry opportunity with {', '.join(key_factors)}, though technical signals are mixed and position requires careful monitoring."

        return reasoning

    def _get_comprehensive_long_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get comprehensive list of factors supporting long position."""
        factors = []

        # Get indicator values
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        close_price = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        williams_r = self._safe_get_value(current_data, 'williams_r', -50)
        bb_position = self._safe_get_value(current_data, 'bb_position', 50)
        support_level = self._safe_get_value(current_data, 'support_level', 0)
        resistance_level = self._safe_get_value(current_data, 'resistance_level', 0)

        # Price vs EMA analysis
        if close_price and ema_20:
            price_vs_ema = ((close_price - ema_20) / ema_20) * 100
            if price_vs_ema > 0:
                factors.append(f"price {abs(price_vs_ema):.2f}% above EMA-20 provides trend support")

        # RSI momentum analysis
        if 30 < rsi_14 < 70:
            factors.append(f"RSI at {rsi_14:.1f} shows healthy momentum without overbought risk")
        elif rsi_14 > 70:
            factors.append(f"RSI at {rsi_14:.1f} indicates strong bullish momentum")

        # MACD momentum confirmation
        if macd_hist > 0:
            factors.append(f"MACD histogram at {macd_hist:.2f} confirms upward momentum")

        # EMA alignment
        if ema_20 and ema_50 and ema_20 > ema_50:
            factors.append("EMA-20 above EMA-50 maintains bullish structure")

        # ADX trend strength
        if adx > 25:
            factors.append(f"ADX at {adx:.1f} shows strong trend development")
        elif adx > 20:
            factors.append(f"ADX at {adx:.1f} indicates developing trend strength")

        # Stochastic positioning
        if 20 < stoch_k < 80:
            factors.append(f"stochastic at {stoch_k:.1f} positioned for continued upside")

        # Support proximity
        if support_level and close_price:
            support_distance = abs(close_price - support_level) / close_price * 100
            if support_distance < 2:
                factors.append("proximity to support provides favorable entry point")

        # Bollinger band positioning
        if bb_position < 80:
            factors.append("bollinger band positioning suggests room for upward movement")

        # Ensure we have at least one factor
        if not factors:
            factors.append("setup offers good entry opportunity")

        return factors

    def _assess_signal_strength(self, current_data: pd.Series, direction: str) -> float:
        """Assess signal strength based on technical indicators."""
        strength_score = 0.0
        total_indicators = 0

        # RSI assessment
        rsi_14 = self._safe_get_value(current_data, 'rsi_14', 50)
        if direction == 'long':
            if 30 < rsi_14 < 70:
                strength_score += 0.8
            elif rsi_14 > 70:
                strength_score += 0.6  # Overbought but still bullish
            elif rsi_14 < 30:
                strength_score += 0.9  # Oversold, good for reversal
        total_indicators += 1

        # MACD assessment
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if direction == 'long' and macd_hist > 0:
            strength_score += 0.8
        elif direction == 'short' and macd_hist < 0:
            strength_score += 0.8
        total_indicators += 1

        # EMA alignment
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        if ema_20 and ema_50:
            if direction == 'long' and ema_20 > ema_50:
                strength_score += 0.7
            elif direction == 'short' and ema_20 < ema_50:
                strength_score += 0.7
            total_indicators += 1

        # ADX trend strength
        adx = self._safe_get_value(current_data, 'adx', 25)
        if adx > 25:
            strength_score += 0.6
        total_indicators += 1

        return strength_score / total_indicators if total_indicators > 0 else 0.5

    def _assess_technical_alignment(self, current_data: pd.Series, direction: str) -> Dict[str, Any]:
        """Assess how well technical indicators align with the given direction."""
        bullish_score = 0
        bearish_score = 0
        total_indicators = 0

        # RSI analysis
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi > 60:
            bullish_score += 1
        elif rsi < 40:
            bearish_score += 1
        total_indicators += 1

        # MACD analysis
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0.1:
            bullish_score += 1
        elif macd_hist < -0.1:
            bearish_score += 1
        total_indicators += 1

        # Moving average analysis
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        if close > ema_20:
            bullish_score += 1
        elif close < ema_20:
            bearish_score += 1
        total_indicators += 1

        # Calculate alignment strength
        if direction == 'long':
            strength = bullish_score / total_indicators
        else:
            strength = bearish_score / total_indicators

        return {
            'strength': strength,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'total_indicators': total_indicators
        }

    def _get_honest_long_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get honest supporting factors for LONG decision with natural language."""
        factors = []

        # Get indicator values
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        support_distance = self._safe_get_value(current_data, 'support_distance', 999)

        # Moving average support
        if close > ema_20 and ema_20 > 0:
            pct_above = ((close - ema_20) / ema_20) * 100
            factors.append(f"price {pct_above:.2f}% above EMA-20 provides trend support")

        # RSI momentum
        if 30 < rsi < 70:
            factors.append(f"RSI at {rsi:.1f} shows healthy momentum without overbought risk")
        elif rsi > 50:
            factors.append(f"RSI at {rsi:.1f} indicates bullish momentum")

        # MACD confirmation
        if macd_hist > 0.05:
            factors.append(f"MACD histogram at {macd_hist:.2f} confirms upward acceleration")
        elif macd_hist > 0:
            factors.append("MACD histogram positive supports upward bias")

        # Trend strength
        if adx > 25:
            factors.append(f"ADX at {adx:.1f} shows strong trend development")

        # Stochastic positioning
        if 20 < stoch_k < 80:
            factors.append(f"Stochastic at {stoch_k:.1f} positioned for continued upside")

        # Support proximity
        if support_distance < 0.5:
            factors.append("proximity to support provides favorable entry point")

        # EMA structure
        if ema_20 > ema_50 and ema_20 > 0:
            factors.append("EMA-20 above EMA-50 maintains bullish structure")

        # Ensure we have enough factors
        if len(factors) < 3:
            factors.extend([
                "technical setup offers good entry opportunity",
                "price structure supports upward movement",
                "momentum indicators show positive bias"
            ])

        return factors[:7]  # Return up to 7 factors

    def _get_conflicting_factors(self, current_data: pd.Series, direction: str) -> List[str]:
        """Get factors that conflict with the given direction."""
        conflicts = []

        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)

        if direction == 'long':
            if rsi > 70:
                conflicts.append("RSI showing overbought conditions")
            if macd_hist < -0.1:
                conflicts.append("MACD histogram showing bearish momentum")
            if close < ema_20:
                conflicts.append("price below key moving average")
        else:  # short
            if rsi < 30:
                conflicts.append("RSI showing oversold conditions")
            if macd_hist > 0.1:
                conflicts.append("MACD histogram showing bullish momentum")
            if close > ema_20:
                conflicts.append("price above key moving average")

        return conflicts[:2]  # Return up to 2 main conflicts

    def _generate_short_reasoning(self, current_data: pd.Series,
                                market_analysis: Dict[str, Any],
                                historical_context: Dict[str, Any]) -> str:
        """Generate honest SHORT decision reasoning that acknowledges mixed signals when present."""

        # Analyze technical indicators to assess signal strength
        signal_assessment = self._assess_technical_alignment(current_data, 'short')

        # Get supporting and conflicting factors
        supporting_factors = self._get_honest_short_factors(current_data, market_analysis)
        conflicting_factors = self._get_conflicting_factors(current_data, 'short')

        # Assess signal strength for short position
        signal_strength = self._assess_signal_strength(current_data, 'short')

        # Build natural reasoning based on signal strength
        if signal_strength >= 0.7:
            # Strong technical alignment
            reasoning = f"Taking short position because {', '.join(supporting_factors[:5])}, and {supporting_factors[5] if len(supporting_factors) > 5 else 'setup offers compelling opportunity'}."
        elif signal_strength >= 0.4:
            # Mixed signals - honest assessment
            main_factors = supporting_factors[:3]
            reasoning = f"Taking short position because {', '.join(main_factors)}, providing favorable setup despite some mixed signals requiring monitoring."
        else:
            # Weak technical alignment - focus on setup quality
            key_factors = supporting_factors[:2]
            reasoning = f"Taking short position because setup offers good entry opportunity with {', '.join(key_factors)}, though technical signals are mixed and position requires careful monitoring."

        return reasoning

    def _get_honest_short_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get honest supporting factors for SHORT decision with natural language."""
        factors = []

        # Get indicator values
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        resistance_distance = self._safe_get_value(current_data, 'resistance_distance', 999)
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)

        # Moving average resistance
        if close < ema_20 and ema_20 > 0:
            pct_below = ((ema_20 - close) / ema_20) * 100
            factors.append(f"price {pct_below:.1f}% below EMA-20 confirms downtrend pressure")

        # RSI momentum
        if rsi > 70:
            factors.append(f"RSI at {rsi:.1f} shows overbought conditions ripe for reversal")
        elif rsi < 50:
            factors.append(f"RSI at {rsi:.1f} indicates bearish momentum")

        # MACD confirmation
        if macd_hist < -0.05:
            factors.append(f"MACD histogram at {macd_hist:.2f} confirms downward acceleration")
        elif macd_hist < 0:
            factors.append("MACD histogram negative supports downward bias")

        # Trend strength
        if adx > 25:
            factors.append(f"ADX at {adx:.1f} shows strong trend development")

        # Stochastic positioning
        if stoch_k > 80:
            factors.append(f"Stochastic at {stoch_k:.1f} in overbought territory signals reversal")
        elif stoch_k < 50:
            factors.append("Stochastic below 50 supports continued downside")

        # Resistance proximity
        if resistance_distance < 0.5:
            factors.append("proximity to resistance provides favorable short entry")

        # Bollinger Band analysis
        if bb_position > 0.8:
            factors.append("price near upper Bollinger Band suggests rejection potential")

        # EMA structure
        if ema_20 < ema_50 and ema_20 > 0:
            factors.append("EMA-20 below EMA-50 maintains bearish structure")

        # Ensure we have enough factors
        if len(factors) < 3:
            factors.extend([
                "technical setup offers good short entry opportunity",
                "price structure supports downward movement",
                "momentum indicators show negative bias"
            ])

        return factors[:7]  # Return up to 7 factors

    def _generate_hold_reasoning(self, current_data: pd.Series,
                               market_analysis: Dict[str, Any],
                               historical_context: Dict[str, Any]) -> str:
        """Generate honest HOLD decision reasoning that explains why no position is taken."""

        # Get factors that support holding
        hold_factors = self._get_honest_hold_factors(current_data, market_analysis)

        # Build natural reasoning
        if len(hold_factors) >= 4:
            reasoning = f"Staying hold because {', '.join(hold_factors[:3])}, and {hold_factors[3]}."
        elif len(hold_factors) >= 2:
            reasoning = f"Staying hold because {hold_factors[0]} and {hold_factors[1]}."
        else:
            reasoning = "Staying hold because current market conditions lack clear directional conviction."

        return reasoning

    def _get_honest_hold_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get honest factors for HOLD decision with natural language."""
        factors = []

        # Get indicator values
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        support_distance = self._safe_get_value(current_data, 'support_distance', 999)
        resistance_distance = self._safe_get_value(current_data, 'resistance_distance', 999)

        # RSI neutral zone
        if 45 < rsi < 55:
            factors.append(f"RSI at {rsi:.1f} in neutral zone lacks directional conviction")
        elif 40 < rsi < 60:
            factors.append(f"RSI at {rsi:.1f} shows mixed signals without clear bias")

        # MACD indecision
        if -0.1 < macd_hist < 0.1:
            factors.append("MACD histogram shows minimal momentum")
        elif abs(macd_hist) < 0.5:
            factors.append("MACD signals lack conviction for directional move")

        # Weak trend
        if adx < 25:
            factors.append(f"ADX at {adx:.1f} indicates weak trend strength")
        elif adx < 30:
            factors.append("trend strength insufficient for high-conviction positioning")

        # Bollinger Band middle range
        if 0.3 < bb_position < 0.7:
            factors.append("price in middle of Bollinger Bands suggests consolidation")

        # Stochastic indecision
        if 40 < stoch_k < 60:
            factors.append("Stochastic in neutral zone lacks directional bias")

        # Support/resistance proximity
        if support_distance < 0.3 and resistance_distance < 0.3:
            factors.append("trapped between key support and resistance levels")
        elif min(support_distance, resistance_distance) < 0.5:
            factors.append("proximity to key levels requires patience for breakout")

        # Default factors if none specific
        if len(factors) < 2:
            factors.extend([
                "mixed technical signals prevent clear directional bias",
                "market conditions favor patience over positioning",
                "waiting for stronger technical confirmation"
            ])

        return factors[:4]  # Return up to 4 factors

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
                "category": "timing",
                "primary_conditions": ["poor timing", "high uncertainty", "weak signals"],
                "waiting_for": ["better entry point", "improved setup", "stronger confirmation"]
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

        # Check for momentum breakout with feature names
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist > 0.1:
            reasons.append(f"MACD histogram at {macd_hist:.2f} confirms strong bullish momentum")

        # Check for trend continuation with feature names
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength > 0.6:
            reasons.append(f"trend strength at {trend_strength:.2f} shows established uptrend continuation")

        # Check for oversold bounce with feature names
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi < 35:
            reasons.append(f"RSI-14 at {rsi:.1f} indicates oversold conditions with bounce potential")

        # Check for EMA support with feature names
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', close)
        if close > ema_20:
            reasons.append(f"price above EMA-20 at {ema_20:.1f} confirms uptrend support")

        # Default to general bullish setup
        if not reasons:
            reasons.append("technical indicators align favorably for upward movement")

        return random.choice(reasons)

    def _get_primary_short_reason(self, current_data: pd.Series, market_analysis: Dict[str, Any], weakness: float) -> str:
        """Get primary reason for SHORT decision based on strongest bearish condition."""
        reasons = []

        # Check for momentum breakdown with feature names
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        if macd_hist < -0.1:
            reasons.append(f"MACD histogram at {macd_hist:.2f} confirms strong bearish momentum")

        # Check for trend reversal with feature names
        trend_strength = self._safe_get_value(current_data, 'trend_strength', 0)
        if trend_strength < -0.6:
            reasons.append(f"trend strength at {trend_strength:.2f} shows established downtrend continuation")

        # Check for overbought rejection with feature names
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        if rsi > 70:
            reasons.append(f"RSI-14 at {rsi:.1f} shows overbought conditions with rejection potential")

        # Check for EMA resistance with feature names
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', close)
        if close < ema_20:
            reasons.append(f"price below EMA-20 at {ema_20:.1f} confirms downtrend pressure")

        # Default to general bearish setup
        if not reasons:
            reasons.append("technical indicators align for downward movement")

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

        # Get comprehensive indicator values for multiple conditions
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', 0)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        di_plus = self._safe_get_value(current_data, 'di_plus', 25)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        support_distance = self._safe_get_value(current_data, 'support_distance', 999)

        # 1. Moving average structure
        if close > ema_20 and ema_20 > 0:
            pct_above = ((close - ema_20) / ema_20) * 100
            factors.append(f"price {pct_above:.2f}% above EMA-20 at {ema_20:.1f} confirms uptrend")
        if ema_20 > ema_50 and ema_20 > 0:
            factors.append("EMA-20 above EMA-50 maintains bullish structure")
        if close > sma_20:
            factors.append("price above SMA-20 shows institutional support")

        # 2. RSI momentum
        if 30 < rsi < 70:
            factors.append(f"RSI at {rsi:.1f} shows healthy momentum without overbought risk")
        elif rsi > 50:
            factors.append(f"RSI at {rsi:.1f} confirms bullish momentum")

        # 3. MACD confirmation
        if macd_hist > 0.05:
            factors.append(f"MACD histogram at {macd_hist:.3f} signals strong acceleration")
        elif macd_hist > 0:
            factors.append("MACD histogram positive confirms upward momentum")

        # 4. ADX trend strength
        if adx > 25 and di_plus > 25:
            factors.append(f"ADX at {adx:.1f} with strong DI+ confirms uptrend")
        elif adx > 20:
            factors.append("developing trend strength supports continuation")

        # 5. Stochastic positioning
        if 20 < stoch_k < 80:
            factors.append(f"Stochastic at {stoch_k:.1f} positioned for continued upside")

        # 6. Support proximity
        if support_distance < 0.5:
            factors.append("positioned near key support provides favorable entry")

        # 7. Bollinger Band analysis
        if 0.2 < bb_position < 0.8:
            factors.append("Bollinger Band position shows room for expansion")

        return factors[:5]  # Return 5 supporting factors like real traders

    def _get_short_supporting_factors(self, current_data: pd.Series, market_analysis: Dict[str, Any]) -> List[str]:
        """Get supporting factors for SHORT decision."""
        factors = []

        # Get comprehensive indicator values for multiple conditions
        close = self._safe_get_value(current_data, 'close', 0)
        ema_20 = self._safe_get_value(current_data, 'ema_20', 0)
        ema_50 = self._safe_get_value(current_data, 'ema_50', 0)
        sma_20 = self._safe_get_value(current_data, 'sma_20', 0)
        rsi = self._safe_get_value(current_data, 'rsi_14', 50)
        macd_hist = self._safe_get_value(current_data, 'macd_histogram', 0)
        adx = self._safe_get_value(current_data, 'adx', 25)
        di_minus = self._safe_get_value(current_data, 'di_minus', 25)
        stoch_k = self._safe_get_value(current_data, 'stoch_k', 50)
        bb_position = self._safe_get_value(current_data, 'bb_position', 0.5)
        resistance_distance = self._safe_get_value(current_data, 'resistance_distance', 999)

        # 1. Moving average resistance
        if close < ema_20 and ema_20 > 0:
            pct_below = ((ema_20 - close) / ema_20) * 100
            factors.append(f"price {pct_below:.1f}% below EMA-20 at {ema_20:.1f} confirms downtrend")
        if ema_20 < ema_50 and ema_20 > 0:
            factors.append("EMA-20 below EMA-50 maintains bearish structure")
        if close < sma_20:
            factors.append("price below SMA-20 shows selling pressure")

        # 2. RSI momentum
        if rsi > 70:
            factors.append(f"RSI at {rsi:.1f} overbought conditions signal reversal")
        elif rsi < 50:
            factors.append(f"RSI at {rsi:.1f} confirms bearish momentum")

        # 3. MACD confirmation
        if macd_hist < -0.05:
            factors.append(f"MACD histogram at {macd_hist:.3f} signals strong bearish acceleration")
        elif macd_hist < 0:
            factors.append("MACD histogram negative confirms downward momentum")

        # 4. ADX trend strength
        if adx > 25 and di_minus > 25:
            factors.append(f"ADX at {adx:.1f} with strong DI- confirms downtrend")

        # 5. Stochastic positioning
        if stoch_k > 80:
            factors.append(f"Stochastic at {stoch_k:.1f} overbought signals reversal")
        elif stoch_k < 50:
            factors.append("Stochastic below 50 supports downside")

        # 6. Resistance proximity
        if resistance_distance < 0.5:
            factors.append("positioned near resistance provides favorable short entry")

        # 7. Bollinger Band analysis
        if bb_position > 0.8:
            factors.append("price near upper Bollinger Band suggests rejection")

        return factors[:5]  # Return 5 supporting factors like real traders

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
        """Get criteria for what we're waiting for in HOLD decision - NO RISK-REWARD REFERENCES."""
        criteria = [
            "waiting for clearer confirmation from momentum indicators",
            "monitoring for breakout above resistance or breakdown below support",
            "awaiting stronger technical signals for directional conviction",
            "watching for volume confirmation of price movement",
            "looking for trend strength improvement before positioning"
        ]

        return random.choice(criteria)
