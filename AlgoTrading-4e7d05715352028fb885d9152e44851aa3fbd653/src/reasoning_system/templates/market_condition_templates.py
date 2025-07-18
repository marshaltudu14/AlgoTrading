#!/usr/bin/env python3
"""
Market Condition Templates for Enhanced Reasoning System
=======================================================

Provides templates for different market conditions (trending, ranging, volatile)
with historical context and natural language variations.
"""

import random
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class MarketConditionTemplates:
    """
    Market condition template system for different market regimes.
    """
    
    def __init__(self):
        """Initialize market condition templates."""
        self.trending_templates = self._load_trending_templates()
        self.ranging_templates = self._load_ranging_templates()
        self.volatile_templates = self._load_volatile_templates()
        self.low_volatility_templates = self._load_low_volatility_templates()
        
        logger.info("MarketConditionTemplates initialized with condition templates")
    
    def get_market_condition_template(self, condition_type: str, strength: str = "moderate") -> Dict[str, Any]:
        """
        Get market condition template based on type and strength.
        
        Args:
            condition_type: "trending", "ranging", "volatile", "low_volatility"
            strength: "weak", "moderate", "strong"
            
        Returns:
            Market condition template dictionary
        """
        if condition_type == "trending":
            templates = self.trending_templates
        elif condition_type == "ranging":
            templates = self.ranging_templates
        elif condition_type == "volatile":
            templates = self.volatile_templates
        elif condition_type == "low_volatility":
            templates = self.low_volatility_templates
        else:
            templates = self.trending_templates  # Default
        
        # Filter by strength if available
        strength_templates = [t for t in templates if t.get("strength", "moderate") == strength]
        if not strength_templates:
            strength_templates = templates
        
        return random.choice(strength_templates)
    
    def _load_trending_templates(self) -> List[Dict[str, Any]]:
        """Load trending market condition templates."""
        return [
            {
                "name": "strong_uptrend",
                "strength": "strong",
                "description": "strong bullish trending market",
                "context_phrases": [
                    "Market structure reveals strong bullish trending conditions",
                    "Current environment demonstrates robust upward momentum",
                    "Technical analysis confirms established bullish trend regime"
                ],
                "characteristics": [
                    "consistent higher highs and higher lows",
                    "strong momentum indicators",
                    "trend-following behavior"
                ],
                "implications": ["trend continuation likely", "momentum strategies favored"]
            },
            {
                "name": "moderate_uptrend",
                "strength": "moderate", 
                "description": "moderate bullish trending market",
                "context_phrases": [
                    "Market structure shows moderate bullish trending characteristics",
                    "Current environment displays steady upward bias",
                    "Technical analysis indicates developing bullish trend"
                ],
                "characteristics": [
                    "generally higher highs and lows",
                    "moderate momentum readings",
                    "selective trend participation"
                ],
                "implications": ["cautious trend following", "selective positioning"]
            },
            {
                "name": "weak_uptrend",
                "strength": "weak",
                "description": "weak bullish trending market",
                "context_phrases": [
                    "Market structure exhibits weak bullish trending tendencies",
                    "Current environment shows tentative upward bias",
                    "Technical analysis suggests fragile bullish trend"
                ],
                "characteristics": [
                    "irregular higher highs and lows",
                    "weak momentum indicators",
                    "trend uncertainty"
                ],
                "implications": ["trend reversal risk", "defensive positioning"]
            },
            {
                "name": "strong_downtrend",
                "strength": "strong",
                "description": "strong bearish trending market",
                "context_phrases": [
                    "Market structure reveals strong bearish trending conditions",
                    "Current environment demonstrates robust downward momentum",
                    "Technical analysis confirms established bearish trend regime"
                ],
                "characteristics": [
                    "consistent lower highs and lower lows",
                    "strong bearish momentum",
                    "trend-following selling"
                ],
                "implications": ["trend continuation likely", "short strategies favored"]
            },
            {
                "name": "moderate_downtrend",
                "strength": "moderate",
                "description": "moderate bearish trending market", 
                "context_phrases": [
                    "Market structure shows moderate bearish trending characteristics",
                    "Current environment displays steady downward bias",
                    "Technical analysis indicates developing bearish trend"
                ],
                "characteristics": [
                    "generally lower highs and lows",
                    "moderate bearish momentum",
                    "selective selling pressure"
                ],
                "implications": ["cautious short positioning", "selective entries"]
            },
            {
                "name": "weak_downtrend",
                "strength": "weak",
                "description": "weak bearish trending market",
                "context_phrases": [
                    "Market structure exhibits weak bearish trending tendencies",
                    "Current environment shows tentative downward bias", 
                    "Technical analysis suggests fragile bearish trend"
                ],
                "characteristics": [
                    "irregular lower highs and lows",
                    "weak bearish momentum",
                    "trend uncertainty"
                ],
                "implications": ["trend reversal risk", "defensive positioning"]
            }
        ]
    
    def _load_ranging_templates(self) -> List[Dict[str, Any]]:
        """Load ranging market condition templates."""
        return [
            {
                "name": "tight_range",
                "strength": "strong",
                "description": "tight horizontal trading range",
                "context_phrases": [
                    "Market structure reveals tight horizontal trading range",
                    "Current environment demonstrates well-defined range boundaries",
                    "Technical analysis confirms established range-bound conditions"
                ],
                "characteristics": [
                    "clear support and resistance levels",
                    "low volatility environment",
                    "mean reversion behavior"
                ],
                "implications": ["range trading strategies", "breakout preparation"]
            },
            {
                "name": "wide_range",
                "strength": "moderate",
                "description": "wide horizontal trading range",
                "context_phrases": [
                    "Market structure shows wide horizontal trading range",
                    "Current environment displays broad range boundaries",
                    "Technical analysis indicates expansive range-bound movement"
                ],
                "characteristics": [
                    "distant support and resistance",
                    "moderate volatility swings",
                    "range oscillation"
                ],
                "implications": ["swing trading opportunities", "range extremes focus"]
            },
            {
                "name": "choppy_range",
                "strength": "weak",
                "description": "choppy range-bound market",
                "context_phrases": [
                    "Market structure exhibits choppy range-bound characteristics",
                    "Current environment shows irregular range behavior",
                    "Technical analysis suggests unstable range conditions"
                ],
                "characteristics": [
                    "unclear range boundaries",
                    "erratic price movement",
                    "false breakout risk"
                ],
                "implications": ["difficult trading environment", "patience required"]
            }
        ]
    
    def _load_volatile_templates(self) -> List[Dict[str, Any]]:
        """Load high volatility market condition templates."""
        return [
            {
                "name": "high_volatility_trending",
                "strength": "strong",
                "description": "high volatility trending market",
                "context_phrases": [
                    "Market structure reveals high volatility trending environment",
                    "Current conditions demonstrate elevated volatility with directional bias",
                    "Technical analysis confirms volatile trending regime"
                ],
                "characteristics": [
                    "large price swings",
                    "strong directional moves",
                    "momentum acceleration"
                ],
                "implications": ["trend following with wider stops", "volatility strategies"]
            },
            {
                "name": "high_volatility_ranging",
                "strength": "moderate",
                "description": "high volatility range-bound market",
                "context_phrases": [
                    "Market structure shows high volatility range-bound conditions",
                    "Current environment displays elevated volatility within range",
                    "Technical analysis indicates volatile range oscillation"
                ],
                "characteristics": [
                    "wide price swings within range",
                    "rapid reversals",
                    "increased noise"
                ],
                "implications": ["range trading with wider parameters", "quick reversals"]
            }
        ]
    
    def _load_low_volatility_templates(self) -> List[Dict[str, Any]]:
        """Load low volatility market condition templates."""
        return [
            {
                "name": "low_volatility_consolidation",
                "strength": "strong",
                "description": "low volatility consolidation phase",
                "context_phrases": [
                    "Market structure reveals low volatility consolidation environment",
                    "Current conditions demonstrate compressed volatility regime",
                    "Technical analysis confirms quiet consolidation phase"
                ],
                "characteristics": [
                    "narrow price ranges",
                    "low momentum readings",
                    "consolidation behavior"
                ],
                "implications": ["breakout preparation", "volatility expansion expected"]
            },
            {
                "name": "low_volatility_drift",
                "strength": "moderate",
                "description": "low volatility directional drift",
                "context_phrases": [
                    "Market structure shows low volatility directional drift",
                    "Current environment displays gradual directional movement",
                    "Technical analysis indicates slow trending behavior"
                ],
                "characteristics": [
                    "gradual price movement",
                    "low momentum",
                    "steady progression"
                ],
                "implications": ["patient trend following", "position building"]
            }
        ]
    
    def get_trending_templates(self) -> List[Dict[str, Any]]:
        """Get all trending market templates."""
        return self.trending_templates
    
    def get_ranging_templates(self) -> List[Dict[str, Any]]:
        """Get all ranging market templates."""
        return self.ranging_templates
    
    def get_volatile_templates(self) -> List[Dict[str, Any]]:
        """Get all volatile market templates."""
        return self.volatile_templates
    
    def get_low_volatility_templates(self) -> List[Dict[str, Any]]:
        """Get all low volatility templates."""
        return self.low_volatility_templates
