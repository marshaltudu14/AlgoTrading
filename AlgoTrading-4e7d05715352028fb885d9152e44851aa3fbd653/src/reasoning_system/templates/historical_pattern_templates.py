#!/usr/bin/env python3
"""
Historical Pattern Templates for Enhanced Reasoning System
=========================================================

Provides templates for historical pattern analysis with realistic trader-like
time horizons (20-50 candles short-term, 100-200 candles medium-term).
"""

import random
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class HistoricalPatternTemplates:
    """
    Historical pattern template system for realistic temporal analysis.
    """
    
    def __init__(self):
        """Initialize historical pattern templates."""
        self.continuation_templates = self._load_continuation_templates()
        self.reversal_templates = self._load_reversal_templates()
        self.consolidation_templates = self._load_consolidation_templates()
        
        # Time horizon phrases
        self.time_horizons = self._load_time_horizon_phrases()
        
        logger.info("HistoricalPatternTemplates initialized with pattern templates")
    
    def get_pattern_template(self, pattern_type: str, time_horizon: str = "short_term") -> Dict[str, Any]:
        """
        Get pattern template based on type and time horizon.
        
        Args:
            pattern_type: "continuation", "reversal", or "consolidation"
            time_horizon: "short_term" (20-50), "medium_term" (100-200)
            
        Returns:
            Pattern template dictionary
        """
        if pattern_type == "continuation":
            template = random.choice(self.continuation_templates)
        elif pattern_type == "reversal":
            template = random.choice(self.reversal_templates)
        elif pattern_type == "consolidation":
            template = random.choice(self.consolidation_templates)
        else:
            template = random.choice(self.continuation_templates)  # Default
        
        # Add time horizon context
        template["time_context"] = self.time_horizons[time_horizon]
        
        return template
    
    def _load_continuation_templates(self) -> List[Dict[str, Any]]:
        """Load trend continuation pattern templates."""
        return [
            {
                "name": "bullish_flag_continuation",
                "description": "bullish flag pattern showing trend continuation",
                "context_phrases": [
                    "Looking at the past {period} candles, price has maintained an ascending trend channel",
                    "The recent {period}-candle pattern shows classic flag formation within the uptrend",
                    "Historical analysis over {period} periods reveals consistent higher highs and higher lows"
                ],
                "implications": ["trend continuation", "momentum resumption"],
                "confidence_factors": ["volume pattern", "trend strength", "pattern completion"]
            },
            {
                "name": "bearish_flag_continuation", 
                "description": "bearish flag pattern showing trend continuation",
                "context_phrases": [
                    "Analysis of the past {period} candles reveals a descending trend channel",
                    "The recent {period}-candle pattern shows classic bear flag formation",
                    "Historical review over {period} periods confirms consistent lower highs and lower lows"
                ],
                "implications": ["trend continuation", "momentum resumption"],
                "confidence_factors": ["volume pattern", "trend strength", "pattern completion"]
            },
            {
                "name": "ascending_triangle_breakout",
                "description": "ascending triangle pattern with bullish breakout potential",
                "context_phrases": [
                    "The past {period} candles have formed an ascending triangle pattern",
                    "Historical analysis shows {period} periods of higher lows against resistance",
                    "Recent {period}-candle formation demonstrates building bullish pressure"
                ],
                "implications": ["upside breakout", "momentum acceleration"],
                "confidence_factors": ["resistance tests", "volume buildup", "momentum convergence"]
            },
            {
                "name": "descending_triangle_breakdown",
                "description": "descending triangle pattern with bearish breakdown potential", 
                "context_phrases": [
                    "The past {period} candles have formed a descending triangle pattern",
                    "Historical analysis shows {period} periods of lower highs against support",
                    "Recent {period}-candle formation demonstrates building bearish pressure"
                ],
                "implications": ["downside breakdown", "momentum acceleration"],
                "confidence_factors": ["support tests", "volume buildup", "momentum convergence"]
            },
            {
                "name": "pullback_completion",
                "description": "healthy pullback completion within established trend",
                "context_phrases": [
                    "The {period}-candle pullback has reached typical retracement levels",
                    "Historical pattern over {period} periods shows classic trend correction",
                    "Analysis of past {period} candles reveals orderly pullback structure"
                ],
                "implications": ["trend resumption", "entry opportunity"],
                "confidence_factors": ["retracement depth", "support holding", "momentum recovery"]
            }
        ]
    
    def _load_reversal_templates(self) -> List[Dict[str, Any]]:
        """Load trend reversal pattern templates."""
        return [
            {
                "name": "double_top_reversal",
                "description": "double top pattern indicating potential trend reversal",
                "context_phrases": [
                    "The past {period} candles have formed a classic double top pattern",
                    "Historical analysis over {period} periods shows failed breakout attempts",
                    "Recent {period}-candle formation reveals weakening bullish momentum"
                ],
                "implications": ["trend reversal", "bearish momentum"],
                "confidence_factors": ["volume divergence", "momentum weakness", "pattern completion"]
            },
            {
                "name": "double_bottom_reversal",
                "description": "double bottom pattern indicating potential trend reversal",
                "context_phrases": [
                    "The past {period} candles have formed a classic double bottom pattern",
                    "Historical analysis over {period} periods shows successful support tests",
                    "Recent {period}-candle formation reveals strengthening bullish momentum"
                ],
                "implications": ["trend reversal", "bullish momentum"],
                "confidence_factors": ["volume confirmation", "momentum strength", "pattern completion"]
            },
            {
                "name": "head_shoulders_reversal",
                "description": "head and shoulders pattern indicating trend reversal",
                "context_phrases": [
                    "The {period}-candle formation shows classic head and shoulders pattern",
                    "Historical analysis reveals {period} periods of distribution pattern",
                    "Past {period} candles demonstrate clear reversal formation"
                ],
                "implications": ["major reversal", "trend change"],
                "confidence_factors": ["neckline break", "volume pattern", "momentum shift"]
            },
            {
                "name": "inverse_head_shoulders",
                "description": "inverse head and shoulders pattern indicating trend reversal",
                "context_phrases": [
                    "The {period}-candle formation shows inverse head and shoulders pattern",
                    "Historical analysis reveals {period} periods of accumulation pattern",
                    "Past {period} candles demonstrate clear bullish reversal formation"
                ],
                "implications": ["major reversal", "trend change"],
                "confidence_factors": ["neckline break", "volume pattern", "momentum shift"]
            },
            {
                "name": "divergence_reversal",
                "description": "momentum divergence indicating potential reversal",
                "context_phrases": [
                    "Analysis of {period} candles reveals bearish divergence between price and momentum",
                    "The past {period} periods show weakening momentum despite higher prices",
                    "Historical review over {period} candles indicates momentum exhaustion"
                ],
                "implications": ["momentum reversal", "trend weakness"],
                "confidence_factors": ["divergence strength", "momentum indicators", "volume pattern"]
            }
        ]
    
    def _load_consolidation_templates(self) -> List[Dict[str, Any]]:
        """Load consolidation/ranging pattern templates."""
        return [
            {
                "name": "horizontal_range",
                "description": "horizontal trading range with defined support and resistance",
                "context_phrases": [
                    "The past {period} candles show price consolidating in horizontal range",
                    "Historical analysis over {period} periods reveals well-defined trading range",
                    "Recent {period}-candle pattern demonstrates range-bound price action"
                ],
                "implications": ["range continuation", "breakout preparation"],
                "confidence_factors": ["range boundaries", "volume pattern", "time duration"]
            },
            {
                "name": "symmetrical_triangle",
                "description": "symmetrical triangle pattern showing indecision",
                "context_phrases": [
                    "The {period}-candle formation shows symmetrical triangle pattern",
                    "Historical analysis reveals {period} periods of converging price action",
                    "Past {period} candles demonstrate decreasing volatility and range compression"
                ],
                "implications": ["directional breakout", "volatility expansion"],
                "confidence_factors": ["pattern completion", "volume contraction", "apex approach"]
            },
            {
                "name": "rectangular_consolidation",
                "description": "rectangular consolidation pattern within trend",
                "context_phrases": [
                    "The past {period} candles show rectangular consolidation pattern",
                    "Historical review over {period} periods reveals sideways price movement",
                    "Analysis of {period}-candle formation shows pause in trending move"
                ],
                "implications": ["trend continuation", "energy building"],
                "confidence_factors": ["range respect", "volume pattern", "trend context"]
            }
        ]
    
    def _load_time_horizon_phrases(self) -> Dict[str, Dict[str, Any]]:
        """Load time horizon context phrases."""
        return {
            "short_term": {
                "period_range": "20-50",
                "description": "immediate trend context",
                "phrases": [
                    "over the immediate 30-candle period",
                    "within the recent 25-candle timeframe", 
                    "across the past 40 trading sessions",
                    "during the last 35-period analysis window"
                ]
            },
            "medium_term": {
                "period_range": "100-200", 
                "description": "broader trend context",
                "phrases": [
                    "over the broader 150-candle analysis period",
                    "within the extended 120-candle timeframe",
                    "across the past 180 trading sessions",
                    "during the comprehensive 160-period review"
                ]
            }
        }
    
    def get_continuation_templates(self) -> List[Dict[str, Any]]:
        """Get all continuation pattern templates."""
        return self.continuation_templates
    
    def get_reversal_templates(self) -> List[Dict[str, Any]]:
        """Get all reversal pattern templates."""
        return self.reversal_templates
    
    def get_consolidation_templates(self) -> List[Dict[str, Any]]:
        """Get all consolidation pattern templates."""
        return self.consolidation_templates
