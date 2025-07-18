#!/usr/bin/env python3
"""
Feature Templates for Enhanced Reasoning System
==============================================

Provides templates for technical indicator analysis with natural language
variations and feature relationship understanding.
"""

import random
from typing import Dict, List, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FeatureTemplates:
    """
    Feature template system for technical indicator analysis.
    """
    
    def __init__(self):
        """Initialize feature templates."""
        self.rsi_templates = self._load_rsi_templates()
        self.macd_templates = self._load_macd_templates()
        self.ma_templates = self._load_ma_templates()
        self.bollinger_templates = self._load_bollinger_templates()
        self.momentum_templates = self._load_momentum_templates()
        self.volatility_templates = self._load_volatility_templates()
        
        logger.info("FeatureTemplates initialized with indicator templates")
    
    def get_rsi_analysis(self, rsi_value: float, historical_context: Dict[str, Any] = None) -> str:
        """Get RSI analysis based on value and historical context."""
        if rsi_value >= 70:
            condition = "overbought"
        elif rsi_value <= 30:
            condition = "oversold"
        elif 45 <= rsi_value <= 55:
            condition = "neutral"
        elif rsi_value > 55:
            condition = "bullish"
        else:
            condition = "bearish"
        
        templates = self.rsi_templates.get(condition, self.rsi_templates["neutral"])
        template = random.choice(templates)
        
        return template.format(rsi_value=rsi_value)
    
    def get_macd_analysis(self, macd: float, signal: float, histogram: float) -> str:
        """Get MACD analysis based on values."""
        if histogram > 0.1:
            condition = "strong_bullish"
        elif histogram > 0:
            condition = "bullish"
        elif histogram < -0.1:
            condition = "strong_bearish"
        elif histogram < 0:
            condition = "bearish"
        else:
            condition = "neutral"
        
        templates = self.macd_templates.get(condition, self.macd_templates["neutral"])
        template = random.choice(templates)
        
        return template.format(histogram=histogram)
    
    def get_ma_analysis(self, current_data: pd.Series) -> str:
        """Get moving average analysis."""
        close = current_data.get('close', 0)
        sma_20 = current_data.get('sma_20', close)
        sma_50 = current_data.get('sma_50', close)
        
        if close > sma_20 > sma_50:
            condition = "bullish_alignment"
        elif close < sma_20 < sma_50:
            condition = "bearish_alignment"
        elif close > sma_20:
            condition = "above_short_term"
        elif close < sma_20:
            condition = "below_short_term"
        else:
            condition = "neutral"
        
        templates = self.ma_templates.get(condition, self.ma_templates["neutral"])
        template = random.choice(templates)
        
        return template
    
    def get_bollinger_analysis(self, bb_position: float, bb_width: float) -> str:
        """Get Bollinger Bands analysis."""
        if bb_position > 0.8:
            position = "upper"
        elif bb_position < 0.2:
            position = "lower"
        else:
            position = "middle"
        
        if bb_width < 0.02:
            width_condition = "squeeze"
        elif bb_width > 0.05:
            width_condition = "expansion"
        else:
            width_condition = "normal"
        
        condition = f"{position}_{width_condition}"
        templates = self.bollinger_templates.get(condition, self.bollinger_templates.get(position, ["price within Bollinger Band range"]))
        template = random.choice(templates)
        
        return template.format(bb_position=bb_position*100)
    
    def _load_rsi_templates(self) -> Dict[str, List[str]]:
        """Load RSI analysis templates."""
        return {
            "overbought": [
                "RSI at {rsi_value:.1f} indicates overbought conditions with potential for pullback",
                "momentum indicator shows overbought reading at {rsi_value:.1f} suggesting caution",
                "RSI overbought at {rsi_value:.1f} warns of possible near-term weakness"
            ],
            "oversold": [
                "RSI at {rsi_value:.1f} indicates oversold conditions with bounce potential",
                "momentum indicator shows oversold reading at {rsi_value:.1f} suggesting opportunity",
                "RSI oversold at {rsi_value:.1f} indicates potential reversal setup"
            ],
            "bullish": [
                "RSI at {rsi_value:.1f} shows healthy bullish momentum with room for advance",
                "momentum indicator displays positive reading at {rsi_value:.1f} supporting upside",
                "RSI bullish at {rsi_value:.1f} indicates sustained momentum potential"
            ],
            "bearish": [
                "RSI at {rsi_value:.1f} shows bearish momentum with downside pressure",
                "momentum indicator displays negative reading at {rsi_value:.1f} supporting decline",
                "RSI bearish at {rsi_value:.1f} indicates sustained weakness potential"
            ],
            "neutral": [
                "RSI at {rsi_value:.1f} remains in neutral territory showing balanced conditions",
                "momentum indicator neutral at {rsi_value:.1f} suggests directional uncertainty",
                "RSI balanced at {rsi_value:.1f} indicates lack of momentum extremes"
            ]
        }
    
    def _load_macd_templates(self) -> Dict[str, List[str]]:
        """Load MACD analysis templates."""
        return {
            "strong_bullish": [
                "MACD histogram strongly positive at {histogram:.3f} confirms bullish momentum acceleration",
                "momentum indicator shows strong bullish divergence with histogram at {histogram:.3f}",
                "MACD strongly bullish with histogram {histogram:.3f} indicating sustained upward pressure"
            ],
            "bullish": [
                "MACD histogram positive at {histogram:.3f} suggests bullish momentum development",
                "momentum indicator shows bullish bias with histogram at {histogram:.3f}",
                "MACD bullish with histogram {histogram:.3f} supporting upward movement"
            ],
            "strong_bearish": [
                "MACD histogram strongly negative at {histogram:.3f} confirms bearish momentum acceleration",
                "momentum indicator shows strong bearish divergence with histogram at {histogram:.3f}",
                "MACD strongly bearish with histogram {histogram:.3f} indicating sustained downward pressure"
            ],
            "bearish": [
                "MACD histogram negative at {histogram:.3f} suggests bearish momentum development",
                "momentum indicator shows bearish bias with histogram at {histogram:.3f}",
                "MACD bearish with histogram {histogram:.3f} supporting downward movement"
            ],
            "neutral": [
                "MACD histogram near zero suggests neutral momentum conditions",
                "momentum indicator shows balanced conditions with minimal directional bias",
                "MACD neutral indicating lack of strong momentum in either direction"
            ]
        }
    
    def _load_ma_templates(self) -> Dict[str, List[str]]:
        """Load moving average analysis templates."""
        return {
            "bullish_alignment": [
                "moving averages show bullish alignment with price above all major levels",
                "all major moving averages aligned bullishly supporting upward trend",
                "price trading above ascending moving average structure"
            ],
            "bearish_alignment": [
                "moving averages show bearish alignment with price below all major levels",
                "all major moving averages aligned bearishly supporting downward trend",
                "price trading below descending moving average structure"
            ],
            "above_short_term": [
                "price trading above short-term moving average support",
                "short-term moving average providing support for current price action",
                "price maintaining position above key short-term moving average"
            ],
            "below_short_term": [
                "price trading below short-term moving average resistance",
                "short-term moving average acting as resistance to price advance",
                "price struggling below key short-term moving average level"
            ],
            "neutral": [
                "moving averages show mixed signals with unclear directional bias",
                "price action around moving average levels suggests indecision",
                "moving average structure indicates neutral market conditions"
            ]
        }
    
    def _load_bollinger_templates(self) -> Dict[str, List[str]]:
        """Load Bollinger Bands analysis templates."""
        return {
            "upper_squeeze": [
                "price near upper Bollinger Band during squeeze suggests breakout potential",
                "upper band test during low volatility indicates building pressure"
            ],
            "upper_expansion": [
                "price at upper Bollinger Band during expansion shows strong momentum",
                "upper band breach during volatility expansion confirms trend strength"
            ],
            "upper_normal": [
                "price near upper Bollinger Band at {bb_position:.0f}% suggests overbought conditions",
                "upper band proximity indicates potential resistance and pullback risk"
            ],
            "lower_squeeze": [
                "price near lower Bollinger Band during squeeze suggests breakout potential",
                "lower band test during low volatility indicates building pressure"
            ],
            "lower_expansion": [
                "price at lower Bollinger Band during expansion shows strong selling pressure",
                "lower band breach during volatility expansion confirms trend weakness"
            ],
            "lower_normal": [
                "price near lower Bollinger Band at {bb_position:.0f}% suggests oversold conditions",
                "lower band proximity indicates potential support and bounce opportunity"
            ],
            "middle_squeeze": [
                "price consolidating around Bollinger Band middle during squeeze phase",
                "middle band trading during low volatility suggests directional uncertainty"
            ],
            "middle_expansion": [
                "price around Bollinger Band middle during expansion shows balanced conditions",
                "middle band area during volatility expansion indicates equilibrium"
            ],
            "middle_normal": [
                "price trading around Bollinger Band middle suggests balanced conditions",
                "middle band area indicates neutral momentum and directional uncertainty"
            ]
        }
    
    def _load_momentum_templates(self) -> Dict[str, List[str]]:
        """Load momentum analysis templates."""
        return {
            "strong_bullish": [
                "momentum indicators show strong bullish acceleration",
                "all momentum measures confirm sustained upward pressure",
                "momentum analysis reveals powerful bullish thrust"
            ],
            "moderate_bullish": [
                "momentum indicators display moderate bullish bias",
                "momentum measures suggest steady upward pressure",
                "momentum analysis shows developing bullish characteristics"
            ],
            "weak_bullish": [
                "momentum indicators show weak bullish tendencies",
                "momentum measures display tentative upward bias",
                "momentum analysis reveals fragile bullish conditions"
            ],
            "strong_bearish": [
                "momentum indicators show strong bearish acceleration",
                "all momentum measures confirm sustained downward pressure",
                "momentum analysis reveals powerful bearish thrust"
            ],
            "moderate_bearish": [
                "momentum indicators display moderate bearish bias",
                "momentum measures suggest steady downward pressure",
                "momentum analysis shows developing bearish characteristics"
            ],
            "weak_bearish": [
                "momentum indicators show weak bearish tendencies",
                "momentum measures display tentative downward bias",
                "momentum analysis reveals fragile bearish conditions"
            ],
            "neutral": [
                "momentum indicators show neutral conditions",
                "momentum measures display balanced characteristics",
                "momentum analysis reveals lack of directional conviction"
            ]
        }
    
    def _load_volatility_templates(self) -> Dict[str, List[str]]:
        """Load volatility analysis templates."""
        return {
            "high": [
                "volatility measures indicate elevated market uncertainty",
                "high volatility environment suggests increased risk and opportunity",
                "volatility analysis reveals expanded price movement potential"
            ],
            "moderate": [
                "volatility measures show moderate market conditions",
                "normal volatility environment suggests typical price behavior",
                "volatility analysis indicates standard market dynamics"
            ],
            "low": [
                "volatility measures indicate compressed market conditions",
                "low volatility environment suggests potential for expansion",
                "volatility analysis reveals quiet market with breakout potential"
            ]
        }
