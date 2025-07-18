#!/usr/bin/env python3
"""
Historical Context Manager
=========================

Manages historical context analysis using a rolling window of previous candles
to provide contextual information for reasoning generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class HistoricalContextManager:
    """
    Manages historical context for reasoning generation using a rolling window
    of previous candles to provide context-aware analysis.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize the historical context manager.
        
        Args:
            window_size: Number of previous candles to analyze for context
        """
        self.window_size = window_size
        self.context_cache = {}
        
        logger.info(f"HistoricalContextManager initialized with window size: {window_size}")
    
    def get_historical_context(self, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """
        Extract comprehensive historical context from previous candles.
        
        Args:
            df: DataFrame with all features
            current_idx: Current row index
            
        Returns:
            Dictionary containing historical context analysis
        """
        # Define the historical window
        start_idx = max(0, current_idx - self.window_size)
        end_idx = current_idx
        
        if end_idx <= start_idx:
            return self._get_minimal_context()
        
        # Extract historical data
        hist_data = df.iloc[start_idx:end_idx]
        
        # Generate cache key for performance
        cache_key = f"{start_idx}_{end_idx}_{len(hist_data)}"
        
        if cache_key in self.context_cache:
            return self.context_cache[cache_key]
        
        # Analyze historical patterns
        context = {
            'trend_analysis': self._analyze_historical_trend(hist_data),
            'volatility_analysis': self._analyze_historical_volatility(hist_data),
            'pattern_frequency': self._analyze_pattern_frequency(hist_data),
            'support_resistance_history': self._analyze_sr_history(hist_data),
            'momentum_analysis': self._analyze_momentum_patterns(hist_data),
            'recent_signals': self._analyze_recent_signals(hist_data),
            'market_regime': self._identify_market_regime(hist_data),
            'price_action_summary': self._analyze_price_action(hist_data)
        }
        
        # Cache the result
        self.context_cache[cache_key] = context
        
        return context
    
    def _get_minimal_context(self) -> Dict[str, Any]:
        """Return minimal context for early data points with insufficient history."""
        return {
            'trend_analysis': {
                'direction': 'neutral',
                'strength': 'weak',
                'consistency': 'low',
                'avg_slope': 0.0
            },
            'volatility_analysis': {
                'level': 'normal',
                'trend': 'stable',
                'current_vs_historical': 1.0
            },
            'pattern_frequency': {
                'bullish_patterns': 0,
                'bearish_patterns': 0,
                'pattern_bias': 'neutral'
            },
            'support_resistance_history': {
                'support_tests': 0,
                'resistance_tests': 0,
                'support_strength': 'unknown',
                'resistance_strength': 'unknown'
            },
            'momentum_analysis': {
                'rsi_trend': 'neutral',
                'macd_trend': 'neutral',
                'momentum_consistency': 'low'
            },
            'recent_signals': {
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'signal_bias': 'neutral'
            },
            'market_regime': 'consolidation',
            'price_action_summary': {
                'volatility_regime': 'normal',
                'trend_regime': 'neutral',
                'dominant_pattern': 'none'
            }
        }
    
    def _analyze_historical_trend(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics over the historical period."""
        if 'trend_slope' not in hist_data.columns or len(hist_data) == 0:
            return {'direction': 'neutral', 'strength': 'weak', 'consistency': 'low', 'avg_slope': 0.0}
        
        slopes = hist_data['trend_slope'].dropna()
        if len(slopes) == 0:
            return {'direction': 'neutral', 'strength': 'weak', 'consistency': 'low', 'avg_slope': 0.0}
        
        # Trend direction analysis
        avg_slope = slopes.mean()
        direction = 'bullish' if avg_slope > 0.1 else 'bearish' if avg_slope < -0.1 else 'neutral'
        
        # Trend strength analysis
        abs_slope = abs(avg_slope)
        strength = 'strong' if abs_slope > 0.5 else 'moderate' if abs_slope > 0.2 else 'weak'
        
        # Trend consistency analysis
        slope_std = slopes.std()
        consistency = 'high' if slope_std < 0.3 else 'moderate' if slope_std < 0.6 else 'low'
        
        # Additional trend metrics
        positive_slopes = (slopes > 0).sum()
        negative_slopes = (slopes < 0).sum()
        trend_persistence = max(positive_slopes, negative_slopes) / len(slopes) if len(slopes) > 0 else 0
        
        return {
            'direction': direction,
            'strength': strength,
            'consistency': consistency,
            'avg_slope': float(avg_slope),
            'trend_persistence': float(trend_persistence),
            'slope_volatility': float(slope_std)
        }
    
    def _analyze_historical_volatility(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns over the historical period."""
        if 'atr' not in hist_data.columns or len(hist_data) == 0:
            return {'level': 'normal', 'trend': 'stable', 'current_vs_historical': 1.0}
        
        atr_values = hist_data['atr'].dropna()
        if len(atr_values) < 10:
            return {'level': 'normal', 'trend': 'stable', 'current_vs_historical': 1.0}
        
        # Current vs historical volatility comparison
        recent_atr = atr_values.tail(10).mean()
        historical_atr = atr_values.mean()
        
        vol_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0
        
        # Volatility level classification
        level = 'high' if vol_ratio > 1.3 else 'low' if vol_ratio < 0.7 else 'normal'
        
        # Volatility trend analysis
        if len(atr_values) >= 20:
            early_atr = atr_values.head(len(atr_values)//2).mean()
            late_atr = atr_values.tail(len(atr_values)//2).mean()
            vol_trend_ratio = late_atr / early_atr if early_atr > 0 else 1.0
            trend = 'increasing' if vol_trend_ratio > 1.1 else 'decreasing' if vol_trend_ratio < 0.9 else 'stable'
        else:
            trend = 'stable'
        
        return {
            'level': level,
            'trend': trend,
            'current_vs_historical': float(vol_ratio),
            'recent_atr': float(recent_atr),
            'historical_atr': float(historical_atr)
        }
    
    def _analyze_pattern_frequency(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze frequency and bias of candlestick patterns."""
        pattern_cols = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']
        
        bullish_patterns = 0
        bearish_patterns = 0
        total_patterns = 0
        
        for col in pattern_cols:
            if col in hist_data.columns:
                pattern_count = hist_data[col].sum()
                total_patterns += pattern_count
                
                if col in ['hammer', 'bullish_engulfing']:
                    bullish_patterns += pattern_count
                elif col in ['bearish_engulfing']:
                    bearish_patterns += pattern_count
                # doji is considered neutral
        
        # Determine pattern bias
        if bullish_patterns > bearish_patterns * 1.5:
            pattern_bias = 'bullish'
        elif bearish_patterns > bullish_patterns * 1.5:
            pattern_bias = 'bearish'
        else:
            pattern_bias = 'neutral'
        
        return {
            'bullish_patterns': int(bullish_patterns),
            'bearish_patterns': int(bearish_patterns),
            'total_patterns': int(total_patterns),
            'pattern_bias': pattern_bias,
            'pattern_frequency': float(total_patterns / len(hist_data)) if len(hist_data) > 0 else 0.0
        }
    
    def _analyze_sr_history(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze support/resistance interaction history."""
        if 'support_distance' not in hist_data.columns or 'resistance_distance' not in hist_data.columns:
            return {
                'support_tests': 0,
                'resistance_tests': 0,
                'support_strength': 'unknown',
                'resistance_strength': 'unknown'
            }
        
        # Count support/resistance tests (when price gets close)
        support_tests = (hist_data['support_distance'] < 1.5).sum()  # Within 1.5% of support
        resistance_tests = (hist_data['resistance_distance'] < 1.5).sum()  # Within 1.5% of resistance
        
        # Analyze strength based on test frequency and bounces
        support_strength = 'strong' if support_tests > 5 else 'moderate' if support_tests > 2 else 'weak'
        resistance_strength = 'strong' if resistance_tests > 5 else 'moderate' if resistance_tests > 2 else 'weak'
        
        return {
            'support_tests': int(support_tests),
            'resistance_tests': int(resistance_tests),
            'support_strength': support_strength,
            'resistance_strength': resistance_strength
        }
    
    def _analyze_momentum_patterns(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicator patterns and trends."""
        momentum_analysis = {
            'rsi_trend': 'neutral',
            'macd_trend': 'neutral',
            'momentum_consistency': 'low'
        }
        
        # RSI trend analysis
        if 'rsi_14' in hist_data.columns:
            rsi_values = hist_data['rsi_14'].dropna()
            if len(rsi_values) >= 10:
                recent_rsi = rsi_values.tail(5).mean()
                if recent_rsi > 60:
                    momentum_analysis['rsi_trend'] = 'bullish'
                elif recent_rsi < 40:
                    momentum_analysis['rsi_trend'] = 'bearish'
        
        # MACD trend analysis
        if 'macd_histogram' in hist_data.columns:
            macd_hist = hist_data['macd_histogram'].dropna()
            if len(macd_hist) >= 5:
                recent_macd = macd_hist.tail(3).mean()
                if recent_macd > 0:
                    momentum_analysis['macd_trend'] = 'bullish'
                elif recent_macd < 0:
                    momentum_analysis['macd_trend'] = 'bearish'
        
        # Momentum consistency
        if momentum_analysis['rsi_trend'] == momentum_analysis['macd_trend'] and momentum_analysis['rsi_trend'] != 'neutral':
            momentum_analysis['momentum_consistency'] = 'high'
        elif momentum_analysis['rsi_trend'] != 'neutral' or momentum_analysis['macd_trend'] != 'neutral':
            momentum_analysis['momentum_consistency'] = 'moderate'
        
        return momentum_analysis
    
    def _analyze_recent_signals(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recent signal distribution and bias."""
        if 'signal' not in hist_data.columns or len(hist_data) == 0:
            return {
                'buy_signals': 0,
                'sell_signals': 0,
                'hold_signals': 0,
                'signal_bias': 'neutral'
            }
        
        # Analyze last 20 signals for recent bias
        recent_signals = hist_data['signal'].tail(20)
        
        buy_signals = int((recent_signals == 1).sum())
        sell_signals = int((recent_signals == 2).sum())
        hold_signals = int((recent_signals == 0).sum())
        
        # Determine signal bias
        if buy_signals > sell_signals * 1.5:
            signal_bias = 'bullish'
        elif sell_signals > buy_signals * 1.5:
            signal_bias = 'bearish'
        else:
            signal_bias = 'neutral'
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'signal_bias': signal_bias,
            'total_signals': len(recent_signals)
        }
    
    def _identify_market_regime(self, hist_data: pd.DataFrame) -> str:
        """Identify current market regime based on historical data."""
        if len(hist_data) == 0:
            return 'consolidation'
        
        # Default regime
        regime = 'consolidation'
        
        # Analyze trend strength and volatility
        if 'trend_strength' in hist_data.columns and 'volatility_20' in hist_data.columns:
            trend_strength = hist_data['trend_strength'].dropna().tail(20).mean()
            volatility = hist_data['volatility_20'].dropna().tail(20).mean()
            
            if not pd.isna(trend_strength) and not pd.isna(volatility):
                if trend_strength > 0.6 and volatility < 2.5:
                    regime = 'trending'
                elif volatility > 4.0:
                    regime = 'volatile'
                elif trend_strength < 0.3 and volatility < 2.0:
                    regime = 'consolidation'
                else:
                    regime = 'transitional'
        
        return regime
    
    def _analyze_price_action(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall price action characteristics."""
        if len(hist_data) == 0:
            return {
                'volatility_regime': 'normal',
                'trend_regime': 'neutral',
                'dominant_pattern': 'none'
            }
        
        summary = {
            'volatility_regime': 'normal',
            'trend_regime': 'neutral',
            'dominant_pattern': 'none'
        }
        
        # Volatility regime
        if 'hl_range' in hist_data.columns:
            avg_range = hist_data['hl_range'].mean()
            if not pd.isna(avg_range):
                if avg_range > 1.5:
                    summary['volatility_regime'] = 'high'
                elif avg_range < 0.5:
                    summary['volatility_regime'] = 'low'
        
        # Trend regime
        if 'trend_direction' in hist_data.columns:
            trend_directions = hist_data['trend_direction'].dropna()
            if len(trend_directions) > 0:
                bullish_count = (trend_directions == 1).sum()
                bearish_count = (trend_directions == -1).sum()
                
                if bullish_count > len(trend_directions) * 0.7:
                    summary['trend_regime'] = 'bullish'
                elif bearish_count > len(trend_directions) * 0.7:
                    summary['trend_regime'] = 'bearish'
        
        # Dominant pattern (simplified)
        pattern_analysis = self._analyze_pattern_frequency(hist_data)
        if pattern_analysis['total_patterns'] > len(hist_data) * 0.1:  # More than 10% pattern frequency
            summary['dominant_pattern'] = pattern_analysis['pattern_bias']
        
        return summary
    
    def clear_cache(self):
        """Clear the context cache to free memory."""
        self.context_cache.clear()
        logger.info("Historical context cache cleared")
