"""
Market Weather Forecaster - Market Classification Module

This module provides market regime classification capabilities, allowing the
autonomous agent to understand the current market "weather" and adapt its
trading strategy accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Warning deduplication to prevent spam
_warning_cache = set()


class MarketRegime(Enum):
    """Enumeration of market regimes."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CONSOLIDATION = "consolidation"


class MarketClassifier:
    """
    Market Weather Forecaster for classifying market regimes.
    
    This class analyzes market data to determine the current market regime,
    helping the autonomous agent understand market conditions and adapt
    its trading strategy accordingly.
    
    The classifier uses multiple technical indicators to determine:
    - Trending: Strong directional movement
    - Ranging: Sideways movement within bounds
    - Volatile: High volatility with erratic price movements
    - Consolidation: Low volatility, tight price range
    """
    
    def __init__(
        self,
        trend_period: int = 20,
        volatility_period: int = 14,
        atr_period: int = 14,
        adx_period: int = 14,
        trend_threshold: float = 25.0,
        volatility_threshold: float = 0.02,
        ranging_threshold: float = 0.5
    ):
        """
        Initialize the Market Classifier.

        Args:
            trend_period: Period for trend calculation (moving averages)
            volatility_period: Period for volatility calculation
            atr_period: Period for Average True Range calculation
            adx_period: Period for ADX (Average Directional Index) calculation
            trend_threshold: ADX threshold for trending markets
            volatility_threshold: Volatility threshold for volatile markets
            ranging_threshold: Threshold for ranging markets (price range ratio)
        """
        self.trend_period = trend_period
        self.volatility_period = volatility_period
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
        self.ranging_threshold = ranging_threshold
        
        # Cache for indicators to avoid recalculation
        self._indicator_cache = {}
        
        logger.info(f"Initialized MarketClassifier with trend_threshold={trend_threshold}, "
                   f"volatility_threshold={volatility_threshold}")
    
    def classify_market(
        self, 
        market_data: Union[pd.DataFrame, np.ndarray, Dict],
        return_confidence: bool = False
    ) -> Union[MarketRegime, Tuple[MarketRegime, float]]:
        """
        Classify the current market regime based on market data.
        
        Args:
            market_data: Market data containing OHLCV information
            return_confidence: Whether to return confidence score
            
        Returns:
            MarketRegime enum value, optionally with confidence score
        """
        # Convert input to standardized format
        ohlcv_data = self._prepare_data(market_data)
        
        required_periods = max(self.trend_period, self.volatility_period, self.adx_period)
        if len(ohlcv_data) < required_periods:
            warning_key = f"insufficient_data_market_classification_{len(ohlcv_data)}_{required_periods}"
            if warning_key not in _warning_cache:
                logger.warning(f"Insufficient data for market classification: have {len(ohlcv_data)} periods, need {required_periods}")
                _warning_cache.add(warning_key)
            # Return default regime with low confidence
            regime = MarketRegime.CONSOLIDATION
            confidence = 0.25  # Low confidence due to insufficient data
            return (regime, confidence) if return_confidence else regime
        
        # Calculate technical indicators
        indicators = self._calculate_indicators(ohlcv_data)
        
        # Classify based on indicators
        regime, confidence = self._classify_from_indicators(indicators)
        
        logger.debug(f"Market classified as {regime.value} with confidence {confidence:.3f}")
        
        return (regime, confidence) if return_confidence else regime
    
    def get_market_features(self, market_data: Union[pd.DataFrame, np.ndarray, Dict]) -> Dict[str, float]:
        """
        Get detailed market features for analysis.
        
        Args:
            market_data: Market data containing OHLCV information
            
        Returns:
            Dictionary of market features and indicators
        """
        ohlcv_data = self._prepare_data(market_data)
        
        if len(ohlcv_data) < max(self.trend_period, self.volatility_period, self.adx_period):
            return {}
        
        indicators = self._calculate_indicators(ohlcv_data)
        regime, confidence = self._classify_from_indicators(indicators)
        
        return {
            'regime': regime.value,
            'confidence': confidence,
            'adx': indicators['adx'],
            'volatility': indicators['volatility'],
            'atr_ratio': indicators['atr_ratio'],
            'trend_strength': indicators['trend_strength'],
            'price_range_ratio': indicators['price_range_ratio'],
            'momentum': indicators['momentum']
        }
    
    def _prepare_data(self, market_data: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        """
        Convert input data to standardized OHLCV numpy array.
        
        Args:
            market_data: Input market data in various formats
            
        Returns:
            Numpy array with shape (n_periods, 5) for OHLCV
        """
        if isinstance(market_data, pd.DataFrame):
            # Assume DataFrame has OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in market_data.columns.str.lower()]
            
            if len(available_cols) >= 4:  # At least OHLC
                data = market_data[available_cols].values
                if data.shape[1] == 4:  # Add dummy volume if missing
                    volume = np.ones((data.shape[0], 1))
                    data = np.hstack([data, volume])
                return data
            else:
                raise ValueError("DataFrame must contain OHLC columns")
        
        elif isinstance(market_data, np.ndarray):
            if market_data.ndim == 1:
                # Single price series, create OHLC from it
                prices = market_data
                ohlc = np.column_stack([prices, prices, prices, prices])
                volume = np.ones((len(prices), 1))
                return np.hstack([ohlc, volume])
            elif market_data.ndim == 2 and market_data.shape[1] >= 4:
                # Already in OHLCV format
                if market_data.shape[1] == 4:  # Add dummy volume
                    volume = np.ones((market_data.shape[0], 1))
                    return np.hstack([market_data, volume])
                return market_data[:, :5]  # Take first 5 columns
            else:
                raise ValueError("Array must have shape (n, 4) or (n, 5) for OHLCV")
        
        elif isinstance(market_data, dict):
            # Extract OHLCV from dictionary
            try:
                open_prices = np.array(market_data.get('open', market_data.get('Open', [])))
                high_prices = np.array(market_data.get('high', market_data.get('High', [])))
                low_prices = np.array(market_data.get('low', market_data.get('Low', [])))
                close_prices = np.array(market_data.get('close', market_data.get('Close', [])))
                volume = np.array(market_data.get('volume', market_data.get('Volume', 
                                                np.ones(len(close_prices)))))
                
                return np.column_stack([open_prices, high_prices, low_prices, close_prices, volume])
            except Exception as e:
                raise ValueError(f"Could not extract OHLCV from dictionary: {e}")
        
        else:
            raise ValueError(f"Unsupported data type: {type(market_data)}")
    
    def _calculate_indicators(self, ohlcv_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate technical indicators for market classification.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            Dictionary of calculated indicators
        """
        open_prices = ohlcv_data[:, 0]
        high_prices = ohlcv_data[:, 1]
        low_prices = ohlcv_data[:, 2]
        close_prices = ohlcv_data[:, 3]
        volume = ohlcv_data[:, 4]
        
        # Calculate ADX (Average Directional Index) for trend strength
        adx = self._calculate_adx(high_prices, low_prices, close_prices)
        
        # Calculate volatility (rolling standard deviation of returns)
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns[-self.volatility_period:]) if len(returns) >= self.volatility_period else 0.0
        
        # Calculate ATR (Average True Range)
        atr = self._calculate_atr(high_prices, low_prices, close_prices)
        atr_ratio = atr / close_prices[-1] if close_prices[-1] != 0 else 0.0
        
        # Calculate trend strength using moving averages
        if len(close_prices) >= self.trend_period:
            ma_short = np.mean(close_prices[-self.trend_period//2:])
            ma_long = np.mean(close_prices[-self.trend_period:])
            trend_strength = abs(ma_short - ma_long) / ma_long if ma_long != 0 else 0.0
        else:
            trend_strength = 0.0
        
        # Calculate price range ratio (recent range vs historical range)
        recent_range = np.max(high_prices[-5:]) - np.min(low_prices[-5:])
        historical_range = np.max(high_prices[-self.trend_period:]) - np.min(low_prices[-self.trend_period:])
        price_range_ratio = recent_range / historical_range if historical_range != 0 else 0.0
        
        # Calculate momentum
        momentum = (close_prices[-1] - close_prices[-min(10, len(close_prices))]) / close_prices[-min(10, len(close_prices))] if len(close_prices) >= 10 else 0.0
        
        return {
            'adx': adx,
            'volatility': volatility,
            'atr_ratio': atr_ratio,
            'trend_strength': trend_strength,
            'price_range_ratio': price_range_ratio,
            'momentum': momentum
        }
    
    def _calculate_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate Average Directional Index (ADX)."""
        if len(high) < self.adx_period + 1:
            return 0.0
        
        # Calculate True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Smooth the values
        if len(tr) >= self.adx_period:
            atr = np.mean(tr[-self.adx_period:])
            adm_plus = np.mean(dm_plus[-self.adx_period:])
            adm_minus = np.mean(dm_minus[-self.adx_period:])
            
            # Calculate DI+ and DI-
            di_plus = (adm_plus / atr) * 100 if atr != 0 else 0
            di_minus = (adm_minus / atr) * 100 if atr != 0 else 0
            
            # Calculate ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100 if (di_plus + di_minus) != 0 else 0
            return dx
        
        return 0.0
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
        """Calculate Average True Range (ATR)."""
        if len(high) < self.atr_period + 1:
            return 0.0
        
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        return np.mean(tr[-self.atr_period:]) if len(tr) >= self.atr_period else 0.0
    
    def _classify_from_indicators(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on calculated indicators.
        
        Args:
            indicators: Dictionary of technical indicators
            
        Returns:
            Tuple of (MarketRegime, confidence_score)
        """
        adx = indicators['adx']
        volatility = indicators['volatility']
        atr_ratio = indicators['atr_ratio']
        trend_strength = indicators['trend_strength']
        price_range_ratio = indicators['price_range_ratio']
        
        # Classification logic
        scores = {
            MarketRegime.TRENDING: 0.0,
            MarketRegime.RANGING: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CONSOLIDATION: 0.0
        }
        
        # Trending market indicators
        if adx > self.trend_threshold:
            scores[MarketRegime.TRENDING] += 0.4
        if trend_strength > 0.02:  # 2% trend strength
            scores[MarketRegime.TRENDING] += 0.3
        if volatility < self.volatility_threshold:  # Low volatility trending
            scores[MarketRegime.TRENDING] += 0.2
        
        # Volatile market indicators
        if volatility > self.volatility_threshold * 2:  # High volatility
            scores[MarketRegime.VOLATILE] += 0.4
        if atr_ratio > 0.03:  # High ATR ratio
            scores[MarketRegime.VOLATILE] += 0.3
        if adx < self.trend_threshold * 0.5:  # Low trend strength
            scores[MarketRegime.VOLATILE] += 0.2
        
        # Ranging market indicators
        if price_range_ratio < self.ranging_threshold:  # Narrow recent range
            scores[MarketRegime.RANGING] += 0.3
        if adx < self.trend_threshold * 0.7 and volatility < self.volatility_threshold * 1.5:
            scores[MarketRegime.RANGING] += 0.4
        if abs(indicators['momentum']) < 0.01:  # Low momentum
            scores[MarketRegime.RANGING] += 0.2
        
        # Consolidation market indicators
        if volatility < self.volatility_threshold * 0.5:  # Very low volatility
            scores[MarketRegime.CONSOLIDATION] += 0.4
        if atr_ratio < 0.01:  # Very low ATR
            scores[MarketRegime.CONSOLIDATION] += 0.3
        if price_range_ratio < self.ranging_threshold * 0.5:  # Very narrow range
            scores[MarketRegime.CONSOLIDATION] += 0.2
        
        # Determine best classification
        best_regime = max(scores, key=scores.get)
        confidence = scores[best_regime]
        
        # Ensure minimum confidence and handle ties
        if confidence < 0.3:
            # Default to consolidation if no clear signal
            best_regime = MarketRegime.CONSOLIDATION
            confidence = 0.5
        
        return best_regime, min(confidence, 1.0)
    
    def get_regime_probabilities(self, market_data: Union[pd.DataFrame, np.ndarray, Dict]) -> Dict[str, float]:
        """
        Get probabilities for all market regimes.
        
        Args:
            market_data: Market data containing OHLCV information
            
        Returns:
            Dictionary mapping regime names to probabilities
        """
        ohlcv_data = self._prepare_data(market_data)
        
        if len(ohlcv_data) < max(self.trend_period, self.volatility_period, self.adx_period):
            # Return uniform probabilities if insufficient data
            return {regime.value: 0.25 for regime in MarketRegime}
        
        indicators = self._calculate_indicators(ohlcv_data)
        
        # Calculate scores for all regimes
        scores = {}
        for regime in MarketRegime:
            regime_indicators = {regime.value: indicators}
            _, score = self._classify_from_indicators(indicators)
            scores[regime.value] = score
        
        # Normalize to probabilities
        total_score = sum(scores.values())
        if total_score > 0:
            probabilities = {regime: score / total_score for regime, score in scores.items()}
        else:
            probabilities = {regime.value: 0.25 for regime in MarketRegime}
        
        return probabilities
