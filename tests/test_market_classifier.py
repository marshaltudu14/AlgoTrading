"""
Unit tests for MarketClassifier.

Tests the functionality of market regime classification including
different market scenarios and data formats.
"""

import pytest
import numpy as np
import pandas as pd
from src.reasoning.market_classifier import MarketClassifier, MarketRegime


class TestMarketClassifier:
    """Test cases for MarketClassifier class."""
    
    def test_market_classifier_initialization(self):
        """Test MarketClassifier initialization with default parameters."""
        classifier = MarketClassifier()
        
        assert classifier.trend_period == 20
        assert classifier.volatility_period == 14
        assert classifier.atr_period == 14
        assert classifier.adx_period == 14
        assert classifier.trend_threshold == 25.0
        assert classifier.volatility_threshold == 0.02
        assert classifier.ranging_threshold == 0.5
    
    def test_market_classifier_custom_parameters(self):
        """Test MarketClassifier initialization with custom parameters."""
        classifier = MarketClassifier(
            trend_period=30,
            volatility_period=20,
            trend_threshold=30.0,
            volatility_threshold=0.03
        )
        
        assert classifier.trend_period == 30
        assert classifier.volatility_period == 20
        assert classifier.trend_threshold == 30.0
        assert classifier.volatility_threshold == 0.03
    
    def test_prepare_data_numpy_array(self):
        """Test data preparation with numpy arrays."""
        classifier = MarketClassifier()
        
        # Test with OHLCV array
        ohlcv_data = np.random.randn(50, 5)
        ohlcv_data[:, :4] = np.abs(ohlcv_data[:, :4]) + 100  # Positive prices
        ohlcv_data[:, 4] = np.abs(ohlcv_data[:, 4]) * 1000  # Positive volume
        
        prepared = classifier._prepare_data(ohlcv_data)
        assert prepared.shape == (50, 5)
        
        # Test with OHLC array (missing volume)
        ohlc_data = ohlcv_data[:, :4]
        prepared = classifier._prepare_data(ohlc_data)
        assert prepared.shape == (50, 5)
        assert np.all(prepared[:, 4] == 1.0)  # Dummy volume
        
        # Test with single price series
        price_series = np.random.randn(30) + 100
        prepared = classifier._prepare_data(price_series)
        assert prepared.shape == (30, 5)
        # OHLC should all be the same (price series)
        assert np.allclose(prepared[:, 0], prepared[:, 1])
        assert np.allclose(prepared[:, 0], prepared[:, 2])
        assert np.allclose(prepared[:, 0], prepared[:, 3])
    
    def test_prepare_data_pandas_dataframe(self):
        """Test data preparation with pandas DataFrame."""
        classifier = MarketClassifier()
        
        # Create sample DataFrame
        data = {
            'open': np.random.randn(40) + 100,
            'high': np.random.randn(40) + 102,
            'low': np.random.randn(40) + 98,
            'close': np.random.randn(40) + 100,
            'volume': np.random.randint(1000, 10000, 40)
        }
        df = pd.DataFrame(data)
        
        prepared = classifier._prepare_data(df)
        assert prepared.shape == (40, 5)
        
        # Test with missing volume
        df_no_volume = df.drop('volume', axis=1)
        prepared = classifier._prepare_data(df_no_volume)
        assert prepared.shape == (40, 5)
        assert np.all(prepared[:, 4] == 1.0)  # Dummy volume
    
    def test_prepare_data_dictionary(self):
        """Test data preparation with dictionary format."""
        classifier = MarketClassifier()
        
        data = {
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }
        
        prepared = classifier._prepare_data(data)
        assert prepared.shape == (5, 5)
        assert prepared[0, 0] == 100  # First open
        assert prepared[-1, 3] == 105  # Last close
    
    def test_classify_market_insufficient_data(self):
        """Test market classification with insufficient data."""
        classifier = MarketClassifier()
        
        # Very small dataset
        small_data = np.random.randn(5, 5) + 100
        regime = classifier.classify_market(small_data)
        
        assert isinstance(regime, MarketRegime)
        assert regime == MarketRegime.CONSOLIDATION  # Default for insufficient data
    
    def test_classify_market_trending_scenario(self):
        """Test market classification for trending scenario."""
        classifier = MarketClassifier(trend_threshold=20.0)
        
        # Create trending data (upward trend)
        n_periods = 50
        base_price = 100
        trend_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            price = base_price + i * 0.5  # Steady upward trend
            noise = np.random.randn() * 0.1
            trend_data[i] = [
                price + noise,  # open
                price + abs(noise) + 0.2,  # high
                price - abs(noise) - 0.2,  # low
                price + noise * 0.5,  # close
                1000  # volume
            ]
        
        regime = classifier.classify_market(trend_data)
        assert isinstance(regime, MarketRegime)
        # Should classify as trending or at least not consolidation
        assert regime in [MarketRegime.TRENDING, MarketRegime.RANGING, MarketRegime.VOLATILE]
    
    def test_classify_market_volatile_scenario(self):
        """Test market classification for volatile scenario."""
        classifier = MarketClassifier(volatility_threshold=0.01)
        
        # Create volatile data
        n_periods = 50
        base_price = 100
        volatile_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            # High volatility with random jumps
            price_change = np.random.randn() * 5  # Large random changes
            price = base_price + price_change
            noise = np.random.randn() * 2
            
            volatile_data[i] = [
                price + noise,  # open
                price + abs(noise) + 2,  # high
                price - abs(noise) - 2,  # low
                price + noise * 0.5,  # close
                1000  # volume
            ]
        
        regime = classifier.classify_market(volatile_data)
        assert isinstance(regime, MarketRegime)
        # Should classify as volatile or at least not consolidation
        assert regime in [MarketRegime.VOLATILE, MarketRegime.RANGING, MarketRegime.TRENDING]
    
    def test_classify_market_consolidation_scenario(self):
        """Test market classification for consolidation scenario."""
        classifier = MarketClassifier(volatility_threshold=0.05)
        
        # Create consolidation data (low volatility, narrow range)
        n_periods = 50
        base_price = 100
        consolidation_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            # Very small price movements
            noise = np.random.randn() * 0.05  # Very small noise
            price = base_price + noise
            
            consolidation_data[i] = [
                price,  # open
                price + 0.1,  # high
                price - 0.1,  # low
                price + noise * 0.1,  # close
                1000  # volume
            ]
        
        regime = classifier.classify_market(consolidation_data)
        assert isinstance(regime, MarketRegime)
        # Should classify as consolidation or ranging
        assert regime in [MarketRegime.CONSOLIDATION, MarketRegime.RANGING]
    
    def test_classify_market_with_confidence(self):
        """Test market classification with confidence scores."""
        classifier = MarketClassifier()
        
        # Create sample data
        data = np.random.randn(30, 5) + 100
        data[:, :4] = np.abs(data[:, :4])  # Positive prices
        data[:, 4] = np.abs(data[:, 4]) * 1000  # Positive volume
        
        regime, confidence = classifier.classify_market(data, return_confidence=True)
        
        assert isinstance(regime, MarketRegime)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    def test_get_market_features(self):
        """Test getting detailed market features."""
        classifier = MarketClassifier()
        
        # Create sample data
        data = np.random.randn(40, 5) + 100
        data[:, :4] = np.abs(data[:, :4])  # Positive prices
        data[:, 4] = np.abs(data[:, 4]) * 1000  # Positive volume
        
        features = classifier.get_market_features(data)
        
        assert isinstance(features, dict)
        assert 'regime' in features
        assert 'confidence' in features
        assert 'adx' in features
        assert 'volatility' in features
        assert 'atr_ratio' in features
        assert 'trend_strength' in features
        assert 'price_range_ratio' in features
        assert 'momentum' in features
        
        # Check that all values are numeric
        for key, value in features.items():
            if key != 'regime':  # regime is string
                assert isinstance(value, (int, float))
                assert not np.isnan(value)
    
    def test_get_market_features_insufficient_data(self):
        """Test getting market features with insufficient data."""
        classifier = MarketClassifier()
        
        # Very small dataset
        small_data = np.random.randn(3, 5) + 100
        features = classifier.get_market_features(small_data)
        
        assert features == {}  # Should return empty dict
    
    def test_calculate_atr(self):
        """Test ATR calculation."""
        classifier = MarketClassifier(atr_period=5)
        
        # Create sample price data
        high = np.array([102, 103, 104, 105, 106, 107])
        low = np.array([98, 99, 100, 101, 102, 103])
        close = np.array([100, 101, 102, 103, 104, 105])
        
        atr = classifier._calculate_atr(high, low, close)
        
        assert isinstance(atr, float)
        assert atr >= 0.0
        assert not np.isnan(atr)
    
    def test_calculate_adx(self):
        """Test ADX calculation."""
        classifier = MarketClassifier(adx_period=5)
        
        # Create sample price data with trend
        high = np.array([102, 103, 104, 105, 106, 107, 108])
        low = np.array([98, 99, 100, 101, 102, 103, 104])
        close = np.array([100, 101, 102, 103, 104, 105, 106])
        
        adx = classifier._calculate_adx(high, low, close)
        
        assert isinstance(adx, float)
        assert adx >= 0.0
        assert not np.isnan(adx)
    
    def test_get_regime_probabilities(self):
        """Test getting probabilities for all regimes."""
        classifier = MarketClassifier()
        
        # Create sample data
        data = np.random.randn(30, 5) + 100
        data[:, :4] = np.abs(data[:, :4])  # Positive prices
        data[:, 4] = np.abs(data[:, 4]) * 1000  # Positive volume
        
        probabilities = classifier.get_regime_probabilities(data)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 4  # Four regimes
        
        # Check that all regime names are present
        regime_names = [regime.value for regime in MarketRegime]
        for regime_name in regime_names:
            assert regime_name in probabilities
            assert isinstance(probabilities[regime_name], float)
            assert 0.0 <= probabilities[regime_name] <= 1.0
        
        # Probabilities should sum to approximately 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.1  # Allow some tolerance
    
    def test_get_regime_probabilities_insufficient_data(self):
        """Test regime probabilities with insufficient data."""
        classifier = MarketClassifier()
        
        # Very small dataset
        small_data = np.random.randn(3, 5) + 100
        probabilities = classifier.get_regime_probabilities(small_data)
        
        assert isinstance(probabilities, dict)
        assert len(probabilities) == 4
        
        # Should return uniform probabilities
        for prob in probabilities.values():
            assert abs(prob - 0.25) < 0.01  # Should be approximately 0.25
    
    def test_market_regimes_enum(self):
        """Test MarketRegime enum values."""
        assert MarketRegime.TRENDING.value == "trending"
        assert MarketRegime.RANGING.value == "ranging"
        assert MarketRegime.VOLATILE.value == "volatile"
        assert MarketRegime.CONSOLIDATION.value == "consolidation"
        
        # Test that all regimes are different
        regimes = list(MarketRegime)
        assert len(regimes) == 4
        assert len(set(regime.value for regime in regimes)) == 4
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data formats."""
        classifier = MarketClassifier()
        
        # Test with invalid data type
        with pytest.raises(ValueError):
            classifier.classify_market("invalid_data")
        
        # Test with empty array - should return default classification, not raise error
        result = classifier.classify_market(np.array([]).reshape(0, 5))
        assert result == MarketRegime.CONSOLIDATION
        
        # Test with wrong shape
        with pytest.raises(ValueError):
            classifier.classify_market(np.random.randn(10, 2))  # Only 2 columns


if __name__ == "__main__":
    pytest.main([__file__])
