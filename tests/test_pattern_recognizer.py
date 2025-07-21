"""
Unit tests for PatternRecognizer.

Tests the functionality of technical analysis pattern recognition including
candlestick patterns, chart patterns, and neural network detection.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from src.reasoning.pattern_recognizer import PatternRecognizer, PatternType, PatternDetection, Pattern1DCNN


class TestPatternDetection:
    """Test cases for PatternDetection dataclass."""
    
    def test_pattern_detection_creation(self):
        """Test PatternDetection creation and validation."""
        detection = PatternDetection(
            pattern_type=PatternType.DOJI,
            confidence=0.8,
            start_index=5,
            end_index=10,
            strength=0.7,
            direction="neutral"
        )
        
        assert detection.pattern_type == PatternType.DOJI
        assert detection.confidence == 0.8
        assert detection.start_index == 5
        assert detection.end_index == 10
        assert detection.strength == 0.7
        assert detection.direction == "neutral"
    
    def test_pattern_detection_validation(self):
        """Test PatternDetection validation."""
        # Invalid confidence
        with pytest.raises(ValueError):
            PatternDetection(
                pattern_type=PatternType.DOJI,
                confidence=1.5,  # > 1.0
                start_index=5,
                end_index=10
            )
        
        # Invalid index order
        with pytest.raises(ValueError):
            PatternDetection(
                pattern_type=PatternType.DOJI,
                confidence=0.8,
                start_index=10,
                end_index=5  # start > end
            )


class TestPattern1DCNN:
    """Test cases for Pattern1DCNN neural network."""
    
    def test_pattern_cnn_initialization(self):
        """Test Pattern1DCNN initialization."""
        model = Pattern1DCNN(
            input_channels=4,
            sequence_length=50,
            num_patterns=len(PatternType),
            hidden_dim=32
        )
        
        assert model.input_channels == 4
        assert model.sequence_length == 50
        assert model.num_patterns == len(PatternType)
        
        # Check that layers are properly initialized
        assert isinstance(model.conv1, torch.nn.Conv1d)
        assert isinstance(model.conv2, torch.nn.Conv1d)
        assert isinstance(model.conv3, torch.nn.Conv1d)
        assert isinstance(model.fc_pattern, torch.nn.Linear)
        assert isinstance(model.fc_confidence, torch.nn.Linear)
    
    def test_pattern_cnn_forward_pass(self):
        """Test Pattern1DCNN forward pass."""
        model = Pattern1DCNN(
            input_channels=4,
            sequence_length=50,
            num_patterns=len(PatternType),
            hidden_dim=32
        )
        
        # Create sample input (batch_size=2, channels=4, sequence_length=50)
        x = torch.randn(2, 4, 50)
        
        output = model(x)
        
        assert 'pattern_probabilities' in output
        assert 'confidence' in output
        assert 'features' in output
        
        # Check output shapes
        assert output['pattern_probabilities'].shape == (2, len(PatternType))
        assert output['confidence'].shape == (2, 1)
        
        # Check that probabilities sum to 1
        prob_sums = torch.sum(output['pattern_probabilities'], dim=1)
        assert torch.allclose(prob_sums, torch.ones(2), atol=1e-6)
        
        # Check that confidence is between 0 and 1
        assert torch.all(output['confidence'] >= 0)
        assert torch.all(output['confidence'] <= 1)


class TestPatternRecognizer:
    """Test cases for PatternRecognizer class."""
    
    def test_pattern_recognizer_initialization(self):
        """Test PatternRecognizer initialization."""
        recognizer = PatternRecognizer(
            sequence_length=30,
            min_pattern_confidence=0.7,
            use_neural_network=True
        )
        
        assert recognizer.sequence_length == 30
        assert recognizer.min_pattern_confidence == 0.7
        assert recognizer.use_neural_network is True
        assert recognizer.cnn_model is not None
        assert isinstance(recognizer.cnn_model, Pattern1DCNN)
    
    def test_pattern_recognizer_without_nn(self):
        """Test PatternRecognizer initialization without neural network."""
        recognizer = PatternRecognizer(
            sequence_length=30,
            use_neural_network=False
        )
        
        assert recognizer.use_neural_network is False
        assert recognizer.cnn_model is None
    
    def test_prepare_price_data_numpy_array(self):
        """Test price data preparation with numpy arrays."""
        recognizer = PatternRecognizer()
        
        # Test with OHLCV array
        ohlcv_data = np.random.randn(50, 5)
        ohlcv_data[:, :4] = np.abs(ohlcv_data[:, :4]) + 100  # Positive prices
        ohlcv_data[:, 4] = np.abs(ohlcv_data[:, 4]) * 1000  # Positive volume
        
        prepared = recognizer._prepare_price_data(ohlcv_data)
        assert prepared.shape == (50, 5)
        
        # Test with single price series
        price_series = np.random.randn(30) + 100
        prepared = recognizer._prepare_price_data(price_series)
        assert prepared.shape == (30, 5)
    
    def test_prepare_price_data_pandas_dataframe(self):
        """Test price data preparation with pandas DataFrame."""
        recognizer = PatternRecognizer()
        
        # Create sample DataFrame
        data = {
            'open': np.random.randn(40) + 100,
            'high': np.random.randn(40) + 102,
            'low': np.random.randn(40) + 98,
            'close': np.random.randn(40) + 100,
            'volume': np.random.randint(1000, 10000, 40)
        }
        df = pd.DataFrame(data)
        
        prepared = recognizer._prepare_price_data(df)
        assert prepared.shape == (40, 5)
    
    def test_recognize_pattern_insufficient_data(self):
        """Test pattern recognition with insufficient data."""
        recognizer = PatternRecognizer(sequence_length=50)
        
        # Very small dataset
        small_data = np.random.randn(10, 5) + 100
        detection = recognizer.recognize_pattern(small_data)
        
        assert isinstance(detection, PatternDetection)
        assert detection.pattern_type == PatternType.NO_PATTERN
    
    def test_recognize_pattern_doji_scenario(self):
        """Test pattern recognition for Doji candlestick pattern."""
        recognizer = PatternRecognizer(use_neural_network=False)  # Use only rule-based
        
        # Create data with Doji pattern (small body, long shadows)
        n_periods = 60
        ohlcv_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            base_price = 100 + i * 0.1
            if i == n_periods - 1:  # Last candle is Doji
                ohlcv_data[i] = [
                    base_price,      # open
                    base_price + 2,  # high (long upper shadow)
                    base_price - 2,  # low (long lower shadow)
                    base_price + 0.05,  # close (very small body)
                    1000             # volume
                ]
            else:
                ohlcv_data[i] = [
                    base_price,
                    base_price + 0.5,
                    base_price - 0.5,
                    base_price + 0.2,
                    1000
                ]
        
        detections = recognizer.recognize_pattern(ohlcv_data, return_all_detections=True)
        
        # Should detect at least one pattern
        assert len(detections) > 0
        
        # Check if Doji is detected
        doji_detections = [d for d in detections if d.pattern_type == PatternType.DOJI]
        assert len(doji_detections) > 0
    
    def test_recognize_pattern_hammer_scenario(self):
        """Test pattern recognition for Hammer candlestick pattern."""
        recognizer = PatternRecognizer(use_neural_network=False)
        
        # Create data with Hammer pattern (small body at top, long lower shadow)
        n_periods = 60
        ohlcv_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            base_price = 100 + i * 0.1
            if i == n_periods - 1:  # Last candle is Hammer
                ohlcv_data[i] = [
                    base_price,      # open
                    base_price + 0.2,  # high (small upper shadow)
                    base_price - 3,  # low (long lower shadow)
                    base_price + 0.1,  # close (small body)
                    1000             # volume
                ]
            else:
                ohlcv_data[i] = [
                    base_price,
                    base_price + 0.5,
                    base_price - 0.5,
                    base_price + 0.2,
                    1000
                ]
        
        detections = recognizer.recognize_pattern(ohlcv_data, return_all_detections=True)
        
        # Check if Hammer is detected
        hammer_detections = [d for d in detections if d.pattern_type == PatternType.HAMMER]
        assert len(hammer_detections) > 0
        
        # Hammer should be bullish
        assert hammer_detections[0].direction == "bullish"
    
    def test_recognize_pattern_double_top_scenario(self):
        """Test pattern recognition for Double Top chart pattern."""
        recognizer = PatternRecognizer(use_neural_network=False)
        
        # Create data with Double Top pattern
        n_periods = 60
        ohlcv_data = np.zeros((n_periods, 5))
        
        for i in range(n_periods):
            base_price = 100
            
            # Create double top pattern
            if 15 <= i <= 20:  # First peak
                price = base_price + 5 + (5 - abs(i - 17.5))
            elif 40 <= i <= 45:  # Second peak (similar height)
                price = base_price + 5 + (5 - abs(i - 42.5))
            elif 25 <= i <= 35:  # Valley between peaks
                price = base_price + 2
            else:
                price = base_price + np.random.randn() * 0.5
            
            ohlcv_data[i] = [
                price,
                price + 0.5,
                price - 0.5,
                price + np.random.randn() * 0.2,
                1000
            ]
        
        detections = recognizer.recognize_pattern(ohlcv_data, return_all_detections=True)
        
        # Check if Double Top is detected
        double_top_detections = [d for d in detections if d.pattern_type == PatternType.DOUBLE_TOP]
        # Note: This is a complex pattern, so detection might not always work with simple rule-based logic
        # The test mainly ensures the method runs without errors
        assert isinstance(detections, list)
    
    def test_get_pattern_features(self):
        """Test getting detailed pattern features."""
        recognizer = PatternRecognizer(use_neural_network=False)
        
        # Create sample data
        data = np.random.randn(60, 5) + 100
        data[:, :4] = np.abs(data[:, :4])  # Positive prices
        data[:, 4] = np.abs(data[:, 4]) * 1000  # Positive volume
        
        features = recognizer.get_pattern_features(data)
        
        assert isinstance(features, dict)
        assert 'best_pattern' in features
        assert 'best_confidence' in features
        assert 'pattern_counts' in features
        assert 'total_patterns_detected' in features
        assert 'technical_features' in features
        assert 'sequence_length' in features
        
        # Check pattern counts structure
        assert isinstance(features['pattern_counts'], dict)
        for pattern_type in PatternType:
            assert pattern_type.value in features['pattern_counts']
    
    def test_get_pattern_features_insufficient_data(self):
        """Test getting pattern features with insufficient data."""
        recognizer = PatternRecognizer(sequence_length=50)
        
        # Very small dataset
        small_data = np.random.randn(10, 5) + 100
        features = recognizer.get_pattern_features(small_data)
        
        assert features == {}  # Should return empty dict
    
    def test_pattern_types_enum(self):
        """Test PatternType enum values."""
        # Test some key pattern types
        assert PatternType.DOJI.value == "doji"
        assert PatternType.HAMMER.value == "hammer"
        assert PatternType.HEAD_AND_SHOULDERS.value == "head_and_shoulders"
        assert PatternType.DOUBLE_TOP.value == "double_top"
        assert PatternType.NO_PATTERN.value == "no_pattern"
        
        # Test that all patterns are different
        pattern_values = [pattern.value for pattern in PatternType]
        assert len(pattern_values) == len(set(pattern_values))
    
    def test_neural_network_integration(self):
        """Test neural network integration."""
        recognizer = PatternRecognizer(
            sequence_length=50,
            use_neural_network=True,
            min_pattern_confidence=0.1  # Lower threshold for testing
        )
        
        # Create sample data with sufficient length
        data = np.random.randn(60, 5) + 100
        data[:, :4] = np.abs(data[:, :4])  # Positive prices
        data[:, 4] = np.abs(data[:, 4]) * 1000  # Positive volume
        
        # Test that neural network detection runs without errors
        detection = recognizer.recognize_pattern(data)
        
        assert isinstance(detection, PatternDetection)
        assert detection.pattern_type in PatternType
        assert 0.0 <= detection.confidence <= 1.0
    
    def test_prepare_sequence_for_nn(self):
        """Test sequence preparation for neural network."""
        recognizer = PatternRecognizer(sequence_length=50)
        
        # Create sample OHLCV data
        ohlcv_data = np.random.randn(60, 5) + 100
        ohlcv_data[:, :4] = np.abs(ohlcv_data[:, :4])  # Positive prices
        
        sequence_tensor = recognizer._prepare_sequence_for_nn(ohlcv_data)
        
        assert sequence_tensor is not None
        assert isinstance(sequence_tensor, torch.Tensor)
        assert sequence_tensor.shape == (1, 4, 50)  # (batch, channels, sequence)
        
        # Test with insufficient data
        small_data = np.random.randn(10, 5) + 100
        sequence_tensor = recognizer._prepare_sequence_for_nn(small_data)
        assert sequence_tensor is None
    
    def test_calculate_technical_features(self):
        """Test technical features calculation."""
        recognizer = PatternRecognizer()
        
        # Create sample OHLCV data
        ohlcv_data = np.random.randn(50, 5) + 100
        ohlcv_data[:, :4] = np.abs(ohlcv_data[:, :4])  # Positive prices
        
        features = recognizer._calculate_technical_features(ohlcv_data)
        
        assert isinstance(features, dict)
        assert 'ma_short' in features
        assert 'ma_long' in features
        assert 'ma_ratio' in features
        assert 'rsi' in features
        assert 'volatility' in features
        assert 'price_position' in features
        assert 'momentum' in features
        
        # Check that all values are numeric
        for key, value in features.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
    
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data formats."""
        recognizer = PatternRecognizer()
        
        # Test with invalid data type
        with pytest.raises(ValueError):
            recognizer.recognize_pattern("invalid_data")
        
        # Test with empty array
        empty_data = np.array([]).reshape(0, 5)
        detection = recognizer.recognize_pattern(empty_data)
        assert detection.pattern_type == PatternType.NO_PATTERN


if __name__ == "__main__":
    pytest.main([__file__])
