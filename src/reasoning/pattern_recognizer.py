"""
Chartist Pattern Recognizer

This module provides technical analysis pattern recognition capabilities,
allowing the autonomous agent to identify classic chart patterns and
candlestick formations from price data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Any
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Enumeration of recognizable chart patterns."""
    # Candlestick patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    
    # Chart patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    
    # No pattern detected
    NO_PATTERN = "no_pattern"


@dataclass
class PatternDetection:
    """
    Represents a detected pattern with metadata.
    """
    pattern_type: PatternType
    confidence: float
    start_index: int
    end_index: int
    strength: float = 0.0
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    
    def __post_init__(self):
        """Validate pattern detection after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.start_index > self.end_index:
            raise ValueError("Start index must be <= end index")


class Pattern1DCNN(nn.Module):
    """
    1D CNN for pattern recognition in price sequences.
    """
    
    def __init__(
        self,
        input_channels: int = 4,  # OHLC
        sequence_length: int = 50,
        num_patterns: int = len(PatternType),
        hidden_dim: int = 64
    ):
        """
        Initialize the 1D CNN for pattern recognition.
        
        Args:
            input_channels: Number of input channels (OHLC = 4)
            sequence_length: Length of input sequences
            num_patterns: Number of pattern types to classify
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_patterns = num_patterns
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

        # Activation functions (define before calculating conv output size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        # Calculate the size after convolutions and pooling
        conv_output_size = self._calculate_conv_output_size()

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_pattern = nn.Linear(hidden_dim // 2, num_patterns)
        self.fc_confidence = nn.Linear(hidden_dim // 2, 1)
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolutions and pooling."""
        # Simulate forward pass to get output size
        x = torch.randn(1, self.input_channels, self.sequence_length)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.numel()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, sequence_length)
            
        Returns:
            Dictionary containing pattern probabilities and confidence
        """
        # Convolutional layers with pooling
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        features = self.relu(self.fc2(x))
        
        # Output heads
        pattern_logits = self.fc_pattern(features)
        pattern_probs = self.softmax(pattern_logits)
        confidence = self.sigmoid(self.fc_confidence(features))
        
        return {
            'pattern_probabilities': pattern_probs,
            'confidence': confidence,
            'features': features
        }


class PatternRecognizer:
    """
    Chartist Pattern Recognizer for identifying technical analysis patterns.
    
    This class combines rule-based pattern detection with neural network
    approaches to identify classic chart patterns and candlestick formations.
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        min_pattern_confidence: float = 0.6,
        use_neural_network: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize the Pattern Recognizer.
        
        Args:
            sequence_length: Length of price sequences to analyze
            min_pattern_confidence: Minimum confidence threshold for pattern detection
            use_neural_network: Whether to use neural network for pattern recognition
            device: Device to run neural network on
        """
        self.sequence_length = sequence_length
        self.min_pattern_confidence = min_pattern_confidence
        self.use_neural_network = use_neural_network
        self.device = device
        
        # Initialize neural network if requested
        if self.use_neural_network:
            self.cnn_model = Pattern1DCNN(
                sequence_length=sequence_length,
                num_patterns=len(PatternType)
            ).to(device)
            self.cnn_model.eval()  # Set to evaluation mode
        else:
            self.cnn_model = None
        
        # Pattern type mapping for neural network output
        self.pattern_types = list(PatternType)
        
        logger.info(f"Initialized PatternRecognizer with sequence_length={sequence_length}, "
                   f"neural_network={use_neural_network}")
    
    def recognize_pattern(
        self,
        price_data: Union[pd.DataFrame, np.ndarray, Dict],
        return_all_detections: bool = False
    ) -> Union[PatternDetection, List[PatternDetection]]:
        """
        Recognize patterns in price data.
        
        Args:
            price_data: Price data containing OHLCV information
            return_all_detections: Whether to return all detected patterns or just the best one
            
        Returns:
            PatternDetection object or list of detections
        """
        # Convert input to standardized format
        ohlcv_data = self._prepare_price_data(price_data)
        
        if len(ohlcv_data) < self.sequence_length:
            logger.warning("Insufficient data for pattern recognition")
            no_pattern = PatternDetection(
                pattern_type=PatternType.NO_PATTERN,
                confidence=1.0,
                start_index=0,
                end_index=len(ohlcv_data) - 1 if len(ohlcv_data) > 0 else 0
            )
            return [no_pattern] if return_all_detections else no_pattern
        
        detections = []
        
        # Rule-based pattern detection
        rule_based_detections = self._detect_patterns_rule_based(ohlcv_data)
        detections.extend(rule_based_detections)
        
        # Neural network pattern detection
        if self.use_neural_network and self.cnn_model is not None:
            nn_detections = self._detect_patterns_neural_network(ohlcv_data)
            detections.extend(nn_detections)
        
        # Filter by confidence threshold
        valid_detections = [
            detection for detection in detections 
            if detection.confidence >= self.min_pattern_confidence
        ]
        
        if not valid_detections:
            no_pattern = PatternDetection(
                pattern_type=PatternType.NO_PATTERN,
                confidence=1.0,
                start_index=0,
                end_index=len(ohlcv_data) - 1
            )
            return [no_pattern] if return_all_detections else no_pattern
        
        if return_all_detections:
            return sorted(valid_detections, key=lambda x: x.confidence, reverse=True)
        else:
            return max(valid_detections, key=lambda x: x.confidence)
    
    def get_pattern_features(
        self, 
        price_data: Union[pd.DataFrame, np.ndarray, Dict]
    ) -> Dict[str, Any]:
        """
        Get detailed pattern features for analysis.
        
        Args:
            price_data: Price data containing OHLCV information
            
        Returns:
            Dictionary of pattern features and statistics
        """
        ohlcv_data = self._prepare_price_data(price_data)
        
        if len(ohlcv_data) < self.sequence_length:
            return {}
        
        # Get all pattern detections
        detections = self.recognize_pattern(price_data, return_all_detections=True)
        
        # Calculate pattern statistics
        pattern_counts = {}
        for pattern_type in PatternType:
            pattern_counts[pattern_type.value] = sum(
                1 for d in detections if d.pattern_type == pattern_type
            )
        
        # Get best detection
        best_detection = detections[0] if detections else None
        
        # Calculate technical indicators for pattern context
        technical_features = self._calculate_technical_features(ohlcv_data)
        
        return {
            'best_pattern': best_detection.pattern_type.value if best_detection else 'no_pattern',
            'best_confidence': best_detection.confidence if best_detection else 0.0,
            'pattern_counts': pattern_counts,
            'total_patterns_detected': len([d for d in detections if d.pattern_type != PatternType.NO_PATTERN]),
            'technical_features': technical_features,
            'sequence_length': len(ohlcv_data)
        }
    
    def _prepare_price_data(self, price_data: Union[pd.DataFrame, np.ndarray, Dict]) -> np.ndarray:
        """
        Convert input data to standardized OHLCV numpy array.
        
        Args:
            price_data: Input price data in various formats
            
        Returns:
            Numpy array with shape (n_periods, 5) for OHLCV
        """
        if isinstance(price_data, pd.DataFrame):
            # Assume DataFrame has OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in price_data.columns.str.lower()]
            
            if len(available_cols) >= 4:  # At least OHLC
                data = price_data[available_cols].values
                if data.shape[1] == 4:  # Add dummy volume if missing
                    volume = np.ones((data.shape[0], 1))
                    data = np.hstack([data, volume])
                return data
            else:
                raise ValueError("DataFrame must contain OHLC columns")
        
        elif isinstance(price_data, np.ndarray):
            if price_data.ndim == 1:
                # Single price series, create OHLC from it
                prices = price_data
                ohlc = np.column_stack([prices, prices, prices, prices])
                volume = np.ones((len(prices), 1))
                return np.hstack([ohlc, volume])
            elif price_data.ndim == 2 and price_data.shape[1] >= 4:
                # Already in OHLCV format
                if price_data.shape[1] == 4:  # Add dummy volume
                    volume = np.ones((price_data.shape[0], 1))
                    return np.hstack([price_data, volume])
                return price_data[:, :5]  # Take first 5 columns
            else:
                raise ValueError("Array must have shape (n, 4) or (n, 5) for OHLCV")
        
        elif isinstance(price_data, dict):
            # Extract OHLCV from dictionary
            try:
                open_prices = np.array(price_data.get('open', price_data.get('Open', [])))
                high_prices = np.array(price_data.get('high', price_data.get('High', [])))
                low_prices = np.array(price_data.get('low', price_data.get('Low', [])))
                close_prices = np.array(price_data.get('close', price_data.get('Close', [])))
                volume = np.array(price_data.get('volume', price_data.get('Volume', 
                                                np.ones(len(close_prices)))))
                
                return np.column_stack([open_prices, high_prices, low_prices, close_prices, volume])
            except Exception as e:
                raise ValueError(f"Could not extract OHLCV from dictionary: {e}")
        
        else:
            raise ValueError(f"Unsupported data type: {type(price_data)}")
    
    def _detect_patterns_rule_based(self, ohlcv_data: np.ndarray) -> List[PatternDetection]:
        """
        Detect patterns using rule-based logic.
        
        Args:
            ohlcv_data: OHLCV data array
            
        Returns:
            List of detected patterns
        """
        detections = []
        
        open_prices = ohlcv_data[:, 0]
        high_prices = ohlcv_data[:, 1]
        low_prices = ohlcv_data[:, 2]
        close_prices = ohlcv_data[:, 3]
        
        # Detect candlestick patterns
        candlestick_detections = self._detect_candlestick_patterns(
            open_prices, high_prices, low_prices, close_prices
        )
        detections.extend(candlestick_detections)
        
        # Detect chart patterns
        chart_detections = self._detect_chart_patterns(
            open_prices, high_prices, low_prices, close_prices
        )
        detections.extend(chart_detections)
        
        return detections

    def _detect_candlestick_patterns(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray
    ) -> List[PatternDetection]:
        """Detect candlestick patterns using rule-based logic."""
        detections = []

        for i in range(1, len(close_prices)):
            # Calculate candlestick properties
            body_size = abs(close_prices[i] - open_prices[i])
            upper_shadow = high_prices[i] - max(open_prices[i], close_prices[i])
            lower_shadow = min(open_prices[i], close_prices[i]) - low_prices[i]
            total_range = high_prices[i] - low_prices[i]

            if total_range == 0:
                continue

            # Doji pattern (small body, long shadows)
            if body_size / total_range < 0.1 and (upper_shadow + lower_shadow) / total_range > 0.7:
                detections.append(PatternDetection(
                    pattern_type=PatternType.DOJI,
                    confidence=0.8,
                    start_index=i,
                    end_index=i,
                    direction="neutral"
                ))

            # Hammer pattern (small body at top, long lower shadow)
            elif (body_size / total_range < 0.3 and
                  lower_shadow / total_range > 0.6 and
                  upper_shadow / total_range < 0.1):
                detections.append(PatternDetection(
                    pattern_type=PatternType.HAMMER,
                    confidence=0.75,
                    start_index=i,
                    end_index=i,
                    direction="bullish"
                ))

            # Shooting star pattern (small body at bottom, long upper shadow)
            elif (body_size / total_range < 0.3 and
                  upper_shadow / total_range > 0.6 and
                  lower_shadow / total_range < 0.1):
                detections.append(PatternDetection(
                    pattern_type=PatternType.SHOOTING_STAR,
                    confidence=0.75,
                    start_index=i,
                    end_index=i,
                    direction="bearish"
                ))

            # Engulfing patterns (requires previous candle)
            if i >= 1:
                prev_body = abs(close_prices[i-1] - open_prices[i-1])
                curr_body = abs(close_prices[i] - open_prices[i])

                # Bullish engulfing
                if (close_prices[i-1] < open_prices[i-1] and  # Previous red candle
                    close_prices[i] > open_prices[i] and      # Current green candle
                    open_prices[i] < close_prices[i-1] and    # Opens below previous close
                    close_prices[i] > open_prices[i-1] and    # Closes above previous open
                    curr_body > prev_body * 1.2):             # Significantly larger body

                    detections.append(PatternDetection(
                        pattern_type=PatternType.ENGULFING_BULLISH,
                        confidence=0.8,
                        start_index=i-1,
                        end_index=i,
                        direction="bullish"
                    ))

                # Bearish engulfing
                elif (close_prices[i-1] > open_prices[i-1] and  # Previous green candle
                      close_prices[i] < open_prices[i] and      # Current red candle
                      open_prices[i] > close_prices[i-1] and    # Opens above previous close
                      close_prices[i] < open_prices[i-1] and    # Closes below previous open
                      curr_body > prev_body * 1.2):             # Significantly larger body

                    detections.append(PatternDetection(
                        pattern_type=PatternType.ENGULFING_BEARISH,
                        confidence=0.8,
                        start_index=i-1,
                        end_index=i,
                        direction="bearish"
                    ))

        return detections

    def _detect_chart_patterns(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray
    ) -> List[PatternDetection]:
        """Detect chart patterns using rule-based logic."""
        detections = []

        # Use a sliding window approach for pattern detection
        window_size = min(20, len(close_prices) // 3)

        for i in range(window_size, len(close_prices) - window_size):
            window_high = high_prices[i-window_size:i+window_size]
            window_low = low_prices[i-window_size:i+window_size]
            window_close = close_prices[i-window_size:i+window_size]

            # Double top pattern
            if self._is_double_top(window_high, window_close):
                detections.append(PatternDetection(
                    pattern_type=PatternType.DOUBLE_TOP,
                    confidence=0.7,
                    start_index=i-window_size,
                    end_index=i+window_size-1,
                    direction="bearish"
                ))

            # Double bottom pattern
            elif self._is_double_bottom(window_low, window_close):
                detections.append(PatternDetection(
                    pattern_type=PatternType.DOUBLE_BOTTOM,
                    confidence=0.7,
                    start_index=i-window_size,
                    end_index=i+window_size-1,
                    direction="bullish"
                ))

            # Head and shoulders pattern
            elif self._is_head_and_shoulders(window_high, window_close):
                detections.append(PatternDetection(
                    pattern_type=PatternType.HEAD_AND_SHOULDERS,
                    confidence=0.75,
                    start_index=i-window_size,
                    end_index=i+window_size-1,
                    direction="bearish"
                ))

        return detections

    def _is_double_top(self, highs: np.ndarray, closes: np.ndarray) -> bool:
        """Check if the pattern resembles a double top."""
        if len(highs) < 10:
            return False

        # Find peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))

        if len(peaks) < 2:
            return False

        # Check if two highest peaks are similar in height
        peaks.sort(key=lambda x: x[1], reverse=True)
        peak1, peak2 = peaks[0], peaks[1]

        height_diff = abs(peak1[1] - peak2[1]) / max(peak1[1], peak2[1])
        return height_diff < 0.02  # Within 2% of each other

    def _is_double_bottom(self, lows: np.ndarray, closes: np.ndarray) -> bool:
        """Check if the pattern resembles a double bottom."""
        if len(lows) < 10:
            return False

        # Find troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))

        if len(troughs) < 2:
            return False

        # Check if two lowest troughs are similar in depth
        troughs.sort(key=lambda x: x[1])
        trough1, trough2 = troughs[0], troughs[1]

        depth_diff = abs(trough1[1] - trough2[1]) / max(trough1[1], trough2[1])
        return depth_diff < 0.02  # Within 2% of each other

    def _is_head_and_shoulders(self, highs: np.ndarray, closes: np.ndarray) -> bool:
        """Check if the pattern resembles a head and shoulders."""
        if len(highs) < 15:
            return False

        # Find peaks
        peaks = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i+1] and
                highs[i] > highs[i-2] and highs[i] > highs[i+2]):
                peaks.append((i, highs[i]))

        if len(peaks) < 3:
            return False

        # Sort peaks by height and check for head and shoulders pattern
        peaks.sort(key=lambda x: x[1], reverse=True)
        head = peaks[0]
        shoulders = peaks[1:3]

        # Check if shoulders are similar height and lower than head
        shoulder_height_diff = abs(shoulders[0][1] - shoulders[1][1]) / max(shoulders[0][1], shoulders[1][1])
        head_shoulder_ratio = head[1] / max(shoulders[0][1], shoulders[1][1])

        return shoulder_height_diff < 0.05 and 1.1 < head_shoulder_ratio < 1.5

    def _detect_patterns_neural_network(self, ohlcv_data: np.ndarray) -> List[PatternDetection]:
        """
        Detect patterns using neural network.

        Args:
            ohlcv_data: OHLCV data array

        Returns:
            List of detected patterns
        """
        if self.cnn_model is None:
            return []

        detections = []

        # Prepare data for neural network
        sequence_data = self._prepare_sequence_for_nn(ohlcv_data)

        if sequence_data is None:
            return detections

        with torch.no_grad():
            # Run inference
            output = self.cnn_model(sequence_data)
            pattern_probs = output['pattern_probabilities'].cpu().numpy()[0]
            confidence = output['confidence'].cpu().numpy()[0][0]

            # Find the most likely pattern
            best_pattern_idx = np.argmax(pattern_probs)
            best_pattern_prob = pattern_probs[best_pattern_idx]

            if best_pattern_prob > 0.5 and confidence > self.min_pattern_confidence:
                pattern_type = self.pattern_types[best_pattern_idx]

                # Determine direction based on pattern type
                direction = "neutral"
                if pattern_type.value in ['hammer', 'engulfing_bullish', 'morning_star', 'double_bottom', 'inverse_head_and_shoulders']:
                    direction = "bullish"
                elif pattern_type.value in ['shooting_star', 'engulfing_bearish', 'evening_star', 'double_top', 'head_and_shoulders']:
                    direction = "bearish"

                detections.append(PatternDetection(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    start_index=len(ohlcv_data) - self.sequence_length,
                    end_index=len(ohlcv_data) - 1,
                    direction=direction,
                    strength=best_pattern_prob
                ))

        return detections

    def _prepare_sequence_for_nn(self, ohlcv_data: np.ndarray) -> Optional[torch.Tensor]:
        """
        Prepare price sequence for neural network input.

        Args:
            ohlcv_data: OHLCV data array

        Returns:
            Tensor ready for neural network input or None if insufficient data
        """
        if len(ohlcv_data) < self.sequence_length:
            return None

        # Take the last sequence_length periods
        sequence = ohlcv_data[-self.sequence_length:, :4]  # Only OHLC

        # Normalize the data (relative to first price)
        base_price = sequence[0, 3]  # First close price
        if base_price == 0:
            return None

        normalized_sequence = sequence / base_price

        # Convert to tensor and reshape for CNN (batch_size, channels, sequence_length)
        tensor_sequence = torch.FloatTensor(normalized_sequence.T).unsqueeze(0).to(self.device)

        return tensor_sequence

    def _calculate_technical_features(self, ohlcv_data: np.ndarray) -> Dict[str, float]:
        """
        Calculate technical indicators for pattern context.

        Args:
            ohlcv_data: OHLCV data array

        Returns:
            Dictionary of technical features
        """
        if len(ohlcv_data) < 20:
            return {}

        close_prices = ohlcv_data[:, 3]
        high_prices = ohlcv_data[:, 1]
        low_prices = ohlcv_data[:, 2]

        # Moving averages
        ma_short = np.mean(close_prices[-10:])
        ma_long = np.mean(close_prices[-20:])

        # RSI calculation (simplified)
        price_changes = np.diff(close_prices[-15:])
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        rsi = 100 - (100 / (1 + (avg_gain / (avg_loss + 1e-10))))

        # Volatility
        price_changes = np.diff(close_prices[-20:])
        if len(price_changes) > 0 and len(close_prices) >= 21:
            volatility = np.std(price_changes / close_prices[-21:-1])
        else:
            volatility = 0.0

        # Price position in recent range
        recent_high = np.max(high_prices[-20:])
        recent_low = np.min(low_prices[-20:])
        current_price = close_prices[-1]
        price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'ma_ratio': ma_short / ma_long if ma_long != 0 else 1.0,
            'rsi': rsi,
            'volatility': volatility,
            'price_position': price_position,
            'momentum': (close_prices[-1] - close_prices[-10]) / close_prices[-10] if close_prices[-10] != 0 else 0.0
        }
