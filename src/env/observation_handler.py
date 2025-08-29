import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class ObservationHandler:
    """Handles observation generation and normalization for the trading environment."""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.high_level_lookback = 128
        self.low_level_lookback = 50
        
        # Track statistics for z-score normalization
        self.feature_stats = {}
        self.observation_history = []
        
        # Observation space parameters
        self.features_per_step = None
        self.account_state_features = 5  # capital, position_quantity, position_entry_price, unrealized_pnl, is_position_open
        self.trailing_features = 1  # distance_to_trail
        self.observation_dim = None
        self.observation_space = None
        
    def initialize_observation_space(self, data: pd.DataFrame, config: Dict = None):
        """Initialize observation space dimensions based on actual data."""
        # CRITICAL: Exclude raw OHLC prices for universal model - only use derived features
        # NOTE: datetime_epoch is INCLUDED as a feature for temporal learning
        excluded_columns = ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in data.columns if col.lower() not in [x.lower() for x in excluded_columns]]
        self.features_per_step = len(feature_columns)
        
        # Get hierarchical processing configuration
        if config:
            hierarchical_processing_config = config.get('hierarchical_reasoning_model', {}).get('hierarchical_processing', {
                'high_level_lookback': 128,
                'low_level_lookback': 50,
                'high_level_features': 64,
                'low_level_features': 32
            })
            
            # Use hierarchical lookback configuration
            self.high_level_lookback = hierarchical_processing_config.get('high_level_lookback', 128)
            self.low_level_lookback = hierarchical_processing_config.get('low_level_lookback', 50)
            
            # For backward compatibility, use the larger lookback as the main lookback
            self.lookback_window = max(self.high_level_lookback, self.low_level_lookback)
        
        self.observation_dim = (self.lookback_window * self.features_per_step) + self.account_state_features + self.trailing_features
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)
        
        return feature_columns
    
    def reset(self):
        """Reset observation handler state."""
        self.feature_stats = {}
        self.observation_history = []
    
    def get_observation(self, data: pd.DataFrame, current_step: int, engine, 
                       current_price: float = None) -> np.ndarray:
        """Generate observation for the current state."""
        # Get market data for the lookback window
        start_index = current_step - self.lookback_window + 1
        end_index = current_step + 1

        # Use ALL available columns for RL training (except datetime if present)
        # NOTE: datetime_epoch is INCLUDED as a feature for temporal learning
        feature_columns = [col for col in data.columns if col.lower() not in ['datetime', 'date', 'time', 'timestamp']]

        # Ensure we don't go out of bounds
        if start_index < 0:
            # Pad with zeros if not enough history
            padding_needed = abs(start_index)
            market_data = np.zeros((self.lookback_window, len(feature_columns)))
            actual_data = data[feature_columns].iloc[0:end_index].values
            market_data[padding_needed:] = actual_data
        else:
            market_data = data[feature_columns].iloc[start_index:end_index].values

        # Get account state with bounds checking
        if current_price is None:
            safe_step = min(current_step, len(data) - 1)
            current_price = data['close'].iloc[safe_step]
        
        account_state = engine.get_account_state(current_price=current_price)

        # Calculate distance to trailing stop
        distance_to_trail = self._calculate_distance_to_trail(current_price, engine)

        # Create raw observation with proper validation
        try:
            # Ensure market_data is properly shaped
            market_features = market_data.flatten().astype(np.float32)

            # Create account state features
            account_features = np.array([
                float(account_state['capital']),
                float(account_state['current_position_quantity']),
                float(account_state['current_position_entry_price']),
                float(account_state['unrealized_pnl']),
                1.0 if account_state['is_position_open'] else 0.0,
                float(distance_to_trail)
            ], dtype=np.float32)

            # Concatenate all features
            raw_observation = np.concatenate([market_features, account_features])

            # Ensure no invalid values
            if not np.isfinite(raw_observation).all():
                # Replace invalid values with zeros
                raw_observation = np.where(np.isfinite(raw_observation), raw_observation, 0.0)

            # Ensure consistent shape
            if self.observation_dim is not None and len(raw_observation) != self.observation_dim:
                if len(raw_observation) < self.observation_dim:
                    # Pad with zeros
                    raw_observation = np.pad(raw_observation, (0, self.observation_dim - len(raw_observation)), 'constant')
                else:
                    # Truncate
                    raw_observation = raw_observation[:self.observation_dim]

            # Store raw observation for statistics
            self.observation_history.append(raw_observation.copy())

            # Apply selective z-score normalization (exclude datetime_epoch)
            normalized_observation = self._apply_selective_zscore_normalization(raw_observation, data)

            return normalized_observation.astype(np.float32)

        except Exception as e:
            print(f"Error creating observation: {e}")
            # Return a zero observation with correct shape
            if self.observation_dim is not None:
                return np.zeros(self.observation_dim, dtype=np.float32)
            else:
                return np.zeros(1246, dtype=np.float32)  # Default fallback

    def _apply_selective_zscore_normalization(self, observation: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """Apply z-score normalization to observation, excluding datetime_epoch features."""
        if len(self.observation_history) < 10:  # Need minimum history
            return observation  # Return raw observation initially

        try:
            # Identify datetime_epoch feature indices
            datetime_epoch_indices = self._get_datetime_epoch_indices(data)

            # Use recent history for statistics (last 100 observations)
            recent_observations = []
            target_shape = observation.shape

            for obs in self.observation_history[-100:]:
                if obs.shape == target_shape:
                    recent_observations.append(obs)

            if len(recent_observations) < 5:  # Need minimum valid history
                return observation

            recent_history = np.stack(recent_observations, axis=0)

            # Calculate mean and std for each feature
            means = np.mean(recent_history, axis=0)
            stds = np.std(recent_history, axis=0)

            # Avoid division by zero
            stds = np.where(stds == 0, 1.0, stds)

            # Apply z-score normalization
            normalized = (observation - means) / stds

            # CRITICAL: Keep datetime_epoch features unnormalized
            for idx in datetime_epoch_indices:
                if idx < len(normalized):
                    normalized[idx] = observation[idx]  # Keep original datetime_epoch value

            # Clip extreme values to prevent instability (except datetime_epoch)
            for i in range(len(normalized)):
                if i not in datetime_epoch_indices:
                    normalized[i] = np.clip(normalized[i], -5.0, 5.0)

            return normalized

        except Exception as e:
            print(f"Warning: Selective z-score normalization failed: {e}, returning raw observation")
            return observation

    def _get_datetime_epoch_indices(self, data: pd.DataFrame) -> List[int]:
        """Get indices of datetime_epoch features in the flattened observation."""
        try:
            # Find datetime_epoch column index in feature_columns
            excluded_columns = ['datetime', 'date', 'time', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in data.columns if col.lower() not in [x.lower() for x in excluded_columns]]

            datetime_epoch_indices = []
            for i, col in enumerate(feature_columns):
                if 'datetime_epoch' in col.lower():
                    # Calculate indices in flattened observation
                    # Each timestep contributes len(feature_columns) features
                    for t in range(self.lookback_window):
                        idx = t * len(feature_columns) + i
                        datetime_epoch_indices.append(idx)

            return datetime_epoch_indices
        except Exception as e:
            print(f"Warning: Could not identify datetime_epoch indices: {e}")
            return []

    def _calculate_distance_to_trail(self, current_price: float, engine) -> float:
        """Calculate normalized distance to trailing stop."""
        if not engine._is_position_open:
            return 0.0  # No position, no trailing stop

        trailing_stop_price = engine._trailing_stop_price
        if trailing_stop_price == 0:
            return 0.0  # No trailing stop set

        # Calculate distance as percentage of current price
        if engine._current_position_quantity > 0:  # Long position
            distance = (current_price - trailing_stop_price) / current_price
        else:  # Short position
            distance = (trailing_stop_price - current_price) / current_price

        # Normalize to reasonable range [-1, 1]
        return np.clip(distance, -1.0, 1.0)