"""
Machine Learning Predictor for Algo Trading
============================================

Uses Random Forest to predict:
1. Price direction (up/down)
2. Volatility (high-low range)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class TradingPredictor:
    """Random Forest based trading predictor"""

    def __init__(self):
        self.direction_model = None
        self.volatility_model = None
        self.features = None
        self.is_trained = False

    def prepare_features_and_targets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and target variables"""
        
        # Create target variables
        # 1. Direction: (next close - current close) / current close
        df['direction'] = df['close'].pct_change().shift(-1)
        
        # 2. Volatility: (next high - next low) / next close
        df['volatility'] = ((df['high'].shift(-1) - df['low'].shift(-1)) / df['close'].shift(-1)) * 100
        
        # Convert direction to 3-state classification
        # Calculate 33rd and 66th percentiles for direction
        direction_threshold_low = df['direction'].quantile(0.33)
        direction_threshold_high = df['direction'].quantile(0.67)
        
        # Create 3-state direction: -1 (strong down), 0 (neutral), 1 (strong up)
        df['direction_3state'] = pd.cut(df['direction'], 
                                       bins=[-np.inf, direction_threshold_low, direction_threshold_high, np.inf],
                                       labels=[-1, 0, 1])
                                       
        # Convert volatility to 3-state classification
        # Calculate 33rd and 66th percentiles for volatility
        volatility_threshold_low = df['volatility'].quantile(0.33)
        volatility_threshold_high = df['volatility'].quantile(0.67)
        
        # Create 3-state volatility: 0 (low), 1 (normal), 2 (high)
        df['volatility_3state'] = pd.cut(df['volatility'], 
                                        bins=[-np.inf, volatility_threshold_low, volatility_threshold_high, np.inf],
                                        labels=[0, 1, 2])
        
        # Remove rows with NaN targets (including categorical NaNs)
        df = df.dropna(subset=['direction_3state', 'volatility_3state'])
        
        # Now convert to int after removing NaNs
        df['direction_3state'] = df['direction_3state'].astype(int)
        df['volatility_3state'] = df['volatility_3state'].astype(int)
        
        # Select features (exclude target variables AND raw price columns)
        exclude_cols = ['direction', 'volatility', 'direction_3state', 'volatility_3state',
                       'volume', 'open', 'high', 'low', 'close', 'atr',  # Exclude raw prices and raw ATR
                       'datetime', 'datetime_readable', 'datetime_epoch', 'Unnamed: 0']
        self.features = [col for col in df.columns if col not in exclude_cols]
        
        X = df[self.features]
        y_direction = df['direction_3state']
        y_volatility = df['volatility_3state']
        
        logger.info(f"Prepared {len(self.features)} features for {len(X)} samples")
        return X, y_direction, y_volatility

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the Random Forest models"""
        
        logger.info("Starting training...")
        
        # Prepare data
        X, y_direction, y_volatility = self.prepare_features_and_targets(df)
        
        # Split data (use time-series split, shuffle=False)
        X_train, X_test, y_dir_train, y_dir_test = train_test_split(
            X, y_direction, test_size=0.2, random_state=42, shuffle=False
        )
        _, _, y_vol_train, y_vol_test = train_test_split(
            X, y_volatility, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Train direction model (classifier)
        logger.info("Training direction classification model...")
        self.direction_model = RandomForestClassifier(
            n_estimators=200,      # Increased from 100
            max_depth=12,          # Increased depth slightly
            min_samples_leaf=4,    # Prevent overfitting
            class_weight='balanced', # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        self.direction_model.fit(X_train, y_dir_train)
        
        # Train volatility model (classifier)
        logger.info("Training volatility classification model...")
        self.volatility_model = RandomForestClassifier(
            n_estimators=200,      # Increased from 100
            max_depth=12,          # Increased depth slightly
            min_samples_leaf=4,    # Prevent overfitting
            class_weight='balanced', # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        self.volatility_model.fit(X_train, y_vol_train)
        
        # Evaluate models
        dir_pred = self.direction_model.predict(X_test)
        vol_pred = self.volatility_model.predict(X_test)
        
        # Classification metrics
        dir_accuracy = accuracy_score(y_dir_test, dir_pred)
        vol_accuracy = accuracy_score(y_vol_test, vol_pred)
        
        logger.info(f"Direction model - Accuracy: {dir_accuracy:.3f}")
        logger.info(f"Volatility model - Accuracy: {vol_accuracy:.3f}")
        
        # Log Feature Importance
        self._log_feature_importance(self.direction_model, "Direction")
        self._log_feature_importance(self.volatility_model, "Volatility")
        
        self.is_trained = True
        
        return {
            'direction_metrics': {'accuracy': dir_accuracy},
            'volatility_metrics': {'accuracy': vol_accuracy}
        }
        
    def _log_feature_importance(self, model, model_name: str):
        """Log top feature importances"""
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            logger.info(f"--- Top 10 Features for {model_name} Model ---")
            for i in range(min(10, len(self.features))):
                logger.info(f"{i+1}. {self.features[indices[i]]}: {importances[indices[i]]:.4f}")
        except Exception as e:
            logger.warning(f"Could not log feature importance: {e}")

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions on new data"""
        
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
            
        # Get features from data columns if not set
        if self.features is None:
            exclude_cols = ['direction', 'volatility', 'direction_3state', 'volatility_3state',
                           'volume', 'open', 'high', 'low', 'close', 'atr',
                           'datetime', 'datetime_readable', 'datetime_epoch', 'Unnamed: 0']
            self.features = [col for col in df.columns if col not in exclude_cols]
            
        # Prepare features
        # Check if all features exist
        missing_features = [f for f in self.features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features in input data: {missing_features}")
            # Add missing features with 0s to avoid crash (though ideally should handle better)
            for f in missing_features:
                df[f] = 0
        
        X = df[self.features].copy()
        
        # Make predictions
        direction_pred = self.direction_model.predict(X)
        volatility_pred = self.volatility_model.predict(X)
        
        # Get prediction probabilities
        direction_proba = self.direction_model.predict_proba(X)
        volatility_proba = self.volatility_model.predict_proba(X)
        
        return {
            'direction_raw': direction_pred,
            'direction_signal': direction_pred,
            'direction_proba': direction_proba,
            'volatility': volatility_pred,
            'volatility_proba': volatility_proba
        }

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict for the most recent data point"""

        if len(df) == 0:
            raise ValueError("Empty dataframe provided")

        # Get only the last row
        latest = df.tail(1)

        # Make prediction
        predictions = self.predict(latest)

        # Get probability for the predicted class
        dir_class_idx = int(predictions['direction_signal'][0])
        vol_class_idx = int(predictions['volatility'][0])

        # Map indices to probability arrays
        dir_probabilities = predictions['direction_proba'][0]
        vol_probabilities = predictions['volatility_proba'][0]

        # Convert -1,0,1 to 0,1,2 for indexing
        dir_prob_idx = dir_class_idx + 1 if dir_class_idx != -1 else 0

        return {
            'direction_prediction': int(predictions['direction_signal'][0]),  # -1, 0, or 1
            'direction_probability': float(dir_probabilities[dir_prob_idx]),
            'volatility_prediction': int(predictions['volatility'][0]),  # 0, 1, or 2
            'volatility_probability': float(vol_probabilities[vol_class_idx]),
            'current_price': float(latest['close'].iloc[-1]),
            'prediction_confidence': self._get_confidence_score(latest)
        }

    def _get_confidence_score(self, df: pd.DataFrame) -> float:
        """Calculate prediction confidence based on probability average"""
        
        if not self.is_trained:
            return 0.0
            
        X = df[self.features].copy()
        
        # Get probabilities
        dir_proba = self.direction_model.predict_proba(X)
        vol_proba = self.volatility_model.predict_proba(X)
        
        # Calculate max probability for each prediction (confidence of the chosen class)
        dir_conf = np.max(dir_proba, axis=1)
        vol_conf = np.max(vol_proba, axis=1)
        
        # Overall confidence is the average of direction and volatility confidence
        confidence = np.mean([dir_conf, vol_conf], axis=0)
        
        # Return scalar if single row, else array
        if len(confidence) == 1:
            return float(confidence[0])
        return confidence

    def save_models(self, path_prefix: str):
        """Save trained models to disk (separate files for each model)"""

        if not self.is_trained:
            raise ValueError("Models not trained yet")

        # Save direction model
        direction_path = f"{path_prefix}_direction.pkl"
        with open(direction_path, 'wb') as f:
            pickle.dump(self.direction_model, f)
        logger.info(f"Direction model saved to {direction_path}")

        # Save volatility model
        volatility_path = f"{path_prefix}_volatility.pkl"
        with open(volatility_path, 'wb') as f:
            pickle.dump(self.volatility_model, f)
        logger.info(f"Volatility model saved to {volatility_path}")

    
    def load_models(self, path_prefix: str):
        """Load trained models from disk (separate files for each model)"""

        try:
            # Load direction model
            direction_path = f"{path_prefix}_direction.pkl"
            with open(direction_path, 'rb') as f:
                self.direction_model = pickle.load(f)

            # Load volatility model
            volatility_path = f"{path_prefix}_volatility.pkl"
            with open(volatility_path, 'rb') as f:
                self.volatility_model = pickle.load(f)

            self.is_trained = True
            self.features = None  # Will be set when predicting from data columns
            logger.info(f"Models loaded from {path_prefix}")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise


def train_and_evaluate(csv_path: str, save_path: str = None) -> TradingPredictor:
    """
    Train the predictor and return the trained model

    Args:
        csv_path: Path to the processed data CSV
        save_path: Optional path to save the trained models

    Returns:
        Trained TradingPredictor instance
    """

    # Load data
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    if 'datetime_readable' in df.columns:
        df['datetime_readable'] = pd.to_datetime(df['datetime_readable'])
        df.set_index('datetime_readable', inplace=True)

    # Initialize and train predictor
    predictor = TradingPredictor()
    metrics = predictor.train(df)

    # Save models if path provided
    if save_path:
        predictor.save_models(save_path)

    return predictor