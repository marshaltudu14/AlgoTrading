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

        # Select features (exclude target variables and OHLCV for now)
        exclude_cols = ['direction', 'volatility', 'direction_3state', 'volatility_3state',
                       'open', 'high', 'low', 'close', 'volume',
                       'datetime', 'datetime_readable', 'datetime_epoch', 'Unnamed: 0']
        self.features = [col for col in df.columns if col not in exclude_cols]

        X = df[self.features]
        y_direction = df['direction_3state']
        y_volatility = df['volatility_3state']

        logger.info(f"Prepared {len(self.features)} features for {len(X)} samples")
        logger.info(f"Direction distribution: -1: {len(df[df['direction_3state'] == -1])}, "
                   f"0: {len(df[df['direction_3state'] == 0])}, "
                   f"1: {len(df[df['direction_3state'] == 1])}")
        logger.info(f"Volatility distribution: 0: {len(df[df['volatility_3state'] == 0])}, "
                   f"1: {len(df[df['volatility_3state'] == 1])}, "
                   f"2: {len(df[df['volatility_3state'] == 2])}")

        return X, y_direction, y_volatility

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the Random Forest models"""

        logger.info("Starting training...")

        # Prepare data
        X, y_direction, y_volatility = self.prepare_features_and_targets(df)

        # Split data
        X_train, X_test, y_dir_train, y_dir_test = train_test_split(
            X, y_direction, test_size=0.2, random_state=42, shuffle=False
        )
        _, _, y_vol_train, y_vol_test = train_test_split(
            X, y_volatility, test_size=0.2, random_state=42, shuffle=False
        )

        # Train direction model (classifier)
        logger.info("Training direction classification model...")
        self.direction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.direction_model.fit(X_train, y_dir_train)

        # Train volatility model (classifier)
        logger.info("Training volatility classification model...")
        self.volatility_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
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

        dir_metrics = {
            'accuracy': dir_accuracy,
            'classification_report': classification_report(y_dir_test, dir_pred, output_dict=True)
        }

        vol_metrics = {
            'accuracy': vol_accuracy,
            'classification_report': classification_report(y_vol_test, vol_pred, output_dict=True)
        }

        logger.info(f"Direction model - Accuracy: {dir_accuracy:.3f}")
        logger.info(f"Volatility model - Accuracy: {vol_accuracy:.3f}")

        self.is_trained = True

        return {
            'direction_metrics': dir_metrics,
            'volatility_metrics': vol_metrics
        }

    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions on new data"""

        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")

        if self.features is None:
            raise ValueError("Features not defined. Train the model first.")

        # Prepare features
        X = df[self.features].copy()

        # Make predictions
        direction_pred = self.direction_model.predict(X)
        volatility_pred = self.volatility_model.predict(X)

        # Get prediction probabilities
        direction_proba = self.direction_model.predict_proba(X)
        volatility_proba = self.volatility_model.predict_proba(X)

        return {
            'direction_raw': direction_pred,  # -1, 0, 1
            'direction_signal': direction_pred,  # Keep original 3-state signal
            'direction_proba': direction_proba,  # Probabilities for each state
            'volatility': volatility_pred,  # 0, 1, 2
            'volatility_proba': volatility_proba  # Probabilities for each state
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
        """Calculate prediction confidence based on model's tree consensus"""

        if not self.is_trained:
            return 0.0

        X = df[self.features].copy()

        # Get predictions from all trees
        direction_trees = np.array([tree.predict(X) for tree in self.direction_model.estimators_])
        volatility_trees = np.array([tree.predict(X) for tree in self.volatility_model.estimators_])

        # Calculate standard deviation (inverse of consensus)
        dir_consensus = 1 - (np.std(direction_trees, axis=0) / np.mean(np.abs(direction_trees), axis=0))
        vol_consensus = 1 - (np.std(volatility_trees, axis=0) / np.mean(volatility_trees, axis=0))

        # Average consensus score
        confidence = np.mean([dir_consensus[0], vol_consensus[0]])

        return float(np.clip(confidence, 0, 1))

    def save_models(self, path: str):
        """Save trained models to disk"""

        if not self.is_trained:
            raise ValueError("Models not trained yet")

        model_data = {
            'direction_model': self.direction_model,
            'volatility_model': self.volatility_model,
            'features': self.features
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Models saved to {path}")

    def load_models(self, path: str):
        """Load trained models from disk"""

        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)

            self.direction_model = model_data['direction_model']
            self.volatility_model = model_data['volatility_model']
            self.features = model_data['features']
            self.is_trained = True

            logger.info(f"Models loaded from {path}")

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