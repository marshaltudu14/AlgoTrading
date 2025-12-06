"""
Machine Learning Predictor for Algo Trading
============================================

Uses Random Forest to predict:
1. Price direction (up/down)
2. Volatility (high-low range)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

        # Remove rows with NaN targets (last row will have NaN)
        df = df.dropna(subset=['direction', 'volatility'])

        # Select features (exclude target variables and OHLCV for now)
        exclude_cols = ['direction', 'volatility', 'open', 'high', 'low', 'close', 'volume',
                       'datetime', 'datetime_readable', 'datetime_epoch', 'Unnamed: 0']
        self.features = [col for col in df.columns if col not in exclude_cols]

        X = df[self.features]
        y_direction = df['direction']
        y_volatility = df['volatility']

        logger.info(f"Prepared {len(self.features)} features for {len(X)} samples")

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

        # Train direction model
        logger.info("Training direction prediction model...")
        self.direction_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.direction_model.fit(X_train, y_dir_train)

        # Train volatility model
        logger.info("Training volatility prediction model...")
        self.volatility_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.volatility_model.fit(X_train, y_vol_train)

        # Evaluate models
        dir_pred = self.direction_model.predict(X_test)
        vol_pred = self.volatility_model.predict(X_test)

        dir_metrics = {
            'mae': mean_absolute_error(y_dir_test, dir_pred),
            'mse': mean_squared_error(y_dir_test, dir_pred),
            'r2': r2_score(y_dir_test, dir_pred)
        }

        vol_metrics = {
            'mae': mean_absolute_error(y_vol_test, vol_pred),
            'mse': mean_squared_error(y_vol_test, vol_pred),
            'r2': r2_score(y_vol_test, vol_pred)
        }

        logger.info(f"Direction model - R²: {dir_metrics['r2']:.3f}, MAE: {dir_metrics['mae']:.4f}")
        logger.info(f"Volatility model - R²: {vol_metrics['r2']:.3f}, MAE: {vol_metrics['mae']:.2f}%")

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

        # Convert direction to binary signal (1 for up, 0 for down)
        direction_signal = (direction_pred > 0).astype(int)

        return {
            'direction_raw': direction_pred,
            'direction_signal': direction_signal,
            'volatility': volatility_pred
        }

    def predict_latest(self, df: pd.DataFrame) -> Dict[str, float]:
        """Predict for the most recent data point"""

        if len(df) == 0:
            raise ValueError("Empty dataframe provided")

        # Get only the last row
        latest = df.tail(1)

        # Make prediction
        predictions = self.predict(latest)

        return {
            'direction_prediction': float(predictions['direction_raw'][0]),
            'direction_signal': int(predictions['direction_signal'][0]),
            'volatility_prediction': float(predictions['volatility'][0]),
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