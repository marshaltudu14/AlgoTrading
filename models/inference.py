"""
Inference code for the TradingTransformer and MoE models.
Used for making predictions in live trading.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union
import os
import json

from models.trading_transformer import TradingTransformer, create_trading_transformer
from models.moe_transformer import MoETransformer, create_moe_transformer
from config import (
    TRANSFORMER_CONFIG,
    MOE_CONFIG,
    DEVICE,
    MODEL_PATH,
    INSTRUMENT_IDS,
    TIMEFRAME_IDS,
    OVERFITTING_CONFIG
)


class TradingTransformerInference:
    """
    Inference wrapper for the TradingTransformer model.
    Handles loading the model and making predictions.
    """
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        config: Dict[str, Any] = TRANSFORMER_CONFIG,
        device: torch.device = DEVICE
    ):
        """
        Initialize the inference wrapper.

        Args:
            model_path: Path to the trained model
            config: Model configuration
            device: Device to run inference on
        """
        self.device = device
        self.config = config
        self.model_path = model_path
        self.model = None

    def load_model(self, state_dim: int, num_instruments: int, num_timeframes: int):
        """
        Load the model from disk.

        Args:
            state_dim: Dimension of the state space
            num_instruments: Number of instruments
            num_timeframes: Number of timeframes
        """
        # Create model
        self.model = create_trading_transformer(
            config=self.config,
            state_dim=state_dim,
            num_instruments=num_instruments,
            num_timeframes=num_timeframes,
            action_dim=3  # hold, buy, sell
        )

        # Load weights
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def predict(
        self,
        features: np.ndarray,
        instrument: str,
        timeframe: int,
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Make a prediction using the model.

        Args:
            features: Input features of shape [window_size, feature_dim]
            instrument: Instrument name
            timeframe: Timeframe value
            deterministic: Whether to use deterministic prediction

        Returns:
            Tuple of (action, extra_info)
        """
        # Check if model is loaded
        if self.model is None:
            state_dim = features.shape[1]
            num_instruments = len(INSTRUMENT_IDS)
            num_timeframes = len(TIMEFRAME_IDS)
            self.load_model(state_dim, num_instruments, num_timeframes)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        instrument_id = torch.tensor([INSTRUMENT_IDS[instrument]], dtype=torch.long).to(self.device)
        timeframe_id = torch.tensor([TIMEFRAME_IDS[timeframe]], dtype=torch.long).to(self.device)

        # Make prediction
        with torch.no_grad():
            actions, extra_info = self.model.get_action(
                features_tensor,
                instrument_id,
                timeframe_id,
                deterministic=deterministic
            )

            # Convert to numpy
            action = actions.item()

            # Convert extra_info tensors to numpy
            for k, v in extra_info.items():
                if isinstance(v, torch.Tensor):
                    extra_info[k] = v.cpu().numpy()

        return action, extra_info


class MoEEnsembleInference:
    """
    Inference wrapper for an ensemble of MoE models.
    Handles loading the models and making predictions.
    """
    def __init__(
        self,
        model_dir: str = os.path.dirname(MODEL_PATH),
        ensemble_size: int = OVERFITTING_CONFIG.get('ensemble_size', 3),
        device: torch.device = DEVICE
    ):
        """
        Initialize the inference wrapper.

        Args:
            model_dir: Directory containing trained models
            ensemble_size: Number of models in the ensemble
            device: Device to run inference on
        """
        self.device = device
        self.model_dir = model_dir
        self.ensemble_size = ensemble_size
        self.models = []
        self.config = None

    def load_models(self, state_dim: int, num_instruments: int, num_timeframes: int):
        """
        Load the ensemble of models.

        Args:
            state_dim: Dimension of the state space
            num_instruments: Number of instruments
            num_timeframes: Number of timeframes
        """
        # Check if ensemble metadata exists
        metadata_path = os.path.join(self.model_dir, 'ensemble_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Use metadata to load models
            self.config = metadata.get('moe_config', MOE_CONFIG)
            model_paths = metadata.get('model_paths', [])

            # Limit to specified ensemble size
            model_paths = model_paths[:self.ensemble_size]

            for model_path in model_paths:
                # Create model
                model = create_moe_transformer(
                    config=self.config,
                    state_dim=state_dim,
                    num_instruments=num_instruments,
                    num_timeframes=num_timeframes,
                    action_dim=3  # hold, buy, sell
                )

                # Load weights
                full_path = os.path.join(self.model_dir, model_path)
                if os.path.exists(full_path):
                    model.load_state_dict(torch.load(full_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                else:
                    print(f"Warning: Model file not found: {full_path}")
        else:
            # Load models based on naming convention
            self.config = MOE_CONFIG
            for i in range(1, self.ensemble_size + 1):
                model_path = os.path.join(self.model_dir, f'moe_model_ensemble_{i}.pt')

                if os.path.exists(model_path):
                    # Create model
                    model = create_moe_transformer(
                        config=self.config,
                        state_dim=state_dim,
                        num_instruments=num_instruments,
                        num_timeframes=num_timeframes,
                        action_dim=3  # hold, buy, sell
                    )

                    # Load weights
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.models.append(model)
                else:
                    print(f"Warning: Model file not found: {model_path}")

        if not self.models:
            # Try loading a single model
            model_path = os.path.join(self.model_dir, 'moe_model.pt')
            if os.path.exists(model_path):
                model = create_moe_transformer(
                    config=self.config,
                    state_dim=state_dim,
                    num_instruments=num_instruments,
                    num_timeframes=num_timeframes,
                    action_dim=3  # hold, buy, sell
                )

                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                self.models.append(model)
            else:
                raise FileNotFoundError(f"No MoE models found in {self.model_dir}")

        print(f"Loaded {len(self.models)} models for ensemble")

    def predict(
        self,
        features: np.ndarray,
        instrument: str,
        timeframe: int,
        deterministic: bool = True
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Make a prediction using the ensemble of models.

        Args:
            features: Input features of shape [window_size, feature_dim]
            instrument: Instrument name
            timeframe: Timeframe value
            deterministic: Whether to use deterministic prediction

        Returns:
            Tuple of (action, extra_info)
        """
        # Check if models are loaded
        if not self.models:
            state_dim = features.shape[1]
            num_instruments = len(INSTRUMENT_IDS)
            num_timeframes = len(TIMEFRAME_IDS)
            self.load_models(state_dim, num_instruments, num_timeframes)

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        instrument_id = torch.tensor([INSTRUMENT_IDS[instrument]], dtype=torch.long).to(self.device)
        timeframe_id = torch.tensor([TIMEFRAME_IDS[timeframe]], dtype=torch.long).to(self.device)

        # Get predictions from all models
        all_actions = []
        all_probs = []
        all_values = []
        all_risks = []

        with torch.no_grad():
            for model in self.models:
                actions, extra_info = model.get_action(
                    features_tensor,
                    instrument_id,
                    timeframe_id,
                    deterministic=deterministic
                )

                all_actions.append(actions)
                all_probs.append(extra_info['action_probs'])
                all_values.append(extra_info['state_values'])
                all_risks.append(extra_info['risk_assessment'])

        # Stack predictions
        actions_stack = torch.stack(all_actions)
        probs_stack = torch.stack(all_probs)
        values_stack = torch.stack(all_values)
        risks_stack = torch.stack(all_risks)

        # Compute ensemble prediction
        if deterministic:
            # Majority vote
            actions, _ = torch.mode(actions_stack, dim=0)
            action = actions.item()
        else:
            # Average probabilities
            avg_probs = torch.mean(probs_stack, dim=0)
            dist = torch.distributions.Categorical(avg_probs)
            actions = dist.sample()
            action = actions.item()

        # Average other metrics
        avg_values = torch.mean(values_stack, dim=0)
        avg_risks = torch.mean(risks_stack, dim=0)

        # Compute uncertainty
        action_uncertainty = torch.std(probs_stack, dim=0)
        value_uncertainty = torch.std(values_stack, dim=0)
        risk_uncertainty = torch.std(risks_stack, dim=0)

        # Create extra info
        extra_info = {
            'action_probs': torch.mean(probs_stack, dim=0).cpu().numpy(),
            'state_values': avg_values.cpu().numpy(),
            'risk_assessment': avg_risks.cpu().numpy(),
            'action_uncertainty': action_uncertainty.cpu().numpy(),
            'value_uncertainty': value_uncertainty.cpu().numpy(),
            'risk_uncertainty': risk_uncertainty.cpu().numpy()
        }

        return action, extra_info


# Global model instances for caching
_transformer_instance = None
_moe_instance = None

def get_model_instance(use_moe: bool = True) -> Union[TradingTransformerInference, MoEEnsembleInference]:
    """
    Get or create a model instance.

    Args:
        use_moe: Whether to use the MoE model

    Returns:
        Model inference instance
    """
    global _transformer_instance, _moe_instance

    if use_moe:
        if _moe_instance is None:
            _moe_instance = MoEEnsembleInference()
        return _moe_instance
    else:
        if _transformer_instance is None:
            _transformer_instance = TradingTransformerInference()
        return _transformer_instance


def predict_action(
    features: np.ndarray,
    instrument: str = 'Nifty',
    timeframe: int = 2,
    use_moe: bool = True
) -> str:
    """
    Predict an action for the given features.

    Args:
        features: Input features
        instrument: Instrument name
        timeframe: Timeframe value
        use_moe: Whether to use the MoE model

    Returns:
        Action string: "HOLD", "BUY_CE", or "BUY_PE"
    """
    # Get model instance
    model = get_model_instance(use_moe=use_moe)

    # Make prediction
    action, extra_info = model.predict(features, instrument, timeframe)

    # Convert action to string
    action_map = {0: "HOLD", 1: "BUY_CE", 2: "BUY_PE"}
    action_str = action_map.get(action, "HOLD")

    # Get risk assessment
    if use_moe:
        risk = extra_info['risk_assessment'][0][0]
        uncertainty = extra_info['risk_uncertainty'][0][0]

        # If risk is too high or uncertainty is high, override to HOLD
        if (risk > 0.7 or uncertainty > 0.2) and action_str.startswith("BUY"):
            action_str = "HOLD"
    else:
        risk = extra_info['risk_assessment'][0][0]

        # If risk is too high, override to HOLD
        if risk > 0.7 and action_str.startswith("BUY"):
            action_str = "HOLD"

    return action_str
