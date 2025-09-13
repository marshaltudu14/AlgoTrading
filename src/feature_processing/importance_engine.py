#!/usr/bin/env python3
"""
Feature Importance Calculation Engine
====================================

A comprehensive system for calculating and analyzing feature importance using multiple methods:
- Attention-based importance extraction from transformer models
- SHAP value calculation for local and global importance
- Permutation importance calculation
- Gradient-based importance calculation

This engine supports both online (real-time) and offline importance calculation,
with optimizations for trading model performance requirements.

Author: AlgoTrading System
Version: 1.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - SHAP-based importance calculations will be disabled")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImportanceMethod(Enum):
    """Supported importance calculation methods."""
    ATTENTION_WEIGHTS = "attention_weights"
    SHAP_VALUES = "shap_values"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    GRADIENT_IMPORTANCE = "gradient_importance"
    INTEGRATED_GRADIENTS = "integrated_gradients"


class ImportanceType(Enum):
    """Types of importance scores."""
    LOCAL = "local"  # Per-prediction importance
    GLOBAL = "global"  # Overall feature importance
    TEMPORAL = "temporal"  # Time-based importance trends


@dataclass
class ImportanceScore:
    """Individual importance score with metadata."""
    feature_name: str
    score: float
    method: ImportanceMethod
    importance_type: ImportanceType
    timestamp: float
    prediction_id: Optional[str] = None
    confidence: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None


@dataclass
class ImportanceResult:
    """Complete importance calculation result."""
    scores: List[ImportanceScore]
    method: ImportanceMethod
    importance_type: ImportanceType
    computation_time: float
    model_info: Dict[str, Any]
    feature_count: int
    summary_stats: Dict[str, float]


class BaseImportanceCalculator(ABC):
    """Base class for importance calculation methods."""

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.computation_stats = {}

    def _get_device(self, device: str) -> str:
        """Determine the best device for computation."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    @abstractmethod
    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           **kwargs) -> ImportanceResult:
        """Calculate importance scores."""
        pass

    def _prepare_input(self, X: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> torch.Tensor:
        """Prepare input data for computation."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(X, torch.Tensor):
            X = X.to(self.device)
        return X


class AttentionImportanceCalculator(BaseImportanceCalculator):
    """Calculate importance based on transformer attention weights."""

    def __init__(self,
                 attention_layers: List[int] = None,
                 attention_heads: List[int] = None,
                 aggregation_method: str = 'mean',
                 device: str = 'auto'):
        super().__init__(device)
        self.attention_layers = attention_layers or list(range(4))  # Default to all 4 layers
        self.attention_heads = attention_heads or list(range(8))    # Default to all 8 heads
        self.aggregation_method = aggregation_method  # 'mean', 'max', 'sum'

    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> ImportanceResult:
        """
        Calculate importance using attention weights.

        Args:
            model: Transformer model with attention mechanisms
            X: Input features
            y: Target values (optional)
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            ImportanceResult with attention-based scores
        """
        start_time = time.time()

        # Prepare input
        X_tensor = self._prepare_input(X)
        model.eval()
        model.to(self.device)

        # Get feature names if not provided
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        importance_scores = []

        with torch.no_grad():
            # Forward pass with attention capture
            outputs = self._forward_with_attention_capture(model, X_tensor)

            if outputs is None or 'attention_weights' not in outputs:
                logger.warning("Model does not return attention weights. Using fallback method.")
                return self._fallback_importance_calculation(model, X_tensor, feature_names)

            attention_weights = outputs['attention_weights']

            # Process attention weights for each layer and head
            for layer_idx in self.attention_layers:
                if layer_idx >= len(attention_weights):
                    continue

                layer_attention = attention_weights[layer_idx]

                for head_idx in self.attention_heads:
                    if head_idx >= layer_attention.shape[1]:
                        continue

                    # Extract attention for this head
                    head_attention = layer_attention[:, head_idx, :, :]

                    # Calculate feature importance from attention patterns
                    feature_importance = self._calculate_attention_importance(
                        head_attention, X_tensor.shape[1]
                    )

                    # Create importance scores for each feature
                    for feat_idx, (feat_name, score) in enumerate(zip(feature_names, feature_importance)):
                        importance_score = ImportanceScore(
                            feature_name=feat_name,
                            score=float(score),
                            method=ImportanceMethod.ATTENTION_WEIGHTS,
                            importance_type=ImportanceType.LOCAL,
                            timestamp=time.time(),
                            additional_info={
                                'layer': layer_idx,
                                'head': head_idx,
                                'aggregation': self.aggregation_method
                            }
                        )
                        importance_scores.append(importance_score)

        # Aggregate scores across all layers/heads
        aggregated_scores = self._aggregate_attention_scores(importance_scores, feature_names)

        computation_time = time.time() - start_time

        return ImportanceResult(
            scores=aggregated_scores,
            method=ImportanceMethod.ATTENTION_WEIGHTS,
            importance_type=ImportanceType.LOCAL,
            computation_time=computation_time,
            model_info={
                'model_type': type(model).__name__,
                'num_layers': len(self.attention_layers),
                'num_heads': len(self.attention_heads),
                'device': self.device
            },
            feature_count=len(feature_names),
            summary_stats=self._calculate_summary_stats(aggregated_scores)
        )

    def _forward_with_attention_capture(self, model: nn.Module, X: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Forward pass that captures attention weights."""
        # This is a simplified implementation - in practice, you'd need to modify
        # the model to return attention weights or use hooks
        try:
            # Try to call model with return_attention=True if supported
            if hasattr(model, 'forward'):
                outputs = model(X, return_attention=True)
                if isinstance(outputs, dict) and 'attention_weights' in outputs:
                    return outputs
        except:
            pass

        return None

    def _calculate_attention_importance(self, attention_matrix: torch.Tensor, num_features: int) -> np.ndarray:
        """Calculate feature importance from attention matrix."""
        # Average attention across sequence length for each feature
        # This is a simplified approach - actual implementation depends on model architecture

        if attention_matrix.dim() == 3:  # [batch, seq_len, seq_len]
            # Average attention across sequence dimension
            feature_importance = attention_matrix.mean(dim=[1, 2])
        elif attention_matrix.dim() == 4:  # [batch, heads, seq_len, seq_len]
            # Average across heads and sequence
            feature_importance = attention_matrix.mean(dim=[1, 2, 3])
        else:
            # Fallback: uniform importance
            feature_importance = torch.ones(attention_matrix.shape[0]) / num_features

        return feature_importance.cpu().numpy()

    def _aggregate_attention_scores(self, scores: List[ImportanceScore], feature_names: List[str]) -> List[ImportanceScore]:
        """Aggregate attention scores across layers and heads."""
        feature_scores = {}

        for score in scores:
            feat_name = score.feature_name
            if feat_name not in feature_scores:
                feature_scores[feat_name] = []
            feature_scores[feat_name].append(score.score)

        aggregated_scores = []
        for feat_name in feature_names:
            if feat_name in feature_scores:
                feat_scores = feature_scores[feat_name]
                if self.aggregation_method == 'mean':
                    agg_score = np.mean(feat_scores)
                elif self.aggregation_method == 'max':
                    agg_score = np.max(feat_scores)
                elif self.aggregation_method == 'sum':
                    agg_score = np.sum(feat_scores)
                else:
                    agg_score = np.mean(feat_scores)

                aggregated_score = ImportanceScore(
                    feature_name=feat_name,
                    score=float(agg_score),
                    method=ImportanceMethod.ATTENTION_WEIGHTS,
                    importance_type=ImportanceType.LOCAL,
                    timestamp=time.time()
                )
                aggregated_scores.append(aggregated_score)

        return aggregated_scores

    def _fallback_importance_calculation(self, model: nn.Module, X: torch.Tensor,
                                       feature_names: List[str]) -> ImportanceResult:
        """Fallback method when attention weights are not available."""
        # Use gradient-based importance as fallback
        gradient_calc = GradientImportanceCalculator(device=self.device)
        return gradient_calc.calculate_importance(model, X, feature_names=feature_names)

    def _calculate_summary_stats(self, scores: List[ImportanceScore]) -> Dict[str, float]:
        """Calculate summary statistics for importance scores."""
        if not scores:
            return {}

        score_values = [s.score for s in scores]
        return {
            'mean': float(np.mean(score_values)),
            'std': float(np.std(score_values)),
            'min': float(np.min(score_values)),
            'max': float(np.max(score_values)),
            'median': float(np.median(score_values))
        }


class SHAPImportanceCalculator(BaseImportanceCalculator):
    """Calculate importance using SHAP values."""

    def __init__(self,
                 background_samples: int = 100,
                 nsamples: int = 'auto',
                 device: str = 'auto'):
        super().__init__(device)
        self.background_samples = background_samples
        self.nsamples = nsamples

        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required for SHAP importance calculations")

    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> ImportanceResult:
        """
        Calculate importance using SHAP values.

        Args:
            model: PyTorch model
            X: Input features
            y: Target values (optional)
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            ImportanceResult with SHAP-based scores
        """
        start_time = time.time()

        # Prepare input
        if isinstance(X, torch.Tensor):
            X_np = X.cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X_np = X.values
            feature_names = feature_names or X.columns.tolist()
        else:
            X_np = X

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X_np.shape[1])]

        # Create background dataset
        background_data = X_np[:self.background_samples]

        try:
            # Create SHAP explainer
            explainer = shap.DeepExplainer(model, background_data)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_np)

            # Calculate importance scores
            importance_scores = self._process_shap_values(shap_values, feature_names)

            computation_time = time.time() - start_time

            return ImportanceResult(
                scores=importance_scores,
                method=ImportanceMethod.SHAP_VALUES,
                importance_type=ImportanceType.LOCAL,
                computation_time=computation_time,
                model_info={
                    'model_type': type(model).__name__,
                    'background_samples': self.background_samples,
                    'device': self.device
                },
                feature_count=len(feature_names),
                summary_stats=self._calculate_summary_stats(importance_scores)
            )

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            # Fallback to permutation importance
            perm_calc = PermutationImportanceCalculator(device=self.device)
            return perm_calc.calculate_importance(model, X, y, feature_names)

    def _process_shap_values(self, shap_values: Union[np.ndarray, List],
                           feature_names: List[str]) -> List[ImportanceScore]:
        """Process SHAP values into importance scores."""
        importance_scores = []

        if isinstance(shap_values, list):
            # Multi-output case
            shap_values = np.abs(np.array(shap_values)).mean(axis=0)
        else:
            # Single output case
            shap_values = np.abs(shap_values)

        # Calculate mean absolute SHAP value for each feature
        mean_shap_values = np.mean(shap_values, axis=0)

        for feat_idx, (feat_name, shap_val) in enumerate(zip(feature_names, mean_shap_values)):
            importance_score = ImportanceScore(
                feature_name=feat_name,
                score=float(shap_val),
                method=ImportanceMethod.SHAP_VALUES,
                importance_type=ImportanceType.LOCAL,
                timestamp=time.time()
            )
            importance_scores.append(importance_score)

        return importance_scores

    def _calculate_summary_stats(self, scores: List[ImportanceScore]) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not scores:
            return {}

        score_values = [s.score for s in scores]
        return {
            'mean': float(np.mean(score_values)),
            'std': float(np.std(score_values)),
            'min': float(np.min(score_values)),
            'max': float(np.max(score_values)),
            'median': float(np.median(score_values))
        }


class PermutationImportanceCalculator(BaseImportanceCalculator):
    """Calculate importance using permutation importance."""

    def __init__(self,
                 n_repeats: int = 10,
                 scoring: str = 'mse',
                 device: str = 'auto'):
        super().__init__(device)
        self.n_repeats = n_repeats
        self.scoring = scoring

    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Union[np.ndarray, torch.Tensor],
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> ImportanceResult:
        """
        Calculate importance using permutation importance.

        Args:
            model: PyTorch model
            X: Input features
            y: Target values
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            ImportanceResult with permutation-based scores
        """
        start_time = time.time()

        # Prepare input
        X_tensor = self._prepare_input(X)
        y_tensor = self._prepare_input(y)

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
        else:
            feature_names = feature_names or [f'feature_{i}' for i in range(X_tensor.shape[1])]

        model.eval()
        model.to(self.device)

        # Calculate baseline score
        baseline_score = self._calculate_baseline_score(model, X_tensor, y_tensor)

        importance_scores = []

        # Calculate importance for each feature
        for feat_idx, feat_name in enumerate(feature_names):
            feature_importance = self._calculate_single_feature_importance(
                model, X_tensor, y_tensor, feat_idx, baseline_score
            )

            importance_score = ImportanceScore(
                feature_name=feat_name,
                score=float(feature_importance),
                method=ImportanceMethod.PERMUTATION_IMPORTANCE,
                importance_type=ImportanceType.GLOBAL,
                timestamp=time.time(),
                additional_info={'n_repeats': self.n_repeats}
            )
            importance_scores.append(importance_score)

        computation_time = time.time() - start_time

        return ImportanceResult(
            scores=importance_scores,
            method=ImportanceMethod.PERMUTATION_IMPORTANCE,
            importance_type=ImportanceType.GLOBAL,
            computation_time=computation_time,
            model_info={
                'model_type': type(model).__name__,
                'n_repeats': self.n_repeats,
                'scoring': self.scoring,
                'device': self.device
            },
            feature_count=len(feature_names),
            summary_stats=self._calculate_summary_stats(importance_scores)
        )

    def _calculate_baseline_score(self, model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
        """Calculate baseline model performance."""
        with torch.no_grad():
            predictions = model(X)
            if predictions.shape != y.shape:
                predictions = predictions.squeeze()
            return self._calculate_score(predictions, y).item()

    def _calculate_single_feature_importance(self, model: nn.Module, X: torch.Tensor,
                                           y: torch.Tensor, feat_idx: int,
                                           baseline_score: float) -> float:
        """Calculate importance for a single feature."""
        importance_scores = []

        for _ in range(self.n_repeats):
            # Permute the feature
            X_permuted = X.clone()
            X_permuted[:, feat_idx] = X_permuted[torch.randperm(X_permuted.shape[0]), feat_idx]

            # Calculate score with permuted feature
            with torch.no_grad():
                predictions = model(X_permuted)
                if predictions.shape != y.shape:
                    predictions = predictions.squeeze()
                permuted_score = self._calculate_score(predictions, y).item()

            # Importance is the decrease in performance
            importance = baseline_score - permuted_score
            importance_scores.append(importance)

        return np.mean(importance_scores)

    def _calculate_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate score based on specified metric."""
        if self.scoring == 'mse':
            return -torch.mean((predictions - targets) ** 2)
        elif self.scoring == 'mae':
            return -torch.mean(torch.abs(predictions - targets))
        elif self.scoring == 'r2':
            ss_res = torch.sum((targets - predictions) ** 2)
            ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
            return 1 - ss_res / ss_tot
        else:
            return -torch.mean((predictions - targets) ** 2)  # Default to MSE

    def _calculate_summary_stats(self, scores: List[ImportanceScore]) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not scores:
            return {}

        score_values = [s.score for s in scores]
        return {
            'mean': float(np.mean(score_values)),
            'std': float(np.std(score_values)),
            'min': float(np.min(score_values)),
            'max': float(np.max(score_values)),
            'median': float(np.median(score_values))
        }


class GradientImportanceCalculator(BaseImportanceCalculator):
    """Calculate importance using gradient-based methods."""

    def __init__(self, device: str = 'auto'):
        super().__init__(device)

    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> ImportanceResult:
        """
        Calculate importance using gradients.

        Args:
            model: PyTorch model
            X: Input features
            y: Target values (optional)
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            ImportanceResult with gradient-based scores
        """
        start_time = time.time()

        # Prepare input
        X_tensor = self._prepare_input(X)
        X_tensor.requires_grad_(True)

        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
        else:
            feature_names = feature_names or [f'feature_{i}' for i in range(X_tensor.shape[1])]

        model.train()
        model.to(self.device)

        importance_scores = []

        # Calculate gradients for each sample
        for i in range(X_tensor.shape[0]):
            sample = X_tensor[i:i+1]

            # Forward pass
            output = model(sample)

            # Backward pass to get gradients
            if output.dim() > 1:
                output = output.mean()  # Use mean for multi-output

            output.backward()

            # Get gradients
            if X_tensor.grad is not None:
                gradients = X_tensor.grad[i].cpu().numpy()
                importance_scores.append(np.abs(gradients))

            # Clear gradients
            model.zero_grad()
            if X_tensor.grad is not None:
                X_tensor.grad.zero_()

        # Average gradients across samples
        if importance_scores:
            avg_importance = np.mean(importance_scores, axis=0)
        else:
            avg_importance = np.zeros(len(feature_names))

        # Create importance scores
        final_scores = []
        for feat_idx, (feat_name, score) in enumerate(zip(feature_names, avg_importance)):
            importance_score = ImportanceScore(
                feature_name=feat_name,
                score=float(score),
                method=ImportanceMethod.GRADIENT_IMPORTANCE,
                importance_type=ImportanceType.LOCAL,
                timestamp=time.time()
            )
            final_scores.append(importance_score)

        computation_time = time.time() - start_time

        return ImportanceResult(
            scores=final_scores,
            method=ImportanceMethod.GRADIENT_IMPORTANCE,
            importance_type=ImportanceType.LOCAL,
            computation_time=computation_time,
            model_info={
                'model_type': type(model).__name__,
                'device': self.device,
                'num_samples': X_tensor.shape[0]
            },
            feature_count=len(feature_names),
            summary_stats=self._calculate_summary_stats(final_scores)
        )

    def _calculate_summary_stats(self, scores: List[ImportanceScore]) -> Dict[str, float]:
        """Calculate summary statistics."""
        if not scores:
            return {}

        score_values = [s.score for s in scores]
        return {
            'mean': float(np.mean(score_values)),
            'std': float(np.std(score_values)),
            'min': float(np.min(score_values)),
            'max': float(np.max(score_values)),
            'median': float(np.median(score_values))
        }


class FeatureImportanceEngine:
    """Main engine for calculating feature importance using multiple methods."""

    def __init__(self,
                 device: str = 'auto',
                 default_methods: List[ImportanceMethod] = None):
        self.device = device
        self.default_methods = default_methods or [
            ImportanceMethod.ATTENTION_WEIGHTS,
            ImportanceMethod.GRADIENT_IMPORTANCE,
            ImportanceMethod.PERMUTATION_IMPORTANCE
        ]

        self.calculators = {
            ImportanceMethod.ATTENTION_WEIGHTS: AttentionImportanceCalculator(device=device),
            ImportanceMethod.SHAP_VALUES: SHAPImportanceCalculator(device=device) if SHAP_AVAILABLE else None,
            ImportanceMethod.PERMUTATION_IMPORTANCE: PermutationImportanceCalculator(device=device),
            ImportanceMethod.GRADIENT_IMPORTANCE: GradientImportanceCalculator(device=device)
        }

        self.results_history = []
        self.computation_stats = {}

    def calculate_importance(self,
                           model: nn.Module,
                           X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                           y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                           methods: Optional[List[ImportanceMethod]] = None,
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> Dict[ImportanceMethod, ImportanceResult]:
        """
        Calculate feature importance using specified methods.

        Args:
            model: PyTorch model
            X: Input features
            y: Target values (required for some methods)
            methods: List of importance methods to use
            feature_names: Names of features
            **kwargs: Additional parameters for specific methods

        Returns:
            Dictionary mapping methods to their results
        """
        methods = methods or self.default_methods
        results = {}

        for method in methods:
            calculator = self.calculators.get(method)
            if calculator is None:
                logger.warning(f"Calculator for method {method.value} is not available")
                continue

            try:
                logger.info(f"Calculating importance using {method.value}")
                result = calculator.calculate_importance(model, X, y, feature_names, **kwargs)
                results[method] = result
                self.results_history.append(result)

                logger.info(f"{method.value} completed in {result.computation_time:.3f}s")

            except Exception as e:
                logger.error(f"Error calculating importance with {method.value}: {e}")
                continue

        return results

    def calculate_global_importance(self,
                                  model: nn.Module,
                                  X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                                  y: Union[np.ndarray, torch.Tensor],
                                  feature_names: Optional[List[str]] = None,
                                  **kwargs) -> Dict[ImportanceMethod, ImportanceResult]:
        """
        Calculate global feature importance (across entire dataset).

        Args:
            model: PyTorch model
            X: Input features
            y: Target values
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping methods to their results
        """
        # Use methods that are suitable for global importance
        global_methods = [
            ImportanceMethod.PERMUTATION_IMPORTANCE,
            ImportanceMethod.SHAP_VALUES
        ]

        return self.calculate_importance(model, X, y, global_methods, feature_names, **kwargs)

    def calculate_local_importance(self,
                                 model: nn.Module,
                                 X: Union[np.ndarray, torch.Tensor, pd.DataFrame],
                                 feature_names: Optional[List[str]] = None,
                                 **kwargs) -> Dict[ImportanceMethod, ImportanceResult]:
        """
        Calculate local feature importance (per prediction).

        Args:
            model: PyTorch model
            X: Input features
            feature_names: Names of features
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping methods to their results
        """
        # Use methods that are suitable for local importance
        local_methods = [
            ImportanceMethod.ATTENTION_WEIGHTS,
            ImportanceMethod.GRADIENT_IMPORTANCE,
            ImportanceMethod.SHAP_VALUES
        ]

        return self.calculate_importance(model, X, None, local_methods, feature_names, **kwargs)

    def get_method_rankings(self, results: Dict[ImportanceMethod, ImportanceResult]) -> Dict[ImportanceMethod, List[Tuple[str, float]]]:
        """
        Get feature rankings for each method.

        Args:
            results: Dictionary of method results

        Returns:
            Dictionary mapping methods to ranked feature lists
        """
        rankings = {}

        for method, result in results.items():
            # Sort features by importance score
            sorted_features = sorted(
                [(score.feature_name, score.score) for score in result.scores],
                key=lambda x: x[1],
                reverse=True
            )
            rankings[method] = sorted_features

        return rankings

    def get_consensus_ranking(self, results: Dict[ImportanceMethod, ImportanceResult]) -> List[Tuple[str, float]]:
        """
        Get consensus ranking across all methods.

        Args:
            results: Dictionary of method results

        Returns:
            Consensus ranking of features
        """
        method_rankings = self.get_method_rankings(results)

        # Collect all features
        all_features = set()
        for ranking in method_rankings.values():
            all_features.update(feat for feat, _ in ranking)

        # Calculate consensus scores
        consensus_scores = {}
        for feature in all_features:
            scores = []
            for method, ranking in method_rankings.items():
                # Find rank of feature in this method
                rank = next((i for i, (feat, _) in enumerate(ranking) if feat == feature), len(ranking))
                # Convert rank to score (higher rank = lower score)
                score = 1.0 / (rank + 1)
                scores.append(score)

            consensus_scores[feature] = np.mean(scores)

        # Sort by consensus score
        consensus_ranking = sorted(consensus_scores.items(), key=lambda x: x[1], reverse=True)
        return consensus_ranking

    def save_results(self, results: Dict[ImportanceMethod, ImportanceResult], filepath: str):
        """Save importance results to file."""
        save_data = {
            'timestamp': time.time(),
            'results': {}
        }

        for method, result in results.items():
            save_data['results'][method.value] = {
                'method': method.value,
                'importance_type': result.importance_type.value,
                'computation_time': result.computation_time,
                'model_info': result.model_info,
                'feature_count': result.feature_count,
                'summary_stats': result.summary_stats,
                'scores': [
                    {
                        'feature_name': score.feature_name,
                        'score': score.score,
                        'method': score.method.value,
                        'importance_type': score.importance_type.value,
                        'timestamp': score.timestamp,
                        'prediction_id': score.prediction_id,
                        'confidence': score.confidence,
                        'additional_info': score.additional_info
                    }
                    for score in result.scores
                ]
            }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    def load_results(self, filepath: str) -> Dict[ImportanceMethod, ImportanceResult]:
        """Load importance results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        results = {}
        for method_data in data['results'].values():
            method = ImportanceMethod(method_data['method'])

            scores = []
            for score_data in method_data['scores']:
                score = ImportanceScore(
                    feature_name=score_data['feature_name'],
                    score=score_data['score'],
                    method=ImportanceMethod(score_data['method']),
                    importance_type=ImportanceType(score_data['importance_type']),
                    timestamp=score_data['timestamp'],
                    prediction_id=score_data['prediction_id'],
                    confidence=score_data['confidence'],
                    additional_info=score_data['additional_info']
                )
                scores.append(score)

            result = ImportanceResult(
                scores=scores,
                method=method,
                importance_type=ImportanceType(method_data['importance_type']),
                computation_time=method_data['computation_time'],
                model_info=method_data['model_info'],
                feature_count=method_data['feature_count'],
                summary_stats=method_data['summary_stats']
            )

            results[method] = result

        logger.info(f"Results loaded from {filepath}")
        return results

    def get_computation_stats(self) -> Dict[str, Any]:
        """Get computation statistics."""
        stats = {
            'total_calculations': len(self.results_history),
            'method_usage': {},
            'average_computation_time': {},
            'feature_count_stats': {}
        }

        if self.results_history:
            for result in self.results_history:
                method = result.method.value
                stats['method_usage'][method] = stats['method_usage'].get(method, 0) + 1
                stats['average_computation_time'][method] = stats['average_computation_time'].get(method, 0) + result.computation_time

            # Calculate averages
            for method in stats['average_computation_time']:
                stats['average_computation_time'][method] /= stats['method_usage'][method]

        return stats