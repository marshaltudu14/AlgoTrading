#!/usr/bin/env python3
"""
Iteration Management System for Research Tracking
=================================================

Manages research iterations, config hashing, and experiment tracking.
Creates new iterations only when architecture or configuration changes.
"""

import os
import json
import hashlib
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class IterationManager:
    """
    Manages research iterations and experiment tracking.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.research_config = config.get('research', {})
        self.results_dir = self.research_config.get('results_dir', 'research/results')
        self.iteration_prefix = self.research_config.get('iteration_prefix', 'iteration_')
        
        # Ensure research directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Current iteration info
        self.current_iteration = None
        self.iteration_dir = None
        self.iteration_hash = None
        
    def _generate_config_hash(self) -> str:
        """Generate a hash of the current configuration for comparison."""
        # Create a stable hash by excluding volatile fields and sorting keys
        config_for_hash = self._get_hashable_config()
        config_str = json.dumps(config_for_hash, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:12]  # Use first 12 chars
    
    def _get_hashable_config(self) -> Dict[str, Any]:
        """Extract configuration elements that affect model architecture and training."""
        hashable_elements = {}
        
        # Model architecture parameters
        if 'hierarchical_reasoning_model' in self.config:
            hashable_elements['hrm'] = self.config['hierarchical_reasoning_model']
        
        if 'model' in self.config:
            model_config = self.config['model'].copy()
            # Exclude paths that don't affect architecture
            model_config.pop('model_path', None)
            hashable_elements['model'] = model_config
        
        # Environment parameters that affect training
        if 'environment' in self.config:
            hashable_elements['environment'] = self.config['environment']
        
        # Risk management parameters
        if 'risk_management' in self.config:
            hashable_elements['risk_management'] = self.config['risk_management']
        
        # Feature generation parameters
        if 'feature_generation' in self.config:
            hashable_elements['feature_generation'] = self.config['feature_generation']
        
        # Training parameters
        if 'training_params' in self.config:
            hashable_elements['training_params'] = self.config['training_params']
            
        return hashable_elements
    
    def _get_feature_info(self, data_loader, symbols: list) -> Dict[str, Any]:
        """Get feature information from the data for comparison."""
        try:
            # Load a sample to get feature count and column names
            if symbols:
                sample_data = data_loader.load_final_data_for_symbol(symbols[0])
                if sample_data is not None and len(sample_data) > 0:
                    return {
                        'feature_count': len(sample_data.columns),
                        'feature_columns': list(sample_data.columns),
                        'data_shape': sample_data.shape
                    }
        except Exception as e:
            logger.warning(f"Could not extract feature info: {e}")
        
        return {'feature_count': 0, 'feature_columns': [], 'data_shape': (0, 0)}
    
    def _find_existing_iterations(self) -> Dict[str, Dict[str, Any]]:
        """Find all existing iterations and their metadata."""
        iterations = {}
        
        if not os.path.exists(self.results_dir):
            return iterations
        
        for item in os.listdir(self.results_dir):
            if item.startswith(self.iteration_prefix):
                iteration_dir = os.path.join(self.results_dir, item)
                metadata_file = os.path.join(iteration_dir, 'iteration_metadata.json')
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        iterations[item] = metadata
                    except Exception as e:
                        logger.warning(f"Could not read metadata for {item}: {e}")
        
        return iterations
    
    def _should_create_new_iteration(self, current_hash: str, feature_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a new iteration should be created."""
        existing_iterations = self._find_existing_iterations()
        
        if not existing_iterations:
            return True, "No existing iterations found"
        
        # Check if current configuration matches any existing iteration
        for iteration_name, metadata in existing_iterations.items():
            existing_hash = metadata.get('config_hash', '')
            existing_feature_count = metadata.get('feature_info', {}).get('feature_count', 0)
            
            # Check for configuration changes
            if existing_hash == current_hash:
                # Check for feature count changes
                if existing_feature_count == feature_info.get('feature_count', 0):
                    return False, f"Configuration matches existing {iteration_name}"
                else:
                    return True, f"Feature count changed: {existing_feature_count} -> {feature_info.get('feature_count', 0)}"
        
        return True, "Configuration hash not found in existing iterations"
    
    def setup_iteration(self, data_loader, symbols: list) -> str:
        """
        Setup a new iteration or use existing one based on configuration.
        
        Returns:
            Path to the iteration directory
        """
        # Generate current configuration hash
        current_hash = self._generate_config_hash()
        self.iteration_hash = current_hash
        
        # Get feature information
        feature_info = self._get_feature_info(data_loader, symbols)
        
        # Check if we need a new iteration
        should_create_new, reason = self._should_create_new_iteration(current_hash, feature_info)
        
        if should_create_new:
            # Create new iteration
            existing_iterations = self._find_existing_iterations()
            iteration_number = len(existing_iterations) + 1
            iteration_name = f"{self.iteration_prefix}{iteration_number:03d}"
            
            logger.info(f"🔬 Creating new iteration: {iteration_name}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   Config hash: {current_hash}")
            logger.info(f"   Feature count: {feature_info.get('feature_count', 0)}")
        else:
            # Use existing iteration (find the matching one)
            existing_iterations = self._find_existing_iterations()
            for iteration_name, metadata in existing_iterations.items():
                if metadata.get('config_hash') == current_hash:
                    logger.info(f"🔄 Using existing iteration: {iteration_name}")
                    logger.info(f"   Reason: {reason}")
                    break
        
        # Set up iteration directory
        self.current_iteration = iteration_name
        self.iteration_dir = os.path.join(self.results_dir, iteration_name)
        os.makedirs(self.iteration_dir, exist_ok=True)
        
        # Save iteration metadata
        self._save_iteration_metadata(current_hash, feature_info, should_create_new)
        
        # Copy current configuration
        self._save_config_snapshot()
        
        return self.iteration_dir
    
    def _save_iteration_metadata(self, config_hash: str, feature_info: Dict[str, Any], is_new: bool):
        """Save metadata about this iteration."""
        metadata = {
            'iteration_name': self.current_iteration,
            'config_hash': config_hash,
            'feature_info': feature_info,
            'created_at': datetime.now().isoformat(),
            'is_new_iteration': is_new,
            'config_elements_hashed': list(self._get_hashable_config().keys())
        }
        
        metadata_file = os.path.join(self.iteration_dir, 'iteration_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _save_config_snapshot(self):
        """Save a complete snapshot of the current configuration."""
        config_file = os.path.join(self.iteration_dir, 'config_snapshot.yaml')
        
        # Copy the original settings.yaml file
        original_config = os.path.join('config', 'settings.yaml')
        if os.path.exists(original_config):
            shutil.copy2(original_config, config_file)
        
        # Also save as JSON for easier parsing
        config_json_file = os.path.join(self.iteration_dir, 'config_snapshot.json')
        with open(config_json_file, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def get_iteration_log_path(self, log_type: str = 'training') -> str:
        """Get the path for a specific log file in the current iteration."""
        if not self.iteration_dir:
            raise ValueError("No iteration has been set up")
        
        log_filename = f"{log_type}_detailed.log"
        return os.path.join(self.iteration_dir, log_filename)
    
    def save_training_metrics(self, metrics: Dict[str, Any]):
        """Save training metrics to the iteration directory."""
        if not self.iteration_dir:
            return
        
        metrics_file = os.path.join(self.iteration_dir, 'training_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def save_model_artifacts(self, model_path: str):
        """Record model information without copying the actual model file (storage optimization)."""
        if not self.iteration_dir or not os.path.exists(model_path):
            return
        
        # Save model metadata instead of copying the large model file
        model_info = {
            'model_path': model_path,
            'model_size_bytes': os.path.getsize(model_path),
            'model_saved_at': datetime.now().isoformat(),
            'note': 'Model stored in central models/ directory to avoid storage duplication'
        }
        
        model_info_file = os.path.join(self.iteration_dir, 'model_info.json')
        with open(model_info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"📁 Model info recorded (model stored centrally at {model_path})")
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current iteration."""
        if not self.current_iteration:
            return {}
        
        return {
            'iteration_name': self.current_iteration,
            'iteration_dir': self.iteration_dir,
            'config_hash': self.iteration_hash,
            'setup_time': datetime.now().isoformat()
        }