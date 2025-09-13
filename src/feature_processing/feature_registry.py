"""
Feature Registry for tracking and managing detected features

This module provides a centralized registry for managing feature metadata,
versions, and changes across the system.
"""

import json
import pickle
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
import logging
from dataclasses import dataclass, field
import hashlib

from .feature_detector import FeatureMetadata, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class FeatureVersion:
    """Version information for a feature"""
    version_id: str
    timestamp: datetime
    metadata: FeatureMetadata
    change_description: str
    parent_version: Optional[str] = None


class FeatureRegistry:
    """
    Centralized registry for managing feature metadata and versions
    """

    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize feature registry

        Args:
            registry_path: Path to registry file
        """
        self.registry_path = Path(registry_path) if registry_path else Path("feature_registry.json")
        self.features: Dict[str, List[FeatureVersion]] = {}
        self.current_versions: Dict[str, str] = {}

        # Load existing registry if file exists
        if self.registry_path.exists():
            self.load_registry()

    def register_feature(self, metadata: FeatureMetadata,
                        change_description: str = "Initial registration") -> str:
        """
        Register a new feature or version of an existing feature

        Args:
            metadata: Feature metadata
            change_description: Description of changes

        Returns:
            Version ID for the registered feature
        """
        feature_name = metadata.name
        version_id = self._generate_version_id(feature_name, metadata)

        # Create feature version
        version = FeatureVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            metadata=metadata,
            change_description=change_description,
            parent_version=self.current_versions.get(feature_name)
        )

        # Add to features dictionary
        if feature_name not in self.features:
            self.features[feature_name] = []

        self.features[feature_name].append(version)
        self.current_versions[feature_name] = version_id

        logger.info(f"Registered feature {feature_name} version {version_id}")
        return version_id

    def get_feature(self, feature_name: str, version: Optional[str] = None) -> Optional[FeatureMetadata]:
        """
        Get feature metadata

        Args:
            feature_name: Name of the feature
            version: Specific version (if None, returns current version)

        Returns:
            Feature metadata or None if not found
        """
        if feature_name not in self.features:
            return None

        if version is None:
            version = self.current_versions.get(feature_name)

        for feature_version in self.features[feature_name]:
            if feature_version.version_id == version:
                return feature_version.metadata

        return None

    def get_feature_history(self, feature_name: str) -> List[FeatureVersion]:
        """
        Get version history for a feature

        Args:
            feature_name: Name of the feature

        Returns:
            List of feature versions
        """
        return self.features.get(feature_name, [])

    def get_all_features(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get all registered features

        Args:
            include_history: Whether to include version history

        Returns:
            Dictionary of all features
        """
        result = {}

        for feature_name, versions in self.features.items():
            current_version = self.current_versions.get(feature_name)
            current_metadata = None

            for version in versions:
                if version.version_id == current_version:
                    current_metadata = version.metadata
                    break

            if current_metadata:
                result[feature_name] = {
                    'current_metadata': current_metadata,
                    'current_version': current_version,
                    'version_count': len(versions)
                }

                if include_history:
                    result[feature_name]['history'] = versions

        return result

    def update_feature(self, feature_name: str, updated_metadata: FeatureMetadata,
                      change_description: str = "Feature updated") -> str:
        """
        Update an existing feature

        Args:
            feature_name: Name of the feature to update
            updated_metadata: Updated feature metadata
            change_description: Description of changes

        Returns:
            New version ID
        """
        if feature_name not in self.features:
            raise ValueError(f"Feature {feature_name} not found in registry")

        # Update timestamp in metadata
        updated_metadata.updated_at = datetime.now()

        # Register new version
        return self.register_feature(updated_metadata, change_description)

    def remove_feature(self, feature_name: str) -> bool:
        """
        Remove a feature from the registry

        Args:
            feature_name: Name of the feature to remove

        Returns:
            True if removed, False if not found
        """
        if feature_name not in self.features:
            return False

        del self.features[feature_name]
        if feature_name in self.current_versions:
            del self.current_versions[feature_name]

        logger.info(f"Removed feature {feature_name} from registry")
        return True

    def get_features_by_type(self, feature_type: FeatureType) -> Dict[str, FeatureMetadata]:
        """
        Get all features of a specific type

        Args:
            feature_type: Feature type to filter by

        Returns:
            Dictionary of features of the specified type
        """
        result = {}

        for feature_name, versions in self.features.items():
            current_version = self.current_versions.get(feature_name)
            if current_version:
                for version in versions:
                    if version.version_id == current_version:
                        if version.metadata.feature_type == feature_type:
                            result[feature_name] = version.metadata
                        break

        return result

    def validate_feature_compatibility(self, feature_names: List[str]) -> Dict[str, Any]:
        """
        Validate compatibility between features

        Args:
            feature_names: List of feature names to validate

        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'feature_count': len(feature_names),
            'compatible_features': [],
            'incompatible_features': []
        }

        for feature_name in feature_names:
            feature = self.get_feature(feature_name)
            if not feature:
                validation_results['issues'].append(f"Feature {feature_name} not found in registry")
                validation_results['incompatible_features'].append(feature_name)
                validation_results['valid'] = False
                continue

            # Check for high null percentage
            if feature.null_percentage > 0.8:
                validation_results['issues'].append(
                    f"Feature {feature_name} has high null percentage: {feature.null_percentage:.2%}"
                )
                validation_results['incompatible_features'].append(feature_name)
                validation_results['valid'] = False
            else:
                validation_results['compatible_features'].append(feature_name)

        return validation_results

    def save_registry(self):
        """Save registry to file"""
        registry_data = {
            'features': {},
            'current_versions': self.current_versions,
            'saved_at': datetime.now().isoformat()
        }

        for feature_name, versions in self.features.items():
            registry_data['features'][feature_name] = []
            for version in versions:
                registry_data['features'][feature_name].append({
                    'version_id': version.version_id,
                    'timestamp': version.timestamp.isoformat(),
                    'metadata': {
                        'name': version.metadata.name,
                        'feature_type': version.metadata.feature_type.value,
                        'dtype': version.metadata.dtype,
                        'null_count': version.metadata.null_count,
                        'null_percentage': version.metadata.null_percentage,
                        'unique_count': version.metadata.unique_count,
                        'min_value': version.metadata.min_value,
                        'max_value': version.metadata.max_value,
                        'mean_value': version.metadata.mean_value,
                        'std_value': version.metadata.std_value,
                        'categories': version.metadata.categories,
                        'is_target': version.metadata.is_target,
                        'is_identifier': version.metadata.is_identifier,
                        'created_at': version.metadata.created_at.isoformat(),
                        'updated_at': version.metadata.updated_at.isoformat()
                    },
                    'change_description': version.change_description,
                    'parent_version': version.parent_version
                })

        with open(self.registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)

        logger.info(f"Registry saved to {self.registry_path}")

    def load_registry(self):
        """Load registry from file"""
        if not self.registry_path.exists():
            logger.warning(f"Registry file {self.registry_path} not found")
            return

        with open(self.registry_path, 'r') as f:
            registry_data = json.load(f)

        self.features = {}
        self.current_versions = registry_data.get('current_versions', {})

        for feature_name, versions_data in registry_data.get('features', {}).items():
            self.features[feature_name] = []
            for version_data in versions_data:
                metadata = FeatureMetadata(
                    name=version_data['metadata']['name'],
                    feature_type=FeatureType(version_data['metadata']['feature_type']),
                    dtype=version_data['metadata']['dtype'],
                    null_count=version_data['metadata']['null_count'],
                    null_percentage=version_data['metadata']['null_percentage'],
                    unique_count=version_data['metadata']['unique_count'],
                    min_value=version_data['metadata']['min_value'],
                    max_value=version_data['metadata']['max_value'],
                    mean_value=version_data['metadata']['mean_value'],
                    std_value=version_data['metadata']['std_value'],
                    categories=version_data['metadata']['categories'],
                    is_target=version_data['metadata']['is_target'],
                    is_identifier=version_data['metadata']['is_identifier'],
                    created_at=datetime.fromisoformat(version_data['metadata']['created_at']),
                    updated_at=datetime.fromisoformat(version_data['metadata']['updated_at'])
                )

                version = FeatureVersion(
                    version_id=version_data['version_id'],
                    timestamp=datetime.fromisoformat(version_data['timestamp']),
                    metadata=metadata,
                    change_description=version_data['change_description'],
                    parent_version=version_data['parent_version']
                )

                self.features[feature_name].append(version)

        logger.info(f"Registry loaded from {self.registry_path}")

    def _generate_version_id(self, feature_name: str, metadata: FeatureMetadata) -> str:
        """Generate unique version ID for a feature"""
        content = f"{feature_name}_{metadata.feature_type.value}_{metadata.dtype}_{metadata.null_count}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def export_summary(self, export_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export registry summary

        Args:
            export_path: Path to save summary (optional)

        Returns:
            Summary dictionary
        """
        summary = {
            'export_timestamp': datetime.now().isoformat(),
            'total_features': len(self.features),
            'total_versions': sum(len(versions) for versions in self.features.values()),
            'feature_types': {},
            'high_null_features': [],
            'target_features': [],
            'identifier_features': []
        }

        # Count by type
        for feature_name, versions in self.features.items():
            current_version = self.current_versions.get(feature_name)
            if current_version:
                for version in versions:
                    if version.version_id == current_version:
                        metadata = version.metadata

                        # Count by type
                        feature_type = metadata.feature_type.value
                        summary['feature_types'][feature_type] = summary['feature_types'].get(feature_type, 0) + 1

                        # High null features
                        if metadata.null_percentage > 0.5:
                            summary['high_null_features'].append(feature_name)

                        # Target features
                        if metadata.is_target:
                            summary['target_features'].append(feature_name)

                        # Identifier features
                        if metadata.is_identifier:
                            summary['identifier_features'].append(feature_name)

                        break

        if export_path:
            with open(export_path, 'w') as f:
                json.dump(summary, f, indent=2)

        return summary