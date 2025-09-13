"""
Hardware utilities for GPU/CPU detection and management.
"""

import torch
import platform
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class HardwareDetector:
    """Detect and manage system hardware capabilities."""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_info = self._get_gpu_info()
        self.memory_info = self._get_memory_info()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        gpu_info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': None,
            'gpu_count': 0,
            'gpus': [],
            'mixed_precision_available': False
        }

        if gpu_info['cuda_available']:
            try:
                gpu_info['cuda_version'] = torch.version.cuda
                gpu_info['gpu_count'] = torch.cuda.device_count()

                for i in range(gpu_info['gpu_count']):
                    gpu = {
                        'id': i,
                        'name': torch.cuda.get_device_name(i),
                        'memory_total': torch.cuda.get_device_properties(i).total_memory,
                        'memory_allocated': torch.cuda.memory_allocated(i),
                        'memory_reserved': torch.cuda.memory_reserved(i),
                        'capability': torch.cuda.get_device_capability(i),
                    }
                    gpu_info['gpus'].append(gpu)

                # Check mixed precision support
                gpu_info['mixed_precision_available'] = (
                    torch.cuda.is_bf16_supported() or
                    any(gpu['capability'] >= (7, 0) for gpu in gpu_info['gpus'])
                )

            except Exception as e:
                logger.warning(f"Error getting GPU info: {e}")

        return gpu_info

    def _get_memory_info(self) -> Dict[str, Any]:
        """Get system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent,
            'swap_total': psutil.swap_memory().total,
            'swap_used': psutil.swap_memory().used,
        }

    def is_gpu_sufficient(self, min_memory_gb: int = 8) -> bool:
        """Check if GPU meets minimum requirements."""
        if not self.gpu_info['cuda_available']:
            return False

        for gpu in self.gpu_info['gpus']:
            memory_gb = gpu['memory_total'] / (1024**3)
            if memory_gb >= min_memory_gb:
                return True

        return False

    def get_recommended_batch_size(self, model_size_mb: int = 100) -> int:
        """Get recommended batch size based on available GPU memory."""
        if not self.gpu_info['cuda_available']:
            return 8  # Conservative CPU batch size

        available_memory_mb = 0
        for gpu in self.gpu_info['gpus']:
            available_memory_mb += (gpu['memory_total'] - gpu['memory_reserved']) / (1024**2)

        # Reserve 1GB for overhead and model
        usable_memory_mb = max(0, available_memory_mb - model_size_mb - 1024)

        # Estimate ~1MB per sample (rough approximation)
        recommended_batch_size = int(usable_memory_mb)

        return min(max(8, recommended_batch_size), 256)  # Clamp between 8 and 256

    def get_optimal_device(self) -> torch.device:
        """Get the optimal device for computation."""
        if self.gpu_info['cuda_available'] and self.gpu_info['gpu_count'] > 0:
            # Return the GPU with most available memory
            best_gpu = max(
                range(self.gpu_info['gpu_count']),
                key=lambda i: torch.cuda.memory_allocated(i)
            )
            return torch.device(f'cuda:{best_gpu}')

        return torch.device('cpu')

    def configure_mixed_precision(self) -> Dict[str, Any]:
        """Configure mixed precision training."""
        config = {
            'enabled': False,
            'dtype': None,
            'scaler': None
        }

        if self.gpu_info['mixed_precision_available']:
            try:
                # Try BF16 first, then FP16
                if torch.cuda.is_bf16_supported():
                    config['dtype'] = torch.bfloat16
                    config['enabled'] = True
                    logger.info("Using BF16 mixed precision")
                else:
                    config['dtype'] = torch.float16
                    config['enabled'] = True
                    config['scaler'] = torch.cuda.amp.GradScaler()
                    logger.info("Using FP16 mixed precision with GradScaler")
            except Exception as e:
                logger.warning(f"Failed to configure mixed precision: {e}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Get complete hardware information as dictionary."""
        return {
            'system': self.system_info,
            'gpu': self.gpu_info,
            'memory': self.memory_info,
            'optimal_device': str(self.get_optimal_device()),
            'recommended_batch_size': self.get_recommended_batch_size(),
            'mixed_precision_config': self.configure_mixed_precision(),
        }

def get_hardware_info() -> Dict[str, Any]:
    """Convenience function to get hardware information."""
    detector = HardwareDetector()
    return detector.to_dict()