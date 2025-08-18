"""
Hardware optimization utilities for efficient training.
"""

import torch
import torch.backends.cudnn as cudnn
import logging
import psutil
import gc
import os
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """
    Comprehensive hardware optimization for PyTorch training.
    """
    
    def __init__(self, enable_optimization: bool = True):
        self.enable_optimization = enable_optimization
        self.device = None
        self.device_info = {}
        self.optimal_batch_sizes = {}
        self.scaler = None  # For mixed precision training
        
        if self.enable_optimization:
            self._initialize_hardware()
    
    def _initialize_hardware(self) -> None:
        """Initialize hardware optimization settings."""
        # Detect and configure device
        self.device = self._get_optimal_device()
        self.device_info = self._get_device_info()
        
        # Enable cuDNN benchmark for performance
        if self.device.type == 'cuda':
            self._enable_cudnn_optimization()
        
        # Log hardware configuration
        self._log_hardware_info()
    
    def _get_optimal_device(self) -> torch.device:
        """
        Get the optimal device for training with graceful fallback.
        Priority: TPU > CUDA > MPS > CPU
        
        Returns:
            torch.device: Optimal device (TPU, CUDA, MPS, or CPU)
        """
        # Check for TPU availability first
        if 'COLAB_TPU_ADDR' in os.environ or 'TPU_NAME' in os.environ:
            try:
                import torch_xla.core.xla_model as xm
                device = xm.xla_device()
                logger.info("TPU detected and configured")
                return device
            except ImportError:
                logger.warning("TPU environment detected but PyTorch XLA not available")
            except Exception as e:
                logger.warning(f"TPU initialization failed: {e}")
        
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"CUDA available: Using GPU {torch.cuda.get_device_name()}")
            return device
        
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("MPS available: Using Apple Silicon GPU")
            return device
        
        # Fallback to CPU
        device = torch.device('cpu')
        logger.info("GPU not available: Using CPU")
        return device
    
    def _get_device_info(self) -> Dict[str, Any]:
        """Get detailed device information."""
        info = {
            'device_type': self.device.type,
            'device_name': str(self.device),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
                'gpu_compute_capability': torch.cuda.get_device_capability(),
                'cudnn_version': torch.backends.cudnn.version(),
                'cuda_version': torch.version.cuda
            })
        elif 'xla' in str(self.device):
            # TPU device
            info.update({
                'tpu_detected': True,
                'tpu_env': os.environ.get('COLAB_TPU_ADDR', os.environ.get('TPU_NAME', 'Unknown'))
            })
        
        return info
    
    def _enable_cudnn_optimization(self) -> None:
        """Enable cuDNN optimizations for CUDA devices."""
        # Only enable cuDNN for CUDA devices, not TPU
        if self.device.type == 'cuda' and torch.backends.cudnn.is_available():
            # Enable benchmark mode for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable deterministic mode if needed (can be slower)
            # torch.backends.cudnn.deterministic = True
            
            logger.info("cuDNN optimization enabled: benchmark=True")
        elif self.device.type != 'cuda':
            logger.info(f"cuDNN optimization skipped for {self.device.type} device")
        else:
            logger.warning("cuDNN not available")
    
    def enable_mixed_precision(self) -> None:
        """Enable mixed precision training for better performance on compatible hardware."""
        if self.device.type == 'cuda':
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("Mixed precision training not available (requires PyTorch 1.6+)")
        elif 'xla' in str(self.device):
            # TPU uses bfloat16 by default, no additional setup needed
            logger.info("TPU uses bfloat16 by default, mixed precision enabled")
        else:
            logger.info("Mixed precision training only available on CUDA devices")
    
    def is_mixed_precision_enabled(self) -> bool:
        """Check if mixed precision training is enabled."""
        return self.scaler is not None
    
    def _log_hardware_info(self) -> None:
        """Log comprehensive hardware information."""
        logger.info("=== Hardware Configuration ===")
        logger.info(f"Device: {self.device_info['device_name']}")
        logger.info(f"CPU cores: {self.device_info['cpu_count']}")
        logger.info(f"System memory: {self.device_info['memory_total_gb']:.1f} GB")
        
        if self.device.type == 'cuda':
            logger.info(f"GPU: {self.device_info['gpu_name']}")
            logger.info(f"GPU memory: {self.device_info['gpu_memory_total_gb']:.1f} GB")
            logger.info(f"CUDA version: {self.device_info['cuda_version']}")
            logger.info(f"cuDNN version: {self.device_info['cudnn_version']}")
        elif 'xla' in str(self.device):
            logger.info(f"TPU environment: {self.device_info.get('tpu_env', 'Detected')}")
        
        logger.info("==============================")
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Optimize model for the target device.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model moved to optimal device
        """
        if not self.enable_optimization:
            return model
        
        # Move model to optimal device
        model = model.to(self.device)
        
        # Enable mixed precision for CUDA
        if self.device.type == 'cuda':
            # Convert model to half precision for compatible layers
            # model = model.half()  # Uncomment if needed
            pass
        
        # Enable multi-GPU training if multiple GPUs are available
        if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        
        # Compile model for PyTorch 2.0+ (if available and C++ compiler present)
        # Disabled for now due to C++ compiler requirements
        # if hasattr(torch, 'compile'):
        #     try:
        #         model = torch.compile(model)
        #         logger.info("Model compiled with torch.compile for optimization")
        #     except Exception as e:
        #         logger.warning(f"Model compilation failed: {e}")
        
        logger.info(f"Model optimized and moved to {self.device}")
        return model
    
    def optimize_tensor(self, tensor: torch.Tensor, non_blocking: bool = True) -> torch.Tensor:
        """
        Optimize tensor for the target device.
        
        Args:
            tensor: Tensor to optimize
            non_blocking: Use non-blocking transfer for CUDA
            
        Returns:
            Optimized tensor moved to optimal device
        """
        if not self.enable_optimization:
            return tensor
        
        # For TPU, we don't use non-blocking transfers
        if 'xla' in str(self.device):
            return tensor.to(self.device)
        
        return tensor.to(self.device, non_blocking=non_blocking and self.device.type == 'cuda')
    
    def get_optimal_batch_size(self, model_name: str, base_batch_size: int = 32, 
                              memory_fraction: float = 0.8) -> int:
        """
        Calculate optimal batch size for the current hardware.
        
        Args:
            model_name: Name of the model for caching
            base_batch_size: Base batch size to start from
            memory_fraction: Fraction of GPU memory to use
            
        Returns:
            Optimal batch size
        """
        if model_name in self.optimal_batch_sizes:
            return self.optimal_batch_sizes[model_name]
        
        if self.device.type == 'cuda':
            optimal_size = self._calculate_gpu_batch_size(base_batch_size, memory_fraction)
        elif 'xla' in str(self.device):
            # TPU typically uses larger batch sizes
            optimal_size = self._calculate_tpu_batch_size(base_batch_size)
        else:
            optimal_size = self._calculate_cpu_batch_size(base_batch_size)
        
        self.optimal_batch_sizes[model_name] = optimal_size
        logger.info(f"Optimal batch size for {model_name}: {optimal_size}")
        
        return optimal_size
    
    def _calculate_gpu_batch_size(self, base_batch_size: int, memory_fraction: float) -> int:
        """Calculate optimal batch size for GPU."""
        try:
            # Get available GPU memory
            gpu_memory_gb = self.device_info.get('gpu_memory_total_gb', 4.0)
            available_memory = gpu_memory_gb * memory_fraction
            
            # Estimate memory per sample (rough heuristic)
            memory_per_sample_mb = 10  # Adjust based on model complexity
            max_batch_size = int((available_memory * 1024) / memory_per_sample_mb)
            
            # Find largest power of 2 that fits
            optimal_size = base_batch_size
            while optimal_size <= max_batch_size and optimal_size < 1024:
                optimal_size *= 2
            
            return optimal_size // 2  # Use previous power of 2
            
        except Exception as e:
            logger.warning(f"GPU batch size calculation failed: {e}")
            return base_batch_size
    
    def _calculate_cpu_batch_size(self, base_batch_size: int) -> int:
        """Calculate optimal batch size for CPU."""
        try:
            # Base on CPU cores and available memory
            cpu_cores = self.device_info.get('cpu_count', 4)
            memory_gb = self.device_info.get('memory_total_gb', 8.0)
            
            # Conservative approach for CPU
            if memory_gb >= 16:
                return min(base_batch_size * 2, 128)
            elif memory_gb >= 8:
                return base_batch_size
            else:
                return max(base_batch_size // 2, 8)
                
        except Exception as e:
            logger.warning(f"CPU batch size calculation failed: {e}")
            return base_batch_size
    
    def _calculate_tpu_batch_size(self, base_batch_size: int) -> int:
        """Calculate optimal batch size for TPU."""
        try:
            # TPUs typically benefit from larger batch sizes
            # Start with 2x the base batch size and scale up
            tpu_batch_size = base_batch_size * 4
            
            # Cap at reasonable maximum for memory constraints
            return min(tpu_batch_size, 1024)
            
        except Exception as e:
            logger.warning(f"TPU batch size calculation failed: {e}")
            return base_batch_size * 2
    
    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        elif 'xla' in str(self.device):
            try:
                import torch_xla.core.xla_model as xm
                xm.mark_step()  # Ensure all computations are finished
                logger.debug("TPU step marked and cache managed")
            except ImportError:
                pass
        
        gc.collect()
        logger.debug("Garbage collection completed")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        usage = {
            'cpu_memory_percent': psutil.virtual_memory().percent,
            'cpu_memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        if self.device.type == 'cuda':
            usage.update({
                'gpu_memory_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_memory_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'gpu_memory_percent': (torch.cuda.memory_allocated() / 
                                     torch.cuda.get_device_properties(0).total_memory) * 100
            })
        elif 'xla' in str(self.device):
            # TPU memory tracking is limited, just indicate TPU usage
            usage.update({
                'tpu_in_use': True
            })
        
        return usage
    
    def monitor_performance(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """
        Monitor performance of a function execution.
        
        Args:
            func: Function to monitor
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (function result, performance metrics)
        """
        import time
        
        # Clear cache before monitoring
        self.clear_cache()
        
        # Record initial state
        start_time = time.time()
        initial_memory = self.get_memory_usage()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record final state
        end_time = time.time()
        final_memory = self.get_memory_usage()
        
        # Calculate metrics
        metrics = {
            'execution_time': end_time - start_time,
            'cpu_memory_delta_gb': final_memory['cpu_memory_used_gb'] - initial_memory['cpu_memory_used_gb']
        }
        
        if self.device.type == 'cuda':
            metrics['gpu_memory_delta_gb'] = (final_memory['gpu_memory_allocated_gb'] - 
                                            initial_memory['gpu_memory_allocated_gb'])
        
        return result, metrics

# Global hardware optimizer instance
_hardware_optimizer = None

def get_hardware_optimizer(enable_optimization: bool = True) -> HardwareOptimizer:
    """Get global hardware optimizer instance."""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer(enable_optimization)
    return _hardware_optimizer

def optimize_for_device(model: torch.nn.Module) -> torch.nn.Module:
    """Convenience function to optimize model for current device."""
    optimizer = get_hardware_optimizer()
    return optimizer.optimize_model(model)

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    """Convenience function to move tensor to optimal device."""
    optimizer = get_hardware_optimizer()
    return optimizer.optimize_tensor(tensor)
