"""
Device Manager for automatic TPU/GPU/CPU selection
Prioritizes TPU > GPU > CPU for optimal training performance
"""
import torch
import logging
import os
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages automatic device selection and optimization"""
    
    def __init__(self):
        self.device = None
        self.device_type = None
        self.device_info = {}
        self._detect_best_device()
    
    def _detect_best_device(self):
        """Detect and select the best available device"""
        
        # Check for TPU first (highest priority)
        if self._check_tpu_available():
            self.device = torch.device('xla')
            self.device_type = 'tpu'
            self._setup_tpu()
            logger.info("TPU detected and selected for training")
            
        # Check for GPU (second priority)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_type = 'gpu'
            self._setup_gpu()
            logger.info(f"GPU detected and selected: {torch.cuda.get_device_name()}")
            
        # Fallback to CPU
        else:
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            self._setup_cpu()
            logger.info("Using CPU for training (TPU/GPU not available)")
    
    def _check_tpu_available(self) -> bool:
        """Check if TPU is available"""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            
            # Try to get TPU device
            device = xm.xla_device()
            self.device_info['tpu_cores'] = xm.xrt_world_size()
            return True
            
        except ImportError:
            logger.debug("torch_xla not installed - TPU not available")
            return False
        except Exception as e:
            logger.debug(f"TPU detection failed: {e}")
            return False
    
    def _setup_tpu(self):
        """Configure TPU-specific settings"""
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            
            self.device_info.update({
                'world_size': xm.xrt_world_size(),
                'ordinal': xm.get_ordinal(),
                'local_ordinal': xm.get_local_ordinal()
            })
            
            # TPU-specific optimizations
            os.environ['XLA_USE_BF16'] = '1'  # Enable bfloat16 for better performance
            
        except Exception as e:
            logger.warning(f"TPU setup failed: {e}")
    
    def _setup_gpu(self):
        """Configure GPU-specific settings"""
        try:
            # Get GPU information
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            
            # Get memory info
            memory_total = torch.cuda.get_device_properties(current_gpu).total_memory
            memory_reserved = torch.cuda.memory_reserved(current_gpu)
            memory_allocated = torch.cuda.memory_allocated(current_gpu)
            memory_free = memory_total - memory_reserved
            
            self.device_info.update({
                'gpu_count': gpu_count,
                'current_gpu': current_gpu,
                'gpu_name': torch.cuda.get_device_name(current_gpu),
                'memory_total': memory_total,
                'memory_total_gb': memory_total / (1024**3),
                'memory_free': memory_free,
                'memory_free_gb': memory_free / (1024**3),
                'compute_capability': torch.cuda.get_device_properties(current_gpu).major
            })
            
            # Enable high-performance optimizations
            torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            torch.backends.cudnn.enabled = True  # Enable cuDNN
            
            # Enable mixed precision if supported (Tensor Cores)
            if torch.cuda.get_device_capability(current_gpu)[0] >= 7:  # V100, T4, RTX series
                self.device_info['mixed_precision_supported'] = True
                self.device_info['tensor_cores'] = True
                logger.info(f"Tensor Cores detected! Mixed precision training available")
            else:
                self.device_info['mixed_precision_supported'] = False
                self.device_info['tensor_cores'] = False
            
            # Clear GPU cache for maximum memory
            torch.cuda.empty_cache()
            
            # Multi-GPU setup if available
            if gpu_count > 1:
                logger.info(f"Multiple GPUs detected ({gpu_count}). Consider using DataParallel.")
                self.device_info['multi_gpu'] = True
            
            logger.info(f"GPU Memory: {self.device_info['memory_total_gb']:.1f}GB total, {self.device_info['memory_free_gb']:.1f}GB available")
            
        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")
    
    def _setup_cpu(self):
        """Configure CPU-specific settings"""
        try:
            # CPU optimizations
            torch.set_num_threads(os.cpu_count())  # Use all CPU cores
            
            self.device_info.update({
                'cpu_cores': os.cpu_count(),
                'num_threads': torch.get_num_threads()
            })
            
        except Exception as e:
            logger.warning(f"CPU setup failed: {e}")
    
    def get_device(self) -> torch.device:
        """Get the selected device"""
        return self.device
    
    def get_device_type(self) -> str:
        """Get device type string"""
        return self.device_type
    
    def get_device_info(self) -> dict:
        """Get detailed device information"""
        return self.device_info
    
    def move_to_device(self, tensor_or_model):
        """Move tensor or model to the selected device"""
        if self.device_type == 'tpu':
            # TPU requires special handling
            import torch_xla.core.xla_model as xm
            return tensor_or_model.to(xm.xla_device())
        else:
            return tensor_or_model.to(self.device)
    
    def optimize_model(self, model):
        """Apply device-specific optimizations to model"""
        model = self.move_to_device(model)
        
        if self.device_type == 'gpu':
            # GPU-specific optimizations
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                try:
                    model = torch.compile(model)  # PyTorch 2.0+ optimization
                    logger.info("Model compiled for GPU optimization")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
            
            # Mixed precision support
            if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
                logger.info("Mixed precision training supported")
                self.device_info['mixed_precision'] = True
        
        elif self.device_type == 'tpu':
            # TPU-specific optimizations
            try:
                import torch_xla.core.xla_model as xm
                # Mark step for TPU optimization
                xm.mark_step()
                logger.info("TPU model optimizations applied")
            except Exception as e:
                logger.warning(f"TPU optimization failed: {e}")
        
        return model
    
    def get_batch_size_recommendation(self, base_batch_size: int = 32) -> int:
        """Get recommended batch size based on device and memory"""
        if self.device_type == 'tpu':
            # TPU works best with larger batch sizes (multiples of 8 per core)
            tpu_cores = self.device_info.get('tpu_cores', 8)
            return max(base_batch_size, tpu_cores * 8)
            
        elif self.device_type == 'gpu':
            # Optimize for available GPU memory
            memory_gb = self.device_info.get('memory_total_gb', 0)
            
            # For 15GB VRAM (T4/V100 in Colab/Kaggle)
            if memory_gb >= 14:  # 15GB VRAM
                return base_batch_size * 8  # 256 batch size for 32 base
            elif memory_gb >= 12:  # 12GB+ VRAM
                return base_batch_size * 6  # 192 batch size
            elif memory_gb >= 8:   # 8GB+ VRAM
                return base_batch_size * 4  # 128 batch size
            elif memory_gb >= 4:   # 4GB+ VRAM
                return base_batch_size * 2  # 64 batch size
            else:  # Entry-level GPU
                return base_batch_size
                
        else:  # CPU
            # Smaller batch sizes for CPU
            return max(base_batch_size // 2, 8)
    
    def get_optimal_training_config(self) -> dict:
        """Get optimal training configuration for current hardware"""
        config = {
            'device': str(self.device),
            'device_type': self.device_type,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            'dataloader_workers': 2,
            'pin_memory': False
        }
        
        if self.device_type == 'gpu':
            memory_gb = self.device_info.get('memory_total_gb', 0)
            
            # Mixed precision for Tensor Core GPUs
            config['mixed_precision'] = self.device_info.get('mixed_precision_supported', False)
            
            # Gradient accumulation for very large effective batch sizes
            if memory_gb >= 14:  # 15GB VRAM
                config['gradient_accumulation_steps'] = 4  # Effective batch size = batch_size * 4
            elif memory_gb >= 12:
                config['gradient_accumulation_steps'] = 2
            
            # Optimize data loading
            config['dataloader_workers'] = min(8, max(2, int(memory_gb // 2)))  # Scale with memory
            config['pin_memory'] = True  # Faster GPU transfer
            config['prefetch_factor'] = 2
            config['persistent_workers'] = True
            
            # Memory optimization settings
            config['memory_optimization'] = {
                'empty_cache_frequency': 10,  # Clear cache every N batches
                'max_memory_fraction': 0.9,   # Use 90% of VRAM
                'gradient_checkpointing': memory_gb < 8  # Enable for low memory
            }
            
        elif self.device_type == 'tpu':
            config['gradient_accumulation_steps'] = 8
            config['dataloader_workers'] = 4
            
        return config
    
    def print_device_summary(self):
        """Print detailed device information"""
        print("\n" + "="*80)
        print("HIGH-PERFORMANCE DEVICE CONFIGURATION")
        print("="*80)
        print(f"Selected Device: {self.device}")
        print(f"Device Type: {self.device_type.upper()}")
        
        if self.device_type == 'tpu':
            print(f"TPU Cores: {self.device_info.get('tpu_cores', 'Unknown')}")
            print(f"World Size: {self.device_info.get('world_size', 'Unknown')}")
            
        elif self.device_type == 'gpu':
            print(f"GPU Name: {self.device_info.get('gpu_name', 'Unknown')}")
            print(f"GPU Memory: {self.device_info.get('memory_total_gb', 0):.1f} GB total")
            print(f"Available Memory: {self.device_info.get('memory_free_gb', 0):.1f} GB")
            print(f"GPU Count: {self.device_info.get('gpu_count', 1)}")
            print(f"Compute Capability: {self.device_info.get('compute_capability', 'Unknown')}")
            print(f"Tensor Cores: {'Available' if self.device_info.get('tensor_cores', False) else 'Not Available'}")
            print(f"Mixed Precision: {'Enabled' if self.device_info.get('mixed_precision_supported', False) else 'Disabled'}")
            
            # Performance recommendations
            config = self.get_optimal_training_config()
            print(f"\nPERFORMANCE OPTIMIZATIONS:")
            print(f"Recommended Batch Size: {self.get_batch_size_recommendation(32)}")
            print(f"Gradient Accumulation: {config['gradient_accumulation_steps']}x")
            print(f"DataLoader Workers: {config['dataloader_workers']}")
            print(f"Pin Memory: {'Enabled' if config['pin_memory'] else 'Disabled'}")
            if config['mixed_precision']:
                print(f"Mixed Precision: FP16 Enabled (Tensor Cores)")
            
        else:  # CPU
            print(f"CPU Cores: {self.device_info.get('cpu_cores', 'Unknown')}")
            print(f"CPU Threads: {self.device_info.get('num_threads', 'Unknown')}")
            print(f"WARNING: Using CPU instead of GPU - Training will be significantly slower!")
        
        print("="*80 + "\n")


# Global device manager instance
_device_manager = None

def get_device_manager() -> DeviceManager:
    """Get the global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager

def get_optimal_device() -> torch.device:
    """Get the optimal device for training"""
    return get_device_manager().get_device()

def move_to_optimal_device(tensor_or_model):
    """Move tensor or model to optimal device"""
    return get_device_manager().move_to_device(tensor_or_model)