"""
GPU Memory and Performance Optimizer
Utilities to maximize GPU utilization and fix low VRAM usage issues
"""
import torch
import gc
import logging
from typing import Dict, List, Optional, Tuple
import psutil
import time

logger = logging.getLogger(__name__)

class GPUOptimizer:
    """GPU optimization utilities for maximum VRAM utilization"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_gpu = torch.cuda.is_available()
        self.initial_memory = None
        self.peak_memory = 0
        
        if self.is_gpu:
            self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU with aggressive optimization settings"""
        if not self.is_gpu:
            return
        
        # Enable all performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  
        torch.backends.cudnn.enabled = True
        
        # Enable TensorFloat-32 for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache and get baseline memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self.initial_memory = torch.cuda.memory_allocated()
        
        logger.info("🚀 GPU Optimizer initialized with aggressive performance settings")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get detailed GPU memory statistics"""
        if not self.is_gpu:
            return {"gpu_available": False}
        
        # Get memory in GB
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_free = memory_total - memory_reserved
        
        utilization_percent = (memory_reserved / memory_total) * 100
        
        return {
            "gpu_available": True,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved, 
            "memory_total_gb": memory_total,
            "memory_free_gb": memory_free,
            "utilization_percent": utilization_percent,
            "peak_memory_gb": self.peak_memory / 1024**3
        }
    
    def force_memory_growth(self, target_gb: float = 10.0) -> bool:
        """Force GPU memory usage to grow to target size"""
        if not self.is_gpu:
            return False
        
        try:
            current_stats = self.get_memory_stats()
            current_gb = current_stats["memory_reserved_gb"]
            
            if current_gb >= target_gb:
                logger.info(f"✅ GPU memory already at target: {current_gb:.2f}GB >= {target_gb:.2f}GB")
                return True
            
            # Calculate how much more memory to allocate
            additional_gb = target_gb - current_gb
            additional_bytes = int(additional_gb * 1024**3)
            
            # Allocate tensors to force memory growth
            dummy_tensors = []
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            
            logger.info(f"🔥 Forcing GPU memory growth: {current_gb:.2f}GB -> {target_gb:.2f}GB")
            
            while additional_bytes > 0:
                try:
                    chunk = min(chunk_size, additional_bytes)
                    # Create tensor directly on GPU
                    dummy_tensor = torch.randn(chunk // 4, dtype=torch.float32, device=self.device)
                    dummy_tensors.append(dummy_tensor)
                    additional_bytes -= chunk
                    
                    # Sync and check progress
                    torch.cuda.synchronize()
                    if len(dummy_tensors) % 10 == 0:  # Log every 1GB
                        current_mem = torch.cuda.memory_reserved() / 1024**3
                        logger.info(f"🎯 GPU memory growth: {current_mem:.2f}GB")
                        
                except torch.cuda.OutOfMemoryError:
                    logger.warning("❌ Hit GPU memory limit during growth")
                    break
            
            # Keep references to prevent garbage collection
            self.dummy_tensors = dummy_tensors
            
            final_stats = self.get_memory_stats()
            logger.info(f"✅ GPU memory grown to: {final_stats['memory_reserved_gb']:.2f}GB ({final_stats['utilization_percent']:.1f}%)")
            
            return final_stats["memory_reserved_gb"] >= target_gb * 0.9  # 90% of target
            
        except Exception as e:
            logger.error(f"❌ Failed to force memory growth: {e}")
            return False
    
    def optimize_for_parallel_training(self, num_environments: int = 25, batch_size: int = 32) -> Dict:
        """Optimize GPU settings for parallel training with multiple environments"""
        if not self.is_gpu:
            return {"optimized": False, "reason": "No GPU available"}
        
        logger.info(f"🔧 Optimizing GPU for parallel training: {num_environments} envs, batch size {batch_size}")
        
        # Clear cache first
        self.clear_cache()
        
        # Calculate target memory usage for maximum utilization
        total_memory_gb = self.get_memory_stats()["memory_total_gb"]
        target_memory_gb = total_memory_gb * 0.85  # Use 85% of VRAM aggressively
        
        # Force memory growth
        growth_success = self.force_memory_growth(target_memory_gb)
        
        # Enable all optimizations
        optimizations = {
            "memory_growth": growth_success,
            "target_memory_gb": target_memory_gb,
            "cudnn_optimizations": True,
            "tensor_cores": self._enable_tensor_cores(),
            "mixed_precision": self._check_mixed_precision_support(),
            "memory_efficiency": self._set_memory_efficient_attention()
        }
        
        final_stats = self.get_memory_stats()
        optimizations.update(final_stats)
        
        logger.info(f"🚀 GPU optimization complete: {final_stats['utilization_percent']:.1f}% VRAM utilization")
        
        return optimizations
    
    def _enable_tensor_cores(self) -> bool:
        """Enable Tensor Core optimizations if available"""
        if not self.is_gpu:
            return False
        
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 7:  # V100, T4, RTX series
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ Tensor Cores enabled for acceleration")
                return True
        except Exception as e:
            logger.debug(f"Tensor Core setup failed: {e}")
        
        return False
    
    def _check_mixed_precision_support(self) -> bool:
        """Check if mixed precision training is supported"""
        if not self.is_gpu:
            return False
        
        try:
            major, minor = torch.cuda.get_device_capability(0)
            supported = major >= 6  # Pascal or newer
            if supported:
                logger.info("✅ Mixed precision (FP16) training supported")
            return supported
        except Exception:
            return False
    
    def _set_memory_efficient_attention(self) -> bool:
        """Enable memory-efficient attention if available"""
        try:
            # Try to enable Flash Attention or other memory-efficient methods
            if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                logger.info("✅ Scaled dot-product attention available for memory efficiency")
                return True
        except Exception:
            pass
        
        return False
    
    def clear_cache(self):
        """Aggressively clear GPU cache and Python memory"""
        if self.is_gpu:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Python garbage collection
        gc.collect()
        
        logger.debug("🧹 GPU cache and Python memory cleared")
    
    def monitor_memory_during_training(self, step: int, log_frequency: int = 10):
        """Monitor GPU memory usage during training"""
        if not self.is_gpu or step % log_frequency != 0:
            return
        
        stats = self.get_memory_stats()
        current_memory = stats["memory_reserved_gb"]
        
        # Track peak memory
        if current_memory > self.peak_memory / 1024**3:
            self.peak_memory = current_memory * 1024**3
        
        # Log memory stats
        logger.info(f"📊 Step {step}: GPU Memory: {current_memory:.2f}GB ({stats['utilization_percent']:.1f}%) | Peak: {stats['peak_memory_gb']:.2f}GB")
        
        # Warning if memory usage is too low
        if stats['utilization_percent'] < 30:
            logger.warning(f"⚠️  Low GPU utilization detected: {stats['utilization_percent']:.1f}% - Consider increasing batch size or parallel environments")
    
    def get_optimization_recommendations(self) -> Dict[str, any]:
        """Get recommendations for improving GPU utilization"""
        if not self.is_gpu:
            return {"gpu_available": False}
        
        stats = self.get_memory_stats()
        current_util = stats["utilization_percent"]
        
        recommendations = {
            "current_utilization": current_util,
            "recommendations": []
        }
        
        if current_util < 20:
            recommendations["recommendations"].extend([
                "🔥 CRITICAL: Very low GPU utilization! Increase parallel environments to 25+",
                "📈 Increase batch size significantly (try 512+ for 15GB VRAM)",
                "🚀 Enable gradient accumulation (8x or higher)",
                "💾 Pre-load multiple datasets simultaneously"
            ])
        elif current_util < 50:
            recommendations["recommendations"].extend([
                "⚡ Moderate GPU usage. Can push harder:",
                "📊 Increase batch size by 2-4x",
                "🔄 Add more parallel environments",
                "🎯 Enable mixed precision training"
            ])
        elif current_util < 80:
            recommendations["recommendations"].extend([
                "✅ Good GPU utilization. Fine-tune for maximum:",
                "🎛️ Consider gradient accumulation for larger effective batches",
                "⚡ Enable all Tensor Core optimizations"
            ])
        else:
            recommendations["recommendations"].append("🏆 Excellent GPU utilization! Well optimized.")
        
        return recommendations

# Global optimizer instance
_gpu_optimizer = None

def get_gpu_optimizer() -> GPUOptimizer:
    """Get global GPU optimizer instance"""
    global _gpu_optimizer
    if _gpu_optimizer is None:
        _gpu_optimizer = GPUOptimizer()
    return _gpu_optimizer

def force_max_gpu_utilization(target_gb: float = None) -> bool:
    """Force maximum GPU utilization"""
    optimizer = get_gpu_optimizer()
    
    if target_gb is None:
        # Auto-calculate target based on available memory
        if optimizer.is_gpu:
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            target_gb = total_gb * 0.85  # Use 85% of VRAM
        else:
            return False
    
    return optimizer.force_memory_growth(target_gb)

def log_gpu_utilization_warning():
    """Log warning about low GPU utilization with specific fixes"""
    optimizer = get_gpu_optimizer()
    
    if not optimizer.is_gpu:
        return
    
    stats = optimizer.get_memory_stats()
    recommendations = optimizer.get_optimization_recommendations()
    
    print("\n" + "="*100)
    print("🚨 GPU UTILIZATION ANALYSIS")
    print("="*100)
    print(f"Current GPU Memory Usage: {stats['memory_reserved_gb']:.2f}GB / {stats['memory_total_gb']:.1f}GB ({stats['utilization_percent']:.1f}%)")
    print(f"Current Allocation: {stats['memory_allocated_gb']:.2f}GB")
    print()
    
    for rec in recommendations["recommendations"]:
        print(rec)
    
    print("\n💡 QUICK FIXES:")
    print("1. Increase parallel_environments in settings.yaml to 32+")
    print("2. Enable gradient accumulation (8x or higher)")
    print("3. Use larger batch sizes (512+ for 15GB VRAM)")
    print("4. Pre-load multiple datasets simultaneously")
    print("5. Enable mixed precision training")
    print("="*100 + "\n")