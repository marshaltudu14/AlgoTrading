"""
GPU Optimization Utilities for Training
"""
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """Handles GPU optimization for training pipeline"""
    
    def __init__(self, device: torch.device, gpu_config: Dict):
        self.device = device
        self.gpu_config = gpu_config
        self.scaler = None
        
        if self.device.type == 'cuda':
            self._init_gpu_optimizations()
    
    def _init_gpu_optimizations(self):
        """Initialize GPU-specific optimizations"""
        logger.info("Initializing GPU optimizations")
        
        # Mixed precision training
        if self.gpu_config.get('mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("  - Mixed precision (AMP) enabled")
        
        # Memory efficient attention
        if self.gpu_config.get('memory_efficient_attention', False):
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("  - Memory efficient attention enabled")
        
        # Tensor core optimization
        if self.gpu_config.get('tensorcore_optimization', False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("  - TensorCore optimization enabled")
        
        # CuDNN benchmarking
        if self.gpu_config.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
            logger.info("  - CuDNN benchmarking enabled")
        
        # Memory management
        if self.gpu_config.get('pin_memory', False):
            logger.info("  - Pinned memory enabled")
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Memory fraction
        max_memory_gb = self.gpu_config.get('max_batch_memory_gb', 12)
        logger.info(f"  - Target memory usage: {max_memory_gb}GB")
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if self.device.type != 'cuda':
            return base_batch_size
        
        try:
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            max_memory_gb = self.gpu_config.get('max_batch_memory_gb', 12)
            target_memory = min(max_memory_gb * 1024**3, total_memory * 0.8)  # Use 80% of available
            
            # Estimate memory per sample (rough approximation)
            memory_per_sample = 1024 * 1024 * 2  # 2MB per sample estimate
            optimal_batch_size = int(target_memory / memory_per_sample)
            
            # Clamp to reasonable range
            optimal_batch_size = max(4, min(optimal_batch_size, 256))
            
            logger.info(f"Calculated optimal batch size: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to calculate optimal batch size: {e}")
            return base_batch_size
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply model optimizations"""
        if self.device.type != 'cuda':
            return model
        
        # Torch compile optimization
        if self.gpu_config.get('torch_compile', False) and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                logger.info("Model optimized with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        return model
    
    def create_mixed_precision_context(self):
        """Create mixed precision context if available"""
        if self.scaler is not None:
            return torch.cuda.amp.autocast()
        else:
            return torch.no_grad()  # Fallback context
    
    def scale_loss_and_backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Handle mixed precision loss scaling and backward pass"""
        if self.scaler is not None:
            # Mixed precision
            self.scaler.scale(loss).backward()
        else:
            # Standard precision
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer, 
                      gradient_clip_norm: Optional[float] = None,
                      model_parameters=None):
        """Handle optimizer step with mixed precision"""
        if self.scaler is not None:
            # Mixed precision optimizer step
            if gradient_clip_norm and model_parameters:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_parameters, gradient_clip_norm)
            
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard precision optimizer step
            if gradient_clip_norm and model_parameters:
                torch.nn.utils.clip_grad_norm_(model_parameters, gradient_clip_norm)
            
            optimizer.step()


class OfflineRLPreprocessor:
    """Handles offline RL preprocessing for batch computation"""
    
    def __init__(self, device: torch.device, gpu_config: Dict):
        self.device = device
        self.gpu_config = gpu_config
        self.offline_config = gpu_config.get('offline_rl', {})
    
    def is_enabled(self) -> bool:
        """Check if offline RL preprocessing is enabled"""
        return self.offline_config.get('enabled', False)
    
    def should_precompute_episodes(self) -> bool:
        """Check if episodes should be precomputed"""
        return self.offline_config.get('precompute_episodes', False)
    
    def should_cache_observations(self) -> bool:
        """Check if observations should be cached"""
        return self.offline_config.get('cache_observations', False)
    
    def should_cache_rewards(self) -> bool:
        """Check if rewards should be cached"""
        return self.offline_config.get('cache_rewards', False)
    
    def should_batch_episode_processing(self) -> bool:
        """Check if episode processing should be batched"""
        return self.offline_config.get('batch_episode_processing', False)
    
    def precompute_dataset_batch(self, 
                               datasets: Dict[str, pd.DataFrame],
                               observation_handler,
                               episode_length: int) -> Dict[str, torch.Tensor]:
        """Precompute observations for entire dataset batch"""
        if not self.should_precompute_episodes():
            return {}
        
        logger.info(f"Precomputing dataset batch for {len(datasets)} instruments")
        
        batch_data = {}
        
        for symbol, data in datasets.items():
            try:
                # Calculate number of episodes
                num_episodes = max(1, (len(data) - 1) // episode_length)
                
                # Precompute observations for this dataset
                episode_observations = []
                
                for episode_idx in range(num_episodes):
                    start_idx = episode_idx * episode_length
                    end_idx = min(start_idx + episode_length, len(data))
                    
                    episode_data = data.iloc[start_idx:end_idx]
                    
                    # Precompute observations for this episode
                    obs_sequence = []
                    lookback = getattr(observation_handler, 'lookback_window', 50)
                    
                    for step in range(lookback, len(episode_data)):
                        try:
                            obs = observation_handler.get_observation(episode_data, step, None)
                            obs_sequence.append(obs)
                        except Exception as e:
                            logger.warning(f"Failed to compute observation at step {step}: {e}")
                            continue
                    
                    if obs_sequence:
                        episode_observations.append(torch.tensor(obs_sequence, 
                                                               dtype=torch.float32, 
                                                               device=self.device))
                
                if episode_observations:
                    # Stack all episodes for this symbol
                    symbol_data = torch.stack(episode_observations)
                    batch_data[symbol] = symbol_data
                    
                    logger.info(f"Precomputed {symbol}: {symbol_data.shape} "
                              f"({num_episodes} episodes)")
                
            except Exception as e:
                logger.error(f"Failed to precompute data for {symbol}: {e}")
        
        total_memory = sum(tensor.numel() * tensor.element_size() 
                          for tensor in batch_data.values()) / 1024**3
        logger.info(f"Total precomputed data: {total_memory:.2f} GB on GPU")
        
        return batch_data
    
    def create_vectorized_batch(self, 
                              batch_data: Dict[str, torch.Tensor],
                              batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Create vectorized batches for parallel processing"""
        if not self.should_batch_episode_processing():
            return []
        
        all_episodes = []
        episode_metadata = []
        
        # Collect all episodes from all instruments
        for symbol, symbol_data in batch_data.items():
            num_episodes = symbol_data.shape[0]
            for episode_idx in range(num_episodes):
                all_episodes.append(symbol_data[episode_idx])
                episode_metadata.append({
                    'symbol': symbol,
                    'episode_idx': episode_idx
                })
        
        # Create batches
        batches = []
        for i in range(0, len(all_episodes), batch_size):
            batch_episodes = all_episodes[i:i+batch_size]
            batch_meta = episode_metadata[i:i+batch_size]
            
            # Pad sequences to same length
            max_length = max(ep.shape[0] for ep in batch_episodes)
            padded_episodes = []
            
            for episode in batch_episodes:
                if episode.shape[0] < max_length:
                    padding = torch.zeros((max_length - episode.shape[0], episode.shape[1]),
                                        device=episode.device, dtype=episode.dtype)
                    padded_episode = torch.cat([episode, padding], dim=0)
                else:
                    padded_episode = episode
                padded_episodes.append(padded_episode)
            
            # Stack into batch
            batch_tensor = torch.stack(padded_episodes)
            
            batches.append({
                'episodes': batch_tensor,
                'metadata': batch_meta,
                'batch_size': len(batch_episodes)
            })
        
        logger.info(f"Created {len(batches)} vectorized batches")
        return batches