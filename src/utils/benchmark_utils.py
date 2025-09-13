"""
Benchmarking utilities for performance testing.
"""

import time
import torch
import numpy as np
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

@contextmanager
def timer(name: str):
    """Context manager for timing operations."""
    start = time.time()
    yield
    end = time.time()
    logger.info(f"{name}: {end - start:.4f} seconds")

class PerformanceBenchmark:
    """Performance benchmarking utility."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def benchmark_matrix_operations(self, sizes: List[int] = [1000, 2000, 5000]):
        """Benchmark matrix operations."""
        results = {}

        for size in sizes:
            logger.info(f"Benchmarking matrix operations of size {size}x{size}")

            # Matrix multiplication
            with timer(f"Matrix multiplication {size}x{size}"):
                a = torch.randn(size, size, device=self.device)
                b = torch.randn(size, size, device=self.device)
                c = a @ b
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

            # SVD
            with timer(f"SVD {size}x{size}"):
                a = torch.randn(size, min(size, 100), device=self.device)
                u, s, v = torch.svd(a)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

            # Eigenvalue decomposition
            with timer(f"Eigenvalue decomposition {size}x{size}"):
                a = torch.randn(size, size, device=self.device)
                a = a @ a.T  # Make symmetric
                eigenvalues = torch.linalg.eigvals(a)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

        return results

    def benchmark_deep_learning_operations(self, batch_sizes: List[int] = [32, 64, 128]):
        """Benchmark deep learning operations."""
        results = {}

        for batch_size in batch_sizes:
            logger.info(f"Benchmarking DL operations with batch size {batch_size}")

            # Linear layer
            with timer(f"Linear layer forward {batch_size}"):
                x = torch.randn(batch_size, 512, device=self.device)
                linear = torch.nn.Linear(512, 1024).to(self.device)
                y = linear(x)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

            # Convolution
            with timer(f"Convolution {batch_size}"):
                x = torch.randn(batch_size, 3, 224, 224, device=self.device)
                conv = torch.nn.Conv2d(3, 64, 3, padding=1).to(self.device)
                y = conv(x)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

            # Transformer attention (simplified)
            with timer(f"Attention {batch_size}"):
                seq_len, d_model = 128, 512
                x = torch.randn(batch_size, seq_len, d_model, device=self.device)
                q = torch.nn.Linear(d_model, d_model).to(self.device)(x)
                k = torch.nn.Linear(d_model, d_model).to(self.device)(x)
                v = torch.nn.Linear(d_model, d_model).to(self.device)(x)

                # Simplified attention
                scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                output = torch.matmul(attn, v)
                torch.cuda.synchronize() if self.device.type == 'cuda' else None

        return results

    def benchmark_memory_bandwidth(self, sizes: List[int] = [1024, 2048, 4096]):
        """Benchmark memory bandwidth."""
        results = {}

        for size in sizes:
            logger.info(f"Benchmarking memory bandwidth with size {size}MB")

            # GPU memory bandwidth if available
            if self.device.type == 'cuda':
                # Test GPU memory bandwidth
                start = time.time()
                for _ in range(10):
                    x = torch.randn(size * 1024 * 256, device=self.device)  # ~size MB
                    y = x * 2
                torch.cuda.synchronize()
                end = time.time()
                bandwidth = (size * 10 * 2) / (end - start)  # MB/s
                results[f'gpu_bandwidth_{size}MB'] = bandwidth

            # Test CPU memory bandwidth
            if self.device.type == 'cpu':
                start = time.time()
                for _ in range(10):
                    x = np.random.randn(size * 1024 * 256).astype(np.float32)
                    y = x * 2
                end = time.time()
                bandwidth = (size * 10 * 2) / (end - start)  # MB/s
                results[f'cpu_bandwidth_{size}MB'] = bandwidth

        return results

    def benchmark_io_operations(self, file_sizes: List[int] = [10, 50, 100]):
        """Benchmark I/O operations."""
        results = {}

        temp_dir = Path("temp_benchmark")
        temp_dir.mkdir(exist_ok=True)

        try:
            for size_mb in file_sizes:
                logger.info(f"Benchmarking I/O with file size {size_mb}MB")

                # Write test
                data = np.random.randn(size_mb * 1024 * 256).astype(np.float32)
                file_path = temp_dir / f"test_{size_mb}MB.npy"

                start = time.time()
                np.save(file_path, data)
                write_time = time.time() - start

                # Read test
                start = time.time()
                loaded_data = np.load(file_path)
                read_time = time.time() - start

                results[f'write_{size_mb}MB'] = size_mb / write_time  # MB/s
                results[f'read_{size_mb}MB'] = size_mb / read_time    # MB/s

        finally:
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

        return results

    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark."""
        logger.info("Starting comprehensive performance benchmark")
        logger.info(f"Device: {self.device}")

        results = {
            'device': str(self.device),
            'timestamp': time.time(),
            'system_info': self._get_system_info()
        }

        # Run benchmarks
        results.update(self.benchmark_matrix_operations())
        results.update(self.benchmark_deep_learning_operations())
        results.update(self.benchmark_memory_bandwidth())
        results.update(self.benchmark_io_operations())

        logger.info("Benchmark completed")
        return results

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        import platform
        return {
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        }

    def save_results(self, results: Dict[str, Any], filename: str):
        """Save benchmark results to file."""
        import json

        # Convert numpy arrays and torch tensors to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj

        results_json = convert_for_json(results)

        with open(filename, 'w') as f:
            json.dump(results_json, f, indent=2)

        logger.info(f"Benchmark results saved to {filename}")

def run_quick_benchmark() -> Dict[str, Any]:
    """Run a quick benchmark for basic performance assessment."""
    benchmark = PerformanceBenchmark()

    # Quick subset of benchmarks
    results = {
        'device': str(benchmark.device),
        'quick_benchmark': True
    }

    # Test basic operations
    with timer("Quick matrix operation"):
        x = torch.randn(1000, 1000, device=benchmark.device)
        y = x @ x.T
        torch.cuda.synchronize() if benchmark.device.type == 'cuda' else None

    with timer("Quick DL operation"):
        x = torch.randn(64, 512, device=benchmark.device)
        linear = torch.nn.Linear(512, 1024).to(benchmark.device)
        y = linear(x)
        torch.cuda.synchronize() if benchmark.device.type == 'cuda' else None

    return results

if __name__ == "__main__":
    # Run quick benchmark when script is executed directly
    results = run_quick_benchmark()
    print("Quick benchmark completed")
    print(f"Device: {results['device']}")