"""
Tests for GPU detection and hardware utilities.
"""

import pytest
import torch
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.hardware_utils import HardwareDetector, get_hardware_info

class TestHardwareDetector:
    """Test HardwareDetector class."""

    def test_init(self):
        """Test HardwareDetector initialization."""
        detector = HardwareDetector()
        assert hasattr(detector, 'system_info')
        assert hasattr(detector, 'gpu_info')
        assert hasattr(detector, 'memory_info')

    def test_system_info(self):
        """Test system information detection."""
        detector = HardwareDetector()
        info = detector.system_info

        assert 'platform' in info
        assert 'python_version' in info
        assert isinstance(info['python_version'], str)

    def test_gpu_info_structure(self):
        """Test GPU information structure."""
        detector = HardwareDetector()
        gpu_info = detector.gpu_info

        required_keys = [
            'cuda_available', 'cuda_version', 'gpu_count',
            'gpus', 'mixed_precision_available'
        ]

        for key in required_keys:
            assert key in gpu_info

        assert isinstance(gpu_info['cuda_available'], bool)
        assert isinstance(gpu_info['gpu_count'], int)
        assert isinstance(gpu_info['gpus'], list)
        assert isinstance(gpu_info['mixed_precision_available'], bool)

    def test_memory_info(self):
        """Test memory information detection."""
        detector = HardwareDetector()
        memory_info = detector.memory_info

        required_keys = ['total', 'available', 'used', 'percent']
        for key in required_keys:
            assert key in memory_info

        assert memory_info['total'] > 0
        assert memory_info['available'] >= 0
        assert memory_info['percent'] >= 0

    def test_is_gpu_sufficient(self):
        """Test GPU sufficiency check."""
        detector = HardwareDetector()

        # Test with no GPU
        with patch.object(detector, 'gpu_info', {'cuda_available': False}):
            assert not detector.is_gpu_sufficient()

        # Test with sufficient GPU
        mock_gpu = {
            'memory_total': 10 * 1024**3  # 10GB
        }
        with patch.object(detector, 'gpu_info', {
            'cuda_available': True,
            'gpus': [mock_gpu]
        }):
            assert detector.is_gpu_sufficient(min_memory_gb=8)

    def test_get_recommended_batch_size(self):
        """Test batch size recommendation."""
        detector = HardwareDetector()

        # Test CPU fallback
        with patch.object(detector, 'gpu_info', {'cuda_available': False}):
            batch_size = detector.get_recommended_batch_size()
            assert batch_size == 8

        # Test GPU recommendation
        mock_gpu = {
            'memory_total': 16 * 1024**3,  # 16GB
            'memory_reserved': 2 * 1024**3   # 2GB reserved
        }
        with patch.object(detector, 'gpu_info', {
            'cuda_available': True,
            'gpus': [mock_gpu]
        }):
            batch_size = detector.get_recommended_batch_size(model_size_mb=100)
            assert 8 <= batch_size <= 256

    def test_get_optimal_device(self):
        """Test optimal device selection."""
        detector = HardwareDetector()

        # Test CPU fallback
        with patch.object(detector, 'gpu_info', {'cuda_available': False}):
            device = detector.get_optimal_device()
            assert device.type == 'cpu'

        # Test GPU selection
        with patch.object(detector, 'gpu_info', {
            'cuda_available': True,
            'gpu_count': 1
        }):
            with patch('torch.cuda.device') as mock_device:
                mock_device.return_value = Mock()
                detector.get_optimal_device()
                mock_device.assert_called_once()

    def test_configure_mixed_precision(self):
        """Test mixed precision configuration."""
        detector = HardwareDetector()

        config = detector.configure_mixed_precision()

        required_keys = ['enabled', 'dtype', 'scaler']
        for key in required_keys:
            assert key in config

        assert isinstance(config['enabled'], bool)

    def test_to_dict(self):
        """Test complete hardware info dictionary."""
        detector = HardwareDetector()
        info_dict = detector.to_dict()

        required_sections = ['system', 'gpu', 'memory', 'optimal_device']
        for section in required_sections:
            assert section in info_dict

def test_get_hardware_info():
    """Test convenience function."""
    hardware_info = get_hardware_info()

    assert isinstance(hardware_info, dict)
    assert 'system' in hardware_info
    assert 'gpu' in hardware_info
    assert 'memory' in hardware_info

class TestGPUFunctionality:
    """Test actual GPU functionality when available."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """Test basic CUDA operations."""
        # Test tensor creation on GPU
        device = torch.device('cuda:0')
        x = torch.randn(10, 10, device=device)
        assert x.device.type == 'cuda'

        # Test computation
        y = x @ x.T
        assert y.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_management(self):
        """Test GPU memory management."""
        initial_memory = torch.cuda.memory_allocated()

        # Allocate some memory
        x = torch.randn(1000, 1000, device='cuda')
        allocated_memory = torch.cuda.memory_allocated()

        assert allocated_memory > initial_memory

        # Clear memory
        del x
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        assert final_memory <= allocated_memory

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mixed_precision_support(self):
        """Test mixed precision support."""
        detector = HardwareDetector()

        if detector.gpu_info['mixed_precision_available']:
            # Test half precision
            x = torch.randn(10, 10, device='cuda')
            x_half = x.half()
            assert x_half.dtype == torch.float16

            # Test BF16 if supported
            if torch.cuda.is_bf16_supported():
                x_bf16 = x.bfloat16()
                assert x_bf16.dtype == torch.bfloat16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_device_properties(self):
        """Test GPU device properties."""
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            assert hasattr(props, 'total_memory')
            assert hasattr(props, 'name')
            assert props.total_memory > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])