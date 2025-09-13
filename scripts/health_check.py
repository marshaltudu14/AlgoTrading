#!/usr/bin/env python3
"""
Health check script for the Transformer Trading environment.

This script performs ongoing health checks to ensure the environment
remains functional after initial setup.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.hardware_utils import get_hardware_info
from utils.benchmark_utils import run_quick_benchmark

def check_import_health():
    """Check if all critical imports are working."""
    critical_modules = [
        'torch', 'torch.nn', 'torch.optim',
        'transformers', 'pandas', 'numpy',
        'sklearn', 'matplotlib', 'seaborn',
        'optuna', 'wandb', 'tensorboard',
        'accelerate', 'omegaconf'
    ]

    failed_imports = []
    for module in critical_modules:
        try:
            __import__(module)
        except ImportError as e:
            failed_imports.append(f"{module}: {e}")

    return len(failed_imports) == 0, failed_imports

def check_gpu_health():
    """Check GPU health and functionality."""
    try:
        import torch
        hardware_info = get_hardware_info()
        gpu_info = hardware_info['gpu']

        if not gpu_info['cuda_available']:
            return True, "GPU not available, falling back to CPU"

        # Test basic GPU operations
        device = torch.device('cuda:0')
        x = torch.randn(100, 100, device=device)
        y = x @ x.T
        torch.cuda.synchronize()

        # Check memory
        memory_allocated = torch.cuda.memory_allocated(device)
        memory_reserved = torch.cuda.memory_reserved(device)

        # Test if memory is reasonable
        if memory_allocated > 1024**3:  # 1GB
            return False, f"Excessive GPU memory allocated: {memory_allocated/1024**3:.2f}GB"

        return True, f"GPU healthy, {memory_allocated/1024**2:.1f}MB allocated"

    except Exception as e:
        return False, f"GPU health check failed: {e}"

def check_memory_health():
    """Check system memory health."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        issues = []

        if memory.percent > 90:
            issues.append(f"High memory usage: {memory.percent:.1f}%")

        if disk.percent > 90:
            issues.append(f"High disk usage: {disk.percent:.1f}%")

        if len(issues) == 0:
            return True, "Memory and disk usage healthy"
        else:
            return False, "; ".join(issues)

    except Exception as e:
        return False, f"Memory health check failed: {e}"

def check_performance_health():
    """Check if performance is within acceptable bounds."""
    try:
        start_time = time.time()
        benchmark_result = run_quick_benchmark()
        elapsed_time = time.time() - start_time

        # If benchmark takes too long, there might be performance issues
        if elapsed_time > 30:  # 30 seconds threshold
            return False, f"Performance degraded: benchmark took {elapsed_time:.1f}s"

        return True, f"Performance healthy: benchmark completed in {elapsed_time:.1f}s"

    except Exception as e:
        return False, f"Performance health check failed: {e}"

def check_configuration_health():
    """Check if configuration files are valid."""
    config_issues = []

    # Check if key files exist
    required_files = [
        'requirements.txt',
        '.env.example',
        'config/example_config.yaml'
    ]

    base_path = Path(__file__).parent.parent
    for file_path in required_files:
        if not (base_path / file_path).exists():
            config_issues.append(f"Missing file: {file_path}")

    # Check if example config is valid YAML
    try:
        import yaml
        with open(base_path / 'config/example_config.yaml', 'r') as f:
            yaml.safe_load(f)
    except Exception as e:
        config_issues.append(f"Invalid config file: {e}")

    if len(config_issues) == 0:
        return True, "Configuration files healthy"
    else:
        return False, "; ".join(config_issues)

def save_health_report(results):
    """Save health check results to file."""
    report_path = Path(__file__).parent / "health_report.json"

    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': results['overall_healthy'],
        'checks': results
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Health report saved to {report_path}")

def main():
    """Main health check function."""
    print("Running Transformer Trading Environment Health Check")
    print("=" * 55)

    checks = {
        'imports': check_import_health(),
        'gpu': check_gpu_health(),
        'memory': check_memory_health(),
        'performance': check_performance_health(),
        'configuration': check_configuration_health()
    }

    # Display results
    for check_name, (healthy, message) in checks.items():
        status = "[OK]" if healthy else "[FAILED]"
        print(f"{status} {check_name.capitalize()}: {message}")

    # Overall status
    overall_healthy = all(healthy for healthy, _ in checks.values())
    if overall_healthy:
        print("\n[OK] Overall health: GOOD")
    else:
        print("\n[FAILED] Overall health: ISSUES DETECTED")

    # Save report
    results = {
        'overall_healthy': overall_healthy,
        'checks': {k: {'healthy': v[0], 'message': v[1]} for k, v in checks.items()}
    }
    save_health_report(results)

    return 0 if overall_healthy else 1

if __name__ == "__main__":
    sys.exit(main())