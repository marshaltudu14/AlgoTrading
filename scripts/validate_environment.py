#!/usr/bin/env python3
"""
Environment validation script for Transformer Trading System.

This script validates that all dependencies are properly installed
and the environment meets minimum requirements.
"""

import sys
import importlib
import warnings
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.hardware_utils import get_hardware_info

def check_imports():
    """Check that all required modules can be imported."""
    required_modules = {
        'torch': '1.9.0',
        'torchvision': '0.10.0',
        'torchaudio': '0.9.0',
        'transformers': '4.0.0',
        'pandas': '1.3.0',
        'numpy': '1.21.0',
        'scikit-learn': '1.0.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'optuna': '2.10.0',
        'wandb': '0.12.0',
        'tensorboard': '2.7.0',
        'accelerate': '0.5.0',
        'psutil': '5.8.0',
        'omegaconf': '2.1.0',
        'pytest': '6.0.0',
        'ruff': '0.0.1',  # ruff doesn't have version attribute
    }

    failed_imports = []
    version_issues = []

    for module, min_version in required_modules.items():
        try:
            if module == 'ruff':
                # Skip version check for ruff
                continue

            imported = importlib.import_module(module)
            if hasattr(imported, '__version__'):
                current_version = imported.__version__
                # Simple version comparison (not semantic)
                if current_version < min_version:
                    version_issues.append(f"{module}: {current_version} < {min_version}")
            elif hasattr(imported, 'version'):
                current_version = imported.version
                if current_version < min_version:
                    version_issues.append(f"{module}: {current_version} < {min_version}")

        except ImportError as e:
            failed_imports.append(f"{module}: {e}")
        except Exception as e:
            failed_imports.append(f"{module}: {e}")

    return failed_imports, version_issues

def check_hardware_requirements():
    """Check hardware requirements."""
    hardware_info = get_hardware_info()
    issues = []

    # Check GPU
    if not hardware_info['gpu']['cuda_available']:
        issues.append("CUDA not available - GPU acceleration disabled")
    else:
        gpu_count = hardware_info['gpu']['gpu_count']
        if gpu_count == 0:
            issues.append("No GPUs detected despite CUDA being available")

        # Check GPU memory
        min_gpu_memory_gb = 8
        sufficient_gpu = False
        for gpu in hardware_info['gpu']['gpus']:
            memory_gb = gpu['memory_total'] / (1024**3)
            if memory_gb >= min_gpu_memory_gb:
                sufficient_gpu = True
                break

        if not sufficient_gpu:
            issues.append(f"No GPU with at least {min_gpu_memory_gb}GB memory")

    # Check system memory
    system_memory_gb = hardware_info['memory']['total'] / (1024**3)
    if system_memory_gb < 16:
        issues.append(f"System memory {system_memory_gb:.1f}GB < 16GB minimum")

    # Check mixed precision support
    if not hardware_info['gpu']['mixed_precision_available']:
        issues.append("Mixed precision training not supported")

    return hardware_info, issues

def check_directory_structure():
    """Check that required directories exist."""
    base_path = Path(__file__).parent.parent
    required_dirs = [
        'src', 'tests', 'data', 'config', 'logs', 'notebooks',
        'src/utils', 'src/models', 'src/data_processing', 'src/auth',
        'tests/unit', 'tests/integration', 'tests/e2e', 'tests/environment'
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not (base_path / dir_path).exists():
            missing_dirs.append(dir_path)

    return missing_dirs

def validate_configuration_files():
    """Check that configuration files exist."""
    base_path = Path(__file__).parent.parent
    required_files = [
        'requirements.txt',
        'setup_env.bat',
        '.env.example',
        'config/example_config.yaml',
        'docs/development-guidelines.md'
    ]

    missing_files = []
    for file_path in required_files:
        if not (base_path / file_path).exists():
            missing_files.append(file_path)

    return missing_files

def main():
    """Main validation function."""
    print("Validating Transformer Trading Environment")
    print("=" * 50)

    all_issues = []
    warnings_list = []

    # Check imports
    print("\nChecking module imports...")
    failed_imports, version_issues = check_imports()
    if failed_imports:
        print("[FAILED] Failed imports:")
        for issue in failed_imports:
            print(f"   {issue}")
            all_issues.append(f"Import: {issue}")
    else:
        print("[OK] All required modules imported successfully")

    if version_issues:
        print("[WARN] Version issues:")
        for issue in version_issues:
            print(f"   {issue}")
            warnings_list.append(f"Version: {issue}")

    # Check hardware
    print("\nChecking hardware requirements...")
    hardware_info, hardware_issues = check_hardware_requirements()
    if hardware_issues:
        print("[FAILED] Hardware issues:")
        for issue in hardware_issues:
            print(f"   {issue}")
            all_issues.append(f"Hardware: {issue}")
    else:
        print("[OK] Hardware requirements met")

    # Check directory structure
    print("\nChecking directory structure...")
    missing_dirs = check_directory_structure()
    if missing_dirs:
        print("[FAILED] Missing directories:")
        for dir_path in missing_dirs:
            print(f"   {dir_path}")
            all_issues.append(f"Directory: {dir_path}")
    else:
        print("[OK] All required directories present")

    # Check configuration files
    print("\nChecking configuration files...")
    missing_files = validate_configuration_files()
    if missing_files:
        print("[FAILED] Missing files:")
        for file_path in missing_files:
            print(f"   {file_path}")
            all_issues.append(f"File: {file_path}")
    else:
        print("[OK] All required files present")

    # Summary
    print("\n" + "=" * 50)
    if all_issues:
        print(f"[FAILED] Validation failed with {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"   • {issue}")
        print("\nPlease resolve these issues before proceeding.")
        return False
    else:
        print("[OK] Environment validation successful!")
        if warnings_list:
            print(f"\n[WARN] {len(warnings_list)} warnings:")
            for warning in warnings_list:
                print(f"   • {warning}")

        # Save hardware info for reference
        hardware_file = Path(__file__).parent / "hardware_info.json"
        with open(hardware_file, 'w') as f:
            json.dump(hardware_info, f, indent=2, default=str)
        print(f"\n[INFO] Hardware info saved to {hardware_file}")

        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)