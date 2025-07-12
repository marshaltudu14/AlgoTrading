#!/usr/bin/env python3
"""
Project Organization Script
===========================

Organizes the AlgoTrading project into a clean, professional structure.
"""

import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the organized directory structure."""
    
    directories = [
        # Core directories
        "src",
        "src/data_processing",
        "src/reasoning_system", 
        "src/auth",
        "src/config",
        
        # Data directories
        "data",
        "data/raw",
        "data/processed", 
        "data/final",
        
        # Scripts and utilities
        "scripts",
        "scripts/data_processing",
        "scripts/testing",
        
        # Documentation
        "docs",
        "docs/api",
        "docs/guides",
        
        # Logs and reports
        "logs",
        "reports",
        "reports/quality",
        "reports/pipeline",
        
        # Configuration
        "config",
        
        # Tests
        "tests",
        "tests/unit",
        "tests/integration",
        
        # Temporary and cache
        "temp",
        "cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def organize_files():
    """Move files to their appropriate locations."""
    
    file_moves = [
        # Core source files
        ("feature_generator.py", "src/data_processing/feature_generator.py"),
        ("reasoning_processor.py", "src/data_processing/reasoning_processor.py"),
        ("data_processing_pipeline.py", "src/data_processing/pipeline.py"),
        ("fyers_auth.py", "src/auth/fyers_auth.py"),
        
        # Configuration files
        ("config.py", "src/config/config.py"),
        ("reasoning_config.py", "src/config/reasoning_config.py"),
        
        # Data directories
        ("historical_data", "data/raw"),
        ("processed_data", "data/processed"),
        ("reasoning_data", "data/processed/reasoning"),
        ("final_data", "data/final"),
        
        # Logs and reports
        ("pipeline_logs", "reports/pipeline"),
        ("quality_reports", "reports/quality"),
        
        # Documentation
        ("mainIdea.md", "docs/mainIdea.md"),
        ("reasoning_system/README_REASONING_SYSTEM.md", "docs/reasoning_system.md"),
        
        # Requirements
        ("requirements.txt", "requirements.txt"),  # Keep in root
    ]
    
    for source, destination in file_moves:
        source_path = Path(source)
        dest_path = Path(destination)
        
        if source_path.exists():
            # Create destination directory if it doesn't exist
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file or directory
            if source_path.is_dir():
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.move(str(source_path), str(dest_path))
            else:
                if dest_path.exists():
                    dest_path.unlink()
                shutil.move(str(source_path), str(dest_path))
            
            print(f"✓ Moved: {source} → {destination}")
        else:
            print(f"⚠ Not found: {source}")

def move_reasoning_system():
    """Move reasoning system to src directory."""
    
    source = Path("reasoning_system")
    dest = Path("src/reasoning_system")
    
    if source.exists():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))
        print(f"✓ Moved: reasoning_system → src/reasoning_system")
    else:
        print("⚠ reasoning_system directory not found")

def clean_up_root():
    """Clean up unnecessary files from root."""
    
    files_to_remove = [
        "__pycache__",
        "context-engineering-intro-main",
        "fyersApi.log",
        "fyersDataSocket.log", 
        "fyersRequests.log",
    ]
    
    for item in files_to_remove:
        item_path = Path(item)
        if item_path.exists():
            if item_path.is_dir():
                shutil.rmtree(item_path)
            else:
                item_path.unlink()
            print(f"✓ Removed: {item}")

def create_init_files():
    """Create __init__.py files for Python packages."""
    
    init_files = [
        "src/__init__.py",
        "src/data_processing/__init__.py",
        "src/auth/__init__.py",
        "src/config/__init__.py",
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/integration/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        if not init_path.exists():
            init_path.write_text('"""Package initialization."""\n')
            print(f"✓ Created: {init_file}")

def create_project_readme():
    """Create main project README."""
    
    readme_content = '''# AlgoTrading System

A comprehensive algorithmic trading system with automated feature generation and human-like reasoning capabilities.

## 🏗️ Project Structure

```
AlgoTrading/
├── src/                          # Source code
│   ├── data_processing/          # Data processing modules
│   │   ├── feature_generator.py  # Technical indicator generation
│   │   ├── reasoning_processor.py # Reasoning generation
│   │   └── pipeline.py           # Integrated processing pipeline
│   ├── reasoning_system/         # Reasoning generation system
│   ├── auth/                     # Authentication modules
│   └── config/                   # Configuration files
├── data/                         # Data storage
│   ├── raw/                      # Raw historical data
│   ├── processed/                # Processed feature data
│   └── final/                    # Final training-ready data
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── tests/                        # Test suites
├── reports/                      # Generated reports
└── logs/                         # Log files
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python src/data_processing/pipeline.py
```

### 3. Run Individual Components
```bash
# Feature generation only
python src/data_processing/feature_generator.py

# Reasoning generation only  
python src/data_processing/reasoning_processor.py
```

## 📊 Data Flow

```
Raw OHLCV Data → Feature Generation → Reasoning Generation → Training Data
```

1. **Raw Data**: Historical OHLCV data in `data/raw/`
2. **Feature Generation**: Technical indicators and signals
3. **Reasoning Generation**: Human-like trading reasoning
4. **Final Data**: Training-ready data with features and reasoning

## 📚 Documentation

- [Main Idea](docs/mainIdea.md) - Project vision and goals
- [Reasoning System](docs/reasoning_system.md) - Detailed reasoning system documentation
- [API Documentation](docs/api/) - Code documentation

## 🔧 Configuration

Configuration files are located in `src/config/`:
- `config.py` - Main system configuration
- `reasoning_config.py` - Reasoning system configuration

## 📈 Features

- **Comprehensive Technical Analysis**: 50+ technical indicators
- **Signal Generation**: Automated buy/sell/hold signals
- **Human-like Reasoning**: Professional trader thinking simulation
- **Quality Assurance**: Multi-layer validation and quality scoring
- **Scalable Architecture**: Modular, extensible design

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/unit/
python -m pytest tests/integration/
```

## 📝 License

This project is part of the AlgoTrading system.
'''
    
    readme_path = Path("README.md")
    readme_path.write_text(readme_content)
    print("✓ Created: README.md")

def main():
    """Main organization function."""
    print("=" * 60)
    print("ORGANIZING ALGOTRADING PROJECT STRUCTURE")
    print("=" * 60)
    
    print("\n1. Creating directory structure...")
    create_directory_structure()
    
    print("\n2. Moving reasoning system...")
    move_reasoning_system()
    
    print("\n3. Organizing files...")
    organize_files()
    
    print("\n4. Cleaning up root directory...")
    clean_up_root()
    
    print("\n5. Creating __init__.py files...")
    create_init_files()
    
    print("\n6. Creating project README...")
    create_project_readme()
    
    print("\n" + "=" * 60)
    print("PROJECT ORGANIZATION COMPLETE!")
    print("=" * 60)
    print("\nProject structure has been organized.")
    print("Main components are now in src/ directory.")
    print("Data is organized in data/ directory.")
    print("Documentation is in docs/ directory.")

if __name__ == "__main__":
    main()
