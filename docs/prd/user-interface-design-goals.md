# User Interface Design Goals

## UX Vision

The HRM migration is a **backend-only architectural change** with **no frontend UI modifications**. All interactions remain through terminal-based commands and configuration files. The focus is on seamless CLI experience and enhanced terminal output showcasing HRM's adaptive reasoning capabilities.

## Interaction Paradigms

**Primary Paradigm: CLI-First Configuration**

- Settings.yaml-driven model configuration
- Terminal-based training and monitoring commands
- Command-line model management and validation

**Secondary Paradigm: Enhanced Terminal Output**

- Rich terminal feedback during training
- Detailed logging and progress indicators
- Improved error messages and debugging information

## Core Terminal Interfaces

**Training Commands:**

```bash
python run_training.py              # Now uses HRM with deep supervision
python run_unified_backtest.py      # Enhanced with HRM hierarchical reasoning
python run_live_bot.py              # HRM-powered adaptive live trading
```

**Configuration Interface:**

- `config/settings.yaml` - Primary configuration file
- Enhanced parameter validation and error reporting
- Clear documentation for HRM dual-module and ACT settings

**Monitoring Interface:**

- Enhanced terminal logging during training
- Progress bars for trajectory processing
- Detailed performance metrics output
- Model checkpointing status updates

## Terminal Output Enhancements

- Clear HRM training progress with hierarchical convergence metrics
- Adaptive computation time status indicators
- Memory usage and performance metrics (O(1) efficiency)
- Enhanced error messages with HRM-specific diagnostics

## No UI/UX Changes Required

- Next.js frontend remains unchanged
- All HRM functionality accessible via existing CLI
- Backend API interfaces preserved for frontend compatibility
- Focus on terminal user experience improvements only
