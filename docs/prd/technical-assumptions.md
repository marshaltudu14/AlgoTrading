# Technical Assumptions

## Repository Structure

- **Preserved Architecture**: Current service-oriented backend structure remains intact
- **New Components**: HRM models added to `src/models/` directory
- **Agent Replacement**: `src/agents/ppo_agent.py` replaced with `src/agents/hrm_agent.py`
- **Configuration Extension**: `config/settings.yaml` extended with HRM dual-module and ACT parameters
- **Model Storage**: New model files stored in `models/` directory alongside existing structure

## Service Architecture

- **API Compatibility**: FastAPI endpoints remain unchanged for frontend compatibility
- **Training Services**: `src/training/` modules updated to support deep supervision training
- **Live Trading**: `src/trading/live_trading_service.py` maintains same interface with HRM backend
- **Backtesting**: `src/trading/backtest_service.py` updated for HRM hierarchical evaluation
- **Data Pipeline**: `src/data_processing/` enhanced to support multi-segment training data

## Technology Stack

- **Python 3.11+**: Current Python version maintained
- **PyTorch**: Continue using PyTorch for model implementation
- **FastAPI**: Backend API framework unchanged
- **Dependencies**: New requirements for HRM (specialized recurrent modules, ACT mechanism, one-step gradient)
- **Testing Framework**: Pytest remains primary testing framework

## Data Pipeline Assumptions

- **Historical Data**: Existing OHLC data format compatible with multi-segment training
- **Hierarchical Processing**: Dual-module reasoning feasible with current data diversity
- **Storage Requirements**: Reduced storage needs due to O(1) memory complexity
- **Memory Constraints**: 8GB RAM sufficient for training and inference

## Model Training Assumptions

- **Deep Supervision Training**: Historical data sufficient for segment-based training
- **Compute Resources**: Current hardware more than adequate for efficient HRM training (27M params)
- **Training Time**: Acceptable training duration (hours vs days for convergence)
- **Data Quality**: Existing market data quality sufficient for sequence modeling

## Integration Assumptions

- **Backward Compatibility**: Existing model loading/saving interfaces adaptable
- **API Contracts**: Current trading API contracts preserved
- **Configuration Migration**: Settings.yaml structure extensible without breaking changes
- **Testing Coverage**: Existing test suite adaptable to Decision Transformer architecture

## Performance Assumptions

- **Inference Speed**: Hierarchical reasoning processable within 50ms latency requirement
- **Memory Usage**: Dramatically reduced model size and memory footprint (27M vs billions of parameters)
- **Scalability**: Architecture supports future multi-instrument training
- **Hardware Compatibility**: Current deployment hardware sufficient

## Risk Mitigation Assumptions

- **Rollback Capability**: Current PPO models preserved as fallback option during transition
- **Validation Framework**: Existing backtesting framework adequate for DT validation
- **Monitoring Systems**: Current logging and monitoring sufficient for DT operations
- **Error Handling**: Existing error handling patterns adaptable to new architecture
