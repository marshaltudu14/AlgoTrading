# RL Trading System: Training & Backtesting Guide

## 🎯 Complete Training Sequence: PPO → MoE → MAML

This system implements a sophisticated 3-stage training sequence designed to create top-quality RL trading agents:

### Stage 1: PPO Baseline (500 episodes)
- **Purpose**: Establish baseline performance and validate environment
- **Algorithm**: Proximal Policy Optimization
- **Episodes**: 500 (optimized for millions of data rows)
- **Success Criteria**: Win Rate ≥ 35%, Profit Factor ≥ 0.8

### Stage 2: MoE Specialization (800 episodes)  
- **Purpose**: Train specialized experts for different market conditions
- **Algorithm**: Mixture of Experts with 4 specialized networks
- **Episodes**: 800 (allows deep specialization)
- **Success Criteria**: Win Rate ≥ 40%, Profit Factor ≥ 1.0

### Stage 3: MAML Meta-Learning (150 iterations)
- **Purpose**: Meta-learning for quick adaptation to new conditions
- **Algorithm**: Model-Agnostic Meta-Learning
- **Iterations**: 150 (comprehensive meta-learning)
- **Success Criteria**: Successful adaptation capability

## 🚀 Quick Start

### 1. Run Complete Training Sequence
```bash
# Train with complete sequence (default)
python run_training.py

# Train specific symbols
python run_training.py --symbols Nifty_2 Bank_Nifty_5

# Train all available symbols
python run_training.py --symbols all
```

### 2. Run Backtesting
```bash
# Process raw data and run backtest
python run_backtest.py --process-data

# Backtest specific symbols
python run_backtest.py --symbols Nifty_2 Bank_Nifty_5

# Generate detailed report
python run_backtest.py --report-file backtest_report.txt
```

## 📊 Optimal Configuration for Large Datasets

### Episode Counts (Optimized for Millions of Rows)
- **PPO**: 500 episodes - Deep pattern learning
- **MoE**: 800 episodes - Expert specialization  
- **MAML**: 150 iterations - Meta-learning mastery

### Key Parameters
```yaml
# PPO Parameters
learning_rate: 0.0001      # Stable learning
batch_size: 128           # Better gradient estimates
gamma: 0.995              # Long-term planning

# MoE Parameters  
num_experts: 4            # Better specialization
expert_hidden_dim: 128    # Complex pattern recognition
diversity_loss_weight: 0.02 # Expert specialization

# MAML Parameters
inner_loop_steps: 7       # Better adaptation
meta_batch_size: 2        # Improved meta-gradients
```

## 🎯 Training Sequence Benefits

### 1. PPO Foundation
- ✅ Stable baseline performance
- ✅ Environment validation
- ✅ Basic pattern recognition

### 2. MoE Specialization
- ✅ Market condition experts
- ✅ Improved performance
- ✅ Robust decision making

### 3. MAML Adaptation
- ✅ Quick adaptation to new symbols
- ✅ Cross-market generalization
- ✅ Production-ready model

## 📁 Model Management

### Saved Models
```
models/
├── {symbol}_ppo_stage1.pth      # PPO baseline
├── {symbol}_moe_stage2.pth      # MoE specialized
├── {symbol}_maml_stage3_final.pth # MAML meta-learned
└── {symbol}_final_model.pth     # Production model
```

### Model Loading Priority (Backtesting)
1. `{symbol}_final_model.pth` (Production)
2. `{symbol}_maml_stage3_final.pth` (MAML)
3. `{symbol}_moe_stage2.pth` (MoE)
4. `{symbol}_ppo_stage1.pth` (PPO)

## 🔧 Advanced Usage

### Custom Training Configuration
Edit `config/training_sequence.yaml` to customize:
- Episode counts per stage
- Success criteria
- Algorithm parameters
- Progression rules

### Raw Data Processing
```bash
# Process all raw data files
python run_backtest.py --process-data --raw-data-dir data/raw

# Custom processing
python run_backtest.py --process-data \
  --raw-data-dir /path/to/raw \
  --processed-data-dir /path/to/processed
```

### Backtesting Options
```bash
# Full backtest with report
python run_backtest.py \
  --process-data \
  --initial-capital 500000 \
  --report-file detailed_report.txt

# Specific symbols only
python run_backtest.py \
  --symbols Nifty_2 Bank_Nifty_5 \
  --initial-capital 200000
```

## 📈 Expected Performance

### Training Time (Approximate)
- **PPO Stage**: 2-4 hours (500 episodes)
- **MoE Stage**: 4-6 hours (800 episodes)  
- **MAML Stage**: 3-5 hours (150 iterations)
- **Total**: 9-15 hours for complete sequence

### Performance Targets
- **Win Rate**: 40-50% (realistic for options trading)
- **Profit Factor**: 1.2-1.8 (good risk-adjusted returns)
- **Max Drawdown**: <25% (controlled risk)
- **Sharpe Ratio**: >1.0 (risk-adjusted performance)

## 🛠️ Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce batch_size in config
2. **Slow Training**: Check GPU availability
3. **Poor Performance**: Increase episodes or adjust success criteria
4. **Model Loading Errors**: Check model file paths

### Performance Optimization
- Use GPU for faster training
- Increase batch sizes for better gradients
- Monitor memory usage during training
- Use early stopping if performance plateaus

## 📊 Monitoring Training

### Key Metrics to Watch
- **Win Rate**: Should improve across stages
- **Profit Factor**: Target >1.0 for profitability
- **Drawdown**: Should decrease with better models
- **Trade Frequency**: Balanced activity

### Success Indicators
- ✅ Consistent improvement across stages
- ✅ Meeting progression criteria
- ✅ Stable performance metrics
- ✅ Successful model saving

## 🎯 Production Deployment

### Final Model Usage
The `{symbol}_final_model.pth` contains the best-performing model after the complete sequence and is ready for:
- Live trading integration
- Further backtesting
- Performance monitoring
- Model updates

### Continuous Improvement
- Retrain periodically with new data
- Monitor live performance vs backtest
- Adjust parameters based on market changes
- Implement ensemble methods for robustness
