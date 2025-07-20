# Training Sequence Guide

## Overview

This guide explains the optimal training sequence for RL trading models and addresses the question of whether to use MAML and MoE as defaults.

## ❌ Current Issue: MAML/MoE as Default

**Problem**: Using MAML and MoE as default training methods is **not optimal** because:

1. **MAML requires pre-trained models**: Meta-learning works best when starting from already competent models
2. **MoE needs baseline performance**: Mixture of Experts requires understanding of what each expert should specialize in
3. **Complex methods need validation**: Starting with complex methods makes it hard to debug issues

## ✅ Recommended Training Sequence

### **Stage 1: PPO Baseline (Foundation)**
```bash
py run_training.py --symbols Nifty_2 --algorithm PPO --episodes 100 --simple
```

**Purpose**: 
- Establish baseline performance
- Validate environment and reward functions
- Learn basic trading patterns
- Debug any fundamental issues

**Success Criteria**:
- Win rate ≥ 35%
- Profit factor ≥ 0.8
- Stable training (no crashes)

### **Stage 2: MoE Specialization (Enhancement)**
```bash
py run_training.py --symbols Nifty_2 --algorithm MoE --episodes 200 --simple
```

**Purpose**:
- Create specialized experts for different market conditions
- Improve performance through specialization
- Learn condition-specific strategies

**Success Criteria**:
- Win rate ≥ 40%
- Profit factor ≥ 1.0
- Better performance than PPO baseline

### **Stage 3: MAML Meta-Learning (Adaptation)**
```bash
py run_training.py --symbols Nifty_2 --algorithm MAML --episodes 50 --simple
```

**Purpose**:
- Enable quick adaptation to new symbols/timeframes
- Improve generalization across market conditions
- Fine-tune adaptation process

**Success Criteria**:
- Fast adaptation (≤ 5 episodes)
- Good cross-symbol performance
- Maintains or improves performance

## 🚀 Complete Sequence (Recommended)

Use the new sequence training feature:

```bash
py run_training.py --symbols Nifty_2 --sequence
```

This automatically runs all three stages in order with proper progression criteria.

## 🔧 Fixed Issues

### P&L Calculation Fixes
- ✅ Fixed trade history field mismatch (`pnl` vs `realized_pnl_this_trade`)
- ✅ Fixed capital tracking to include brokerage costs
- ✅ Fixed MAML training summary metrics calculation
- ✅ Improved capital history tracking across meta-iterations

### Training Improvements
- ✅ Added proper training sequence management
- ✅ Added success criteria for stage progression
- ✅ Added comprehensive training reports
- ✅ Added model transfer between stages

## 📊 Expected Results

After fixes, you should see:

### Realistic P&L Values
```
📊 Current Trade P&L: ₹-1375.00  (realistic loss)
📊 Current Trade P&L: ₹2857.50   (realistic profit)
💰 Total P&L from Initial: ₹-14795.69  (cumulative)
🏦 Capital: ₹85204.31  (consistent with P&L)
```

### Proper Metrics
```
Win Rate: 33.3%
Total Trades: 3
Profit Factor: inf (when no losses)
Total Return: -32.46%
Max Drawdown: 32.46%
```

## 🎯 Recommendations

1. **Start with PPO**: Always begin with PPO to establish baseline
2. **Progress sequentially**: Only move to next stage after success criteria are met
3. **Use sequence training**: Use `--sequence` flag for automated progression
4. **Monitor metrics**: Pay attention to win rate, profit factor, and drawdown
5. **Validate environment**: Ensure P&L calculations are realistic before advanced training

## 🔍 Troubleshooting

### If PPO Stage Fails
- Check data quality and preprocessing
- Verify environment reward function
- Adjust hyperparameters (learning rate, episodes)
- Check for data leakage or look-ahead bias

### If MoE Stage Fails
- Ensure PPO baseline is working
- Check expert specialization (are experts learning different strategies?)
- Adjust number of experts or diversity loss weight

### If MAML Stage Fails
- Verify MoE model is loaded correctly
- Check meta-learning hyperparameters
- Ensure sufficient task diversity for meta-learning

## 📈 Performance Expectations

### PPO Baseline
- Win Rate: 35-45%
- Profit Factor: 0.8-1.2
- Sharpe Ratio: 0.1-0.5

### MoE Enhancement
- Win Rate: 40-50%
- Profit Factor: 1.0-1.5
- Sharpe Ratio: 0.2-0.7

### MAML Meta-Learning
- Quick adaptation: 3-10 episodes
- Cross-symbol performance: 80%+ of single-symbol
- Improved generalization

## 🎯 Conclusion

**Answer to your question**: No, MAML and MoE should **not** be the default training methods. The correct approach is:

1. **PPO first** (foundation)
2. **MoE second** (specialization) 
3. **MAML third** (meta-learning)

This sequence ensures each stage builds on the previous one's success, making debugging easier and results more reliable.
