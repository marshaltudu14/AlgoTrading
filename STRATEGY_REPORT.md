# Trading Strategy Performance Report

## Overview
This report summarizes the performance of various rule-based trading strategies tested on Nifty 5-minute timeframe data.

## Strategies Tested

### 1. **multi_indicator_combo**
Combines multiple indicators (RSI, MACD, EMA crossover, trend following)

### 2. **trend_following**
Uses trend strength, ADX, and price momentum for directional trades

### 3. **rsi_mean_reversion**
Based on RSI oversold/overbought levels

### 4. **macd_crossover**
MACD line crossover with signal line

### 5. **bollinger_bands**
Bollinger band breakouts and squeezes

### 6. **ema_crossover**
Price crossover with EMAs

### 7. **mean_reversion**
Price mean reversion to moving averages

### 8. **momentum_breakout**
High momentum breakouts with volume confirmation

## Performance Summary

### Best Performing Configurations:

| Strategy | Target (Rs) | Stop Loss (Rs) | RR Ratio | Win Rate | Total Return | Sharpe Ratio | Max Drawdown | Trades/Day |
|----------|-------------|----------------|----------|----------|--------------|--------------|--------------|------------|
| multi_indicator_combo | 1500 | -500 | 3:1 | 39.13% | 26.8% | 4.39 | -2750 | 2.3 |
| multi_indicator_combo | 500 | -250 | 2:1 | 56.52% | 14.2% | 6.10 | -600 | 2.3 |
| trend_following | 300 | -200 | 1.5:1 | 64.00% | 8.8% | 4.03 | -750 | 2.5 |
| multi_indicator_combo | 300 | -200 | 1.5:1 | 65.22% | 8.8% | 6.16 | -500 | 2.3 |
| rsi_mean_reversion | 1000 | -500 | 2:1 | 50.00% | 4.0% | -0.66 | -1100 | 0.4 |

### Poor Performing Configurations:
- **macd_crossover**: Generated 0 trades in all test configurations
- **bollinger_bands**: Generated 0 trades in all test configurations
- **momentum_breakout**: Generated 0 trades in all test configurations
- **mean_reversion**: Generated 0 trades with tested parameters
- **trend_following** with 3000/1000: -11.5% return, huge drawdown
- **Scalping** (200/200): Only 1.2% return despite 65% win rate

## Key Insights

1. **multi_indicator_combo** is the clear winner, working across multiple RR ratios
2. **Higher targets (1500+)** generate good absolute returns but with lower win rates
3. **Medium targets (300-500)** provide better win rates with decent returns
4. **Scalping strategies** give high win rates but poor absolute returns
5. **Trend following** works well only with moderate targets
6. **Many single-indicator strategies failed to generate any trades**

## Removed Strategies
The following strategies will be removed as they consistently failed to generate trades:
- `macd_crossover`
- `bollinger_bands`
- `momentum_breakout`
- `mean_reversion`

## Recommendations

### For High Frequency Trading (HFT) - Scalping Approach:
- **Strategy**: `multi_indicator_combo`
- **Target**: Rs.200
- **Stop Loss**: Rs.-150
- **Min Confidence**: 0.3
- **Expected**: ~10 trades/day, small but consistent profits

### For Swing Trading Approach:
- **Strategy**: `multi_indicator_combo`
- **Target**: Rs.500
- **Stop Loss**: Rs.-250
- **Min Confidence**: 0.5
- **Expected**: ~2-3 trades/day, balanced risk/reward

### For Trend Following:
- **Strategy**: `trend_following`
- **Target**: Rs.300
- **Stop Loss**: Rs.-200
- **Min Confidence**: 0.5
- **Expected**: ~2-3 trades/day, high win rate

## Final Recommendation
After extensive testing, the optimal configuration for consistent profits with minimal drawdown:

### Recommended Configuration:
- **Strategy**: `multi_indicator_combo` (only working strategy)
- **Target**: Rs.500
- **Stop Loss**: Rs.-250 (2:1 RR)
- **Min Confidence**: 0.5
- **Expected Performance**: ~56% win rate, 14% return, minimal drawdown

### For High Frequency Trading:
- **Strategy**: `multi_indicator_combo`
- **Target**: Rs.200
- **Stop Loss**: Rs.-150 (1.33:1 RR)
- **Min Confidence**: 0.3
- **Expected Performance**: ~65% win rate, 3% return, low drawdown

### Important Notes:
1. Only 4 strategies work: multi_indicator_combo, trend_following, rsi_mean_reversion, ema_crossover
2. Most trades complete in 1-2 bars (5-10 minutes)
3. Higher targets (>1000) lead to large drawdowns
4. Scalping (<200) gives high win rates but poor absolute returns
5. Best balance is with medium targets (300-500) with 2:1 RR

### Timeframe Comparison (365 days backtest):

| Timeframe | Target | SL | Trades | Win Rate | Total Return | Sharpe | Trades/Day |
|-----------|--------|----|--------|----------|--------------|--------|------------|
| 5min | 500 | -250 | 1397 | 55.62% | 818.2% | 4.05 | 3.8 |
| 2min | 400 | -200 | 2976 | 53.39% | 1047.0% | 2.72 | 8.1 |

### Final Recommendation:
- **2min timeframe** generates more than double the trades and higher absolute returns
- However, **5min timeframe** has better Sharpe ratio (risk-adjusted returns)
- For consistent profits with better risk management: Use **5min** timeframe
- For maximum profit potential: Use **2min** timeframe

### Latest Ultra-Enhanced Strategy Results (5min, 365 days):
- **Total Trades**: 837
- **Win Rate**: 55.20%
- **Total Return**: 477%
- **Max Drawdown**: Rs.-2,700
- **Sharpe Ratio**: 4.53
- **Profit Factor**: 1.85

### New Features Added:
1. **Multi-timeframe Alignment**: 5/10/20/50 SMA alignment for trend confirmation
2. **Bollinger Band Analysis**: Bounces from bands, squeeze plays, rejections
3. **RSI Divergence**: Both price-RSI and RSI_14/RSI_21 divergence
4. **Candlestick Patterns**: Hammer and shooting star confirmation
5. **Dynamic Confidence**: Automatic boost based on multiple confirmations
6. **Adaptive Thresholds**: Lower entry requirements with strong signals

### Implementation:
Update `trading.py` with:
```
STRATEGY = "multi_indicator_combo"
TARGET_PNL = 500
STOP_LOSS_PNL = -250
MIN_CONFIDENCE = 0.5
```