# 📊 Comprehensive Backtesting Documentation

## Overview

This document explains the complete backtesting system implementation, covering both the RL training environment and the realistic backtesting system.

## 🔄 Two Backtesting Systems

### 1. RL Training Backtesting (`run_backtest.py`)
- **Purpose**: Validate trained models using the same environment as training
- **Data Source**: Static processed files from `data/final/`
- **Environment**: Uses `TradingEnv` with episode-based processing
- **Use Case**: Quick validation that model learned correctly

### 2. Realistic Backtesting (`run_realistic_backtest.py`)
- **Purpose**: Test real-world performance with actual market conditions
- **Data Source**: Real-time data from Fyers API (Bank Nifty Index)
- **Environment**: Row-by-row processing with realistic position management
- **Use Case**: Evaluate actual trading performance

## 🎯 Realistic Backtesting Logic Flow

### Data Flow
```
1. Fetch Real-time Data (Fyers API)
   ↓
2. Process Features (Technical Indicators)
   ↓
3. Row-by-Row Processing
   ↓
4. Model Prediction for Each Row
   ↓
5. Position Management & Risk Controls
   ↓
6. Trade Execution & P&L Calculation
```

### Prediction & Action Flow

#### 1. **Model Prediction**
- **Input**: Observation vector (1246 dimensions) from last 20 rows
- **Output**: `[action_type, quantity]`
  - `action_type`: 0=BUY_LONG, 1=SELL_SHORT, 2=CLOSE_LONG, 3=CLOSE_SHORT, 4=HOLD
  - `quantity`: Predicted position size (integer)

#### 2. **Position Size Calculation**
```python
# Capital-aware position sizing
max_position_pct = 0.1  # 10% of capital max
max_capital_for_trade = capital * max_position_pct
max_quantity = int(max_capital_for_trade / current_price)
final_quantity = min(predicted_quantity, max_quantity)
```

#### 3. **Entry Logic**
When model predicts BUY_LONG (0) or SELL_SHORT (1):

**Long Entry:**
```python
position = +quantity
entry_price = current_price
stop_loss_price = entry_price - ATR
target_price = entry_price + (2 * ATR)  # 2:1 RR
trailing_stop_price = entry_price * (1 - 0.015)  # 1.5% trailing
capital -= quantity * current_price  # Deduct cost
```

**Short Entry:**
```python
position = -quantity  
entry_price = current_price
stop_loss_price = entry_price + ATR
target_price = entry_price - (2 * ATR)  # 2:1 RR
trailing_stop_price = entry_price * (1 + 0.015)  # 1.5% trailing
capital += quantity * current_price  # Add proceeds
```

#### 4. **Exit Logic**
Positions can be closed by:

**A. Model Signal:**
- Model predicts CLOSE_LONG (2) or CLOSE_SHORT (3)

**B. Automatic Risk Management:**
- **Stop Loss Hit**: Price crosses SL level
- **Target Profit Hit**: Price reaches TP level  
- **Trailing Stop Hit**: Price crosses trailing SL

**C. End of Data:**
- Any open position closed at final price

#### 5. **P&L Calculation**
```python
# Long Position P&L
pnl = quantity * (exit_price - entry_price)

# Short Position P&L  
pnl = quantity * (entry_price - exit_price)

# Update Capital
capital += pnl (for long) or capital -= cost (for short cover)
```

## 🛡️ Risk Management Features

### 1. **Stop Loss & Target Profit**
- **SL Distance**: 1 ATR from entry price
- **TP Distance**: 2 ATR from entry price (2:1 risk-reward)
- **Automatic Execution**: No model intervention needed

### 2. **Trailing Stop Loss**
- **Percentage**: 1.5% from peak price
- **Dynamic Update**: Moves with favorable price movement
- **Protection**: Locks in profits during trends

### 3. **Position Sizing**
- **Maximum Risk**: 10% of capital per trade
- **Capital Awareness**: Prevents over-leveraging
- **Integer Quantities**: Realistic position sizes

### 4. **Price-Based Calculations**
- **No Premium Calculation**: Uses actual index price ± ATR
- **Realistic Spreads**: ATR represents natural price movement
- **Market Impact**: Simplified but realistic

## 📈 Data Sources & Symbols

### Real-time Data (Fyers API)
```python
# Correct Index Symbols
'banknifty': 'NSE:NIFTYBANK-INDEX'    # Bank Nifty Index
'nifty': 'NSE:NIFTY50-INDEX'          # Nifty 50 Index  
'sensex': 'BSE:SENSEX-INDEX'          # BSE Sensex Index
```

### Configuration Parameters
```yaml
# From config/backtesting_config.yaml
data_source:
  default_symbol: "banknifty"
  default_timeframe: "5"      # 5-minute candles
  default_days: 30            # 30 days of data

trading:
  risk_management:
    stop_loss_percentage: 0.02      # 2% SL
    target_profit_percentage: 0.04  # 4% TP  
    trailing_stop_percentage: 0.015 # 1.5% trailing

  position_sizing:
    max_position_percentage: 0.1    # 10% max capital per trade
    min_quantity: 1                 # Minimum 1 unit
```

## 🔧 Usage Examples

### Run Realistic Backtesting
```bash
# Default: Bank Nifty, 5min, 30 days
python run_realistic_backtest.py

# Custom parameters
python run_realistic_backtest.py --symbol banknifty --timeframe 5 --days 7

# Different symbol
python run_realistic_backtest.py --symbol nifty --timeframe 15 --days 14
```

### Run RL Training Backtesting  
```bash
# Static data backtesting
python run_backtest.py

# Real-time data with RL environment
python run_backtest.py --realtime --symbol banknifty
```

## 📊 Output & Results

### Trade Log Example
```
🟢 LONG ENTRY: 5 @ ₹56,850.00
   SL: ₹56,750.00, TP: ₹57,050.00
🟢 LONG EXIT: 5 @ ₹57,100.00 | Reason: TARGET_PROFIT_HIT
   P&L: ₹1,250.00, Capital: ₹101,250.00
```

### Final Results
```
📊 REALISTIC BACKTEST RESULTS
==================================================
Initial Capital: ₹100,000.00
Final Capital:   ₹105,250.00
Total P&L:       ₹5,250.00
Total Return:    5.25%
Total Trades:    12
Win Rate:        66.67%
Profit Factor:   2.15
Max Drawdown:    3.25%
==================================================
```

## 🎯 Key Differences from RL Environment

| Aspect | RL Environment | Realistic Backtesting |
|--------|----------------|----------------------|
| **Data Processing** | Episode-based (1000 steps) | Row-by-row sequential |
| **Position Management** | Simplified for training | Full risk management |
| **Exit Logic** | Model-only decisions | Model + automatic SL/TP |
| **Capital Management** | Abstract rewards | Real capital tracking |
| **Price Calculations** | Proxy premiums | Actual price ± ATR |
| **Trade Validation** | Training-focused | Production-realistic |

## 🚀 Next Steps

1. **Model Optimization**: Tune model for realistic conditions
2. **Strategy Enhancement**: Add more sophisticated entry/exit rules
3. **Risk Management**: Implement portfolio-level risk controls
4. **Performance Analysis**: Add detailed trade analytics
5. **Live Trading**: Integrate with actual trading APIs

This realistic backtesting system provides a bridge between RL training and actual trading, giving you confidence in your model's real-world performance.
