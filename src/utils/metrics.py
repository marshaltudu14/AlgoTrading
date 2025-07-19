import pandas as pd
import numpy as np
from typing import List, Dict

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std(ddof=0)

def calculate_total_pnl(trade_history: List[Dict]) -> float:
    return sum(trade.get('pnl', 0) for trade in trade_history)

def calculate_profit_factor(trade_history: List[Dict]) -> float:
    gross_profit = sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0)
    gross_loss = sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) < 0)
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return abs(gross_profit / gross_loss)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return abs(drawdown.min())

def calculate_win_rate(trade_history: List[Dict]) -> float:
    if not trade_history:
        return 0.0
    winning_trades = sum(1 for trade in trade_history if trade.get('pnl', 0) > 0)
    return winning_trades / len(trade_history)

def calculate_avg_pnl_per_trade(trade_history: List[Dict]) -> float:
    if not trade_history:
        return 0.0
    return sum(trade.get('pnl', 0) for trade in trade_history) / len(trade_history)

def calculate_num_trades(trade_history: List[Dict]) -> int:
    return len(trade_history)
