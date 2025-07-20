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

def calculate_comprehensive_metrics(trade_history: List[Dict], capital_history: List[float],
                                  initial_capital: float, total_episodes: int,
                                  total_reward: float = 0.0) -> Dict:
    """
    Calculate comprehensive trading metrics for end-of-training summary.

    Args:
        trade_history: List of trade dictionaries with 'pnl' key
        capital_history: List of capital values over time
        initial_capital: Starting capital amount
        total_episodes: Number of training episodes
        total_reward: Total reward accumulated during training

    Returns:
        Dictionary with all key trading metrics
    """
    if not trade_history:
        return {
            'win_rate': 0.0,
            'total_trades': 0,
            'total_pnl': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_percentage': 0.0,
            'total_return_percentage': 0.0,
            'avg_pnl_per_trade': 0.0,
            'avg_reward_per_episode': 0.0,
            'total_reward': total_reward
        }

    # Basic metrics
    win_rate = calculate_win_rate(trade_history)
    total_trades = calculate_num_trades(trade_history)
    total_pnl = calculate_total_pnl(trade_history)
    profit_factor = calculate_profit_factor(trade_history)
    avg_pnl_per_trade = calculate_avg_pnl_per_trade(trade_history)

    # Returns and Sharpe ratio
    if capital_history and len(capital_history) > 1:
        final_capital = capital_history[-1]
        total_return_percentage = ((final_capital - initial_capital) / initial_capital) * 100

        # Calculate period returns for Sharpe ratio
        returns = []
        for i in range(1, len(capital_history)):
            if capital_history[i-1] > 0:
                period_return = (capital_history[i] - capital_history[i-1]) / capital_history[i-1]
                returns.append(period_return)

        if returns:
            returns_series = pd.Series(returns)
            sharpe_ratio = calculate_sharpe_ratio(returns_series)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        capital_series = pd.Series(capital_history)
        max_drawdown_percentage = calculate_max_drawdown(capital_series) * 100
    else:
        total_return_percentage = 0.0
        sharpe_ratio = 0.0
        max_drawdown_percentage = 0.0

    # Average reward per episode
    avg_reward_per_episode = total_reward / total_episodes if total_episodes > 0 else 0.0

    return {
        'win_rate': win_rate,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_percentage': max_drawdown_percentage,
        'total_return_percentage': total_return_percentage,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'avg_reward_per_episode': avg_reward_per_episode,
        'total_reward': total_reward
    }
