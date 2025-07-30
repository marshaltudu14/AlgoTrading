import pandas as pd
import numpy as np
from typing import List, Dict

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std(ddof=0)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio using downside deviation."""
    if returns.empty:
        return 0.0

    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0

    downside_deviation = downside_returns.std(ddof=0)
    if downside_deviation == 0:
        return 0.0

    return excess_returns.mean() / downside_deviation

def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (Annual Return / Max Drawdown)."""
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0
    return abs(total_return / max_drawdown)

def calculate_recovery_factor(total_return: float, max_drawdown: float) -> float:
    """Calculate recovery factor (Net Profit / Max Drawdown)."""
    if max_drawdown == 0:
        return float('inf') if total_return > 0 else 0.0
    return abs(total_return / max_drawdown)

def calculate_expectancy(trade_history: List[Dict]) -> float:
    """Calculate expectancy per trade."""
    if not trade_history:
        return 0.0

    winning_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) > 0]
    losing_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) < 0]

    if not winning_trades and not losing_trades:
        return 0.0

    win_rate = len(winning_trades) / len(trade_history)
    loss_rate = len(losing_trades) / len(trade_history)

    avg_win = np.mean(winning_trades) if winning_trades else 0.0
    avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0

    return (win_rate * avg_win) - (loss_rate * avg_loss)

def calculate_total_pnl(trade_history: List[Dict]) -> float:
    return sum(trade.get('pnl', 0) for trade in trade_history)

def calculate_profit_factor(trade_history: List[Dict]) -> float:
    gross_profit = sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) > 0)
    gross_loss = sum(trade.get('pnl', 0) for trade in trade_history if trade.get('pnl', 0) < 0)
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return abs(gross_profit / gross_loss)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown as a negative percentage.
    Returns negative value for drawdowns, 0.0 for no drawdown.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return 0.0

    # Calculate running maximum (peak)
    peak = equity_curve.expanding(min_periods=1).max()

    # Calculate drawdown as percentage from peak
    drawdown = (equity_curve - peak) / peak

    # Return the minimum (most negative) drawdown
    # This will be negative for actual drawdowns, 0 or positive for no drawdown
    min_drawdown = drawdown.min()

    # Ensure we return a negative value for actual drawdowns
    if min_drawdown >= 0:
        return 0.0  # No drawdown
    else:
        return min_drawdown  # Return negative value

def calculate_win_rate(trade_history: List[Dict]) -> float:
    if not trade_history:
        return 0.0

    # Only consider closing trades for win rate calculation
    closing_trades = [trade for trade in trade_history if trade.get('trade_type') == 'CLOSE']
    if not closing_trades:
        return 0.0

    winning_trades = sum(1 for trade in closing_trades if trade.get('pnl', 0) > 0)
    return winning_trades / len(closing_trades)

def calculate_avg_pnl_per_trade(trade_history: List[Dict]) -> float:
    if not trade_history:
        return 0.0

    # Only consider closing trades for P&L calculation
    closing_trades = [trade for trade in trade_history if trade.get('trade_type') == 'CLOSE']
    if not closing_trades:
        return 0.0

    return sum(trade.get('pnl', 0) for trade in closing_trades) / len(closing_trades)

def calculate_num_trades(trade_history: List[Dict]) -> int:
    # Only count closing trades as completed trades
    closing_trades = [trade for trade in trade_history if trade.get('trade_type') == 'CLOSE']
    return len(closing_trades)

def calculate_avg_winning_trade(trade_history: List[Dict]) -> float:
    """Calculate average winning trade amount."""
    winning_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) > 0]
    return np.mean(winning_trades) if winning_trades else 0.0

def calculate_avg_losing_trade(trade_history: List[Dict]) -> float:
    """Calculate average losing trade amount."""
    losing_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) < 0]
    return np.mean(losing_trades) if losing_trades else 0.0

def calculate_largest_winning_trade(trade_history: List[Dict]) -> float:
    """Calculate largest winning trade."""
    winning_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) > 0]
    return max(winning_trades) if winning_trades else 0.0

def calculate_largest_losing_trade(trade_history: List[Dict]) -> float:
    """Calculate largest losing trade."""
    losing_trades = [trade['pnl'] for trade in trade_history if trade.get('pnl', 0) < 0]
    return min(losing_trades) if losing_trades else 0.0

def calculate_consecutive_wins_losses(trade_history: List[Dict]) -> Dict[str, int]:
    """Calculate maximum consecutive wins and losses."""
    if not trade_history:
        return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0}

    max_wins = 0
    max_losses = 0
    current_wins = 0
    current_losses = 0

    for trade in trade_history:
        pnl = trade.get('pnl', 0)
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_wins = max(max_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_losses = max(max_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return {'max_consecutive_wins': max_wins, 'max_consecutive_losses': max_losses}

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
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'recovery_factor': 0.0,
            'expectancy': 0.0,
            'max_drawdown_percentage': 0.0,
            'total_return_percentage': 0.0,
            'avg_pnl_per_trade': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            'largest_winning_trade': 0.0,
            'largest_losing_trade': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'avg_reward_per_episode': 0.0,
            'total_reward': total_reward
        }

    # Basic metrics
    win_rate = calculate_win_rate(trade_history)
    total_trades = calculate_num_trades(trade_history)
    total_pnl = calculate_total_pnl(trade_history)
    profit_factor = calculate_profit_factor(trade_history)
    avg_pnl_per_trade = calculate_avg_pnl_per_trade(trade_history)

    # Trade statistics
    avg_winning_trade = calculate_avg_winning_trade(trade_history)
    avg_losing_trade = calculate_avg_losing_trade(trade_history)
    largest_winning_trade = calculate_largest_winning_trade(trade_history)
    largest_losing_trade = calculate_largest_losing_trade(trade_history)
    consecutive_stats = calculate_consecutive_wins_losses(trade_history)
    expectancy = calculate_expectancy(trade_history)

    # Returns and advanced ratios
    if capital_history and len(capital_history) > 1:
        final_capital = capital_history[-1]
        total_return_percentage = ((final_capital - initial_capital) / initial_capital) * 100

        # Calculate period returns for ratio calculations
        returns = []
        for i in range(1, len(capital_history)):
            if capital_history[i-1] > 0:
                period_return = (capital_history[i] - capital_history[i-1]) / capital_history[i-1]
                returns.append(period_return)

        if returns:
            returns_series = pd.Series(returns)
            sharpe_ratio = calculate_sharpe_ratio(returns_series)
            sortino_ratio = calculate_sortino_ratio(returns_series)
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Max drawdown (already returns negative percentage)
        capital_series = pd.Series(capital_history)
        max_drawdown_percentage = calculate_max_drawdown(capital_series) * 100

        # Calmar and Recovery ratios
        calmar_ratio = calculate_calmar_ratio(total_return_percentage, abs(max_drawdown_percentage))
        recovery_factor = calculate_recovery_factor(total_pnl, abs(max_drawdown_percentage))
    else:
        total_return_percentage = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
        max_drawdown_percentage = 0.0
        calmar_ratio = 0.0
        recovery_factor = 0.0

    # Average reward per episode
    avg_reward_per_episode = total_reward / total_episodes if total_episodes > 0 else 0.0

    return {
        'win_rate': win_rate,
        'total_trades': total_trades,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'recovery_factor': recovery_factor,
        'expectancy': expectancy,
        'max_drawdown_percentage': max_drawdown_percentage,
        'total_return_percentage': total_return_percentage,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'avg_winning_trade': avg_winning_trade,
        'avg_losing_trade': avg_losing_trade,
        'largest_winning_trade': largest_winning_trade,
        'largest_losing_trade': largest_losing_trade,
        'max_consecutive_wins': consecutive_stats['max_consecutive_wins'],
        'max_consecutive_losses': consecutive_stats['max_consecutive_losses'],
        'avg_reward_per_episode': avg_reward_per_episode,
        'total_reward': total_reward
    }
