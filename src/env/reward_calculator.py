import numpy as np
from typing import Dict, List, Optional


class RewardCalculator:
    """Handles all reward calculation logic for the trading environment."""
    
    def __init__(self, reward_function: str = "pnl", symbol: str = None):
        self.reward_function = reward_function
        # No instrument-specific normalization - universal scaling only
        self.reward_normalization_factor = 1.0
        
        # Load action and reward configuration for efficient access
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        self.config = config_loader.get_config()  # Store config for later use
        action_config = self.config.get('actions', {})
        self.action_types = action_config.get('action_types', {
            'BUY_LONG': 0, 'SELL_SHORT': 1, 'CLOSE_LONG': 2, 'CLOSE_SHORT': 3, 'HOLD': 4
        })
        
        # Load reward scaling configuration
        reward_config = self.config.get('rewards', {})
        self.global_scaling_factor = reward_config.get('global_scaling_factor', 1.0)
        self.scale_reward_shaping = reward_config.get('scale_reward_shaping', True)
        
        # Track for reward shaping
        self.idle_steps = 0
        self.trade_count = 0
        self.last_action_type = self.action_types.get('HOLD', 4)  # Start with HOLD
        self.previous_trailing_stop = 0.0  # Track trailing stop improvement
        
        # Track returns for sophisticated reward calculations
        self.returns_history = []
        self.equity_history = []
        
    
    def reset(self, initial_capital: float):
        """Reset reward calculator state."""
        self.returns_history = []
        self.equity_history = [initial_capital]
        self.idle_steps = 0
        self.trade_count = 0
        self.last_action_type = 4
        self.previous_trailing_stop = 0.0
    
    def update_tracking_data(self, action_type: int, current_capital: float):
        """Update tracking data for reward calculation."""
        # Update equity history and returns
        self.equity_history.append(current_capital)
        if len(self.equity_history) > 1:
            step_return = (current_capital - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(step_return)
        
        # Track action for reward shaping
        if action_type == self.action_types.get('HOLD', 4):  # HOLD
            self.idle_steps += 1
        else:
            self.idle_steps = 0
            self.trade_count += 1
        
        self.last_action_type = action_type
    
    def calculate_reward(self, current_capital: float, prev_capital: float, 
                        engine) -> float:
        """Calculate reward based on selected reward function."""
        if self.reward_function == "pnl":
            # Include unrealized P&L in reward calculation
            account_state = engine.get_account_state()
            total_pnl_change = (current_capital + account_state['unrealized_pnl']) - prev_capital
            return total_pnl_change
        elif self.reward_function == "sharpe":
            return self._calculate_sharpe_ratio()
        elif self.reward_function == "sortino":
            return self._calculate_sortino_ratio()
        elif self.reward_function == "profit_factor":
            return self._calculate_profit_factor()
        elif self.reward_function == "trading_focused":
            return self._calculate_trading_focused_reward(current_capital, prev_capital, engine)
        elif self.reward_function == "enhanced_trading_focused":
            return self._calculate_enhanced_trading_focused_reward(current_capital, prev_capital, engine)
        else:
            # Default to P&L including unrealized gains/losses
            account_state = engine.get_account_state()
            total_pnl_change = (current_capital + account_state['unrealized_pnl']) - prev_capital
            return total_pnl_change
    
    def apply_reward_shaping(self, base_reward: float, action_type: int, 
                           current_capital: float, prev_capital: float, 
                           engine, current_price: float = None) -> float:
        """Apply reward shaping to guide agent behavior."""
        shaped_reward = base_reward

        # Calculate shaping scale factor - if global scaling is enabled, scale shaping factors too
        shaping_scale = self.global_scaling_factor if self.scale_reward_shaping else 1.0

        # Penalty for idleness (holding no position for too long) 
        if action_type == self.action_types.get('HOLD', 4) and self.idle_steps > 10:  # HOLD for more than 10 steps
            if not engine.get_account_state()['is_position_open']:
                shaped_reward -= (0.1 * shaping_scale) * (self.idle_steps - 10)  # Increasing penalty

        # Enhanced bonus/penalty for trade outcomes
        close_actions = [self.action_types.get('CLOSE_LONG', 2), self.action_types.get('CLOSE_SHORT', 3)]
        if action_type in close_actions:  # CLOSE_LONG or CLOSE_SHORT
            pnl_change = current_capital - prev_capital
            if pnl_change > 0:
                # Larger bonus for profitable trades to encourage winning
                shaped_reward += min(pnl_change * 0.01, 5.0 * shaping_scale)  # Scale with profit, cap scaled
            else:
                # Penalty for losing trades to discourage bad exits
                shaped_reward += max(pnl_change * 0.005, -2.0 * shaping_scale)  # Scale with loss, cap scaled

        # Penalty for over-trading (too many trades in short period)
        if self.trade_count > 0:
            lookback_window = getattr(self, 'lookback_window', 50)
            current_step = getattr(self, 'current_step', 0)
            recent_trade_rate = self.trade_count / max(1, current_step - lookback_window + 1)
            if recent_trade_rate > 0.3:  # More than 30% of steps are trades
                shaped_reward -= (0.5 * shaping_scale) * (recent_trade_rate - 0.3)  # Scaled over-trading penalty

        # Bonus for maintaining profitable positions
        account_state = engine.get_account_state()
        if account_state['is_position_open'] and action_type == self.action_types.get('HOLD', 4):  # HOLD with open position
            unrealized_pnl = account_state['unrealized_pnl']
            if unrealized_pnl > 0:
                # Small bonus for holding profitable positions
                shaped_reward += min(unrealized_pnl * 0.001, 0.5 * shaping_scale)  # Scale with unrealized profit

        # Trailing stop reward shaping
        if current_price is not None:
            trailing_reward = self._calculate_trailing_stop_reward_shaping(action_type, engine, current_price)
            shaped_reward += trailing_reward

        # Reward/penalty for closing positions before stop loss hit
        if current_price is not None:
            sl_reward = self._calculate_stop_loss_proximity_reward(action_type, engine, current_price)
            shaped_reward += sl_reward

        return shaped_reward
    
    
    def normalize_reward(self, reward: float, capital_pct_change: float = None) -> float:
        """
        Normalize reward to -100 to +100 range using percentage-based approach.
        
        Args:
            reward: Raw reward (usually P&L change)
            capital_pct_change: Percentage change in capital for context-aware scaling
            
        Returns:
            Normalized reward in -100 to +100 range
        """
        # Load reward configuration
        reward_config = self.config.get('rewards', {})
        reward_range = reward_config.get('reward_range', {'min': -100.0, 'max': 100.0})
        clipping_config = reward_config.get('reward_clipping', {'enabled': True, 'clip_percentile': 95})
        
        # If we have percentage capital change, use it for more meaningful scaling
        if capital_pct_change is not None:
            # Scale percentage change to reward range
            # 10% gain = +100 reward, 10% loss = -100 reward
            # This creates meaningful rewards regardless of instrument price levels
            scaled_reward = capital_pct_change * 10  # 1% = 10 reward points
            
            # Apply clipping to prevent extreme outliers from dominating
            if clipping_config.get('enabled', True):
                clip_value = reward_range['max']  # Use max range as clip value
                scaled_reward = np.clip(scaled_reward, -clip_value, clip_value)
        else:
            # Fallback: normalize raw reward using global scaling factor
            scaled_reward = reward * self.global_scaling_factor
            
        # Final clipping to reward range
        normalized_reward = np.clip(scaled_reward, reward_range['min'], reward_range['max'])
        
        return float(normalized_reward)
    
    def calculate_percentage_pnl(self, current_capital: float, previous_capital: float) -> float:
        """
        Calculate percentage-based P&L change.
        This is instrument-agnostic and provides meaningful reward scaling.
        
        Args:
            current_capital: Current capital amount
            previous_capital: Previous capital amount
            
        Returns:
            Percentage change in capital
        """
        if previous_capital == 0:
            return 0.0
            
        pct_change = ((current_capital - previous_capital) / previous_capital) * 100
        return float(pct_change)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio based on recent returns."""
        if len(self.returns_history) < 10:  # Need minimum history
            return 0.0

        returns = np.array(self.returns_history[-30:])  # Use last 30 returns
        if np.std(returns) == 0:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = mean_return / std_return
        return sharpe * 10  # Scale for RL training

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(self.returns_history) < 10:
            return 0.0

        returns = np.array(self.returns_history[-30:])
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return mean_return * 10  # No downside, return scaled mean

        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        sortino = mean_return / downside_std
        return sortino * 10  # Scale for RL training

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.returns_history) < 5:
            return 0.0

        returns = np.array(self.returns_history[-30:])
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return gross_profit * 10 if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return (profit_factor - 1.0) * 5  # Center around 0 and scale

    def _calculate_trading_focused_reward(self, current_capital: float, prev_capital: float, engine) -> float:
        """Calculate reward focused on key trading metrics: profit factor, drawdown, win rate."""
        # Include unrealized P&L in base reward calculation
        account_state = engine.get_account_state()
        base_reward = (current_capital + account_state['unrealized_pnl']) - prev_capital

        # Get current trade history for real-time metrics
        trade_history = engine.get_trade_history()

        if len(trade_history) < 2:
            return base_reward  # Not enough trades for metrics

        # Calculate real-time profit factor (only for closing trades)
        recent_trades = trade_history[-20:]  # Last 20 trades for better sample size
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']

        profit_factor_bonus = 0.0
        if len(closing_trades) >= 5:  # Need minimum trades for meaningful PF
            gross_profit = sum(trade['pnl'] for trade in closing_trades if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in closing_trades if trade['pnl'] < 0))

            if gross_loss > 0:
                pf = gross_profit / gross_loss
                # ENHANCED: Much stronger focus on profit factor since profits matter most
                if pf > 3.0:  # Exceptional profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # Massive bonus for exceptional performance
                elif pf > 2.0:  # Excellent profit factor
                    profit_factor_bonus = (pf - 1.0) * 75   # Very high bonus
                elif pf > 1.5:  # Good profit factor
                    profit_factor_bonus = (pf - 1.0) * 50   # High bonus
                elif pf > 1.2:  # Decent profit factor
                    profit_factor_bonus = (pf - 1.0) * 30   # Moderate bonus
                elif pf > 1.0:  # Barely profitable
                    profit_factor_bonus = (pf - 1.0) * 15   # Small bonus
                elif pf < 0.5:  # Terrible profit factor
                    profit_factor_bonus = (pf - 1.0) * 80   # Severe penalty
                elif pf < 0.7:  # Very poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 60   # High penalty
                else:  # Poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 40   # Moderate penalty

        # Calculate real-time win rate (only for closing trades)
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']
        win_rate_bonus = 0.0
        if closing_trades:
            win_rate = sum(1 for trade in closing_trades if trade['pnl'] > 0) / len(closing_trades)
            # REDUCED: Lower emphasis on win rate since profit factor matters more
            if win_rate > 0.7:  # Excellent win rate
                win_rate_bonus = (win_rate - 0.5) * 30  # Moderate bonus
            elif win_rate > 0.6:  # Good win rate
                win_rate_bonus = (win_rate - 0.5) * 20  # Small bonus
            elif win_rate > 0.5:  # Decent win rate
                win_rate_bonus = (win_rate - 0.5) * 10  # Very small bonus
            elif win_rate < 0.3:  # Very poor win rate
                win_rate_bonus = (win_rate - 0.5) * 25  # Moderate penalty
            elif win_rate < 0.4:  # Poor win rate
                win_rate_bonus = (win_rate - 0.5) * 15  # Small penalty

        # Calculate drawdown penalty using existing equity_history
        drawdown_penalty = 0.0
        if len(self.equity_history) > 5:
            recent_capitals = self.equity_history[-20:]  # Last 20 steps
            peak = max(recent_capitals)
            current_dd = (peak - current_capital) / peak if peak > 0 else 0
            drawdown_penalty = -current_dd * 100 if current_dd > 0.05 else 0  # Penalty if DD > 5%

        total_reward = base_reward + profit_factor_bonus + win_rate_bonus + drawdown_penalty
        return total_reward

    def _calculate_enhanced_trading_focused_reward(self, current_capital: float, prev_capital: float, engine) -> float:
        """Enhanced reward function targeting 70%+ win rate with better profit factor."""
        # Include unrealized P&L in base reward calculation
        account_state = engine.get_account_state()
        base_reward = (current_capital + account_state['unrealized_pnl']) - prev_capital

        # Get current trade history for real-time metrics
        trade_history = engine.get_trade_history()

        if len(trade_history) < 2:
            return base_reward  # Not enough trades for metrics

        # Calculate real-time profit factor with enhanced bonuses
        recent_trades = trade_history[-30:]  # Increased sample size for better metrics
        closing_trades = [trade for trade in recent_trades if trade.get('trade_type') == 'CLOSE']

        profit_factor_bonus = 0.0
        if len(closing_trades) >= 3:  # Reduced minimum for faster feedback
            gross_profit = sum(trade['pnl'] for trade in closing_trades if trade['pnl'] > 0)
            gross_loss = abs(sum(trade['pnl'] for trade in closing_trades if trade['pnl'] < 0))

            if gross_loss > 0:
                pf = gross_profit / gross_loss
                # ULTRA-ENHANCED: Massive focus on profit factor for 70%+ performance
                if pf > 4.0:  # Exceptional profit factor
                    profit_factor_bonus = (pf - 1.0) * 200  # Massive bonus
                elif pf > 3.0:  # Excellent profit factor
                    profit_factor_bonus = (pf - 1.0) * 150  # Very high bonus
                elif pf > 2.5:  # Very good profit factor
                    profit_factor_bonus = (pf - 1.0) * 120  # High bonus
                elif pf > 2.0:  # Good profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # Good bonus
                elif pf > 1.5:  # Decent profit factor
                    profit_factor_bonus = (pf - 1.0) * 80   # Moderate bonus
                elif pf > 1.2:  # Barely good
                    profit_factor_bonus = (pf - 1.0) * 50   # Small bonus
                elif pf > 1.0:  # Barely profitable
                    profit_factor_bonus = (pf - 1.0) * 25   # Tiny bonus
                elif pf < 0.4:  # Terrible profit factor
                    profit_factor_bonus = (pf - 1.0) * 150  # Severe penalty
                elif pf < 0.6:  # Very poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 100  # High penalty
                else:  # Poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 75   # Moderate penalty

        # Enhanced win rate calculation targeting 70%+
        win_rate_bonus = 0.0
        if closing_trades:
            win_rate = sum(1 for trade in closing_trades if trade['pnl'] > 0) / len(closing_trades)
            # ENHANCED: Strong bonuses for high win rates (targeting 70%+)
            if win_rate >= 0.8:  # Exceptional win rate (80%+)
                win_rate_bonus = (win_rate - 0.5) * 100  # Massive bonus
            elif win_rate >= 0.7:  # Target win rate (70%+)
                win_rate_bonus = (win_rate - 0.5) * 80   # High bonus
            elif win_rate >= 0.6:  # Good win rate
                win_rate_bonus = (win_rate - 0.5) * 60   # Moderate bonus
            elif win_rate >= 0.5:  # Decent win rate
                win_rate_bonus = (win_rate - 0.5) * 40   # Small bonus
            elif win_rate < 0.3:  # Very poor win rate
                win_rate_bonus = (win_rate - 0.5) * 60   # High penalty
            elif win_rate < 0.4:  # Poor win rate
                win_rate_bonus = (win_rate - 0.5) * 40   # Moderate penalty

        # Enhanced drawdown penalty for better risk management
        drawdown_penalty = 0.0
        if len(self.equity_history) > 5:
            recent_capitals = self.equity_history[-30:]  # Longer history for better DD calculation
            peak = max(recent_capitals)
            current_dd = (peak - current_capital) / peak if peak > 0 else 0
            # Stricter drawdown penalties
            if current_dd > 0.1:  # 10%+ drawdown
                drawdown_penalty = -current_dd * 200  # Severe penalty
            elif current_dd > 0.05:  # 5%+ drawdown
                drawdown_penalty = -current_dd * 100  # High penalty
            elif current_dd > 0.02:  # 2%+ drawdown
                drawdown_penalty = -current_dd * 50   # Moderate penalty

        # Risk-reward ratio bonus
        risk_reward_bonus = 0.0
        if len(closing_trades) >= 3:
            avg_win = np.mean([trade['pnl'] for trade in closing_trades if trade['pnl'] > 0]) if any(trade['pnl'] > 0 for trade in closing_trades) else 0
            avg_loss = abs(np.mean([trade['pnl'] for trade in closing_trades if trade['pnl'] < 0])) if any(trade['pnl'] < 0 for trade in closing_trades) else 1

            if avg_loss > 0:
                risk_reward_ratio = avg_win / avg_loss
                if risk_reward_ratio > 3.0:  # Excellent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 20
                elif risk_reward_ratio > 2.0:  # Good risk-reward
                    risk_reward_bonus = risk_reward_ratio * 15
                elif risk_reward_ratio > 1.5:  # Decent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 10

        total_reward = base_reward + profit_factor_bonus + win_rate_bonus + drawdown_penalty + risk_reward_bonus
        return total_reward

    def _calculate_trailing_stop_reward_shaping(self, action_type: int, engine, current_price: float) -> float:
        """Calculate reward shaping for trailing stops."""
        if not engine._is_position_open:
            return 0.0

        current_trailing_stop = engine._trailing_stop_price
        reward_adjustment = 0.0

        # Bonus for holding profitable position as trailing stop improves
        if action_type == self.action_types.get('HOLD', 4):  # HOLD action
            if self.previous_trailing_stop > 0 and current_trailing_stop > 0:
                # For long positions, trailing stop moving up is good
                if engine._current_position_quantity > 0:
                    if current_trailing_stop > self.previous_trailing_stop:
                        reward_adjustment += 0.1  # Bonus for improving trailing stop
                # For short positions, trailing stop moving down is good
                else:
                    if current_trailing_stop < self.previous_trailing_stop:
                        reward_adjustment += 0.1  # Bonus for improving trailing stop

        # Penalty for closing profitable position prematurely when trend is strong
        elif action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
            distance_to_trail = self._calculate_distance_to_trail(current_price, engine)

            # If distance to trail is large (trend is strong) and position is profitable
            if distance_to_trail > 0.02:  # More than 2% away from trailing stop
                unrealized_pnl = engine.get_account_state()['unrealized_pnl']
                if unrealized_pnl > 0:  # Position is profitable
                    reward_adjustment -= 0.3  # Penalty for premature closing

        # Update previous trailing stop for next iteration
        self.previous_trailing_stop = current_trailing_stop

        return reward_adjustment

    def _calculate_stop_loss_proximity_reward(self, action_type: int, engine, current_price: float) -> float:
        """Calculate reward shaping based on stop loss proximity."""
        if not engine._is_position_open:
            return 0.0

        stop_loss_price = engine._stop_loss_price
        position_quantity = engine._current_position_quantity

        # Calculate distance to stop loss as a percentage of entry price
        account_state = engine.get_account_state()
        entry_price = account_state['current_position_entry_price']
        
        if entry_price == 0:
            return 0.0

        # Calculate distance to stop loss
        if position_quantity > 0:  # Long position
            distance_to_sl = (current_price - stop_loss_price) / entry_price
        else:  # Short position
            distance_to_sl = (stop_loss_price - current_price) / entry_price

        reward_adjustment = 0.0

        # If the agent is closing a position when it's close to stop loss, give a small bonus
        if action_type in [2, 3]:  # CLOSE_LONG or CLOSE_SHORT
            # If we're within 5% of the stop loss, give a small bonus for closing proactively
            if 0 < distance_to_sl < 0.05:  # Within 5% of stop loss
                reward_adjustment += 0.2  # Small bonus for closing before SL hit
            # If we're very close to stop loss (within 1%), give a larger bonus
            elif 0 < distance_to_sl < 0.01:  # Within 1% of stop loss
                reward_adjustment += 0.5  # Larger bonus for closing very close to SL

        # If the agent is holding when very close to stop loss, give a small penalty
        elif action_type == self.action_types.get('HOLD', 4):  # HOLD action
            # If we're very close to stop loss (within 1%), give a small penalty for not closing
            if 0 < distance_to_sl < 0.01:  # Within 1% of stop loss
                reward_adjustment -= 0.1  # Small penalty for holding when close to SL

        return reward_adjustment

    def _calculate_distance_to_trail(self, current_price: float, engine) -> float:
        """Calculate normalized distance to trailing stop."""
        if not engine._is_position_open:
            return 0.0  # No position, no trailing stop

        trailing_stop_price = engine._trailing_stop_price
        if trailing_stop_price == 0:
            return 0.0  # No trailing stop set

        # Calculate distance as percentage of current price
        if engine._current_position_quantity > 0:  # Long position
            distance = (current_price - trailing_stop_price) / current_price
        else:  # Short position
            distance = (trailing_stop_price - current_price) / current_price

        # Normalize to reasonable range [-1, 1]
        return np.clip(distance, -1.0, 1.0)