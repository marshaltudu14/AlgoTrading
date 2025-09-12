import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union


class RewardCalculator:
    """Handles all reward calculation logic for the trading environment."""
    
    def __init__(self, reward_function: str = "pnl", symbol: str = None, 
                 device: str = "cpu", enable_gpu_optimization: bool = False):
        self.reward_function = reward_function
        # No instrument-specific normalization - universal scaling only
        self.reward_normalization_factor = 1.0
        
        # GPU optimization setup
        self.device = torch.device(device)
        self.enable_gpu_optimization = enable_gpu_optimization and (device != "cpu")
        
        # Load action and reward configuration for efficient access
        from src.utils.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        self.config = config_loader.get_config()  # Store config for later use
        action_config = self.config.get('actions', {})
        self.action_types = action_config.get('action_types', {
            'BUY_LONG': 0, 'SELL_SHORT': 1, 'CLOSE_LONG': 2, 'CLOSE_SHORT': 3, 'HOLD': 4
        })
        
        # Load reward scaling configuration - ENHANCED 10X SCALING
        reward_config = self.config.get('rewards', {})
        self.global_scaling_factor = reward_config.get('global_scaling_factor', 10.0)  # INCREASED from 1.0 to 10.0
        self.scale_reward_shaping = reward_config.get('scale_reward_shaping', True)
        
        # Additional reward scaling for better training signals
        self.reward_multiplier = 10.0  # 10x multiplier for all rewards
        
        # Track for reward shaping
        self.idle_steps = 0
        self.trade_count = 0
        self.last_action_type = self.action_types.get('HOLD', 4)  # Start with HOLD
        self.previous_trailing_stop = 0.0  # Track trailing stop improvement
        
        # Track returns for sophisticated reward calculations
        self.returns_history = []
        self.equity_history = []
        
        # GPU-optimized computation cache
        self._gpu_cache = {}
        if self.enable_gpu_optimization:
            self._init_gpu_tensors()
    
    def _init_gpu_tensors(self):
        """Initialize GPU tensors for vectorized computation"""
        self._gpu_cache['returns_tensor'] = torch.empty(0, device=self.device, dtype=torch.float32)
        self._gpu_cache['equity_tensor'] = torch.empty(0, device=self.device, dtype=torch.float32)
        self._gpu_cache['action_tensor'] = torch.empty(0, device=self.device, dtype=torch.long)
    
    def vectorized_reward_calculation(self, 
                                    capital_batch: Union[torch.Tensor, np.ndarray],
                                    prev_capital_batch: Union[torch.Tensor, np.ndarray],
                                    action_batch: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Vectorized reward calculation for batch processing"""
        if not self.enable_gpu_optimization:
            # Fallback to individual calculations
            rewards = []
            for i in range(len(capital_batch)):
                reward = self._calculate_percentage_pnl_simple(
                    float(capital_batch[i]), float(prev_capital_batch[i])
                )
                rewards.append(reward)
            return torch.tensor(rewards, device=self.device, dtype=torch.float32)
        
        # Convert to GPU tensors
        if isinstance(capital_batch, np.ndarray):
            capital_batch = torch.tensor(capital_batch, device=self.device, dtype=torch.float32)
        if isinstance(prev_capital_batch, np.ndarray):
            prev_capital_batch = torch.tensor(prev_capital_batch, device=self.device, dtype=torch.float32)
        if isinstance(action_batch, np.ndarray):
            action_batch = torch.tensor(action_batch, device=self.device, dtype=torch.long)
        
        # Ensure tensors are on correct device
        capital_batch = capital_batch.to(self.device)
        prev_capital_batch = prev_capital_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        
        # Vectorized percentage-based P&L calculation
        pnl_percentage = (capital_batch - prev_capital_batch) / prev_capital_batch
        
        # Apply global scaling (vectorized)
        scaled_rewards = pnl_percentage * self.global_scaling_factor * self.reward_multiplier
        
        # Vectorized reward shaping based on actions
        hold_mask = (action_batch == self.action_types.get('HOLD', 4))
        trade_mask = ~hold_mask
        
        # Apply small penalty for excessive holding (vectorized)
        hold_penalty = torch.where(hold_mask, -0.001, 0.0)
        
        # Apply bonus for active trading (vectorized) 
        trade_bonus = torch.where(trade_mask, 0.01, 0.0)
        
        # Combine all components
        final_rewards = scaled_rewards + hold_penalty + trade_bonus
        
        return final_rewards
    
    def _calculate_percentage_pnl_simple(self, current_capital: float, prev_capital: float) -> float:
        """Simple percentage P&L calculation for fallback"""
        if prev_capital <= 0:
            return 0.0
        
        pnl_percentage = (current_capital - prev_capital) / prev_capital
        return pnl_percentage * self.global_scaling_factor * self.reward_multiplier
    
    def reset(self, initial_capital: float):
        """Reset reward calculator state."""
        self.returns_history = []
        self.equity_history = [initial_capital]
        self.idle_steps = 0  # Reset idle counter
        self.trade_count = 0
        self.last_action_type = self.action_types.get('HOLD', 4)  # Start with HOLD
        self.previous_trailing_stop = 0.0
    
    def update_tracking_data(self, action_type: int, current_capital: float, engine):
        """Update tracking data for reward calculation."""
        # Update equity history and returns
        self.equity_history.append(current_capital)
        if len(self.equity_history) > 1:
            step_return = (current_capital - self.equity_history[-2]) / self.equity_history[-2]
            self.returns_history.append(step_return)
        
        # Track action for reward shaping - FIXED: Reset idle steps properly
        if action_type == self.action_types.get('HOLD', 4):  # HOLD
            # Only increment idle steps if no position is open
            # When position is open, HOLD is not idle - it's position management
            account_state = getattr(engine, '_account_state', {'is_position_open': False})
            if hasattr(engine, 'get_account_state'):
                account_state = engine.get_account_state()
            
            if not account_state.get('is_position_open', False):
                self.idle_steps += 1
            else:
                self.idle_steps = 0  # Reset when managing open position
        else:
            self.idle_steps = 0  # Reset idle counter when taking action
            self.trade_count += 1
        
        self.last_action_type = action_type
    
    def calculate_reward(self, current_capital: float, prev_capital: float, 
                        engine) -> float:
        """Calculate reward based on selected reward function using percentage returns with 10x scaling."""
        base_reward = 0.0
        
        if self.reward_function == "pnl":
            # Use percentage-based P&L for instrument-agnostic rewards
            base_reward = self._calculate_percentage_pnl_reward(current_capital, prev_capital, engine)
        elif self.reward_function == "sharpe":
            base_reward = self._calculate_sharpe_ratio()
        elif self.reward_function == "sortino":
            base_reward = self._calculate_sortino_ratio()
        elif self.reward_function == "profit_factor":
            base_reward = self._calculate_profit_factor()
        elif self.reward_function == "trading_focused":
            base_reward = self._calculate_enhanced_trading_focused_reward(current_capital, prev_capital, engine)
        elif self.reward_function == "enhanced_trading_focused":
            base_reward = self._calculate_enhanced_trading_focused_reward(current_capital, prev_capital, engine)
        else:
            # Default to percentage-based P&L including unrealized gains/losses
            base_reward = self._calculate_percentage_pnl_reward(current_capital, prev_capital, engine)
        
        # Apply 10x scaling to all rewards for better training signals
        scaled_reward = base_reward * self.reward_multiplier * self.global_scaling_factor
        return scaled_reward
    
    def apply_reward_shaping(self, base_reward: float, action_type: int, 
                           current_capital: float, prev_capital: float, 
                           engine, current_price: float = None) -> float:
        """Apply reward shaping to guide agent behavior."""
        shaped_reward = base_reward

        # Calculate shaping scale factor - if global scaling is enabled, scale shaping factors too
        shaping_scale = self.global_scaling_factor if self.scale_reward_shaping else 1.0

        # REMOVED: Negative HOLD penalty that was causing reward decay
        # No penalty for holding when no position is open - this is neutral, not negative
        # The agent should not be penalized for waiting for good opportunities
        # Original problematic code:
        # if action_type == self.action_types.get('HOLD', 4) and self.idle_steps > 10:  # HOLD for more than 10 steps
        #     if not engine.get_account_state()['is_position_open']:
        #         shaped_reward -= (0.1 * shaping_scale) * (self.idle_steps - 10)  # Increasing penalty

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

        # ENHANCED: Bonus for maintaining profitable positions (encourages position management)
        account_state = engine.get_account_state()
        if account_state['is_position_open'] and action_type == self.action_types.get('HOLD', 4):  # HOLD with open position
            unrealized_pnl = account_state['unrealized_pnl']
            if unrealized_pnl > 0:
                # Enhanced bonus for holding profitable positions - this is good behavior!
                profit_bonus = min(unrealized_pnl * 0.01, 2.0 * shaping_scale)  # Increased from 0.001 and 0.5
                shaped_reward += profit_bonus
            # No penalty for holding losing positions - stop losses will handle risk

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
        Return percentage-based reward without clipping.
        True percentage-based rewards don't need artificial limits.
        
        Args:
            reward: Percentage-based reward (1% = 10 points)
            capital_pct_change: Percentage change in capital (optional, for compatibility)
            
        Returns:
            Raw percentage-based reward (no clipping)
        """
        # With percentage-based rewards, no normalization/clipping needed
        # Large moves should get proportionally large rewards
        return float(reward)
    
    def _calculate_percentage_pnl_reward(self, current_capital: float, prev_capital: float, engine) -> float:
        """
        Calculate simple percentage-based reward using entry and exit prices.
        Reward = ((exit_price - entry_price) / entry_price) * 100 * 10
        
        Args:
            current_capital: Current capital amount
            prev_capital: Previous capital amount  
            engine: Trading engine for trade history
            
        Returns:
            Percentage-based reward (1% price move = 10 reward points)
        """
        # Get the latest trade to extract entry and exit prices
        trade_history = engine.get_trade_history()
        
        if not trade_history:
            # No trades yet, return zero
            return 0.0
            
        latest_trade = trade_history[-1]
        
        # Check if this is a closing trade
        action = latest_trade.get('action')
        if action in ['CLOSE_LONG', 'CLOSE_SHORT']:
            exit_price = latest_trade.get('price')  # Current price (exit price)
            
            # Find the corresponding opening trade to get entry price
            entry_price = None
            for trade in reversed(trade_history[:-1]):  # Look backwards, exclude current trade
                if trade.get('action') in ['BUY_LONG', 'SELL_SHORT']:
                    entry_price = trade.get('price')
                    break
            
            if entry_price and exit_price and entry_price > 0:
                if action == 'CLOSE_LONG':
                    # Long position: profit when exit > entry
                    price_pct_change = ((exit_price - entry_price) / entry_price) * 100
                elif action == 'CLOSE_SHORT':
                    # Short position: profit when exit < entry  
                    price_pct_change = ((entry_price - exit_price) / entry_price) * 100
                else:
                    # Fallback
                    price_pct_change = ((exit_price - entry_price) / entry_price) * 100
                
                # Return percentage change as reward with enhanced scaling
                # 1% price move now = 10.0 reward points (10x scaling)
                return float(price_pct_change)
        
        # For non-closing trades or missing price info, fallback to capital-based
        step_pnl = current_capital - prev_capital
        if abs(step_pnl) < 0.01:
            return 0.0
            
        # Simple capital percentage for opening trades, holds, etc.
        initial_capital = getattr(engine, '_initial_capital', 100000.0)
        capital_pct_change = (step_pnl / initial_capital) * 100
        
        # Return consistent with price-based rewards with enhanced scaling
        return float(capital_pct_change)
    
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

        # GPU-optimized computation when available
        import torch
        if torch.cuda.is_available():
            # GPU: Use tensor operations
            returns = torch.tensor(self.returns_history[-30:], dtype=torch.float32, device='cuda')
            if torch.std(returns).item() == 0:
                return 0.0
            mean_return = torch.mean(returns).item()
            std_return = torch.std(returns).item()
        else:
            # CPU: Use numpy when GPU unavailable
            import numpy as np
            returns = np.array(self.returns_history[-30:])
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

        # GPU-optimized computation when available
        import torch
        if torch.cuda.is_available():
            # GPU: Use tensor operations
            returns = torch.tensor(self.returns_history[-30:], dtype=torch.float32, device='cuda')
            mean_return = torch.mean(returns).item()
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return mean_return * 10
            downside_std = torch.std(downside_returns).item()
        else:
            # CPU: Use numpy when GPU unavailable
            import numpy as np
            returns = np.array(self.returns_history[-30:])
            mean_return = np.mean(returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return mean_return * 10
            downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0

        sortino = mean_return / downside_std
        return sortino * 10  # Scale for RL training

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        if len(self.returns_history) < 5:
            return 0.0

        # GPU-optimized computation when available
        import torch
        if torch.cuda.is_available():
            # GPU: Use tensor operations
            returns = torch.tensor(self.returns_history[-30:], dtype=torch.float32, device='cuda')
            gross_profit = torch.sum(returns[returns > 0]).item()
            gross_loss = abs(torch.sum(returns[returns < 0]).item())
        else:
            # CPU: Use numpy when GPU unavailable
            import numpy as np
            returns = np.array(self.returns_history[-30:])
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))

        if gross_loss == 0:
            return gross_profit * 10 if gross_profit > 0 else 0.0

        profit_factor = gross_profit / gross_loss
        return (profit_factor - 1.0) * 5  # Center around 0 and scale

    def _calculate_streak_bonus(self, engine) -> float:
        """Calculate bonus/penalty based on current winning/losing streaks"""
        trade_history = engine.get_trade_history()
        if not trade_history:
            return 0.0
        
        # Get only closing trades
        closing_trades = [trade for trade in trade_history if trade.get('trade_type') == 'CLOSE']
        if len(closing_trades) < 2:
            return 0.0
        
        # Calculate current streak (looking backwards from most recent trade)
        current_streak = 0
        streak_type = None  # 'win' or 'loss'
        
        for trade in reversed(closing_trades):
            trade_pnl = trade.get('pnl', 0)
            is_win = trade_pnl > 0
            
            if streak_type is None:
                # First trade sets the streak type
                streak_type = 'win' if is_win else 'loss'
                current_streak = 1
            elif (streak_type == 'win' and is_win) or (streak_type == 'loss' and not is_win):
                # Streak continues
                current_streak += 1
            else:
                # Streak broken
                break
        
        # Apply streak bonuses/penalties
        if streak_type == 'win':
            # Winning streak bonuses (exponential growth to encourage consistency)
            if current_streak >= 5:
                return min(15.0 * (current_streak - 4), 50.0)  # Cap at +50
            elif current_streak >= 3:
                return 5.0 * (current_streak - 2)  # +5 for 3rd win, +10 for 4th
            else:
                return 0.0  # No bonus for streaks < 3
        else:
            # Losing streak penalties (motivate to break the pattern)
            if current_streak >= 4:
                return max(-10.0 * (current_streak - 3), -40.0)  # Cap at -40
            elif current_streak >= 2:
                return -3.0 * (current_streak - 1)  # -3 for 2nd loss, -6 for 3rd
            else:
                return 0.0  # No penalty for single loss
        
        return 0.0

    def _calculate_enhanced_trading_focused_reward(self, current_capital: float, prev_capital: float, engine) -> float:
        """Enhanced reward function targeting 50%+ win rate with better profit factor (realistic with 1:2 RR)."""
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
                # BALANCED: Strong focus on profit factor with reasonable multipliers
                if pf > 4.0:  # Exceptional profit factor
                    profit_factor_bonus = (pf - 1.0) * 50   # High bonus (PF=4.5 → +150)
                elif pf > 3.0:  # Excellent profit factor
                    profit_factor_bonus = (pf - 1.0) * 40   # Good bonus (PF=3.5 → +100)
                elif pf > 2.5:  # Very good profit factor
                    profit_factor_bonus = (pf - 1.0) * 30   # Moderate bonus (PF=3.0 → +60)
                elif pf > 2.0:  # Good profit factor
                    profit_factor_bonus = (pf - 1.0) * 25   # Small bonus (PF=2.5 → +37.5)
                elif pf > 1.5:  # Decent profit factor
                    profit_factor_bonus = (pf - 1.0) * 20   # Tiny bonus (PF=2.0 → +20)
                elif pf > 1.2:  # Barely good
                    profit_factor_bonus = (pf - 1.0) * 15   # Minimal bonus (PF=1.5 → +7.5)
                elif pf > 1.0:  # Barely profitable
                    profit_factor_bonus = (pf - 1.0) * 10   # Very small bonus (PF=1.2 → +2)
                elif pf < 0.4:  # Terrible profit factor
                    profit_factor_bonus = (pf - 1.0) * 60   # Severe penalty (PF=0.3 → -42)
                elif pf < 0.6:  # Very poor profit factor
                    profit_factor_bonus = (pf - 1.0) * 40   # High penalty (PF=0.5 → -20)
                else:  # Poor profit factor (0.6-1.0)
                    profit_factor_bonus = (pf - 1.0) * 25   # Moderate penalty (PF=0.8 → -5)

        # Enhanced win rate calculation targeting 50%+ (realistic with 1:2 RR)
        win_rate_bonus = 0.0
        if closing_trades:
            win_rate = sum(1 for trade in closing_trades if trade['pnl'] > 0) / len(closing_trades)
            # ENHANCED: Strong bonuses for high win rates (50%+ profitable with 1:2 RR)
            if win_rate >= 0.7:  # Exceptional win rate (70%+)
                win_rate_bonus = (win_rate - 0.5) * 50  # Massive bonus for exceptional accuracy
            elif win_rate >= 0.6:  # Excellent win rate (60-69%)
                win_rate_bonus = (win_rate - 0.5) * 40  # High bonus
            elif win_rate >= 0.5:  # Good win rate (50-59%) - profitable with 1:2 RR
                win_rate_bonus = (win_rate - 0.5) * 30  # Moderate bonus
            elif win_rate >= 0.4:  # Decent win rate (40-49%)
                win_rate_bonus = (win_rate - 0.5) * 10  # Small bonus
            elif win_rate < 0.3:  # Very poor win rate (<30%)
                win_rate_bonus = (win_rate - 0.5) * 40  # High penalty
            elif win_rate < 0.4:  # Poor win rate (30-39%)
                win_rate_bonus = (win_rate - 0.5) * 25  # Moderate penalty

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
            # GPU-optimized computation when available
            import torch
            winning_trades = [trade['pnl'] for trade in closing_trades if trade['pnl'] > 0]
            losing_trades = [trade['pnl'] for trade in closing_trades if trade['pnl'] < 0]
            
            if torch.cuda.is_available():
                # GPU: Use tensor operations
                avg_win = torch.tensor(winning_trades, device='cuda').mean().item() if winning_trades else 0
                avg_loss = abs(torch.tensor(losing_trades, device='cuda').mean().item()) if losing_trades else 1
            else:
                # CPU: Use numpy when GPU unavailable
                import numpy as np
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = abs(np.mean(losing_trades)) if losing_trades else 1

            if avg_loss > 0:
                risk_reward_ratio = avg_win / avg_loss
                if risk_reward_ratio > 3.0:  # Excellent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 20
                elif risk_reward_ratio > 2.0:  # Good risk-reward
                    risk_reward_bonus = risk_reward_ratio * 15
                elif risk_reward_ratio > 1.5:  # Decent risk-reward
                    risk_reward_bonus = risk_reward_ratio * 10

        # Calculate streak bonus/penalty
        streak_bonus = self._calculate_streak_bonus(engine)

        # Apply 10x scaling to enhanced trading focused reward
        total_reward = (base_reward + profit_factor_bonus + win_rate_bonus + drawdown_penalty + risk_reward_bonus + streak_bonus)
        scaled_total_reward = total_reward * self.reward_multiplier
        return scaled_total_reward

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

        # Check for invalid current price
        if current_price <= 0:
            return 0.0
            
        # Calculate distance as percentage of current price
        if engine._current_position_quantity > 0:  # Long position
            distance = (current_price - trailing_stop_price) / current_price
        else:  # Short position
            distance = (trailing_stop_price - current_price) / current_price

        # Normalize to reasonable range [-1, 1]
        return np.clip(distance, -1.0, 1.0)