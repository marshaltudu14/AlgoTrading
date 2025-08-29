from typing import Tuple, Optional
from .trading_mode import TradingMode


class TerminationManager:
    """Handles termination conditions for the trading environment."""
    
    def __init__(self, mode: TradingMode, max_drawdown_pct: float = 0.20):
        self.mode = mode
        self.max_drawdown_pct = max_drawdown_pct
        self.peak_equity = 0.0
        self.episode_end_step = None
        
    def reset(self, initial_capital: float, episode_end_step: Optional[int] = None):
        """Reset termination manager state."""
        self.peak_equity = initial_capital
        self.episode_end_step = episode_end_step
        
    def update_peak_equity(self, current_capital: float):
        """Update peak equity tracking."""
        if current_capital > self.peak_equity:
            self.peak_equity = current_capital
    
    def check_termination_conditions(self, current_step: int, data_length: int, 
                                   current_capital: float) -> Tuple[bool, Optional[str]]:
        """Check if episode should terminate due to risk management conditions or strategic episode end."""

        # Check if we've reached the end of available data
        if current_step >= data_length - 1:
            return True, f"end_of_data_step_{current_step}"

        if self.mode == TradingMode.TRAINING:
            # TRAINING mode: Use episode-based termination
            # Check strategic episode end (if using data feeding strategy)
            if self.episode_end_step is not None and current_step >= self.episode_end_step:
                return True, f"strategic_episode_end_step_{current_step}"

            # Update peak equity
            self.update_peak_equity(current_capital)

            # Check maximum drawdown
            current_drawdown = (self.peak_equity - current_capital) / self.peak_equity
            if current_drawdown > self.max_drawdown_pct:
                return True, f"max_drawdown_exceeded_{current_drawdown:.2%}"

            # No capital constraints for index trading - removed insufficient capital check

        else:
            # BACKTESTING/LIVE modes: No early termination, process all data
            # Update peak equity for tracking
            self.update_peak_equity(current_capital)

        return False, None
    
    def force_close_positions(self, engine, data, current_step: int):
        """Force close any open positions at the end of an episode."""
        account_state = engine.get_account_state()

        if account_state['is_position_open']:
            current_price = data['close'].iloc[min(current_step, len(data) - 1)]
            position_quantity = account_state['current_position_quantity']

            if position_quantity > 0:
                # Close long position
                engine.execute_trade("CLOSE_LONG", current_price, abs(position_quantity))
            elif position_quantity < 0:
                # Close short position
                engine.execute_trade("CLOSE_SHORT", current_price, abs(position_quantity))