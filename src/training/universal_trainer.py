#!/usr/bin/env python3
"""
Universal Trainer with Timeframe-Aware Symbol Rotation
======================================================

This trainer rotates through symbols with timeframe progression (higher to lower)
to create a universal model that can handle diverse market conditions with 
curriculum learning benefits.
"""

import logging
import random
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.backtesting.environment import TradingEnv
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import DataLoader
from src.utils.metrics import (
    calculate_win_rate, calculate_total_pnl, calculate_num_trades,
    calculate_profit_factor, calculate_avg_pnl_per_trade
)

logger = logging.getLogger(__name__)


class UniversalTrainer:
    """
    Universal trainer with timeframe-aware symbol rotation for curriculum learning.
    Progresses from higher timeframes (less noisy) to lower timeframes (more noisy).
    """
    
    def __init__(self, agent, symbols: List[str], data_loader: DataLoader, 
                 num_episodes: int, log_interval: int = 10, config: Dict = None,
                 research_logger: 'ResearchLogger' = None):
        self.agent = agent
        self.symbols = symbols
        self.data_loader = data_loader
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.config = config or {}
        self.research_logger = research_logger
        
        # Group symbols by timeframe for batch processing
        self.symbol_groups = self._group_symbols_by_timeframe()
        self.current_group_index = 0
        self.current_symbol_index_in_group = 0
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.total_reward = 0.0
        self.symbol_episode_count = {symbol: 0 for symbol in symbols}
        self.cumulative_trade_history = []
        self.cumulative_capital_history = []
        
        logger.info(f"🎯 Universal Trainer initialized with {len(symbols)} symbols across {len(self.symbol_groups)} timeframe groups")
        for timeframe, group_symbols in self.symbol_groups.items():
            logger.info(f"   📊 Timeframe {timeframe}: {len(group_symbols)} symbols")

    def _group_symbols_by_timeframe(self) -> Dict[str, List[str]]:
        """Group symbols by their timeframe for batch processing."""
        groups = {}
        for symbol in self.symbols:
            # Extract timeframe from symbol (assuming format like SYMBOL_TIMEFRAME)
            parts = symbol.split('_')
            if len(parts) >= 2:
                timeframe = parts[-1]  # Last part is timeframe
            else:
                timeframe = "unknown"
            
            if timeframe not in groups:
                groups[timeframe] = []
            groups[timeframe].append(symbol)
        
        # Sort timeframes numerically for processing order (lower timeframes first)
        sorted_groups = {}
        try:
            # Try to sort numerically
            sorted_timeframes = sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            for tf in sorted_timeframes:
                sorted_groups[tf] = groups[tf]
        except:
            # Fallback to alphabetical sorting
            for tf in sorted(groups.keys()):
                sorted_groups[tf] = groups[tf]
        
        return sorted_groups

    def _sort_symbols_by_timeframe(self, symbols: List[str]) -> List[str]:
        """
        Sort symbols by timeframe (highest to lowest) for curriculum learning.
        Expects symbols in format: 'symbol_timeframe' (e.g., 'Bankex_180', 'RELIANCE_1')
        """
        def extract_timeframe(symbol: str) -> int:
            # Extract timeframe from symbol name (e.g., 'Bankex_180' -> 180)
            parts = symbol.split('_')
            if len(parts) >= 2:
                try:
                    return int(parts[-1])  # Last part should be timeframe
                except ValueError:
                    pass
            return 0  # Default timeframe for symbols without explicit timeframe
        
        # Sort by timeframe in descending order (highest to lowest)
        sorted_symbols = sorted(symbols, key=extract_timeframe, reverse=True)
        
        # Group by timeframe for logging
        timeframe_groups = {}
        for symbol in sorted_symbols:
            tf = extract_timeframe(symbol)
            if tf not in timeframe_groups:
                timeframe_groups[tf] = []
            timeframe_groups[tf].append(symbol)
        
        logger.info("📊 Timeframe-based symbol grouping (curriculum order):")
        for tf in sorted(timeframe_groups.keys(), reverse=True):
            logger.info(f"   {tf}min timeframe: {timeframe_groups[tf]}")
        
        return sorted_symbols

    def _get_next_symbol(self, episode: int) -> str:
        """Get the next symbol for training using batch timeframe rotation strategy."""
        # Get current timeframe group
        timeframe_groups = list(self.symbol_groups.keys())
        current_timeframe = timeframe_groups[self.current_group_index]
        current_group_symbols = self.symbol_groups[current_timeframe]
        
        # Get next symbol within the current group
        symbol = current_group_symbols[self.current_symbol_index_in_group]
        self.symbol_episode_count[symbol] += 1
        
        # Move to next symbol in group
        self.current_symbol_index_in_group += 1
        
        # If we've processed all symbols in current group, move to next timeframe group
        if self.current_symbol_index_in_group >= len(current_group_symbols):
            self.current_symbol_index_in_group = 0
            self.current_group_index = (self.current_group_index + 1) % len(timeframe_groups)
            
            # Log group completion
            logger.info(f"✅ Completed processing all symbols for timeframe {current_timeframe}")
        
        return symbol

    def _convert_epoch_to_readable(self, epoch_timestamp: float) -> str:
        """Convert epoch timestamp to readable datetime format."""
        try:
            if pd.isna(epoch_timestamp) or epoch_timestamp == 0:
                return "N/A"
            dt = datetime.fromtimestamp(epoch_timestamp)
            return dt.strftime('%d %B %H:%M:%S')
        except (ValueError, OSError, OverflowError):
            return f"Epoch_{int(epoch_timestamp)}"

    def _calculate_real_time_win_rate(self) -> float:
        """Calculate real-time win rate from cumulative trades."""
        if not self.cumulative_trade_history:
            return 0.0

        closing_trades = [trade for trade in self.cumulative_trade_history if trade.get('trade_type') == 'CLOSE']
        if not closing_trades:
            return 0.0

        winning_trades = sum(1 for trade in closing_trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(closing_trades)

    def _calculate_win_rate_from_trades(self, trades: list) -> float:
        """Calculate win rate from a given list of trades."""
        if not trades:
            return 0.0

        closing_trades = [trade for trade in trades if trade.get('trade_type') == 'CLOSE']
        if not closing_trades:
            return 0.0

        winning_trades = sum(1 for trade in closing_trades if trade.get('pnl', 0) > 0)
        return winning_trades / len(closing_trades)

    def _get_exit_reason_from_info(self, info: Dict) -> str:
        """Extract exit reason from step info."""
        if 'exit_reason' in info:
            return info['exit_reason']
        elif 'engine_info' in info and 'exit_reason' in info['engine_info']:
            return info['engine_info']['exit_reason']
        return ""

    def train(self) -> None:
        """Run the universal training loop with symbol rotation."""
        logger.info(f"🚀 Starting Universal Training for {self.num_episodes} episodes")
        
        env_config = self.config.get('environment', {})
        initial_capital = env_config.get('initial_capital', 100000.0)
        
        # Initialize research logging
        if self.research_logger:
            self.research_logger.start_training(self.num_episodes, self.symbols)
        
        for episode in range(self.num_episodes):
            # Get symbol for this episode
            current_symbol = self._get_next_symbol(episode)
            
            # Get current timeframe for logging
            timeframe_groups = list(self.symbol_groups.keys())
            current_timeframe = timeframe_groups[self.current_group_index]
            
            logger.info(f"📊 Episode {episode + 1}/{self.num_episodes} | Symbol: {current_symbol} | Timeframe: {current_timeframe}")
            
            # Initialize research logging for episode
            if self.research_logger:
                self.research_logger.start_episode(episode + 1, current_symbol, initial_capital)
            
            # Create environment for current symbol
            env = TradingEnv(
                data_loader=self.data_loader,
                symbol=current_symbol,
                initial_capital=initial_capital,
                lookback_window=env_config.get('lookback_window', 128),  # Updated to 128 for comprehensive market view
                episode_length=env_config.get('episode_length', 5000),   # Updated to 5000 for better market exposure
                reward_function=env_config.get('reward_function', "trading_focused"),
                use_streaming=env_config.get('use_streaming', False),
                trailing_stop_percentage=env_config.get('trailing_stop_percentage', 0.02)
            )
            
            # Run episode
            episode_reward, episode_trades = self._run_episode(env, episode + 1, current_symbol)
            
            # Track episode metrics
            self.episode_rewards.append(episode_reward)
            self.total_reward += episode_reward
            
            # Track capital history
            current_capital = env.engine.get_account_state()['capital']
            self.capital_history.append(current_capital)
            self.cumulative_capital_history.append(current_capital)
            
            # Accumulate trade history from this episode
            if episode_trades:
                # Add episode number and symbol to each trade for tracking
                for trade in episode_trades:
                    trade['episode'] = episode + 1
                    trade['symbol'] = current_symbol
                self.cumulative_trade_history.extend(episode_trades)
            
            # Track symbol performance
            self.symbol_performance[current_symbol].append(episode_reward)
            
            # Calculate episode-specific metrics for research logging
            episode_win_rate = self._calculate_win_rate_from_trades(episode_trades) if episode_trades else 0.0
            final_capital = env.engine.get_account_state()['capital']
            
            episode_metrics = {
                'episode': episode + 1,
                'symbol': current_symbol,
                'episode_reward': episode_reward,
                'episode_win_rate': episode_win_rate,
                'episode_trades': len([t for t in episode_trades if t.get('trade_type') == 'CLOSE']) if episode_trades else 0,
                'final_capital': final_capital,
                'total_pnl': final_capital - initial_capital
            }
            
            # End episode research logging
            if self.research_logger:
                self.research_logger.end_episode(episode_metrics)
            
            # Log progress at intervals
            if (episode + 1) % self.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
                account_state = env.engine.get_account_state()
                
                # Calculate win rate and other metrics from cumulative trades
                if self.cumulative_trade_history:
                    win_rate = calculate_win_rate(self.cumulative_trade_history)
                    total_pnl = calculate_total_pnl(self.cumulative_trade_history)
                    num_trades = calculate_num_trades(self.cumulative_trade_history)
                    
                    self._log_progress_detailed(episode + 1, episode_reward, avg_reward,
                                              win_rate, total_pnl, num_trades, account_state['capital'])
                else:
                    self._log_progress(episode + 1, episode_reward, avg_reward, None)
        
        # Create training summary for research logging
        training_summary = self._create_training_summary(initial_capital)
        
        # End research logging with model analysis
        if self.research_logger:
            self.research_logger.end_training(training_summary, model=self.agent)
        
        # Display comprehensive training summary
        self._display_training_summary(initial_capital)

    def _run_episode(self, env: TradingEnv, episode_num: int, symbol: str) -> tuple:
        """Run a single episode and return reward and trades."""
        obs = env.reset()
        done = False
        episode_reward = 0.0
        step_count = 0
        experiences = []
        
        while not done:
            # Get current account state and price from environment for action masking
            account_state = env.engine.get_account_state()
            current_position_quantity = account_state['current_position_quantity']
            available_capital = account_state['capital']
            current_price = env.data['close'].iloc[env.current_step] # Assuming current_step is valid index

            # Select action with masking context and get probabilities for logging
            if hasattr(self.agent, 'select_action') and 'return_probabilities' in self.agent.select_action.__code__.co_varnames:
                action_type, quantity, action_probs = self.agent.select_action(
                    obs,
                    available_capital=available_capital,
                    current_position_quantity=current_position_quantity,
                    current_price=current_price,
                    instrument=env.instrument, # Pass the instrument object
                    return_probabilities=True
                )
                action = [action_type, quantity]
            else:
                action = self.agent.select_action(
                    obs,
                    available_capital=available_capital,
                    current_position_quantity=current_position_quantity,
                    current_price=current_price,
                    instrument=env.instrument # Pass the instrument object
                )
                action_probs = None
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            truncated = False  # Our environment doesn't use truncated
            
            # Store experience
            experience = (obs, action, reward, next_obs, done)
            experiences.append(experience)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # Enhanced step logging with real-time metrics (every step for debugging)
            self._log_step_details(env, episode_num, step_count, action, reward, info, symbol, action_probs)
        
        # HRM uses deep supervision during forward passes, not experience-based learning
        
        # Get episode trades
        episode_trades = env.engine.get_trade_history()
        
        return episode_reward, episode_trades

    def _log_step_details(self, env: TradingEnv, episode: int, step: int, action: list, 
                         reward: float, info: dict, symbol: str, action_probs: np.ndarray = None) -> None:
        """Log detailed step information with real-time metrics."""
        # Get current datetime from environment data
        current_datetime = "N/A"
        
        try:
            if hasattr(env, 'data') and env.data is not None and env.current_step < len(env.data):
                # Get epoch timestamp and convert to readable
                if 'datetime_epoch' in env.data.columns:
                    epoch_timestamp = env.data['datetime_epoch'].iloc[env.current_step]
                    current_datetime = self._convert_epoch_to_readable(epoch_timestamp)
                else:
                    current_datetime = f"Step_{env.current_step}"
        except Exception as e:
            current_datetime = f"Step_{env.current_step}"
        
        # Get account state
        account_state = env.engine.get_account_state()
        action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
        action_name = action_names[action[0]] if action[0] < len(action_names) else "UNKNOWN"
        
        # ACTION DEBUGGING: Log raw model output for analysis
        raw_action_type = action[0] if len(action) > 0 else "NONE"
        raw_quantity = action[1] if len(action) > 1 else "NONE"
        logger.debug(f"RAW MODEL OUTPUT: action_type={raw_action_type}, quantity={raw_quantity}")
        
        # Calculate real-time win rate from current episode trades only (not cumulative)
        current_episode_trades = env.engine.get_trade_history()
        
        # TRADE DEBUGGING: Check if trades are being executed but not logged
        trade_count = len(current_episode_trades)
        logger.debug(f"CURRENT TRADE COUNT: {trade_count} trades in episode")
        episode_win_rate = self._calculate_win_rate_from_trades(current_episode_trades)
        
        # Get trading position information
        current_position = account_state.get('current_position_quantity', 0)
        entry_price = "N/A"
        sl_price = "N/A" 
        target_price = "N/A"
        
        # Get position details if in a position from the engine directly
        if current_position != 0 and hasattr(env.engine, '_current_position_entry_price'):
            entry_price = env.engine._current_position_entry_price
            sl_price = env.engine._stop_loss_price
            target_price = env.engine._target_profit_price
        
        # Prepare step data for research logger  
        env_config = self.config.get('environment', {}) if hasattr(self, 'config') else {}
        initial_capital = env_config.get('initial_capital', 100000.0)
        step_data = {
            'datetime': current_datetime,
            'instrument': symbol,
            'action_name': action_name,
            'quantity': action[1] if len(action) > 1 else 0,
            'episode_win_rate': episode_win_rate,
            'initial_capital': initial_capital,
            'current_capital': account_state['capital'],
            'entry_price': entry_price,
            'sl_price': sl_price,
            'target_price': target_price,
            'reward': reward,
            'exit_reason': self._get_exit_reason_from_info(info)
        }
        
        # Add action probabilities as percentages if available
        if action_probs is not None:
            action_names = ["BUY", "SELL", "CLOSE_L", "CLOSE_S", "HOLD"]
            for i, prob in enumerate(action_probs):
                if i < len(action_names):
                    step_data[f'prob_{action_names[i]}'] = f'{prob*100:.1f}%'
        
        # Use research logger if available
        if hasattr(self, 'research_logger') and self.research_logger:
            self.research_logger.log_step(step, step_data)
        else:
            # Fallback to standard logging
            timeframe = self._extract_timeframe(symbol)
            logger.info(f"🎯 Ep {episode} | Step {step} | {current_datetime} | {symbol} ({timeframe}) | "
                       f"Action: {action_name} | Capital: ₹{account_state['capital']:.2f} | "
                       f"Position: {account_state['current_position_quantity']} | "
                       f"Reward: {reward:.4f} | Win Rate: {episode_win_rate:.1%}")
    
    def _extract_timeframe(self, symbol: str) -> str:
        """Extract timeframe from symbol name."""
        parts = symbol.split('_')
        if len(parts) >= 2:
            try:
                return f"{parts[-1]}min"
            except ValueError:
                pass
        return "?min"

    def _log_progress_detailed(self, episode: int, episode_reward: float, avg_reward: float,
                              win_rate: float, total_pnl: float, num_trades: int, capital: float) -> None:
        """Log detailed training progress with trading metrics."""
        logger.info(f"📊 Episode {episode} Training Progress:")
        logger.info(f"  💰 Reward: {episode_reward:.2f} (Avg: {avg_reward:.2f})")
        logger.info(f"  📈 Trading: {num_trades} trades, Win Rate: {win_rate:.1%}")
        logger.info(f"  💵 P&L: ₹{total_pnl:.2f}, Capital: ₹{capital:.2f}")
        logger.info(f"  🔄 Symbol Distribution: {dict(self.symbol_episode_count)}")
        logger.info("-" * 60)

    def _log_progress(self, episode: int, total_reward: float, avg_reward: float, loss: float = None) -> None:
        """Simple progress logging."""
        log_string = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}"
        if loss is not None:
            log_string += f", Loss: {loss:.4f}"
        logger.info(log_string)

    def _display_training_summary(self, initial_capital: float) -> None:
        """Display comprehensive training summary."""
        logger.info("=" * 80)
        logger.info("UNIVERSAL TRAINING SUMMARY")
        logger.info("=" * 80)
        
        total_episodes_run = len(self.episode_rewards)
        final_capital = self.cumulative_capital_history[-1] if self.cumulative_capital_history else initial_capital
        
        logger.info(f"📊 Episodes Completed: {total_episodes_run}")
        logger.info(f"💰 Capital: ₹{initial_capital:,.2f} → ₹{final_capital:,.2f}")
        logger.info(f"📈 Total Return: ₹{final_capital - initial_capital:,.2f} ({((final_capital/initial_capital - 1) * 100):.2f}%)")
        
        # Symbol distribution
        logger.info(f"🔄 Symbol Episodes: {dict(self.symbol_episode_count)}")
        
        # Trade statistics
        if self.cumulative_trade_history:
            closing_trades = [trade for trade in self.cumulative_trade_history if trade.get('trade_type') == 'CLOSE']
            win_rate = calculate_win_rate(self.cumulative_trade_history)
            total_trades = len(closing_trades)
            
            logger.info(f"📊 Total Trades: {total_trades}")
            logger.info(f"🎯 Final Win Rate: {win_rate:.1%}")
            
            if total_trades > 0:
                profit_factor = calculate_profit_factor(closing_trades)
                avg_pnl_per_trade = calculate_avg_pnl_per_trade(self.cumulative_trade_history)
                logger.info(f"💹 Profit Factor: {profit_factor:.2f}")
                logger.info(f"💰 Avg P&L per Trade: ₹{avg_pnl_per_trade:.2f}")
        
        logger.info("=" * 80)
    
    def _create_training_summary(self, initial_capital: float) -> Dict[str, Any]:
        """Create a comprehensive training summary."""
        total_episodes_run = len(self.episode_rewards)
        final_capital = self.cumulative_capital_history[-1] if self.cumulative_capital_history else initial_capital
        
        summary = {
            'total_episodes': total_episodes_run,
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': final_capital - initial_capital,
            'total_return_pct': ((final_capital/initial_capital - 1) * 100) if initial_capital > 0 else 0,
            'symbol_distribution': dict(self.symbol_episode_count),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }
        
        # Add trading statistics if available
        if self.cumulative_trade_history:
            closing_trades = [trade for trade in self.cumulative_trade_history if trade.get('trade_type') == 'CLOSE']
            win_rate = calculate_win_rate(self.cumulative_trade_history)
            total_trades = len(closing_trades)
            
            summary.update({
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': calculate_profit_factor(closing_trades) if total_trades > 0 else 0,
                'avg_pnl_per_trade': calculate_avg_pnl_per_trade(self.cumulative_trade_history) if total_trades > 0 else 0
            })
        
        return summary
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary for external use."""
        initial_capital = self.config.get('environment', {}).get('initial_capital', 100000.0)
        return self._create_training_summary(initial_capital)
