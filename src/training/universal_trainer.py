#!/usr/bin/env python3
"""
Universal Trainer for PPO Agent with Symbol Rotation
===================================================

This trainer rotates through different symbols per episode to create
a universal model that can handle diverse market conditions.
"""

import logging
import random
import numpy as np
import pandas as pd
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
    Universal trainer that rotates symbols per episode for diverse market exposure.
    """
    
    def __init__(self, agent: BaseAgent, symbols: List[str], data_loader: DataLoader, 
                 num_episodes: int = 100, log_interval: int = 10, config: dict = None):
        self.agent = agent
        self.symbols = symbols
        self.data_loader = data_loader
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.config = config or {}
        
        # Enhanced tracking for comprehensive metrics
        self.capital_history = []
        self.episode_rewards = []
        self.total_reward = 0.0
        self.cumulative_trade_history = []  # Accumulate trades across all episodes
        self.cumulative_capital_history = []  # Track capital across all episodes
        
        # Real-time win rate tracking
        self.step_win_rates = []
        self.step_trade_counts = []
        
        # Symbol rotation tracking
        self.symbol_episode_count = {symbol: 0 for symbol in symbols}
        self.symbol_performance = {symbol: [] for symbol in symbols}
        
        logger.info(f"🎯 Universal Trainer initialized with {len(symbols)} symbols")
        logger.info(f"   Symbols: {symbols}")
        logger.info(f"   Episodes: {num_episodes}")
        logger.info(f"   Log interval: {log_interval}")

    def _get_next_symbol(self, episode: int) -> str:
        """Get the next symbol for training using rotation strategy."""
        # Simple round-robin rotation
        symbol = self.symbols[episode % len(self.symbols)]
        self.symbol_episode_count[symbol] += 1
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
        
        for episode in range(self.num_episodes):
            # Get symbol for this episode
            current_symbol = self._get_next_symbol(episode)
            logger.info(f"📊 Episode {episode + 1}/{self.num_episodes} | Symbol: {current_symbol}")
            
            # Create environment for current symbol
            env = TradingEnv(
                data_loader=self.data_loader,
                symbol=current_symbol,
                initial_capital=initial_capital,
                lookback_window=env_config.get('lookback_window', 50),
                episode_length=env_config.get('episode_length', 500),
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
            # Select action
            action = self.agent.select_action(obs)
            
            # Take step
            next_obs, reward, done, info = env.step(action)
            truncated = False  # Our environment doesn't use truncated
            
            # Store experience
            experience = (obs, action, reward, next_obs, done)
            experiences.append(experience)
            
            obs = next_obs
            episode_reward += reward
            step_count += 1
            
            # Enhanced step logging with real-time metrics
            if step_count <= 20 or step_count % 10 == 0:  # Log first 20 steps then every 10th
                self._log_step_details(env, episode_num, step_count, action, reward, info, symbol)
        
        # Learn from experiences
        if hasattr(self.agent, 'learn'):
            self.agent.learn(experiences)
        
        # Get episode trades
        episode_trades = env.engine.get_trade_history()
        
        return episode_reward, episode_trades

    def _log_step_details(self, env: TradingEnv, episode: int, step: int, action: list, 
                         reward: float, info: dict, symbol: str) -> None:
        """Log detailed step information with real-time metrics."""
        # Get current datetime from environment data
        current_datetime = "N/A"
        epoch_feature = "N/A"
        
        try:
            if hasattr(env, 'data') and env.data is not None and env.current_step < len(env.data):
                # Get epoch timestamp and convert to readable
                if 'datetime_epoch' in env.data.columns:
                    epoch_timestamp = env.data['datetime_epoch'].iloc[env.current_step]
                    current_datetime = self._convert_epoch_to_readable(epoch_timestamp)
                    epoch_feature = int(epoch_timestamp)
                else:
                    current_datetime = f"Step_{env.current_step}"
        except Exception as e:
            current_datetime = f"Step_{env.current_step}"
        
        # Get account state
        account_state = env.engine.get_account_state()
        action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
        action_name = action_names[action[0]] if action[0] < len(action_names) else "UNKNOWN"
        
        # Calculate real-time win rate from current episode trades + cumulative
        current_episode_trades = env.engine.get_trade_history()
        all_trades = self.cumulative_trade_history + current_episode_trades
        real_time_win_rate = self._calculate_win_rate_from_trades(all_trades)
        total_trades = len([t for t in all_trades if t.get('trade_type') == 'CLOSE'])
        
        # Get exit reason if available
        exit_reason = self._get_exit_reason_from_info(info)
        exit_info = f" | Exit: {exit_reason}" if exit_reason else ""
        
        # Enhanced logging with real-time metrics
        logger.info(f"🎯 Ep {episode} | Step {step} | {current_datetime} | {symbol} | "
                   f"Action: {action_name} | Capital: ₹{account_state['capital']:.2f} | "
                   f"Position: {account_state['current_position_quantity']} | "
                   f"Reward: {reward:.4f} | Win Rate: {real_time_win_rate:.1%} | "
                   f"Trades: {total_trades}{exit_info}")

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
