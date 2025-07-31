#!/usr/bin/env python3
"""
Curriculum Trainer for PPO Agent with Timeframe Progression
==========================================================

This trainer implements curriculum learning by progressing from higher timeframes
(less noisy) to lower timeframes (more noisy), rotating symbols within each batch.
"""

import logging
import random
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from src.backtesting.environment import TradingEnv
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import DataLoader
from src.utils.metrics import (
    calculate_win_rate, calculate_total_pnl, calculate_num_trades,
    calculate_profit_factor, calculate_avg_pnl_per_trade
)

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """
    Curriculum trainer that progresses from higher to lower timeframes for better learning.
    """
    
    def __init__(self, agent: BaseAgent, data_loader: DataLoader, 
                 num_episodes: int = 100, log_interval: int = 10, config: dict = None):
        self.agent = agent
        self.data_loader = data_loader
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.config = config or {}
        
        # Discover available symbols and timeframes
        self.curriculum_batches = self._discover_curriculum_data()
        
        # Enhanced tracking for comprehensive metrics
        self.capital_history = []
        self.episode_rewards = []
        self.total_reward = 0.0
        self.cumulative_trade_history = []  # Accumulate trades across all episodes
        self.cumulative_capital_history = []  # Track capital across all episodes
        
        # Real-time win rate tracking
        self.step_win_rates = []
        self.step_trade_counts = []
        
        # Curriculum tracking
        self.current_batch_index = 0
        self.episodes_per_batch = num_episodes  # Each timeframe gets FULL episode count
        self.total_curriculum_episodes = num_episodes * len(self.curriculum_batches) if self.curriculum_batches else num_episodes
        self.batch_performance = []

        logger.info(f"🎓 Curriculum Trainer initialized")
        logger.info(f"   Episodes per timeframe: {self.episodes_per_batch}")
        logger.info(f"   Curriculum batches: {len(self.curriculum_batches)}")
        logger.info(f"   Total curriculum episodes: {self.total_curriculum_episodes}")
        for i, batch in enumerate(self.curriculum_batches):
            logger.info(f"   Batch {i+1}: Timeframe {batch['timeframe']}min - {len(batch['symbols'])} symbols")

    def _discover_curriculum_data(self) -> List[Dict[str, Any]]:
        """Discover available data files and organize into curriculum batches."""
        final_dir = "data/final"
        if not os.path.exists(final_dir):
            logger.warning(f"Data directory {final_dir} not found")
            return []
        
        # Find all CSV files and extract symbols/timeframes
        files = [f for f in os.listdir(final_dir) if f.endswith('.csv')]
        symbols_timeframes = {}
        
        for file in files:
            # Extract symbol and timeframe from filename
            # Pattern: features_Symbol_Timeframe.csv
            match = re.match(r'features_(.+)_(\d+)\.csv', file)
            if match:
                symbol = match.group(1)
                timeframe = int(match.group(2))
                
                if timeframe not in symbols_timeframes:
                    symbols_timeframes[timeframe] = []
                symbols_timeframes[timeframe].append(f"{symbol}_{timeframe}")
        
        # Create curriculum batches sorted by timeframe (highest to lowest)
        curriculum_batches = []
        for timeframe in sorted(symbols_timeframes.keys(), reverse=True):
            batch = {
                'timeframe': timeframe,
                'symbols': symbols_timeframes[timeframe],
                'description': f"{timeframe}min timeframe"
            }
            curriculum_batches.append(batch)
        
        if not curriculum_batches:
            logger.warning("No curriculum data found, creating fallback batch")
            # Fallback: use any available symbols
            all_symbols = []
            for file in files:
                if file.startswith('features_'):
                    symbol_part = file.replace('features_', '').replace('.csv', '')
                    all_symbols.append(symbol_part)
            
            if all_symbols:
                curriculum_batches = [{
                    'timeframe': 'mixed',
                    'symbols': all_symbols,
                    'description': 'Mixed timeframe fallback'
                }]
        
        return curriculum_batches

    def _get_current_batch(self, episode: int) -> Dict[str, Any]:
        """Get the current curriculum batch based on episode number."""
        if not self.curriculum_batches:
            return {'timeframe': 'unknown', 'symbols': [], 'description': 'No data'}

        # Calculate which batch we should be in (each batch gets full episodes)
        batch_index = min(episode // self.episodes_per_batch, len(self.curriculum_batches) - 1)
        return self.curriculum_batches[batch_index]

    def _get_next_symbol(self, episode: int) -> str:
        """Get the next symbol for training using curriculum strategy."""
        current_batch = self._get_current_batch(episode)
        symbols = current_batch['symbols']
        
        if not symbols:
            logger.warning("No symbols available in current batch")
            return "fallback_symbol"
        
        # Rotate through symbols within the current batch
        symbol_index = episode % len(symbols)
        return symbols[symbol_index]

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

    def train(self) -> Dict[str, Any]:
        """
        Execute curriculum training with timeframe progression.
        
        Returns:
            Training results and metrics
        """
        logger.info("🎓 Starting Curriculum Training")
        logger.info("=" * 60)
        logger.info(f"📊 Total curriculum episodes: {self.total_curriculum_episodes}")
        logger.info(f"📊 Episodes per timeframe: {self.episodes_per_batch}")
        logger.info(f"📊 Training across {len(self.curriculum_batches)} timeframe batches")

        if not self.curriculum_batches:
            logger.error("No curriculum batches available for training")
            return {"error": "No training data available"}

        for episode in range(self.total_curriculum_episodes):
            # Get current curriculum batch and symbol
            current_batch = self._get_current_batch(episode)
            symbol = self._get_next_symbol(episode)
            
            # Check if we've moved to a new batch
            new_batch_index = min(episode // self.episodes_per_batch, len(self.curriculum_batches) - 1)
            if new_batch_index != self.current_batch_index:
                self.current_batch_index = new_batch_index
                logger.info(f"🎓 Curriculum Progress: Moving to Batch {new_batch_index + 1}")
                logger.info(f"   {current_batch['description']} - {len(current_batch['symbols'])} symbols")
            
            # Train on the selected symbol
            episode_result = self._train_episode(episode, symbol, current_batch)
            
            # Log progress
            if (episode + 1) % self.log_interval == 0:
                self._log_training_progress(episode + 1, current_batch)
        
        # Final training summary
        return self._generate_training_summary()

    def _train_episode(self, episode: int, symbol: str, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single episode on the specified symbol."""
        try:
            # Initialize environment for the specific symbol
            env = TradingEnv(
                data_loader=self.data_loader,
                symbol=symbol,
                initial_capital=self.config.get('initial_capital', 100000),
                lookback_window=self.config.get('lookback_window', 50),
                episode_length=self.config.get('episode_length', 500),
                use_streaming=self.config.get('use_streaming', False),
                reward_function=self.config.get('reward_function', "trading_focused"),
                trailing_stop_percentage=self.config.get('trailing_stop_percentage', 0.02)
            )

            # Reset environment and get initial observation
            obs = env.reset()
            episode_reward = 0.0
            step_count = 0

            while True:
                # Agent selects action
                action_type, quantity = self.agent.select_action(obs)
                action = (action_type, quantity)

                # Execute action in environment
                next_obs, reward, done, info = env.step(action)

                # PPOAgent handles experience storage internally during update()

                # Update tracking
                episode_reward += reward
                step_count += 1

                # Get current state for logging
                current_price = env.data.iloc[env.current_step]['close'] if env.current_step < len(env.data) else 0
                account_state = env.engine.get_account_state(current_price=current_price)

                # Real-time win rate calculation
                current_trades = env.engine.get_trade_history()
                self.cumulative_trade_history.extend(current_trades)
                real_time_win_rate = self._calculate_real_time_win_rate()
                total_trades = len([t for t in self.cumulative_trade_history if t.get('trade_type') == 'CLOSE'])

                # Convert epoch timestamp to readable datetime
                current_datetime = self._convert_epoch_to_readable(env.data.iloc[env.current_step]['datetime_epoch'])

                # Get action name and exit info
                action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
                action_name = action_names[action_type] if 0 <= action_type < len(action_names) else "UNKNOWN"

                exit_info = ""
                if 'exit_reason' in info:
                    exit_info = f" | Exit: {info['exit_reason']}"

                # Enhanced step logging with curriculum info
                logger.info(f"🎓 Ep {episode+1} | Step {step_count} | {current_datetime} | {symbol} | "
                           f"Batch: {batch_info['description']} | Action: {action_name} | "
                           f"Capital: ₹{account_state['capital']:.2f} | "
                           f"Position: {account_state['current_position_quantity']} | "
                           f"Reward: {reward:.4f} | Win Rate: {real_time_win_rate:.1%} | "
                           f"Trades: {total_trades}{exit_info}")

                obs = next_obs

                if done:
                    break

            # Update agent (learning step)
            if hasattr(self.agent, 'update'):
                self.agent.update()

            # Store episode results
            final_capital = account_state['capital']
            self.capital_history.append(final_capital)
            self.episode_rewards.append(episode_reward)
            self.total_reward += episode_reward

            return {
                'episode': episode,
                'symbol': symbol,
                'batch': batch_info['description'],
                'final_capital': final_capital,
                'episode_reward': episode_reward,
                'steps': step_count,
                'win_rate': real_time_win_rate,
                'total_trades': total_trades
            }

        except Exception as e:
            logger.error(f"Error in episode {episode} with symbol {symbol}: {e}")
            return {
                'episode': episode,
                'symbol': symbol,
                'batch': batch_info['description'],
                'error': str(e)
            }

    def _log_training_progress(self, episode: int, current_batch: Dict[str, Any]):
        """Log training progress with curriculum information."""
        if not self.capital_history:
            return

        # Calculate metrics
        avg_capital = np.mean(self.capital_history[-self.log_interval:])
        total_return = ((self.capital_history[-1] - 100000) / 100000) * 100
        avg_reward = np.mean(self.episode_rewards[-self.log_interval:])
        current_win_rate = self._calculate_real_time_win_rate()
        total_trades = len([t for t in self.cumulative_trade_history if t.get('trade_type') == 'CLOSE'])

        logger.info("🎓 Curriculum Training Progress")
        logger.info("=" * 50)
        logger.info(f"Episodes: {episode + 1}/{self.total_curriculum_episodes}")
        logger.info(f"Current Batch: {current_batch['description']}")
        logger.info(f"Batch Progress: {(episode % self.episodes_per_batch) + 1}/{self.episodes_per_batch}")
        logger.info(f"Average Capital (last {self.log_interval}): ₹{avg_capital:,.2f}")
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Win Rate: {current_win_rate:.1%}")
        logger.info(f"Total Trades: {total_trades}")

    def _generate_training_summary(self) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        if not self.capital_history:
            return {"error": "No training data available"}

        # Calculate final metrics
        initial_capital = 100000
        final_capital = self.capital_history[-1]
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        final_win_rate = self._calculate_real_time_win_rate()
        total_trades = len([t for t in self.cumulative_trade_history if t.get('trade_type') == 'CLOSE'])

        # Batch performance analysis
        batch_summary = []
        for i, batch in enumerate(self.curriculum_batches):
            start_ep = i * self.episodes_per_batch
            end_ep = min((i + 1) * self.episodes_per_batch, len(self.capital_history))

            if start_ep < len(self.capital_history):
                batch_capitals = self.capital_history[start_ep:end_ep]
                batch_avg = np.mean(batch_capitals) if batch_capitals else 0
                batch_summary.append({
                    'batch': batch['description'],
                    'episodes': f"{start_ep+1}-{end_ep}",
                    'avg_capital': batch_avg,
                    'symbols': len(batch['symbols'])
                })

        summary = {
            'total_episodes': self.num_episodes,
            'curriculum_batches': len(self.curriculum_batches),
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'final_win_rate': final_win_rate,
            'total_trades': total_trades,
            'batch_performance': batch_summary
        }

        logger.info("🎓 Curriculum Training Summary")
        logger.info("=" * 50)
        logger.info(f"Total Episodes: {summary['total_episodes']}")
        logger.info(f"Curriculum Batches: {summary['curriculum_batches']}")
        logger.info(f"Final Capital: ₹{summary['final_capital']:,.2f}")
        logger.info(f"Total Return: {summary['total_return_pct']:.2f}%")
        logger.info(f"Final Win Rate: {summary['final_win_rate']:.1%}")
        logger.info(f"Total Trades: {summary['total_trades']}")

        logger.info("\nBatch Performance:")
        for batch in batch_summary:
            logger.info(f"  {batch['batch']}: Avg Capital ₹{batch['avg_capital']:,.2f} "
                       f"(Episodes {batch['episodes']}, {batch['symbols']} symbols)")

        return summary
