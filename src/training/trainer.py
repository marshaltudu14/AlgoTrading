import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import copy
import logging

from src.backtesting.environment import TradingEnv
from src.agents.base_agent import BaseAgent
from src.utils.data_loader import DataLoader

from src.utils.metrics import (
    calculate_sharpe_ratio,
    calculate_total_pnl,
    calculate_profit_factor,
    calculate_max_drawdown,
    calculate_win_rate,
    calculate_avg_pnl_per_trade,
    calculate_num_trades,
    calculate_comprehensive_metrics
)

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, agent: BaseAgent, env: 'TradingEnv' = None, log_interval: int = 10, num_episodes: int = 100):
        self.agent = agent
        self.log_interval = log_interval
        self.env = env
        self.num_episodes = num_episodes
        import logging
        logging.getLogger('src.backtesting.engine').setLevel(logging.INFO)

        # Enhanced tracking for comprehensive metrics
        self.capital_history = []
        self.episode_rewards = []
        self.total_reward = 0.0
        self.cumulative_trade_history = []  # Accumulate trades across all episodes
        self.cumulative_capital_history = []  # Track capital across all episodes

    def train(self, data_loader: DataLoader, symbol: str, initial_capital: float, env: 'TradingEnv' = None) -> None:
        # Use provided environment or create a new one
        if env is not None:
            self.env = env
        else:
            # Initialize the environment for the specific symbol with consistent configuration
            self.env = TradingEnv(
                data_loader=data_loader,
                symbol=symbol,
                initial_capital=initial_capital,
                lookback_window=20,
                episode_length=500,
                use_streaming=False  # Use full data for consistent dimensions
            )

        # Initialize tracking
        initial_capital = self.env.engine.get_account_state()['capital']
        self.capital_history = [initial_capital]

        # Initialize cumulative tracking if not already done
        if not self.cumulative_capital_history:
            self.cumulative_capital_history = [initial_capital]

        for episode in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            experiences = []

            while not done:
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                experiences.append((observation, action, reward, next_observation, done))
                observation = next_observation
                episode_reward += reward

            if hasattr(self.agent, 'learn'):
                self.agent.learn(experiences)

            # Track episode metrics
            self.episode_rewards.append(episode_reward)
            self.total_reward += episode_reward

            # Track capital history
            current_capital = self.env.engine.get_account_state()['capital']
            self.capital_history.append(current_capital)
            self.cumulative_capital_history.append(current_capital)

            # Accumulate trade history from this episode
            episode_trades = self.env.engine.get_trade_history()
            if episode_trades:
                # Add episode number to each trade for tracking
                for trade in episode_trades:
                    trade['episode'] = episode + 1
                self.cumulative_trade_history.extend(episode_trades)

            if (episode + 1) % self.log_interval == 0:
                # Calculate proper metrics using cumulative data
                avg_reward = self.total_reward / (episode + 1) if episode > 0 else self.total_reward

                # Get trading metrics from cumulative trade history
                account_state = self.env.engine.get_account_state()

                # Calculate win rate and other metrics from cumulative trades
                if self.cumulative_trade_history:
                    win_rate = calculate_win_rate(self.cumulative_trade_history)
                    total_pnl = calculate_total_pnl(self.cumulative_trade_history)
                    num_trades = calculate_num_trades(self.cumulative_trade_history)

                    self._log_progress_detailed(episode + 1, self.total_reward, avg_reward,
                                              win_rate, total_pnl, num_trades, account_state['capital'])
                else:
                    self._log_progress(episode + 1, episode_reward, avg_reward, None)

        # Display comprehensive training summary
        self._display_training_summary(initial_capital)

    def _log_progress(self, episode: int, total_reward: float, avg_reward: float, loss: float = None) -> None:
        log_string = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}"
        if loss is not None:
            log_string += f", Loss: {loss:.4f}"
        print(log_string)

    def _log_progress_detailed(self, episode: int, episode_reward: float, avg_reward: float,
                              win_rate: float, total_pnl: float, num_trades: int, capital: float) -> None:
        """Log detailed training progress with trading metrics."""
        import os
        detailed_logging = os.environ.get('DETAILED_BACKTEST_LOGGING', 'false').lower() == 'true'

        if detailed_logging:
            logger.info(f"📊 Episode {episode} Training Progress:")
            logger.info(f"  💰 Reward: {episode_reward:.2f} (Avg: {avg_reward:.2f})")
            logger.info(f"  📈 Trading: {num_trades} trades, Win Rate: {win_rate:.1%}")
            logger.info(f"  💵 P&L: ₹{total_pnl:.2f}, Capital: ₹{capital:.2f}")
        else:
            # Concise logging - just episode and key metrics
            logger.info(f"📊 Ep {episode}: Reward {episode_reward:.1f}, {num_trades} trades, Win {win_rate:.0%}, Capital ₹{capital:.0f}")

        # Show recent trading decisions
        if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'get_trade_summary'):
            trade_summary = self.env.engine.get_trade_summary()
            if 'total_decisions' in trade_summary:
                logger.info(f"  🎯 Decisions: {trade_summary['total_decisions']} total, "
                           f"{trade_summary['position_opens']} opens, "
                           f"{trade_summary['position_closes']} closes, "
                           f"{trade_summary['hold_actions']} holds")

                if trade_summary['exit_reasons']:
                    exit_summary = ", ".join([f"{reason}: {count}" for reason, count in trade_summary['exit_reasons'].items()])
                    logger.info(f"  🚪 Exit reasons: {exit_summary}")

        logger.info("-" * 60)

    def _display_training_summary(self, initial_capital: float) -> None:
        """Display comprehensive training summary with all key metrics."""
        print("\n" + "=" * 80)
        print("🎯 COMPREHENSIVE TRAINING SUMMARY")
        print("=" * 80)

        # Get final trading data using cumulative history
        final_account = self.env.engine.get_account_state()

        # Calculate comprehensive metrics using cumulative data
        total_episodes_run = len(self.episode_rewards)
        metrics = calculate_comprehensive_metrics(
            trade_history=self.cumulative_trade_history,
            capital_history=self.cumulative_capital_history,
            initial_capital=initial_capital,
            total_episodes=total_episodes_run,
            total_reward=self.total_reward
        )

        # Display metrics in organized sections
        print(f"\n📊 TRADING PERFORMANCE:")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Avg P&L per Trade: ₹{metrics['avg_pnl_per_trade']:.2f}")

        print(f"\n💰 FINANCIAL METRICS:")
        print(f"   Initial Capital: ₹{initial_capital:,.2f}")
        print(f"   Final Capital: ₹{final_account['capital']:,.2f}")
        print(f"   Total P&L: ₹{metrics['total_pnl']:,.2f}")
        print(f"   Total Return: {metrics['total_return_percentage']:.2f}%")
        print(f"   Max Drawdown: {metrics['max_drawdown_percentage']:.2f}%")

        print(f"\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\n🤖 TRAINING METRICS:")
        print(f"   Total Episodes: {self.num_episodes}")
        print(f"   Total Reward: {metrics['total_reward']:.2f}")
        print(f"   Avg Reward per Episode: {metrics['avg_reward_per_episode']:.2f}")

        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if metrics['win_rate'] >= 0.6:
            print(f"   ✅ Excellent Win Rate ({metrics['win_rate']:.1%})")
        elif metrics['win_rate'] >= 0.5:
            print(f"   ✅ Good Win Rate ({metrics['win_rate']:.1%})")
        else:
            print(f"   ⚠️ Low Win Rate ({metrics['win_rate']:.1%}) - Consider strategy adjustment")

        if metrics['profit_factor'] >= 1.5:
            print(f"   ✅ Excellent Profit Factor ({metrics['profit_factor']:.2f})")
        elif metrics['profit_factor'] >= 1.0:
            print(f"   ✅ Profitable Strategy ({metrics['profit_factor']:.2f})")
        else:
            print(f"   ❌ Losing Strategy ({metrics['profit_factor']:.2f}) - Needs improvement")

        if metrics['sharpe_ratio'] >= 1.0:
            print(f"   ✅ Good Risk-Adjusted Returns (Sharpe: {metrics['sharpe_ratio']:.3f})")
        elif metrics['sharpe_ratio'] >= 0.5:
            print(f"   ⚠️ Moderate Risk-Adjusted Returns (Sharpe: {metrics['sharpe_ratio']:.3f})")
        else:
            print(f"   ❌ Poor Risk-Adjusted Returns (Sharpe: {metrics['sharpe_ratio']:.3f})")

        print("=" * 80)
