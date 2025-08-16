import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import copy
import logging
import yaml
import os

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

        # Load configuration
        self.config = self._load_config()

        # Enhanced tracking for comprehensive metrics
        self.capital_history = []
        self.episode_rewards = []
        self.total_reward = 0.0
        self.cumulative_trade_history = []  # Accumulate trades across all episodes
        self.cumulative_capital_history = []  # Track capital across all episodes

    def _load_config(self) -> dict:
        """Load configuration from settings.yaml"""
        config_path = "config/settings.yaml"
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration if file loading fails"""
        return {
            'environment': {
                'initial_capital': 100000.0,
                'lookback_window': 50,
                'episode_length': 500,
                'use_streaming': False,
                'reward_function': "trading_focused",
                'trailing_stop_percentage': 0.02
            }
        }

    def train(self, data_loader: DataLoader, symbol: str, initial_capital: float, env: 'TradingEnv' = None) -> None:
        # Use provided environment or create a new one
        if env is not None:
            self.env = env
        else:
            # Get environment configuration
            env_config = self.config.get('environment', {})

            # Initialize the environment for the specific symbol with configuration
            self.env = TradingEnv(
                data_loader=data_loader,
                symbol=symbol,
                initial_capital=env_config.get('initial_capital', initial_capital),
                lookback_window=env_config.get('lookback_window', 50),
                episode_length=env_config.get('episode_length', 500),
                use_streaming=env_config.get('use_streaming', False),
                reward_function=env_config.get('reward_function', "trading_focused"),
                trailing_stop_percentage=env_config.get('trailing_stop_percentage', 0.02),
                smart_action_filtering=env_config.get('smart_action_filtering', False)
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

            step_count = 0
            while not done:
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                experiences.append((observation, action, reward, next_observation, done))
                observation = next_observation
                episode_reward += reward
                step_count += 1

                # Log EVERY SINGLE STEP with datetime as requested by user
                current_datetime = "N/A"
                epoch_feature = "N/A"
                try:
                    if hasattr(self.env, 'data') and self.env.data is not None and self.env.current_step < len(self.env.data):
                        # Use the datetime index (readable format)
                        current_datetime = str(self.env.data.index[self.env.current_step])
                        # Also show epoch feature for verification
                        epoch_feature = self.env.data['datetime_epoch'].iloc[self.env.current_step] if 'datetime_epoch' in self.env.data.columns else "N/A"
                except Exception as e:
                    current_datetime = f"Step_{self.env.current_step}"
                    epoch_feature = "N/A"

                account_state = self.env.engine.get_account_state()
                action_names = ["BUY_LONG", "SELL_SHORT", "CLOSE_LONG", "CLOSE_SHORT", "HOLD"]
                action_name = action_names[action[0]] if action[0] < len(action_names) else "UNKNOWN"
                logger.info(f"🎯 Episode {episode + 1} | Step {step_count} | {current_datetime} | Epoch: {epoch_feature} | Action: {action_name} | Capital: ₹{account_state['capital']:.2f} | Position: {account_state['current_position_quantity']} | Reward: {reward:.4f}")

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

        # Calculate metrics properly - use only last episode for capital-based metrics
        # but cumulative for trade statistics
        total_episodes_run = len(self.episode_rewards)

        # For trade statistics, use cumulative data but calculate averages
        if self.cumulative_trade_history:
            closing_trades = [trade for trade in self.cumulative_trade_history if trade.get('trade_type') == 'CLOSE']
            avg_trades_per_episode = len(closing_trades) / total_episodes_run if total_episodes_run > 0 else 0
            win_rate = calculate_win_rate(self.cumulative_trade_history)
            profit_factor = calculate_profit_factor(closing_trades)
            avg_pnl_per_trade = calculate_avg_pnl_per_trade(self.cumulative_trade_history)
            total_trades = len(closing_trades)
        else:
            avg_trades_per_episode = 0
            win_rate = 0.0
            profit_factor = 0.0
            avg_pnl_per_trade = 0.0
            total_trades = 0

        # For capital-based metrics, use only the final episode's performance
        final_capital = final_account['capital']
        total_return_percentage = ((final_capital - initial_capital) / initial_capital) * 100

        # Calculate average reward per episode
        avg_reward_per_episode = self.total_reward / total_episodes_run if total_episodes_run > 0 else 0.0

        # Display metrics in organized sections
        print(f"\n📊 TRADING PERFORMANCE:")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Avg Trades per Episode: {avg_trades_per_episode:.1f}")
        print(f"   Profit Factor: {profit_factor:.2f}")
        print(f"   Avg P&L per Trade: ₹{avg_pnl_per_trade:.2f}")

        print(f"\n💰 FINANCIAL METRICS (Final Episode):")
        print(f"   Initial Capital: ₹{initial_capital:,.2f}")
        print(f"   Final Capital: ₹{final_capital:,.2f}")
        print(f"   Episode P&L: ₹{final_capital - initial_capital:,.2f}")
        print(f"   Episode Return: {total_return_percentage:.2f}%")

        # Calculate simple max drawdown for final episode
        if len(self.capital_history) > 1:
            episode_capitals = self.capital_history[-100:]  # Last episode's capital history
            peak = max(episode_capitals)
            trough = min(episode_capitals)
            max_drawdown_pct = ((trough - peak) / peak) * 100 if peak > 0 else 0
            print(f"   Max Drawdown (Final Episode): {max_drawdown_pct:.2f}%")
        else:
            print(f"   Max Drawdown (Final Episode): 0.00%")

        print(f"\n📈 RISK METRICS:")
        # Calculate Sharpe ratio for final episode only
        if len(self.capital_history) > 1:
            episode_returns = []
            for i in range(1, len(self.capital_history)):
                ret = (self.capital_history[i] - self.capital_history[i-1]) / self.capital_history[i-1]
                episode_returns.append(ret)

            if episode_returns:
                returns_series = pd.Series(episode_returns)
                sharpe_ratio = calculate_sharpe_ratio(returns_series)
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0

        print(f"   Sharpe Ratio (Final Episode): {sharpe_ratio:.3f}")
        print(f"   Profit Factor: {profit_factor:.2f}")

        print(f"\n🤖 TRAINING METRICS:")
        print(f"   Total Episodes: {total_episodes_run}")
        print(f"   Total Reward: {self.total_reward:.2f}")
        print(f"   Avg Reward per Episode: {avg_reward_per_episode:.2f}")

        # Performance assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if win_rate >= 0.6:
            print(f"   ✅ Excellent Win Rate ({win_rate:.1%})")
        elif win_rate >= 0.5:
            print(f"   ✅ Good Win Rate ({win_rate:.1%})")
        else:
            print(f"   ⚠️ Low Win Rate ({win_rate:.1%}) - Consider strategy adjustment")

        if profit_factor >= 1.5:
            print(f"   ✅ Excellent Profit Factor ({profit_factor:.2f})")
        elif profit_factor >= 1.0:
            print(f"   ✅ Profitable Strategy ({profit_factor:.2f})")
        else:
            print(f"   ❌ Losing Strategy ({profit_factor:.2f}) - Needs improvement")

        if sharpe_ratio >= 1.0:
            print(f"   ✅ Good Risk-Adjusted Returns (Sharpe: {sharpe_ratio:.3f})")
        elif sharpe_ratio >= 0.5:
            print(f"   ⚠️ Moderate Risk-Adjusted Returns (Sharpe: {sharpe_ratio:.3f})")
        else:
            print(f"   ❌ Poor Risk-Adjusted Returns (Sharpe: {sharpe_ratio:.3f})")

        print("=" * 80)
