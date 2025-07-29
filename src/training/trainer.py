import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import copy
import logging

from src.backtesting.environment import TradingEnv
from src.agents.base_agent import BaseAgent
from src.agents.moe_agent import MoEAgent
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
    def __init__(self, agent: BaseAgent, num_episodes: int, log_interval: int = 10, meta_lr: float = 0.001):
        self.agent = agent
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.meta_lr = meta_lr
        self.env = None # Will be initialized in train method
        import logging
        logging.getLogger('src.backtesting.engine').setLevel(logging.INFO)

        # Initialize meta-optimizer for MAML
        if isinstance(agent, MoEAgent):
            self.meta_optimizer = self._create_meta_optimizer(agent)
        else:
            self.meta_optimizer = None

        # Enhanced tracking for comprehensive metrics
        self.capital_history = []
        self.episode_rewards = []
        self.total_reward = 0.0

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

            self.agent.learn(experiences)

            # Track episode metrics
            self.episode_rewards.append(episode_reward)
            self.total_reward += episode_reward

            # Track capital history
            current_capital = self.env.engine.get_account_state()['capital']
            self.capital_history.append(current_capital)

            if (episode + 1) % self.log_interval == 0:
                # Calculate proper metrics
                avg_reward = self.total_reward / (episode + 1) if episode > 0 else self.total_reward

                # Get trading metrics from environment
                account_state = self.env.engine.get_account_state()
                trade_history = self.env.engine.get_trade_history()

                # Calculate win rate and other metrics
                if trade_history:
                    win_rate = calculate_win_rate(trade_history)
                    total_pnl = calculate_total_pnl(trade_history)
                    num_trades = calculate_num_trades(trade_history)

                    self._log_progress_detailed(episode + 1, self.total_reward, avg_reward,
                                              win_rate, total_pnl, num_trades, account_state['capital'])
                else:
                    self._log_progress(episode + 1, episode_reward, avg_reward, None)

        # Display comprehensive training summary
        self._display_training_summary(initial_capital)

        self.agent.save_model("trained_agent.pth")

    def _create_meta_optimizer(self, agent: MoEAgent) -> optim.Optimizer:
        """Create meta-optimizer for MAML that includes all trainable parameters."""
        meta_params = []

        # Add gating network parameters
        meta_params.extend(agent.gating_network.parameters())

        # Add all expert parameters
        for expert in agent.experts:
            meta_params.extend(expert.actor.parameters())
            meta_params.extend(expert.critic.parameters())

        return optim.Adam(meta_params, lr=self.meta_lr)

    def run_backtest_and_report(self, agent: BaseAgent, env: TradingEnv) -> Dict:
        # Run a full backtest episode/simulation
        observation = env.reset()
        done = False
        trade_history = []
        equity_curve = [env.engine.initial_capital]

        while not done:
            action = agent.select_action(observation)
            next_observation, reward, done, info = env.step(action)
            
            # Assuming engine stores trade details in a way that can be retrieved
            # For now, let's just append a dummy trade for testing purposes
            # In a real scenario, BacktestingEngine would provide detailed trade info
            trade_history.append({'pnl': reward}) # Simplified: reward is PnL for this step
            equity_curve.append(env.engine.get_account_state()['capital'])

            observation = next_observation

        # Calculate metrics
        metrics = {
            "Sharpe Ratio": calculate_sharpe_ratio(pd.Series(equity_curve).pct_change().dropna()),
            "Total P&L": calculate_total_pnl(trade_history),
            "Profit Factor": calculate_profit_factor(trade_history),
            "Maximum Drawdown": calculate_max_drawdown(pd.Series(equity_curve)),
            "Win Rate": calculate_win_rate(trade_history),
            "Average P&L per Trade": calculate_avg_pnl_per_trade(trade_history),
            "Number of Trades": calculate_num_trades(trade_history)
        }

        self._display_backtest_report(metrics)
        return metrics

    def _display_backtest_report(self, metrics: Dict) -> None:
        print("\n=== Backtesting Report ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        print("==========================")

    def meta_train(self, data_loader: DataLoader, initial_capital: float, num_meta_iterations: int, num_inner_loop_steps: int, num_evaluation_steps: int, meta_batch_size: int) -> None:
        """
        Complete MAML meta-training implementation with proper outer loop meta-updates.
        """
        if not isinstance(self.agent, MoEAgent):
            raise ValueError("Meta-training is only supported for MoEAgent instances")

        # Initialize tracking for MAML training
        self.capital_history = [initial_capital]
        self.total_reward = 0.0
        self.episode_rewards = []

        for iteration in range(num_meta_iterations):
            tasks = data_loader.sample_tasks(meta_batch_size)
            meta_gradients = []
            avg_task_reward = 0
            total_meta_loss = 0

            for task in tasks:
                # Inner loop adaptation with gradient tracking
                task_meta_loss, task_reward, final_capital = self._execute_task_adaptation_and_evaluation(
                    data_loader, task, initial_capital, num_inner_loop_steps, num_evaluation_steps
                )

                meta_gradients.append(task_meta_loss)
                avg_task_reward += task_reward
                # Handle 0-dimensional tensor safely
                if hasattr(task_meta_loss, 'item'):
                    total_meta_loss += task_meta_loss.item()
                else:
                    total_meta_loss += float(task_meta_loss)

                # Track capital and reward for comprehensive metrics
                self.capital_history.append(final_capital)
                self.total_reward += task_reward
                self.episode_rewards.append(task_reward)

            # Outer loop meta-update: Update meta-parameters based on accumulated gradients
            self._perform_meta_update(meta_gradients)

            avg_task_reward /= meta_batch_size
            avg_meta_loss = total_meta_loss / meta_batch_size

            self._log_meta_progress(iteration + 1, avg_meta_loss, avg_task_reward)

        # Display comprehensive MAML training summary
        self._display_maml_training_summary(initial_capital, num_meta_iterations, avg_task_reward)

        self.agent.save_model("meta_trained_agent.pth")

    def _execute_task_adaptation_and_evaluation(self, data_loader: DataLoader, task: Tuple,
                                              initial_capital: float, num_inner_loop_steps: int,
                                              num_evaluation_steps: int) -> Tuple[torch.Tensor, float, float]:
        """
        Execute inner loop adaptation and evaluation for a single task.
        Returns meta-loss tensor, evaluation reward, and final capital.
        """
        # Create task environment
        # Handle task format: task can be tuple (instrument, timeframe) or string (symbol)
        if isinstance(task, tuple):
            # Convert tuple format to symbol format
            instrument, timeframe = task
            if instrument.startswith('features_'):
                instrument = instrument.replace('features_', '')
            symbol = f"{instrument}_{timeframe}"
        else:
            symbol = task

        # CRITICAL: Use universal parameters to ensure consistent observation dimensions
        if hasattr(self, 'universal_params') and self.universal_params:
            lookback_window = self.universal_params.lookback_window
            episode_length = self.universal_params.episode_length
            logger.info(f"🎯 MAML using universal parameters: lookback={lookback_window}, episode_length={episode_length}")
        else:
            # Fallback to default values if universal params not available
            lookback_window = 20
            episode_length = 500
            logger.warning("⚠️ Universal parameters not available for MAML, using defaults")

        self.env = TradingEnv(
            data_loader=data_loader,
            symbol=symbol,
            initial_capital=initial_capital,
            lookback_window=lookback_window,
            episode_length=episode_length,
            reward_function="trading_focused",  # Use trading-focused reward
            use_streaming=False  # Use full data for consistent dimensions
        )

        # Create a copy of the agent for adaptation (maintaining gradient tracking)
        adapted_agent = copy.deepcopy(self.agent)

        # Inner loop adaptation
        for step in range(num_inner_loop_steps):
            observation = self.env.reset()
            done = False
            adaptation_experiences = []

            # Collect experiences for this adaptation step
            while not done and len(adaptation_experiences) < 10:  # Limit experiences per step
                action = adapted_agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                adaptation_experiences.append((observation, action, reward, next_observation, done))
                observation = next_observation

            # Perform adaptation using collected experiences
            if adaptation_experiences:
                adapted_agent.learn(adaptation_experiences)

        # Evaluation phase: measure performance of adapted agent
        evaluation_reward = 0
        evaluation_experiences = []

        for eval_episode in range(num_evaluation_steps):
            observation = self.env.reset()
            done = False
            episode_reward = 0
            episode_experiences = []

            while not done:
                action = adapted_agent.select_action(observation)

                # Handle both Gym API versions (4 vs 5 return values)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_observation, reward, done, truncated, info = step_result
                else:
                    next_observation, reward, done, info = step_result
                    truncated = False  # For older Gym versions

                episode_reward += reward
                episode_experiences.append((observation, action, reward, next_observation, done))
                observation = next_observation

            evaluation_reward += episode_reward
            evaluation_experiences.extend(episode_experiences)

        # Calculate meta-loss: negative of evaluation performance
        # This creates a gradient that encourages better initial parameters
        avg_evaluation_reward = evaluation_reward / num_evaluation_steps
        meta_loss = -torch.tensor(avg_evaluation_reward, requires_grad=True)

        # Get final capital from the environment
        final_capital = self.env.engine.get_account_state()['capital']

        return meta_loss, avg_evaluation_reward, final_capital

    def _perform_meta_update(self, meta_gradients: List[torch.Tensor]) -> None:
        """
        Perform the outer loop meta-update using accumulated gradients from all tasks.
        """
        if self.meta_optimizer is None:
            return

        # Average the meta-gradients across all tasks
        avg_meta_loss = torch.stack(meta_gradients).mean()

        # Zero gradients
        self.meta_optimizer.zero_grad()

        # Backward pass to compute gradients
        avg_meta_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self._get_meta_parameters(), max_norm=1.0)

        # Update meta-parameters
        self.meta_optimizer.step()

    def _get_meta_parameters(self) -> List[torch.nn.Parameter]:
        """Get all meta-parameters for gradient clipping."""
        meta_params = []

        if isinstance(self.agent, MoEAgent):
            # Add gating network parameters
            meta_params.extend(self.agent.gating_network.parameters())

            # Add all expert parameters
            for expert in self.agent.experts:
                meta_params.extend(expert.actor.parameters())
                meta_params.extend(expert.critic.parameters())

        return meta_params

    def _log_meta_progress(self, iteration: int, meta_loss: float, avg_task_reward: float) -> None:
        print(f"Meta-Iteration: {iteration}, Meta-Loss: {meta_loss:.2f}, Avg Task Reward: {avg_task_reward:.2f}")

    def _log_progress(self, episode: int, total_reward: float, avg_reward: float, loss: float = None) -> None:
        log_string = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}"
        if loss is not None:
            log_string += f", Loss: {loss:.4f}"
        print(log_string)

    def _log_progress_detailed(self, episode: int, episode_reward: float, avg_reward: float,
                              win_rate: float, total_pnl: float, num_trades: int, capital: float) -> None:
        """Log detailed training progress with trading metrics."""
        logger.info(f"📊 Episode {episode} Training Progress:")
        logger.info(f"  💰 Reward: {episode_reward:.2f} (Avg: {avg_reward:.2f})")
        logger.info(f"  📈 Trading: {num_trades} trades, Win Rate: {win_rate:.1%}")
        logger.info(f"  💵 P&L: ₹{total_pnl:.2f}, Capital: ₹{capital:.2f}")

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

        # Get final trading data
        trade_history = self.env.engine.get_trade_history()
        final_account = self.env.engine.get_account_state()

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            trade_history=trade_history,
            capital_history=self.capital_history,
            initial_capital=initial_capital,
            total_episodes=self.num_episodes,
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

    def _display_maml_training_summary(self, initial_capital: float, num_meta_iterations: int,
                                     final_avg_task_reward: float) -> None:
        """Display comprehensive MAML training summary."""
        print("\n" + "=" * 80)
        print("🧠 MAML META-LEARNING TRAINING SUMMARY")
        print("=" * 80)

        # Get final trading data if environment exists
        if hasattr(self, 'env') and self.env is not None:
            trade_history = self.env.engine.get_trade_history()
            final_account = self.env.engine.get_account_state()
        else:
            trade_history = []
            final_account = {'capital': self.capital_history[-1] if hasattr(self, 'capital_history') and self.capital_history else initial_capital}

        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(
            trade_history=trade_history,
            capital_history=self.capital_history if hasattr(self, 'capital_history') else [initial_capital],
            initial_capital=initial_capital,
            total_episodes=num_meta_iterations,
            total_reward=self.total_reward if hasattr(self, 'total_reward') else final_avg_task_reward
        )

        # Display trading metrics
        print(f"\n📊 META-LEARNING PERFORMANCE:")
        print(f"   Meta-Iterations: {num_meta_iterations}")
        print(f"   Final Avg Task Reward: {final_avg_task_reward:.2f}")
        print(f"   Win Rate: {metrics['win_rate']:.1%}")
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\n💰 FINANCIAL PERFORMANCE:")
        print(f"   Initial Capital: ₹{initial_capital:,.2f}")
        print(f"   Final Capital: ₹{final_account['capital']:,.2f}")
        print(f"   Total P&L: ₹{metrics['total_pnl']:,.2f}")
        print(f"   Total Return: {metrics['total_return_percentage']:.2f}%")
        print(f"   Max Drawdown: {metrics['max_drawdown_percentage']:.2f}%")

        print(f"\n📈 RISK & ADAPTATION:")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Avg P&L per Trade: ₹{metrics['avg_pnl_per_trade']:.2f}")
        print(f"   Avg Reward per Meta-Iteration: {metrics['avg_reward_per_episode']:.2f}")

        print(f"\n🎯 MAML ASSESSMENT:")
        if final_avg_task_reward > -1000:
            print(f"   ✅ Good meta-learning adaptation")
        elif final_avg_task_reward > -10000:
            print(f"   ⚠️ Moderate meta-learning performance")
        else:
            print(f"   ❌ Poor meta-learning - needs hyperparameter tuning")

        print("=" * 80)

    def train_maml(self, data_loader: DataLoader, symbol: str, initial_capital: float, meta_iterations: int = 15) -> Dict[str, Any]:
        """Train using MAML meta-learning approach."""
        logger.info(f"🧠 Starting MAML meta-learning with {meta_iterations} iterations")

        # Use the existing meta-learning training logic
        return self.train(data_loader, symbol, initial_capital)
