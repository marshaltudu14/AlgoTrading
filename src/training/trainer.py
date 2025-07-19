import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

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
    calculate_num_trades
)

class Trainer:
    def __init__(self, agent: BaseAgent, num_episodes: int, log_interval: int = 10):
        self.agent = agent
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.env = None # Will be initialized in train method

    def train(self, data_loader: DataLoader, symbol: str, initial_capital: float) -> None:
        # Initialize the environment for the specific symbol
        self.env = TradingEnv(data_loader, symbol, initial_capital)

        for episode in range(self.num_episodes):
            observation = self.env.reset()
            done = False
            total_reward = 0
            experiences = []

            while not done:
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.env.step(action)
                experiences.append((observation, action, reward, next_observation, done))
                observation = next_observation
                total_reward += reward

            self.agent.learn(experiences)

            if (episode + 1) % self.log_interval == 0:
                # For now, passing dummy values for avg_reward and loss
                self._log_progress(episode + 1, total_reward, total_reward, None)

        self.agent.save_model("trained_agent.pth")

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
        for iteration in range(num_meta_iterations):
            tasks = data_loader.sample_tasks(meta_batch_size)
            meta_loss = 0
            avg_task_reward = 0

            for task in tasks:
                # Inner loop adaptation
                task_data = data_loader.get_task_data(task[0], task[1])
                # Temporarily set environment data to task_data for adaptation
                self.env = TradingEnv(data_loader=data_loader, symbol=task[0], initial_capital=initial_capital)

                adapted_agent = self.agent # Initial adapted agent is the current meta-agent
                for _ in range(num_inner_loop_steps):
                    observation = self.env.reset()
                    done = False
                    while not done:
                        action = adapted_agent.select_action(observation)
                        next_observation, reward, done, info = self.env.step(action)
                        adapted_agent = adapted_agent.adapt(observation, action, reward, next_observation, done, 1) # Perform one gradient step
                        observation = next_observation

                # Evaluate adapted agent
                evaluation_reward = 0
                for _ in range(num_evaluation_steps):
                    observation = self.env.reset()
                    done = False
                    episode_reward = 0
                    while not done:
                        action = adapted_agent.select_action(observation)
                        next_observation, reward, done, info = self.env.step(action)
                        episode_reward += reward
                        observation = next_observation
                    evaluation_reward += episode_reward
                avg_task_reward += evaluation_reward / num_evaluation_steps

                # Calculate meta-loss (simplified: using negative evaluation reward)
                meta_loss += -evaluation_reward

            # Outer loop meta-update
            # This part would typically involve backpropagating through the adaptation process
            # and updating the meta-parameters. For this placeholder, we'll just log the meta-loss.
            self._log_meta_progress(iteration + 1, meta_loss / meta_batch_size, avg_task_reward / meta_batch_size)
        self.agent.save_model("meta_trained_agent.pth")

    def _log_meta_progress(self, iteration: int, meta_loss: float, avg_task_reward: float) -> None:
        print(f"Meta-Iteration: {iteration}, Meta-Loss: {meta_loss:.2f}, Avg Task Reward: {avg_task_reward:.2f}")

    def _log_progress(self, episode: int, total_reward: float, avg_reward: float, loss: float = None) -> None:
        log_string = f"Episode: {episode}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}"
        if loss is not None:
            log_string += f", Loss: {loss:.4f}"
        print(log_string, end='\r')
