import pytest
from unittest.mock import Mock
from src.training.trainer import Trainer
from src.backtesting.environment import TradingEnv
from src.agents.base_agent import BaseAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.moe_agent import MoEAgent
from src.utils.data_loader import DataLoader
import numpy as np
import pandas as pd
import os

def test_trainer_initialization():
    mock_env = Mock(spec=TradingEnv)
    mock_agent = Mock(spec=BaseAgent)
    num_episodes = 100
    log_interval = 10

    trainer = Trainer(mock_env, mock_agent, num_episodes, log_interval)

    assert trainer.env == mock_env
    assert trainer.agent == mock_agent
    assert trainer.num_episodes == num_episodes
    assert trainer.log_interval == log_interval

def test_trainer_train_loop():
    mock_env = Mock(spec=TradingEnv)
    mock_agent = Mock(spec=BaseAgent)
    num_episodes = 5
    log_interval = 1

    # Configure mock_env behavior
    mock_env.reset.return_value = np.array([0.0])
    mock_env.step.side_effect = [
        (np.array([0.1]), 1.0, False, {}),  # Episode 1, step 1
        (np.array([0.2]), 2.0, True, {}),   # Episode 1, step 2 (done)
        (np.array([0.3]), 1.5, False, {}),  # Episode 2, step 1
        (np.array([0.4]), 2.5, True, {}),   # Episode 2, step 2 (done)
        (np.array([0.5]), 1.0, False, {}),  # Episode 3, step 1
        (np.array([0.6]), 2.0, True, {}),   # Episode 3, step 2 (done)
        (np.array([0.7]), 1.0, False, {}),  # Episode 4, step 1
        (np.array([0.8]), 2.0, True, {}),   # Episode 4, step 2 (done)
        (np.array([0.9]), 1.0, False, {}),  # Episode 5, step 1
        (np.array([1.0]), 2.0, True, {}),   # Episode 5, step 2 (done)
    ]

    # Configure mock_agent behavior
    mock_agent.select_action.return_value = 0
    mock_agent.learn.return_value = None
    mock_agent.save_model.return_value = None

    trainer = Trainer(mock_env, mock_agent, num_episodes, log_interval)
    trainer.train()

    assert mock_env.reset.call_count == num_episodes
    assert mock_agent.select_action.call_count > 0
    assert mock_env.step.call_count > 0
    assert mock_agent.learn.call_count == num_episodes
    assert mock_agent.save_model.call_count == 1

def test_moe_training_integration():
    mock_env = Mock(spec=TradingEnv)
    mock_moe_agent = Mock(spec=MoEAgent)
    num_episodes = 3
    log_interval = 1

    mock_env.reset.return_value = np.array([0.0])
    mock_env.step.side_effect = [
        (np.array([0.1]), 1.0, False, {}),
        (np.array([0.2]), 2.0, True, {}),
        (np.array([0.3]), 1.5, False, {}),
        (np.array([0.4]), 2.5, True, {}),
        (np.array([0.5]), 1.0, False, {}),
        (np.array([0.6]), 2.0, True, {}),
    ]

    mock_moe_agent.select_action.return_value = 0
    mock_moe_agent.learn.return_value = None
    mock_moe_agent.save_model.return_value = None

    trainer = Trainer(mock_env, mock_moe_agent, num_episodes, log_interval)
    trainer.train()

    assert mock_env.reset.call_count == num_episodes
    assert mock_moe_agent.select_action.call_count > 0
    assert mock_env.step.call_count > 0
    assert mock_moe_agent.learn.call_count == num_episodes
    assert mock_moe_agent.save_model.call_count == 1

def test_trainer_meta_train_loop():
    mock_data_loader = Mock(spec=DataLoader)
    mock_data_loader.get_available_tasks.return_value = [("Nifty", "5"), ("Bank_Nifty", "5")]
    mock_data_loader.sample_tasks.return_value = [("Nifty", "5")]
    mock_data_loader.get_task_data.return_value = pd.DataFrame({
        'datetime': ['2023-01-01', '2023-01-02'],
        'open': [100, 101],
        'high': [105, 106],
        'low': [99, 100],
        'close': [103, 104]
    })

    mock_env = Mock(spec=TradingEnv)
    mock_env.data_loader = mock_data_loader # Attach mock data_loader to mock_env
    mock_env.data = pd.DataFrame() # Mock the data attribute
    mock_env.reset.return_value = np.array([0.0])
    mock_env.step.side_effect = [
        (np.array([0.1]), 1.0, False, {}),
        (np.array([0.2]), 2.0, True, {}),
        (np.array([0.3]), 1.5, False, {}),
        (np.array([0.4]), 2.5, True, {}),
    ]

    mock_agent = Mock(spec=PPOAgent)
    mock_agent.select_action.return_value = 0
    mock_agent.adapt.return_value = mock_agent # adapted agent is just the mock agent itself
    mock_agent.save_model.return_value = None

    num_meta_iterations = 1
    num_inner_loop_steps = 1
    num_evaluation_steps = 1
    meta_batch_size = 1

    trainer = Trainer(mock_env, mock_agent, num_episodes=1, log_interval=1) # num_episodes and log_interval not directly used by meta_train
    trainer.meta_train(num_meta_iterations, num_inner_loop_steps, num_evaluation_steps, meta_batch_size)

    mock_data_loader.sample_tasks.assert_called_once_with(meta_batch_size)
    mock_data_loader.get_task_data.assert_called_once()
    mock_env.reset.call_count > 0
    mock_agent.select_action.call_count > 0
    mock_env.step.call_count > 0
    mock_agent.adapt.call_count > 0
    mock_agent.save_model.assert_called_once_with("meta_trained_agent.pth")

def test_comprehensive_backtesting_report():
    mock_env = Mock(spec=TradingEnv)
    mock_agent = Mock(spec=BaseAgent)

    # Mock engine and its methods
    mock_engine = Mock()
    mock_engine.initial_capital = 10000.0
    mock_engine.get_account_state.side_effect = [
        {'capital': 10000.0, 'realized_pnl': 0, 'unrealized_pnl': 0},
        {'capital': 10100.0, 'realized_pnl': 100, 'unrealized_pnl': 0},
        {'capital': 10050.0, 'realized_pnl': 50, 'unrealized_pnl': 0},
        {'capital': 10250.0, 'realized_pnl': 250, 'unrealized_pnl': 0},
    ]
    mock_env.engine = mock_engine

    # Mock env step to simulate trades and PnL
    mock_env.reset.return_value = np.array([0.0])
    mock_env.step.side_effect = [
        (np.array([0.1]), 100.0, False, {}), # PnL 100
        (np.array([0.2]), -50.0, False, {}), # PnL -50
        (np.array([0.3]), 200.0, True, {}),  # PnL 200, done
    ]

    mock_agent.select_action.return_value = 0

    trainer = Trainer(mock_env, mock_agent, num_episodes=1, log_interval=1)
    
    # Capture print output
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        metrics = trainer.run_backtest_and_report(mock_agent, mock_env)
    output = f.getvalue()

    assert "=== Backtesting Report ===" in output
    assert "Total P&L: 250.00" in output
    assert "Profit Factor: 6.00" in output # (100+200) / 50
    assert "Win Rate: 0.67" in output # 2 winning trades out of 3
    assert "Number of Trades: 3" in output
    assert "Sharpe Ratio" in output # Check if it's calculated
    assert "Maximum Drawdown" in output # Check if it's calculated
    assert "Average P&L per Trade" in output # Check if it's calculated

    assert metrics["Total P&L"] == 250.0
    assert metrics["Number of Trades"] == 3

def test_real_time_training_progress_updates():
    mock_env = Mock(spec=TradingEnv)
    mock_agent = Mock(spec=BaseAgent)
    num_episodes = 3
    log_interval = 1

    mock_env.reset.return_value = np.array([0.0])
    mock_env.step.side_effect = [
        (np.array([0.1]), 1.0, False, {}),
        (np.array([0.2]), 2.0, True, {}),
        (np.array([0.3]), 1.5, False, {}),
        (np.array([0.4]), 2.5, True, {}),
        (np.array([0.5]), 1.0, False, {}),
        (np.array([0.6]), 2.0, True, {}),
    ]

    mock_agent.select_action.return_value = 0
    mock_agent.learn.return_value = None
    mock_agent.save_model.return_value = None

    trainer = Trainer(mock_env, mock_agent, num_episodes, log_interval)

    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        trainer.train()
    output = f.getvalue()

    # Check if the output contains the expected progress updates
    assert "Episode: 1, Total Reward: 3.00, Avg Reward: 3.00" in output
    assert "Episode: 2, Total Reward: 4.00, Avg Reward: 4.00" in output
    assert "Episode: 3, Total Reward: 3.00, Avg Reward: 3.00" in output