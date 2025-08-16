import pytest
import pandas as pd
import os
from unittest.mock import patch, mock_open
import logging

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

from src.config.settings import get_settings

# Configure logging to capture messages during tests
@pytest.fixture(autouse=True)
def caplog_fixture(caplog):
    caplog.set_level(logging.INFO)

# Mock data for testing
settings = get_settings()
paths_config = settings.get('paths', {})
MOCK_FINAL_DATA_DIR = paths_config.get('final_data_dir', 'data/final')
MOCK_RAW_DATA_DIR = paths_config.get('raw_data_dir', 'data/raw')

MOCK_FINAL_FILES = {
    "Nifty_5.csv": pd.DataFrame({'datetime': ['2023-01-01'], 'open': [100], 'high': [105], 'low': [99], 'close': [103]}),
    "Bank_Nifty_5.csv": pd.DataFrame({'datetime': ['2023-01-01'], 'open': [200], 'high': [205], 'low': [199], 'close': [203]}),
    "Sensex_15.csv": pd.DataFrame({'datetime': ['2023-01-01'], 'open': [300], 'high': [305], 'low': [299], 'close': [303]}),
}

MOCK_RAW_FILES = {
    "Nifty_2.csv": pd.DataFrame({'datetime': ['2023-01-01'], 'open': [100], 'high': [105], 'low': [99], 'close': [103]}),
}

@pytest.fixture
def mock_data_loader():
    with patch('os.listdir', side_effect=lambda x: list(MOCK_FINAL_FILES.keys()) if x == MOCK_FINAL_DATA_DIR else list(MOCK_RAW_FILES.keys()) if x == MOCK_RAW_DATA_DIR else []):
        with patch('pandas.read_csv', side_effect=lambda x: MOCK_FINAL_FILES[os.path.basename(x)] if os.path.basename(x) in MOCK_FINAL_FILES else MOCK_RAW_FILES[os.path.basename(x)]):
            yield DataLoader(MOCK_FINAL_DATA_DIR, MOCK_RAW_DATA_DIR)

def test_load_all_processed_data_success(mock_data_loader):
    df = mock_data_loader.load_all_processed_data()
    assert not df.empty
    assert len(df) == sum(len(f) for f in MOCK_FINAL_FILES.values())

def test_load_all_processed_data_empty_dir(caplog):
    with patch('os.listdir', return_value=[]):
        loader = DataLoader(MOCK_FINAL_DATA_DIR)
        df = loader.load_all_processed_data()
        assert df.empty
        assert "No CSV files found" in caplog.text

def test_load_all_processed_data_corrupted_file(caplog):
    with patch('os.listdir', return_value=["corrupted.csv"]):
        with patch('pandas.read_csv', side_effect=Exception("Corrupted file")):
            loader = DataLoader(MOCK_FINAL_DATA_DIR)
            df = loader.load_all_processed_data()
            assert df.empty # Or handle partial load based on implementation
            assert "Could not read corrupted.csv" in caplog.text

def test_load_raw_data_for_symbol_success(mock_data_loader):
    df = mock_data_loader.load_raw_data_for_symbol("Nifty_2")
    assert not df.empty
    assert 'datetime' in df.columns
    assert 'open' in df.columns

def test_load_raw_data_for_symbol_file_not_found(caplog):
    with patch('os.path.exists', return_value=False):
        loader = DataLoader(MOCK_RAW_DATA_DIR)
        df = loader.load_raw_data_for_symbol("NonExistent")
        assert df.empty
        assert "File not found for symbol" in caplog.text

def test_load_raw_data_for_symbol_invalid_ohlc(caplog):
    invalid_df = pd.DataFrame({
        'datetime': ['2023-01-01'],
        'open': [100],
        'high': [90], # high < low
        'low': [95],
        'close': [98]
    })
    with patch('os.listdir', return_value=["Invalid.csv"]):
        with patch('pandas.read_csv', return_value=invalid_df):
            loader = DataLoader(MOCK_RAW_DATA_DIR)
            df = loader.load_raw_data_for_symbol("Invalid")
            assert not df.empty
            assert "OHLC validation failed" in caplog.text

def test_get_available_tasks(mock_data_loader):
    tasks = mock_data_loader.get_available_tasks()
    expected_tasks = [
        ('Bank_Nifty', '5'),
        ('Nifty', '5'),
        ('Sensex', '15')
    ]
    assert sorted(tasks) == sorted(expected_tasks)

def test_sample_tasks_sufficient(mock_data_loader):
    with patch('random.sample', return_value=[('Nifty', '5'), ('Bank_Nifty', '5')]):
        sampled_tasks = mock_data_loader.sample_tasks(2)
        assert len(sampled_tasks) == 2
        assert ('Nifty', '5') in sampled_tasks
        assert ('Bank_Nifty', '5') in sampled_tasks

def test_sample_tasks_insufficient(mock_data_loader, caplog):
    with patch('random.sample', side_effect=lambda population, k: population):
        sampled_tasks = mock_data_loader.sample_tasks(10) # Request more than available
        assert len(sampled_tasks) == 3 # Only 3 available
        assert "Requested 10 tasks, but only 3 are available" in caplog.text

def test_get_task_data_success(mock_data_loader):
    df = mock_data_loader.get_task_data('Nifty', '5')
    assert not df.empty
    assert len(df) == 1

def test_get_task_data_file_not_found(caplog):
    with patch('os.path.exists', return_value=False):
        loader = DataLoader(MOCK_FINAL_DATA_DIR)
        df = loader.get_task_data('NonExistent', '1')
        assert df.empty
        assert "Task data file not found" in caplog.text

# Tests for metrics.py

def test_calculate_sharpe_ratio():
    returns = pd.Series([0.01, 0.02, -0.01, 0.005])
    assert calculate_sharpe_ratio(returns) == pytest.approx(0.57735, rel=1e-3)
    assert calculate_sharpe_ratio(pd.Series([0,0,0])) == 0.0

def test_calculate_total_pnl():
    trade_history = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}]
    assert calculate_total_pnl(trade_history) == 250
    assert calculate_total_pnl([]) == 0

def test_calculate_profit_factor():
    trade_history = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -25}]
    assert calculate_profit_factor(trade_history) == pytest.approx(300 / 75)
    assert calculate_profit_factor([{'pnl': 100}]) == float('inf')
    assert calculate_profit_factor([{'pnl': -100}]) == 0.0
    assert calculate_profit_factor([]) == 0.0

def test_calculate_max_drawdown():
    equity_curve = pd.Series([100, 120, 110, 130, 100, 150])
    assert calculate_max_drawdown(equity_curve) == pytest.approx(0.23076, rel=1e-3) # (130-100)/130
    assert calculate_max_drawdown(pd.Series([])) == 0.0

def test_calculate_win_rate():
    trade_history = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': -25}]
    assert calculate_win_rate(trade_history) == 0.5
    assert calculate_win_rate([]) == 0.0

def test_calculate_avg_pnl_per_trade():
    trade_history = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}]
    assert calculate_avg_pnl_per_trade(trade_history) == pytest.approx(250 / 3)
    assert calculate_avg_pnl_per_trade([]) == 0.0

def test_calculate_num_trades():
    trade_history = [{'pnl': 100}, {'pnl': -50}, {'pnl': 200}]
    assert calculate_num_trades(trade_history) == 3
    assert calculate_num_trades([]) == 0
