"""
python evaluate_model.py --model models/rl2_multitask_chunk_9.zip --eval_days 90

Evaluation script for trained RL trading model.
Runs the saved PPO or RecurrentPPO model across all instrument-timeframe environments
and collects performance metrics: final capital, max drawdown, trade count, win rate.
Outputs results to console and saves to CSV.
"""
import pandas as pd
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from envs.trading_env import TradingEnv
from config import INSTRUMENTS, TIMEFRAMES
import config
from broker.broker_api import authenticate_fyers
from envs.data_fetcher import fetch_candle_data
from data_processing.processor import process_df
import random


def load_model(model_path):
    try:
        return PPO.load(model_path)
    except Exception:
        return RecurrentPPO.load(model_path)


def evaluate_model(model_path):
    model = load_model(model_path)
    records = []
    for inst in INSTRUMENTS:
        print(f"Evaluating instrument: {inst}")
        for tf in TIMEFRAMES:
            print(f"  Timeframe: {tf}...", end=" ")
            env = TradingEnv(inst, tf)
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                steps += 1
            print(f"done in {steps} steps. Final capital: {info.get('capital')}")
            records.append({
                'instrument': inst,
                'timeframe': tf,
                'final_capital': info.get('capital'),
                'max_drawdown': info.get('max_drawdown'),
                'trade_count': info.get('trade_count'),
                'win_rate': info.get('win_rate')
            })
    df = pd.DataFrame(records)
    print("\nEvaluation complete. DataFrame created.")
    return df


if __name__ == '__main__':
    # Hardcoded model and evaluation window
    model = load_model('models/rl2_multitask_chunk_9.zip')
    days = 90
    print(f"Starting unseen evaluation: last {days} days")
    # Authenticate once
    fyers, _ = authenticate_fyers(
        config.APP_ID, config.SECRET_KEY, config.REDIRECT_URI,
        config.FYERS_USER, config.FYERS_PIN, config.FYERS_TOTP
    )
    # Sample 5 random instrument-timeframe pairs
    pairs = [(inst, tf) for inst in config.INSTRUMENTS for tf in config.TIMEFRAMES]
    selected = random.sample(pairs, 5)
    print(f"Selected pairs: {selected}")
    records = []
    for inst, tf in selected:
        symbol = config.INSTRUMENTS[inst]
        print(f"Fetching {days} days for {inst} @ {tf}m... ", end="")
        raw = fetch_candle_data(fyers, days, symbol, tf)
        processed = process_df(raw)
        # Override env with fresh data
        env = TradingEnv(inst, tf)
        env.df = processed
        env.data = processed[env.feature_cols].values.astype(env.data.dtype)
        obs, _ = env.reset()
        done, steps = False, 0
        hold_count = buy_count = sell_count = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # count actions
            if action == 0:
                hold_count += 1
            elif action == 1:
                buy_count += 1
            else:
                sell_count += 1
            obs, _, done, _, info = env.step(action)
            steps += 1
        print(f"done in {steps} steps. Final capital: {info.get('capital')}, holds:{hold_count}, buys:{buy_count}, sells:{sell_count}")
        records.append({
            'instrument': inst,
            'timeframe': tf,
            'final_capital': info.get('capital'),
            'max_drawdown': info.get('max_drawdown'),
            'trade_count': info.get('trade_count'),
            'win_rate': info.get('win_rate'),
            'holds': hold_count,
            'buys': buy_count,
            'sells': sell_count
        })
    # Print results only
    df = pd.DataFrame(records)
    print(df)
