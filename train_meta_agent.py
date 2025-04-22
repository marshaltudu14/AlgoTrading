import os
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
import itertools
import math
import gc

from config import INSTRUMENTS, TIMEFRAMES
from historical_data_fetcher import fetch_historical_data
from data_processing.processor import process_all
from envs.trading_env import TradingEnv


def make_env(instrument, timeframe, log_dir=None):
    """Factory for TradingEnv with optional monitoring"""
    def _init():
        env = TradingEnv(instrument, timeframe)
        if log_dir:
            fname = f"{instrument.replace(' ','_')}_{timeframe}.csv"
            env = Monitor(env, filename=os.path.join(log_dir, fname))
        return env
    return _init


def train_multitask_ppo(total_timesteps=100_000):
    """Train PPO across all instrument-timeframe tasks."""
    # Fetch and process data
    fetch_historical_data()
    process_all()
    # Build vectorized multi-task env without logging
    env_fns = [make_env(instr, tf) for instr in INSTRUMENTS for tf in TIMEFRAMES]
    vec_env = DummyVecEnv(env_fns)

    model = PPO(
        'MlpPolicy', vec_env, verbose=1,
        ent_coef=float(os.getenv('PPO_ENT_COEF', 0.01)),
        gamma=float(os.getenv('PPO_GAMMA', 0.99)),
        learning_rate=float(os.getenv('PPO_LR', 3e-4))
    )
    model.learn(total_timesteps=total_timesteps)
    os.makedirs('models', exist_ok=True)
    model.save('models/ppo_multitask')
    return model


def train_rl2(total_timesteps=100_000, chunk_size=None):
    """Train RL^2 with RecurrentPPO across all tasks in manageable chunks."""
    # Prepare task list
    tasks = [(instr, tf) for instr in INSTRUMENTS for tf in TIMEFRAMES]
    num_tasks = len(tasks)
    # Determine chunk size
    if chunk_size is None:
        chunk_size = int(os.getenv("RL_CHUNK_SIZE", 8))
    num_chunks = math.ceil(num_tasks / chunk_size)
    per_chunk_timesteps = total_timesteps // num_chunks
    print(f"[train_rl2] Total tasks: {num_tasks}, chunk_size: {chunk_size}, num_chunks: {num_chunks}, steps/chunk: {per_chunk_timesteps}")
    model = None
    for idx in range(num_chunks):
        # Slice tasks for this chunk
        start, end = idx * chunk_size, (idx + 1) * chunk_size
        chunk_tasks = tasks[start:end]
        print(f"[train_rl2] Chunk {idx+1}/{num_chunks} tasks: {chunk_tasks}")
        # Fetch and process data once at first chunk
        if idx == 0:
            print("[train_rl2] Fetching and processing all data...")
            fetch_historical_data()
            process_all()
            print("[train_rl2] Data ready.")
        # Build environments for this chunk
        env_fns = [make_env(instr, tf) for instr, tf in chunk_tasks]
        vec_env = DummyVecEnv(env_fns)
        # Initialize or update model
        model_path = f'models/rl2_multitask_chunk_{idx}.zip'
        if model is None:
            print(f"[train_rl2] Initializing RecurrentPPO for chunk {idx+1}...")
            model = RecurrentPPO(
                'MlpLstmPolicy', vec_env, verbose=1,
                ent_coef=float(os.getenv('RL2_ENT_COEF', 0.01)),
                gamma=float(os.getenv('RL2_GAMMA', 0.99)),
                learning_rate=float(os.getenv('RL2_LR', 3e-4))
            )
        else:
            prev_envs = model.get_env().num_envs if model.get_env() is not None else None
            curr_envs = vec_env.num_envs
            if prev_envs == curr_envs:
                print(f"[train_rl2] Updating environment for existing model, chunk {idx+1}...")
                model.set_env(vec_env)
            else:
                # Reload model with new env if env count changes
                prev_model_path = f'models/rl2_multitask_chunk_{idx}.zip'
                if os.path.exists(prev_model_path):
                    print(f"[train_rl2] Reloading model from {prev_model_path} with new env for chunk {idx+1}...")
                    model = RecurrentPPO.load(prev_model_path, env=vec_env, verbose=1)
                else:
                    raise RuntimeError(f"Expected model checkpoint at {prev_model_path} for chunk {idx}")
        # Train
        print(f"[train_rl2] Training on chunk {idx+1} for {per_chunk_timesteps} timesteps...")
        try:
            model.learn(total_timesteps=per_chunk_timesteps, reset_num_timesteps=False)
            print(f"[train_rl2] Chunk {idx+1} training complete.")
        except Exception as e:
            print(f"[train_rl2] Exception in training chunk {idx+1}: {e}")
            raise
        # Save intermediate model
        os.makedirs('models', exist_ok=True)
        model.save(f'models/rl2_multitask_chunk_{idx+1}')
        print(f"[train_rl2] Saved model for chunk {idx+1}.")
        # Cleanup to free memory
        del vec_env
        gc.collect()
    print("[train_rl2] All chunks trained. Final model ready.")
    return model



if __name__ == '__main__':
    train_rl2()
