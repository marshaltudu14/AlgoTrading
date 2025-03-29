import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Assuming rl_environment.py is in the src directory
from src.rl_environment import TradingEnv # No longer need constants from here
# Import config variables
from src.config import (
    RL_MODEL_SAVE_DIR, RL_EVAL_MODEL_FILENAME, RL_N_EVAL_EPISODES,
    RL_DETERMINISTIC_EVAL, RL_EVAL_DATA_DIR, RL_EVAL_STATS_FILE,
    RL_PROCESSED_DATA_DIR, RL_STATS_FILE # Need defaults for comparison
)

# --- Configuration ---
# Constants are now imported from src.config
# --- End Configuration ---

def check_files_exist():
    """Checks if required model and stats files exist."""
    model_path = os.path.join(RL_MODEL_SAVE_DIR, RL_EVAL_MODEL_FILENAME) # Use config vars
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model has been trained and saved correctly, or update RL_EVAL_MODEL_FILENAME in src/config.py.")
        return False
    if not os.path.exists(RL_EVAL_STATS_FILE): # Use config var
        print(f"Error: Normalization stats file '{RL_EVAL_STATS_FILE}' not found.")
        print("Ensure the stats file exists for the evaluation data (run preprocess_norm_stats.py if needed).")
        return False
    if not os.path.exists(RL_EVAL_DATA_DIR): # Use config var
        print(f"Error: Evaluation data directory '{RL_EVAL_DATA_DIR}' not found.")
        return False
    return True

if __name__ == "__main__":
    if not check_files_exist():
        exit()

    print(f"Starting evaluation of model: {RL_EVAL_MODEL_FILENAME}") # Use config var

    # --- Environment Setup for Evaluation ---
    # Create a single environment for evaluation
    # Important: Use the same configuration (lookback, features, etc.) as training
    # but point to the correct data and stats file for evaluation.
    print("Creating evaluation environment...")
    try:
        # Modify TradingEnv temporarily to use EVAL paths if needed,
        # or pass paths as arguments if modifying the class
        # For simplicity here, we assume TradingEnv uses the constants from src.config,
        # ensure RL_EVAL_DATA_DIR and RL_EVAL_STATS_FILE point correctly in src.config
        # if you want to evaluate on different data/stats than the training defaults.
        if RL_EVAL_DATA_DIR != RL_PROCESSED_DATA_DIR or RL_EVAL_STATS_FILE != RL_STATS_FILE:
             print("--- Running OUT-OF-SAMPLE Evaluation ---")
             print(f"  Data Dir: {RL_EVAL_DATA_DIR}")
             print(f"  Stats File: {RL_EVAL_STATS_FILE}")
             # IMPORTANT: Ensure TradingEnv uses these paths. If TradingEnv only reads
             # from src.config defaults, you might need to temporarily modify src.config
             # or modify TradingEnv to accept paths as arguments.
             # For now, we assume TradingEnv uses the defaults from src.config,
             # so this check is mainly informational unless TradingEnv is modified.
        else:
             print("--- Running IN-SAMPLE Evaluation (using training data) ---")
             # Ideally, pass these paths to TradingEnv constructor if modified to accept them.

        eval_env = TradingEnv(render_mode='none') # No rendering during evaluation
        # Optional: Wrap with Monitor to capture episode stats during evaluation
        # eval_env = Monitor(eval_env)

        print("Evaluation environment created.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Load Model ---
    model_path = os.path.join(RL_MODEL_SAVE_DIR, RL_EVAL_MODEL_FILENAME) # Use config vars
    print(f"Loading model from {model_path}...")
    try:
        # Provide the environment instance to PPO.load
        # Ensure the environment uses the correct paths (EVAL_DATA_DIR, EVAL_STATS_FILE)
        # If TradingEnv wasn't modified to take args, it will use defaults from config.
        model = PPO.load(model_path, env=eval_env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Evaluation ---
    print(f"Running evaluation for {RL_N_EVAL_EPISODES} episodes...") # Use config var
    try:
        # Use Stable Baselines3 evaluate_policy helper function
        episode_rewards, episode_lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=RL_N_EVAL_EPISODES, # Use config var
            deterministic=RL_DETERMINISTIC_EVAL, # Use config var
            return_episode_rewards=True, # Get rewards per episode
            warn=True
        )

        print("\n--- Evaluation Results ---")
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_len = np.mean(episode_lengths)

        print(f"Evaluation over {RL_N_EVAL_EPISODES} episodes:") # Use config var
        print(f"Mean reward (Final Sharpe - Cost Penalty): {mean_reward:.4f} +/- {std_reward:.4f}")
        print(f"Mean episode length: {mean_len:.2f} steps")

        # TODO: Add more detailed evaluation metrics:
        # - Run one long episode and track portfolio value, drawdown, trades, etc.
        # - Calculate overall Sharpe, Sortino, Max Drawdown, Win Rate etc.
        # This requires modifying the evaluation loop or adding callbacks.

    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        eval_env.close()

    print("\nEvaluation finished.")

# Note:
# *   This script loads a specified model checkpoint. You'll need to update `RL_EVAL_MODEL_FILENAME` in src/config.py to the actual filename of the model you want to evaluate (e.g., after running `train_agent.py`).
# *   It uses the `evaluate_policy` helper from Stable Baselines3, which runs the agent for a set number of episodes and returns the mean/std of the final episode rewards (our Sharpe - Cost Penalty).
# *   It's currently configured to evaluate using the *training* data directory and stats file (in-sample). Remember to change `RL_EVAL_DATA_DIR` and `RL_EVAL_STATS_FILE` in src/config.py when you have proper test data for out-of-sample evaluation.
# *   More detailed metrics (like max drawdown, overall PnL curve) would require a custom evaluation loop instead of just `evaluate_policy`.
