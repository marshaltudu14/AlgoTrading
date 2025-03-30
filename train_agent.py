import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Assuming rl_environment.py is in the src directory
from src.rl_environment import TradingEnv # STATS_FILE is no longer needed here
# Import config variables
from src.config import (
    RL_LOG_DIR, RL_MODEL_SAVE_DIR, RL_MODEL_FILENAME, RL_TOTAL_TIMESTEPS,
    RL_CHECKPOINT_FREQ, RL_N_ENVS, RL_STATS_FILE # Import RL_STATS_FILE for check
)

# --- Configuration ---
# Constants are now imported from src.config
# LOG_DIR = "logs/rl_logs/"
# MODEL_SAVE_DIR = "models/rl_models/"
# MODEL_FILENAME = "ppo_trading_agent"
# TOTAL_TIMESTEPS = 1_000_000
# CHECKPOINT_FREQ = 50_000
# N_ENVS = 4

# Ensure log and model directories exist (already done in config.py)
# os.makedirs(RL_LOG_DIR, exist_ok=True) # Handled in config.py
# os.makedirs(RL_MODEL_SAVE_DIR, exist_ok=True) # Handled in config.py
# --- End Configuration ---

def check_stats_file():
    """Checks if the normalization stats file exists."""
    if not os.path.exists(RL_STATS_FILE): # Use config variable
        print(f"Error: Normalization stats file '{RL_STATS_FILE}' not found.")
        print("Please run preprocess_norm_stats.py first.")
        return False
    return True

if __name__ == "__main__":
    if not check_stats_file():
        exit()

    print("Starting RL agent training...")
    start_time = time.time()

    # --- Environment Setup ---
    # Create vectorized environments for parallel training
    # Use SubprocVecEnv for true parallelism, DummyVecEnv for debugging
    # Wrap each env with Monitor for logging episode stats
    print(f"Creating {RL_N_ENVS} parallel environments...")
    try:
        # Define a function to create a single environment instance
        def make_env():
            env = TradingEnv()
            env = Monitor(env, RL_LOG_DIR) # Use config variable
            return env

        # Create the vectorized environment
        # Use SubprocVecEnv if __name__ == '__main__' check is working correctly
        # Otherwise, fall back to DummyVecEnv which is slower but safer on some systems
        if RL_N_ENVS > 1:
             env = SubprocVecEnv([make_env for _ in range(RL_N_ENVS)])
        else:
             env = DummyVecEnv([make_env])

        print("Environments created successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback
        traceback.print_exc()
        exit()


    # --- Callbacks ---
    # Save a checkpoint of the model every `RL_CHECKPOINT_FREQ` steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(RL_CHECKPOINT_FREQ // RL_N_ENVS, 1), # Adjust frequency for vec envs
        save_path=RL_MODEL_SAVE_DIR, # Use config variable
        name_prefix=RL_MODEL_FILENAME, # Use config variable
        save_replay_buffer=True,
        save_vecnormalize=True, # Important if using VecNormalize wrapper
    )

    # Optional: Evaluation Callback (requires a separate evaluation environment)
    # eval_env = Monitor(TradingEnv()) # Create a separate env for evaluation
    # eval_callback = EvalCallback(eval_env, best_model_save_path=RL_MODEL_SAVE_DIR + 'best_model/',
    #                              log_path=RL_LOG_DIR + 'eval_logs/', eval_freq=max(RL_CHECKPOINT_FREQ // RL_N_ENVS, 1),
    #                              deterministic=True, render=False)


    # --- Model Definition ---
    # Using Multi-Layer Perceptron (MLP) policy network initially
    # Can experiment with 'MlpLstmPolicy' or custom CNN/Transformer later
    # Adjust hyperparameters as needed (learning_rate, n_steps, batch_size, etc.)
    print("Defining PPO model...")
    try:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1, # Print training progress
            tensorboard_log=RL_LOG_DIR, # Use config variable
            # --- Hyperparameters ---
            learning_rate=0.0003, # Default: 0.0003
            n_steps=2048,         # Default: 2048 (Steps per env before update)
            batch_size=64,          # Default: 64
            n_epochs=10,            # Default: 10
            gamma=0.99,           # Default: 0.99 (Discount factor)
            gae_lambda=0.95,        # Default: 0.95
            clip_range=0.2,         # Default: 0.2
            ent_coef=0.01,          # Default: 0.0 - Increased to encourage exploration
            vf_coef=0.5,          # Default: 0.5 (Value function coefficient)
            max_grad_norm=0.5,      # Default: 0.5
            # -----------------------
            device="auto" # Use GPU if available, otherwise CPU
        )
        print("PPO model defined with custom hyperparameters (ent_coef=0.01).")
    except Exception as e:
        print(f"Error defining PPO model: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # --- Training ---
    print(f"Starting training for {RL_TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(
            total_timesteps=RL_TOTAL_TIMESTEPS, # Use config variable
            callback=checkpoint_callback, # Add eval_callback here if using
            log_interval=1, # Log stats every episode
            progress_bar=True
        )
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Save Final Model ---
        final_model_path = os.path.join(RL_MODEL_SAVE_DIR, f"{RL_MODEL_FILENAME}_final") # Use config variables
        print(f"\nSaving final model to {final_model_path}")
        model.save(final_model_path)
        env.close() # Close the environment

    end_time = time.time()
    print(f"\nTraining finished.")
    print(f"Total training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Model checkpoints saved in: {RL_MODEL_SAVE_DIR}") # Use config variable
    print(f"TensorBoard logs saved in: {RL_LOG_DIR}") # Use config variable
    print(f"To view logs, run: tensorboard --logdir {RL_LOG_DIR}") # Use config variable

# Note:
# *   This script sets up basic PPO training with `MlpPolicy`. Hyperparameters are commented out but are crucial for good performance and will likely need tuning.
# *   It uses `SubprocVecEnv` for parallel environment execution, which speeds up training significantly on multi-core CPUs.
# *   It includes a `CheckpointCallback` to save the model periodically during training.
# *   An optional `EvalCallback` structure is commented out; implementing proper evaluation requires setting up a separate, dedicated evaluation environment using unseen data.
# *   Error handling is included for environment creation and training.
# *   Make sure the `src` directory is correctly recognized as a package (e.g., by having an `__init__.py` file, which you already do).
