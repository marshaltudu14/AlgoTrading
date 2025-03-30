import os
import pandas as pd
import numpy as np
import argparse
import json
import traceback
from datetime import datetime
from fyers_apiv3 import fyersModel
from stable_baselines3 import PPO

# Import project modules
from src.rl_environment import TradingEnv, MARKET_FEATURES # Import features needed for stats calc
from src.data_handler import fetch_candle_data, FullFeaturePipeline
from src.fyers_auth import get_fyers_access_token
from src.config import (
    RL_MODEL_SAVE_DIR, RL_EVAL_MODEL_FILENAME, RL_N_EVAL_EPISODES,
    RL_DETERMINISTIC_EVAL, INDEX_MAPPING, APP_ID # Need INDEX_MAPPING and APP_ID
)

# --- Helper Function for On-the-Fly Normalization Stats ---
def calculate_norm_stats(df: pd.DataFrame, features: list) -> dict:
    """
    Calculates mean and std deviation for specified features in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        features (list): A list of feature column names to calculate stats for.

    Returns:
        dict: A dictionary where keys are feature names and values are
              dicts containing 'mean' and 'std'.
    """
    stats = {}
    print("Calculating normalization stats for fetched data...")
    for feature in features:
        if feature in df.columns:
            # Ensure data is numeric, coercing errors and filling NaNs with 0 before calc
            # This prevents stats calculation from failing on non-numeric data
            numeric_data = pd.to_numeric(df[feature], errors='coerce').fillna(0)
            mean = np.mean(numeric_data)
            std = np.std(numeric_data)
            stats[feature] = {'mean': float(mean), 'std': float(std)}
            # print(f"  Stats for {feature}: Mean={mean:.4f}, Std={std:.4f}") # Optional: Debug print
        else:
            # Handle derived features like 'close_ema50_diff' - use base 'close' stats
            if feature == 'close_ema50_diff' and 'close' in stats:
                 stats[feature] = stats['close'] # Use pre-calculated close stats
                 print(f"  Using 'close' stats for derived feature: {feature}")
            elif feature == 'close_ema200_diff' and 'close' in stats:
                 stats[feature] = stats['close'] # Use pre-calculated close stats
                 print(f"  Using 'close' stats for derived feature: {feature}")
            else:
                 print(f"Warning: Feature '{feature}' not found in DataFrame for stats calculation. Skipping.")
                 # Optionally add default stats (mean=0, std=1) if needed by env
                 # stats[feature] = {'mean': 0.0, 'std': 1.0}
    print("Normalization stats calculation complete.")
    return stats

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL trading agent on fetched data.")
    parser.add_argument("--instrument", type=str, default="Nifty", help="Index name (e.g., Nifty, Bank_Nifty). Must be in config.INDEX_MAPPING.")
    parser.add_argument("--timeframe", type=int, default=5, help="Candle timeframe in minutes (e.g., 5, 15).")
    parser.add_argument("--days", type=int, default=60, help="Number of past days of data to fetch for evaluation (max ~100).")
    parser.add_argument("--model", type=str, default=RL_EVAL_MODEL_FILENAME, help=f"Filename of the model to load from {RL_MODEL_SAVE_DIR}.")
    parser.add_argument("--episodes", type=int, default=RL_N_EVAL_EPISODES, help="Number of evaluation episodes to run.")
    parser.add_argument("--deterministic", action='store_true', default=RL_DETERMINISTIC_EVAL, help="Use deterministic actions for evaluation.")
    parser.add_argument("--no-deterministic", dest='deterministic', action='store_false', help="Use stochastic actions for evaluation.")


    args = parser.parse_args()

    # --- Validate Instrument ---
    if args.instrument not in INDEX_MAPPING:
        print(f"Error: Instrument '{args.instrument}' not found in INDEX_MAPPING in src/config.py.")
        print(f"Available instruments: {list(INDEX_MAPPING.keys())}")
        exit()
    fyers_symbol = INDEX_MAPPING[args.instrument]["symbol"]

    # --- Validate Model File ---
    model_path = os.path.join(RL_MODEL_SAVE_DIR, args.model)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()

    print(f"--- Starting Evaluation ---")
    print(f" Instrument: {args.instrument} ({fyers_symbol})")
    print(f" Timeframe:  {args.timeframe} min")
    print(f" Fetch Days: {args.days}")
    print(f" Model:      {args.model}")
    print(f" Episodes:   {args.episodes}")
    print(f" Deterministic: {args.deterministic}")
    print("---------------------------")

    # --- 1. Authentication ---
    fyers = None
    try:
        print("Attempting Fyers authentication...")
        access_token = get_fyers_access_token()
        if not access_token:
            print("Authentication failed. Exiting.")
            exit()
        fyers = fyersModel.FyersModel(client_id=APP_ID, token=access_token)
        profile = fyers.get_profile() # Verify connection
        print(f"Authentication successful. FY_ID: {profile.get('data', {}).get('fy_id')}")
    except Exception as e:
        print(f"Error during authentication: {e}")
        traceback.print_exc()
        exit()

    # --- 2. Fetch Data ---
    raw_df = fetch_candle_data(
        fyers_instance=fyers,
        symbol=fyers_symbol,
        resolution=str(args.timeframe), # Ensure string
        days_to_fetch=args.days
    )

    if raw_df is None or raw_df.empty:
        print("Failed to fetch data or no data returned. Exiting.")
        exit()

    # --- 3. Process Data ---
    print("Processing fetched data...")
    try:
        pipeline = FullFeaturePipeline(raw_df)
        pipeline.run_pipeline()
        processed_df = pipeline.get_processed_df()
    except Exception as e:
        print(f"Error processing fetched data: {e}")
        traceback.print_exc()
        exit()

    if processed_df is None or processed_df.empty:
        print("Processing resulted in empty DataFrame. Exiting.")
        exit()

    # --- 4. Calculate Normalization Stats ---
    # Use MARKET_FEATURES defined in rl_environment
    try:
        # Calculate stats for the features
        norm_stats_features = calculate_norm_stats(processed_df, MARKET_FEATURES)

        # Get lot size for the instrument
        lot_size = INDEX_MAPPING[args.instrument].get("quantity", 1)
        print(f"Using Lot Size: {lot_size} for {args.instrument}")

        # Combine feature stats with metadata (lot size) for the environment override
        norm_stats_override = {
             **norm_stats_features,
             "_metadata": {"lot_size": lot_size}
        }

        # Basic check if stats were generated for features
        if not norm_stats_features or not all(f in norm_stats_features for f in MARKET_FEATURES if f in processed_df.columns):
             # Check if base features exist for derived ones if direct ones are missing
             missing_direct = [f for f in MARKET_FEATURES if f in processed_df.columns and f not in norm_stats_features]
             missing_derived_base = []
             if 'close_ema50_diff' in MARKET_FEATURES and 'close_ema50_diff' not in norm_stats_features and 'close' not in norm_stats_features:
                 missing_derived_base.append('close_ema50_diff (base: close)')
             if 'close_ema200_diff' in MARKET_FEATURES and 'close_ema200_diff' not in norm_stats_features and 'close' not in norm_stats_features:
                 missing_derived_base.append('close_ema200_diff (base: close)')

             if missing_direct or missing_derived_base:
                 print(f"Error: Failed to calculate normalization stats for all required features.")
                 if missing_direct: print(f"  Missing direct stats for: {missing_direct}")
                 if missing_derived_base: print(f"  Missing base stats for derived: {missing_derived_base}")
                 exit()
             else:
                 print("Note: Some features might be missing from data or using base stats.")


    except Exception as e:
        print(f"Error calculating normalization stats: {e}")
        traceback.print_exc()
        exit()

    # --- 5. Create Environment ---
    print("Creating evaluation environment with fetched data...")
    try:
        # Pass the processed data and the combined stats override (features + metadata)
        eval_env = TradingEnv(
            render_mode='none',
            data_df=processed_df,
            norm_stats_override=norm_stats_override
        )
        print("Evaluation environment created.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        traceback.print_exc()
        exit()

    # --- 6. Load Model ---
    print(f"Loading model from {model_path}...")
    try:
        # Pass the custom environment instance to load
        model = PPO.load(model_path, env=eval_env)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        exit()

    # --- 7. Evaluation Loop ---
    print(f"Running evaluation for {args.episodes} episodes...")
    all_episode_infos = []
    # Custom Evaluation Loop
    try:
        for episode in range(args.episodes):
            obs, info = eval_env.reset() # Reset uses the data_df provided in __init__
            done = False
            episode_steps = 0
            while not done:
                action, _states = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_steps += 1
                # Optional: Add rendering or progress indicator here if needed

            # Append the final info dictionary containing episode stats
            all_episode_infos.append(info)
            print(f"  Episode {episode + 1}/{args.episodes} finished. Steps: {episode_steps}, Final Capital: {info.get('final_capital', 'N/A'):.2f}, Win Rate: {info.get('win_rate', 'N/A'):.2%}, Sharpe: {info.get('episode_sharpe', 'N/A'):.4f}")

        # --- 8. Aggregate and Print Results ---
        print("\n--- Aggregated Evaluation Results ---")

        if not all_episode_infos:
             print("No episode data collected.")
             exit()

        # Extract metrics from infos, handling potential missing keys gracefully
        final_capitals = [info.get('final_capital', np.nan) for info in all_episode_infos if info]
        win_rates = [info.get('win_rate', np.nan) for info in all_episode_infos if info]
        total_trades_list = [info.get('total_trades', np.nan) for info in all_episode_infos if info]
        total_costs = [info.get('total_cost', np.nan) for info in all_episode_infos if info]
        sharpes = [info.get('episode_sharpe', np.nan) for info in all_episode_infos if info]
        winning_trades_list = [info.get('winning_trades', np.nan) for info in all_episode_infos if info]
        losing_trades_list = [info.get('losing_trades', np.nan) for info in all_episode_infos if info]

        # Calculate mean and std, handling potential NaNs and empty lists
        mean_final_capital = np.nanmean(final_capitals) if final_capitals else np.nan
        std_final_capital = np.nanstd(final_capitals) if final_capitals else np.nan
        mean_win_rate = np.nanmean(win_rates) if win_rates else np.nan
        std_win_rate = np.nanstd(win_rates) if win_rates else np.nan
        mean_total_trades = np.nanmean(total_trades_list) if total_trades_list else np.nan
        std_total_trades = np.nanstd(total_trades_list) if total_trades_list else np.nan
        mean_total_cost = np.nanmean(total_costs) if total_costs else np.nan
        std_total_cost = np.nanstd(total_costs) if total_costs else np.nan
        mean_sharpe = np.nanmean(sharpes) if sharpes else np.nan
        std_sharpe = np.nanstd(sharpes) if sharpes else np.nan
        mean_winning_trades = np.nanmean(winning_trades_list) if winning_trades_list else np.nan
        mean_losing_trades = np.nanmean(losing_trades_list) if losing_trades_list else np.nan

        # Get initial balance from the environment instance (assuming it's accessible)
        initial_balance = eval_env.initial_balance if hasattr(eval_env, 'initial_balance') else 'N/A'

        print(f"Evaluation over {len(all_episode_infos)} episodes on fetched data:")
        print(f"  Initial Capital:    {initial_balance:.2f}" if isinstance(initial_balance, (int, float)) else f"  Initial Capital:    {initial_balance}")
        print(f"  Mean Final Capital: {mean_final_capital:.2f} +/- {std_final_capital:.2f}")
        print(f"  Mean Win Rate:      {mean_win_rate:.2%} +/- {std_win_rate:.2%}")
        print(f"  Mean Total Trades:  {mean_total_trades:.2f} +/- {std_total_trades:.2f}")
        print(f"  Mean Winning Trades:{mean_winning_trades:.2f}")
        print(f"  Mean Losing Trades: {mean_losing_trades:.2f}")
        print(f"  Mean Total Cost:    {mean_total_cost:.2f} +/- {std_total_cost:.2f}")
        print(f"  Mean Sharpe Ratio:  {mean_sharpe:.4f} +/- {std_sharpe:.4f}")
        # Note: Sharpe here is before cost penalty, as calculated in env

    except Exception as e:
        print(f"\nAn error occurred during evaluation loop: {e}")
        traceback.print_exc()
    finally:
        if 'eval_env' in locals() and eval_env is not None:
            eval_env.close()

    print("\nEvaluation finished.")
