import os
import pandas as pd
from src.training.trainer import Trainer
from src.backtesting.environment import TradingEnv
from src.agents.ppo_agent import PPOAgent
from src.utils.data_loader import DataLoader

# Configuration
DATA_FINAL_DIR = "data/final"
INITIAL_CAPITAL = 1000000.0
NUM_EPISODES = 100
LOG_INTERVAL = 10

# Initialize DataLoader
data_loader = DataLoader(final_data_dir=DATA_FINAL_DIR)

# Get available symbols from data/final
available_files = os.listdir(DATA_FINAL_DIR)
symbols_to_train = []
for filename in available_files:
    if filename.startswith("features_") and filename.endswith(".csv"):
        # Extract symbol from filename (e.g., "features_Bank_Nifty_5.csv" -> "Bank_Nifty_5")
        symbol = filename[len("features_"):-len(".csv")]
        symbols_to_train.append(symbol)

if not symbols_to_train:
    print(f"No feature files found in {DATA_FINAL_DIR}. Please ensure data is processed.")
else:
    print(f"Found {len(symbols_to_train)} symbols to train: {symbols_to_train}")

    # Train for each symbol
    for symbol in symbols_to_train:
        print(f"\n--- Starting training for symbol: {symbol} ---")
        # Initialize agent (using PPOAgent as an example)
        # These dimensions should ideally come from a config or be inferred from data
        observation_dim = 10 # Placeholder, adjust based on actual feature count
        action_dim = 5 # BUY_LONG, SELL_SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        hidden_dim = 64
        lr_actor = 0.001
        lr_critic = 0.001
        gamma = 0.99
        epsilon_clip = 0.2
        k_epochs = 3

        agent = PPOAgent(observation_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, epsilon_clip, k_epochs)

        # Initialize Trainer
        trainer = Trainer(agent, NUM_EPISODES, LOG_INTERVAL)

        # Run training
        trainer.train(data_loader, symbol, INITIAL_CAPITAL)

        print(f"--- Training for symbol {symbol} completed ---")

    print("\nAll training sessions completed.")
