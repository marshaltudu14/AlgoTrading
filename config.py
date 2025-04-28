import os
import torch

# Data fetch/process config
HIST_DIR = os.getenv("HIST_DIR", "historical_data")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "processed_data")

# Instruments and timeframes
INSTRUMENTS = {
    'Bankex': 'BSE:BANKEX-INDEX',
    'Finnifty': 'NSE:FINNIFTY-INDEX',
    'Bank Nifty': 'NSE:NIFTYBANK-INDEX',
    'Nifty': 'NSE:NIFTY50-INDEX',
    'Sensex': 'BSE:SENSEX-INDEX'
}
TIMEFRAMES = [1,2,3,5,10,15,20,30,45,60,120,180,240]

# Instrument and timeframe IDs for embeddings
INSTRUMENT_IDS = {name: idx for idx, name in enumerate(INSTRUMENTS.keys())}
TIMEFRAME_IDS = {tf: idx for idx, tf in enumerate(TIMEFRAMES)}

# Lot quantities per instrument
QUANTITIES = {
    'Bankex': 15,
    'Finnifty': 40,
    'Bank Nifty': 30,
    'Nifty': 75,
    'Sensex': 20
}

# Historical fetch days
DAYS = 365

# Credentials
APP_ID = os.getenv("FY_APP_ID", "TS79V3NXK1-100")
SECRET_KEY = os.getenv("FY_SECRET_KEY", "KQCPB0FJ74")
REDIRECT_URI = os.getenv("FY_REDIRECT_URI", "https://google.com")
FYERS_USER = os.getenv("FYERS_USER", "XM22383")
FYERS_PIN = os.getenv("FYERS_PIN", "4628")
FYERS_TOTP = os.getenv("FYERS_TOTP", "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW")

# RL environment and trading config
INITIAL_CAPITAL = int(os.getenv("INITIAL_CAPITAL", 500000))  # 5L
BROKERAGE_ENTRY = float(os.getenv("BROKERAGE_ENTRY", 20.0))
BROKERAGE_EXIT = float(os.getenv("BROKERAGE_EXIT", 20.0))
# RLHF reward model weight
RLHF_WEIGHT = float(os.getenv("RLHF_WEIGHT", 1.0))  # weight for learned human preference reward

# Risk-reward ratio for labeling
RR_RATIO = 2

# Live trading settings
WINDOW_SIZE      = int(os.getenv("WINDOW_SIZE", 50))
LIVE_FETCH_DAYS  = int(os.getenv("LIVE_FETCH_DAYS", 15))
MODEL_PATH       = os.getenv("MODEL_PATH", "models/trading_transformer.pt")
FEATURES_DIR     = os.getenv("FEATURES_DIR", PROCESSED_DIR)
STRIKE_STEP      = int(os.getenv("STRIKE_STEP", 50))
SL_ATR_MULT      = float(os.getenv("SL_ATR_MULT", 1.0))
TP_ATR_MULT      = float(os.getenv("TP_ATR_MULT", RR_RATIO))
OPTION_MASTER_FO = os.getenv("OPTION_MASTER_FO", "https://public.fyers.in/sym_details/NSE_FO_sym_master.json")

# Advanced model configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer model config
TRANSFORMER_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "dropout": 0.1,
    "max_seq_len": WINDOW_SIZE,
    "num_regimes": 3,  # Number of market regimes to detect
}

# Mixture of Experts model config
MOE_CONFIG = {
    "hidden_dim": 128,
    "num_layers": 4,
    "num_heads": 4,
    "dropout": 0.1,
    "max_seq_len": WINDOW_SIZE,
    "num_regimes": 3,
    "num_experts": 4,        # Number of expert networks
    "expert_hidden_dim": 128,
    "k": 2,                  # Number of experts to use for each input
    "noisy_gating": True,    # Add noise to gating for exploration
}

# Training config
TRAINING_CONFIG = {
    "batch_size": 64,
    "bc_epochs": 50,         # Behavioral cloning epochs
    "rm_epochs": 30,         # Reward modeling epochs
    "rl_epochs": 100,        # RL fine-tuning epochs
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "kl_weight": 0.1,        # KL divergence weight for RL fine-tuning
    "exploration_weight": 0.05,  # Exploration bonus weight
    "sl_penalty_weight": 2.0,    # Extra penalty for stop-loss hits
    "validation_split": 0.2,
    "early_stopping_patience": 10,
    "use_moe": True,         # Whether to use MoE model
    "balance_instruments": True,  # Balance data across instruments
    "balance_signals": True,      # Balance data across signal types
    "curriculum_learning": True,  # Use curriculum learning
    "adversarial_training": True, # Use adversarial training
}

# RLHF config
RLHF_CONFIG = {
    "reward_lr": 5e-5,
    "reward_hidden_dim": 64,
    "reward_layers": 2,
    "preference_noise": 0.1,  # Noise added to preferences for robustness
    "asymmetric_loss": True,  # Use asymmetric loss for SL vs target hits
    "target_weight": 1.0,     # Weight for target hits
    "sl_weight": 2.0,         # Weight for SL hits (higher penalty)
}

# Anti-overfitting config
OVERFITTING_CONFIG = {
    "dropout": 0.2,
    "weight_decay": 1e-5,
    "early_stopping": True,
    "data_augmentation": True,
    "ensemble_size": 3,      # Number of models in ensemble
    "cross_validation_folds": 5,
}
