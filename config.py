import os

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
MODEL_PATH       = os.getenv("MODEL_PATH", "models/ppo_multitask.zip")
FEATURES_DIR     = os.getenv("FEATURES_DIR", PROCESSED_DIR)
STRIKE_STEP      = int(os.getenv("STRIKE_STEP", 50))
SL_ATR_MULT      = float(os.getenv("SL_ATR_MULT", 1.0))
TP_ATR_MULT      = float(os.getenv("TP_ATR_MULT", RR_RATIO))
OPTION_MASTER_FO = os.getenv("OPTION_MASTER_FO", "https://public.fyers.in/sym_details/NSE_FO_sym_master.json")
