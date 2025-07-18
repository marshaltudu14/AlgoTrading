#!/usr/bin/env python3
"""
Configuration file for the Feature Generator and Signal Generation System
========================================================================

This file contains all configurable parameters for technical analysis
feature generation and signal calculation.
"""

# === FEATURE GENERATION CONFIGURATION ===
FEATURE_CONFIG = {
    # Moving Average Periods
    'sma_periods': [5, 10, 20, 50, 100, 200],
    'ema_periods': [5, 10, 20, 50, 100, 200],
    
    # RSI Periods
    'rsi_periods': [14, 21],
    
    # MACD Parameters
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    
    # Bollinger Bands
    'bb_period': 20,
    'bb_std_dev': 2.0,
    
    # ATR Period
    'atr_period': 14,
    
    # Stochastic Parameters
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    
    # Williams %R Period
    'williams_r_period': 14,
    
    # CCI Period
    'cci_period': 20,
    
    # ADX Period
    'adx_period': 14,
    
    # Momentum and ROC Periods
    'momentum_period': 10,
    'roc_period': 10,
    
    # TRIX Period
    'trix_period': 14,
    
    # Volatility Calculation Periods
    'volatility_periods': [10, 20],
}

# === DATA PROCESSING CONFIGURATION ===
DATA_CONFIG = {
    # File Processing
    'input_folder': 'data/raw',
    'output_folder': 'data/final',
    'file_pattern': '*.csv',
    
    # Data Cleaning
    'remove_volume_column': True,      # Remove volume column (not always reliable)
    'timezone': 'Asia/Kolkata',        # Convert timestamps to this timezone
    'round_decimals': 2,               # Round numeric features to this many decimal places
    
    # Data Validation
    'required_columns': ['datetime', 'open', 'high', 'low', 'close'],
    'validate_ohlc': True,             # Validate OHLC relationships (high >= low, etc.)
    'remove_duplicates': True,         # Remove duplicate timestamps
    'drop_nan_rows': True,             # Remove rows with NaN values after processing
}

# === LOGGING CONFIGURATION ===
LOGGING_CONFIG = {
    'level': 'INFO',                   # DEBUG, INFO, WARNING, ERROR
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'log_to_file': False,              # Set to True to log to file
    'log_file': 'feature_generator.log',
}

# === PERFORMANCE CONFIGURATION ===
PERFORMANCE_CONFIG = {
    'batch_size': 10000,               # Process data in batches for large files
    'parallel_processing': False,      # Enable parallel processing (experimental)
    'memory_optimization': True,       # Optimize memory usage for large datasets
}

# === VALIDATION CONFIGURATION ===
VALIDATION_CONFIG = {
    'feature_correlation_check': False, # Check for highly correlated features
    'max_correlation_threshold': 0.95, # Maximum allowed correlation between features
}



def get_config():
    """
    Get the complete configuration dictionary.
    
    Returns:
        dict: Complete configuration with all sections
    """
    return {
        'features': FEATURE_CONFIG,
        'data': DATA_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'validation': VALIDATION_CONFIG,
    }

def print_config():
    """Print the current configuration in a readable format."""
    config = get_config()
    
    print("=" * 80)
    print("FEATURE GENERATOR CONFIGURATION")
    print("=" * 80)
    
    for section_name, section_config in config.items():
        print(f"\n[{section_name.upper()}]")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_config()
