#!/usr/bin/env python3
"""
Configuration Loader for Backtesting
Loads and manages backtesting configuration from YAML files
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class BacktestingConfigLoader:
    """
    Loads and manages backtesting configuration.
    """
    
    def __init__(self, config_path: str = "config/backtesting_config.yaml"):
        """
        Initialize the configuration loader.
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = None
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            dict: Loaded configuration
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                self.config = self._get_default_config()
                return self.config
            
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}")
            self.config = self._get_default_config()
            return self.config
    
    def get_data_source_config(self) -> Dict[str, Any]:
        """Get data source configuration."""
        return self.config.get('data_source', {})
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config.get('environment', {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration."""
        return self.config.get('trading', {})
    
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for a specific symbol.
        
        Args:
            symbol (str): Symbol name
            
        Returns:
            dict: Symbol-specific configuration
        """
        symbols_config = self.config.get('symbols', {})
        symbol_lower = symbol.lower()
        
        if symbol_lower in symbols_config:
            return symbols_config[symbol_lower]
        
        # Return default configuration if symbol not found
        return {
            'fyers_symbol': symbol,
            'lot_size': 1,
            'tick_size': 0.05,
            'timeframe': '5',
            'days': 30
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_testing_config(self) -> Dict[str, Any]:
        """Get testing configuration."""
        return self.config.get('testing', {})
    
    def is_testing_mode(self) -> bool:
        """Check if testing mode is enabled."""
        testing_config = self.get_testing_config()
        return testing_config.get('enabled', False)
    
    def get_backtesting_params(self, symbol: str = None, testing_mode: bool = False) -> Dict[str, Any]:
        """
        Get complete backtesting parameters for a symbol.
        
        Args:
            symbol (str): Symbol name (uses default if None)
            testing_mode (bool): Whether to use testing parameters
            
        Returns:
            dict: Complete backtesting parameters
        """
        data_source = self.get_data_source_config()
        environment = self.get_environment_config()
        trading = self.get_trading_config()
        testing = self.get_testing_config()
        
        # Use provided symbol or default
        symbol = symbol or data_source.get('default_symbol', 'banknifty')
        symbol_config = self.get_symbol_config(symbol)
        
        # Base parameters
        params = {
            # Data parameters
            'symbol': symbol,
            'fyers_symbol': symbol_config.get('fyers_symbol', symbol),
            'timeframe': symbol_config.get('timeframe', data_source.get('default_timeframe', '5')),
            'days': symbol_config.get('days', data_source.get('default_days', 30)),
            
            # Environment parameters
            'initial_capital': environment.get('initial_capital', 100000),
            'episode_length': environment.get('episode_length', 1000),
            'lookback_window': environment.get('lookback_window', 20),
            'use_streaming': environment.get('use_streaming', False),
            'detailed_logging': environment.get('detailed_logging', False),
            
            # Trading parameters
            'lot_size': symbol_config.get('lot_size', 1),
            'tick_size': symbol_config.get('tick_size', 0.05),
            
            # Risk management
            'stop_loss_percentage': trading.get('risk_management', {}).get('stop_loss_percentage', 0.02),
            'target_profit_percentage': trading.get('risk_management', {}).get('target_profit_percentage', 0.04),
            'trailing_stop_percentage': trading.get('risk_management', {}).get('trailing_stop_percentage', 0.015),
            
            # Model parameters
            'model_path': self.get_model_config().get('model_path', 'models/universal_final_model.pth'),
            'model_type': self.get_model_config().get('model_type', 'ppo'),
        }
        
        # Override with testing parameters if in testing mode
        if testing_mode or self.is_testing_mode():
            params.update({
                'days': testing.get('test_days', 7),
                'episode_length': testing.get('test_episode_length', 100),
                'detailed_logging': True
            })
            logger.info("🧪 Using testing mode parameters")
        
        return params
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration if file loading fails.
        
        Returns:
            dict: Default configuration
        """
        return {
            'data_source': {
                'provider': 'fyers',
                'default_symbol': 'Nifty',
                'default_timeframe': '5',
                'default_days': 30,
                'min_data_points': 100
            },
            'environment': {
                'initial_capital': 100000,
                'episode_length': 1000,
                'lookback_window': 20,
                'use_streaming': False,
                'detailed_logging': False
            },
            'trading': {
                'risk_management': {
                    'stop_loss_percentage': 0.02,
                    'target_profit_percentage': 0.04,
                    'trailing_stop_percentage': 0.015
                }
            },
            'symbols': {
                'Nifty': {
                    'fyers_symbol': 'NSE:NIFTY50-INDEX',
                    'lot_size': 75,
                    'tick_size': 0.05,
                    'timeframe': '5',
                    'days': 30
                }
            },
            'model': {
                'model_path': 'models/universal_final_model.pth',
                'model_type': 'ppo'
            },
            'output': {
                'results_dir': 'results/backtesting',
                'save_trade_logs': True,
                'report_format': 'txt'
            },
            'testing': {
                'enabled': False,
                'test_days': 7,
                'test_episode_length': 100
            }
        }
    
    def save_config(self, config_path: str = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path (str): Path to save config (uses default if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            save_path = config_path or self.config_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config to {save_path}: {e}")
            return False
