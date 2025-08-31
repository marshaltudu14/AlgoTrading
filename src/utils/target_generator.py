"""
Target Generator for HRM Trading Model
Generates expert targets from technical indicators for supervised learning
"""

import torch
import numpy as np
from typing import Dict, Any
import pandas as pd
from src.config.settings import get_settings


class TargetGenerator:
    """Generate expert targets for HRM training"""
    
    def __init__(self):
        self.settings = get_settings()
        self.action_types = self.settings['actions']['action_types']
        self.risk_config = self.settings.get('risk_management', {})
        
    def generate_market_understanding_guidance(self, row: pd.Series) -> Dict[str, float]:
        """
        Instead of hardcoded actions, provide market understanding guidance
        Let HRM discover action patterns from historical sequences
        """
        
        guidance = {}
        
        # Market momentum indicators (normalized 0-1, let model learn patterns)
        trend_strength = row.get('trend_strength', 0.5) if pd.notna(row.get('trend_strength', np.nan)) else 0.5
        trend_direction = row.get('trend_direction', 0.5) if pd.notna(row.get('trend_direction', np.nan)) else 0.5
        
        guidance['momentum_signal'] = trend_strength * (trend_direction - 0.5) * 2  # -1 to 1
        
        # Volatility context (let model learn when to be cautious vs aggressive)
        volatility = row.get('volatility_20', 1.0) if pd.notna(row.get('volatility_20', np.nan)) else 1.0
        guidance['volatility_context'] = min(2.0, volatility)  # Cap at 2x normal
        
        # Price momentum indicators (let model discover patterns)
        rsi_14 = row.get('rsi_14', 50) if pd.notna(row.get('rsi_14', np.nan)) else 50
        guidance['momentum_oscillator'] = (rsi_14 - 50) / 50  # -1 to 1
        
        # MACD momentum (trend following context)
        macd_hist = row.get('macd_histogram', 0) if pd.notna(row.get('macd_histogram', np.nan)) else 0
        macd_signal = row.get('macd_signal', 0) if pd.notna(row.get('macd_signal', np.nan)) else 0
        guidance['trend_momentum'] = np.tanh(macd_hist / max(abs(macd_signal), 1))  # Normalized
        
        # Support/resistance context
        bb_position = row.get('bb_position', 0.5) if pd.notna(row.get('bb_position', np.nan)) else 0.5
        guidance['price_position'] = (bb_position - 0.5) * 2  # -1 to 1
        
        return guidance
    
    
    def generate_all_targets(self, row: pd.Series, account_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate ONLY market understanding guidance
        Remove regime and risk targets - let HRM learn from environment rewards
        """
        # Use config values instead of hardcoding
        initial_capital = self.settings.get('environment', {}).get('initial_capital', 100000.0)
        
        if account_state is None:
            account_state = {'capital': initial_capital, 'current_position_quantity': 0.0}
            
        targets = {
            'market_understanding': self.generate_market_understanding_guidance(row)
            # REMOVED: market_regime - let H-module learn from 100 candles  
            # REMOVED: risk_context - let model learn from environment rewards
        }
        
        return targets


def create_training_targets(info: Dict, model_outputs: Dict, data_row: pd.Series = None) -> Dict[str, torch.Tensor]:
    """
    Create ONLY market understanding guidance for HRM learning
    Let HRM learn regimes and risk from environment rewards and 100-candle sequences
    """
    
    if data_row is None:
        # Use model's own predictions for consistency learning
        return create_self_supervised_targets(info, model_outputs)
    
    # Generate market understanding guidance only
    target_generator = TargetGenerator()
    
    # Use config values instead of hardcoding
    from src.config.settings import get_settings
    settings = get_settings()
    initial_capital = settings.get('environment', {}).get('initial_capital', 100000.0)
    account_state = info.get('account_state', {'capital': initial_capital, 'current_position_quantity': 0.0})
    
    guidance = target_generator.generate_all_targets(data_row, account_state)
    
    targets = {}
    
    # ONLY market understanding guidance (help model interpret market context)
    market_guidance = guidance['market_understanding']
    guidance_tensor = torch.tensor([
        market_guidance['momentum_signal'],
        market_guidance['volatility_context'], 
        market_guidance['momentum_oscillator'],
        market_guidance['trend_momentum'],
        market_guidance['price_position']
    ], dtype=torch.float)
    targets['market_understanding'] = guidance_tensor.unsqueeze(0)  # Add batch dim
    
    # REMOVED: regime targets - let H-module learn from 100 historical candles
    # REMOVED: risk targets - let model learn from environment reward feedback
    
    return targets


def create_self_supervised_targets(info: Dict, model_outputs: Dict) -> Dict[str, torch.Tensor]:
    """
    Self-supervised learning when market data not available
    Use model's own market understanding predictions for consistency training
    """
    targets = {}
    
    # ONLY self-supervised market understanding (model learns from its own context interpretation)
    if 'market_understanding_outputs' in model_outputs:
        understanding_tensor = model_outputs['market_understanding_outputs'].detach()
        if understanding_tensor.dim() == 1:
            understanding_tensor = understanding_tensor.unsqueeze(0)
        targets['market_understanding'] = understanding_tensor
    
    # REMOVED: regime self-supervision - let H-module discover regimes naturally
    # REMOVED: risk context self-supervision - let model learn from environment rewards
    
    return targets