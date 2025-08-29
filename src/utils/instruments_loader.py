"""
Utility to load and manage instrument and timeframe configurations
"""
import yaml
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class InstrumentsLoader:
    """Loads instrument and timeframe configurations with dynamic embedding support"""
    
    def __init__(self, instruments_path: str = "config/instruments.yaml"):
        self.instruments_path = instruments_path
        self._config = None
        self._instrument_id_map = {}
        self._timeframe_id_map = {}
        self._load_config()
    
    def _load_config(self):
        """Load instruments configuration"""
        try:
            with open(self.instruments_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Build ID mappings
            for instrument in self._config['instruments']:
                self._instrument_id_map[instrument['symbol']] = instrument['id']
            
            for timeframe in self._config['timeframes']:
                self._timeframe_id_map[timeframe['name']] = timeframe['id']
                
        except Exception as e:
            raise ValueError(f"Failed to load instruments config from {self.instruments_path}: {e}")
    
    def get_instrument_id(self, symbol: str) -> Optional[int]:
        """Get instrument ID for a given symbol"""
        return self._instrument_id_map.get(symbol)
    
    def get_timeframe_id(self, timeframe: str) -> Optional[int]:
        """Get timeframe ID for a given timeframe"""
        return self._timeframe_id_map.get(timeframe)
    
    def get_num_instruments(self) -> int:
        """Get total number of instruments"""
        return len(self._config['instruments'])
    
    def get_num_timeframes(self) -> int:
        """Get total number of timeframes"""
        return len(self._config['timeframes'])
    
    def get_instrument_by_id(self, instrument_id: int) -> Optional[Dict]:
        """Get instrument config by ID"""
        for instrument in self._config['instruments']:
            if instrument['id'] == instrument_id:
                return instrument
        return None
    
    def get_timeframe_by_id(self, timeframe_id: int) -> Optional[Dict]:
        """Get timeframe config by ID"""
        for timeframe in self._config['timeframes']:
            if timeframe['id'] == timeframe_id:
                return timeframe
        return None
    
    def get_all_instruments(self) -> List[Dict]:
        """Get all instrument configurations"""
        return self._config['instruments']
    
    def get_all_timeframes(self) -> List[Dict]:
        """Get all timeframe configurations"""
        return self._config['timeframes']
    
    def get_embedding_dimensions(self) -> Tuple[int, int]:
        """Get embedding dimensions for instruments and timeframes"""
        return self.get_num_instruments(), self.get_num_timeframes()
    
    def validate_symbol_timeframe(self, symbol: str, timeframe: str) -> Tuple[bool, Optional[int], Optional[int]]:
        """Validate and get IDs for symbol and timeframe"""
        instrument_id = self.get_instrument_id(symbol)
        timeframe_id = self.get_timeframe_id(timeframe)
        
        valid = instrument_id is not None and timeframe_id is not None
        return valid, instrument_id, timeframe_id


# Global instance
_instruments_loader = None

def get_instruments_loader() -> InstrumentsLoader:
    """Get global instruments loader instance"""
    global _instruments_loader
    if _instruments_loader is None:
        _instruments_loader = InstrumentsLoader()
    return _instruments_loader