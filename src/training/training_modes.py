"""
Training Mode Management for Debug vs Final Training
"""
import logging
from typing import Dict, List
from enum import Enum

logger = logging.getLogger(__name__)


class TrainingMode(Enum):
    DEBUG = "debug"
    FINAL = "final"


class TrainingModeManager:
    """Manages training mode configuration and behavior"""
    
    def __init__(self, training_mode_config: Dict, gpu_config: Dict):
        self.training_mode_config = training_mode_config
        self.gpu_config = gpu_config
        
        # Determine current mode
        current_mode = training_mode_config.get('mode', 'debug')
        self.mode = TrainingMode.DEBUG if current_mode == 'debug' else TrainingMode.FINAL
        
        # Get mode-specific config
        self.mode_config = training_mode_config.get(current_mode, {})
        
    def is_debug_mode(self) -> bool:
        """Check if in debug mode"""
        return self.mode == TrainingMode.DEBUG
    
    def is_final_mode(self) -> bool:
        """Check if in final mode"""
        return self.mode == TrainingMode.FINAL
    
    def should_show_detailed_logging(self) -> bool:
        """Whether to show detailed step-by-step logging"""
        return self.mode_config.get('detailed_logging', False)
    
    def should_show_step_by_step_logging(self) -> bool:
        """Whether to show step-by-step logging"""
        return self.mode_config.get('step_by_step_logging', False)
    
    def should_show_reward_calculations(self) -> bool:
        """Whether to show reward calculations"""
        return self.mode_config.get('show_reward_calculations', False)
    
    def should_show_position_changes(self) -> bool:
        """Whether to show position changes"""
        return self.mode_config.get('show_position_changes', False)
    
    def get_max_instruments_parallel(self) -> int:
        """Get maximum number of instruments to process in parallel"""
        return self.mode_config.get('max_instruments_parallel', 1 if self.is_debug_mode() else 16)
    
    def get_log_frequency(self) -> int:
        """Get logging frequency"""
        return self.mode_config.get('log_frequency', 1 if self.is_debug_mode() else 25)
    
    def should_use_single_instrument_mode(self) -> bool:
        """Whether to process one instrument at a time for debugging"""
        return self.mode_config.get('single_instrument_mode', False)
    
    def should_batch_all_instruments(self) -> bool:
        """Whether to batch all instruments for maximum throughput"""
        return self.mode_config.get('batch_all_instruments', False)
    
    def should_use_offline_rl_preprocessing(self) -> bool:
        """Whether to use offline RL preprocessing"""
        return self.mode_config.get('offline_rl_preprocessing', False)
    
    def should_use_gpu_optimization(self) -> bool:
        """Whether to use GPU optimization"""
        return self.mode_config.get('gpu_optimization', False)
    
    def get_training_info(self) -> Dict:
        """Get training mode information for logging"""
        return {
            'mode': self.mode.value,
            'detailed_logging': self.should_show_detailed_logging(),
            'step_by_step_logging': self.should_show_step_by_step_logging(),
            'max_instruments_parallel': self.get_max_instruments_parallel(),
            'log_frequency': self.get_log_frequency(),
            'single_instrument_mode': self.should_use_single_instrument_mode(),
            'batch_all_instruments': self.should_batch_all_instruments(),
            'offline_rl_preprocessing': self.should_use_offline_rl_preprocessing(),
            'gpu_optimization': self.should_use_gpu_optimization()
        }
    
    def log_training_mode_info(self):
        """Log current training mode configuration"""
        info = self.get_training_info()
        logger.info(f"Training Mode: {info['mode'].upper()}")
        
        if self.is_debug_mode():
            logger.info("DEBUG MODE SETTINGS:")
            logger.info(f"  - Single instrument mode: {info['single_instrument_mode']}")
            logger.info(f"  - Detailed logging: {info['detailed_logging']}")
            logger.info(f"  - Step-by-step logging: {info['step_by_step_logging']}")
            logger.info(f"  - Max instruments parallel: {info['max_instruments_parallel']}")
            logger.info(f"  - Log frequency: {info['log_frequency']}")
        else:
            logger.info("FINAL MODE SETTINGS:")
            logger.info(f"  - Batch all instruments: {info['batch_all_instruments']}")
            logger.info(f"  - Offline RL preprocessing: {info['offline_rl_preprocessing']}")
            logger.info(f"  - GPU optimization: {info['gpu_optimization']}")
            logger.info(f"  - Max instruments parallel: {info['max_instruments_parallel']}")
            logger.info(f"  - Log frequency: {info['log_frequency']}")


def get_instruments_for_mode(available_instruments: List[str], 
                           mode_manager: TrainingModeManager) -> List[str]:
    """Get the list of instruments to process based on training mode"""
    
    if mode_manager.should_use_single_instrument_mode():
        # Debug mode: process only first instrument
        return available_instruments[:1]
    else:
        # Final mode: process all instruments
        return available_instruments