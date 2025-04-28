"""
Logging setup for the AlgoTrading system.
Centralizes logging configuration for consistent logging across the application.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    app_name="algotrading",
    console_output=True
):
    """
    Set up logging for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_dir: Directory to store log files (default: logs)
        app_name: Name of the application for log file naming
        console_output: Whether to output logs to console
        
    Returns:
        Logger instance
    """
    # Create log directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir_path / f"{app_name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    )
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name):
    """
    Get a logger with the given name.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create default logger
logger = setup_logging()


if __name__ == "__main__":
    # Test logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
