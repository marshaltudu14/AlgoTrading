import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class TradeLogger:
    """Service for logging trades to a JSON file"""

    def __init__(self, log_file: str = "tradelog.json"):
        self.log_file = Path(log_file)
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the log file if it doesn't exist"""
        if not self.log_file.exists():
            with open(self.log_file, "w") as f:
                json.dump([], f)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade to the JSON file"""
        try:
            with open(self.log_file, "r+") as f:
                log_content = json.load(f)
                log_content.append(trade_data)
                f.seek(0)
                json.dump(log_content, f, indent=4)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to log trade: {e}")
