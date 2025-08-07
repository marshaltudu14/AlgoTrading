import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class UserActionLogger:
    """Service for logging user actions to a file"""

    def __init__(self, log_file: str = "logs/user_actions.log"):
        self.log_file = Path(log_file)
        self._initialize_log_file()

    def _initialize_log_file(self):
        """Initialize the log file and directory if they don't exist"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch()

    def log_action(self, user_id: str, action_type: str, details: Dict[str, Any]):
        """Log a user action to the file"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "action_type": action_type,
                "details": details
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            logger.error(f"Failed to log user action: {e}")
