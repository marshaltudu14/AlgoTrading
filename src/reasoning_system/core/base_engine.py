from typing import Dict, Any

class BaseReasoningEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config