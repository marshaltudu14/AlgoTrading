import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Service for calculating and storing trading metrics"""

    def __init__(self, metrics_file: str = "metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics = self._load_metrics()

    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics from the JSON file"""
        if not self.metrics_file.exists():
            return {
                "totalTrades": 0,
                "winRate": 0,
                "totalPnl": 0,
                "averagePnlPerTrade": 0,
                "todayPnL": 0,
                "lastTradeTime": None
            }
        
        try:
            with open(self.metrics_file, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load metrics: {e}")
            return {}

    def update_metrics(self, trade_data: Dict[str, Any]):
        """Update metrics based on a new trade"""
        self.metrics["totalTrades"] += 1
        self.metrics["totalPnl"] += trade_data["pnl"]
        self.metrics["averagePnlPerTrade"] = self.metrics["totalPnl"] / self.metrics["totalTrades"]
        self.metrics["todayPnL"] += trade_data["pnl"]
        self.metrics["lastTradeTime"] = trade_data["exit_time"]

        if trade_data["pnl"] > 0:
            # Recalculate win rate
            wins = self.metrics.get("wins", 0) + 1
            self.metrics["wins"] = wins
            self.metrics["winRate"] = (wins / self.metrics["totalTrades"]) * 100

        self._save_metrics()

    def _save_metrics(self):
        """Save metrics to the JSON file"""
        try:
            with open(self.metrics_file, "w") as f:
                json.dump(self.metrics, f, indent=4)
        except IOError as e:
            logger.error(f"Failed to save metrics: {e}")
