import pandas as pd
from typing import Dict, Any

class HistoricalContextManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context_window_size = config.get('context_window_size', 100) # Default to 100 if not in config

    def get_historical_context(self, df: pd.DataFrame, current_index: int) -> pd.DataFrame:
        """
        Retrieves a window of historical data preceding the current_index.
        """
        start_index = max(0, current_index - self.context_window_size)
        historical_df = df.iloc[start_index:current_index]
        return historical_df
