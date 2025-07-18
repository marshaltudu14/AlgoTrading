#!/usr/bin/env python3
"""
Live Data Processor
"""

import pandas as pd
from src.data_processing.feature_generator import DynamicFileProcessor
from src.reasoning_system.core.enhanced_orchestrator import EnhancedReasoningOrchestrator

class LiveDataProcessor:
    def __init__(self, config):
        self.feature_generator = DynamicFileProcessor()
        self.reasoning_orchestrator = EnhancedReasoningOrchestrator(config)

    def process_data(self, df):
        # 1. Generate features and combine with original data
        df_with_features = self.feature_generator.process_dataframe(df)

        # 2. Generate reasoning
        df_with_reasoning = self.reasoning_orchestrator.process_dataframe(df_with_features)

        return df_with_reasoning
