"""
Automated Trading Reasoning Generation System
============================================

A modular system for generating human-like trading reasoning that simulates
professional trader thinking patterns using historical context and technical analysis.

Modules:
- core: Core reasoning engine and orchestration
- context: Historical context management and analysis
- engines: Specialized reasoning engines (pattern, context, psychology, etc.)
- generators: Text generation and language processing
- utils: Utility functions and helpers
- config: Configuration management

Author: AlgoTrading System
Version: 1.0
"""

from .core.enhanced_orchestrator import EnhancedReasoningOrchestrator
from .core.base_engine import BaseReasoningEngine
from .context.historical_context_manager import HistoricalContextManager
from .engines.historical_pattern_engine import HistoricalPatternEngine
from .engines.context_analysis_engine import ContextAnalysisEngine
from .engines.psychology_assessment_engine import PsychologyAssessmentEngine
from .engines.execution_decision_engine import ExecutionDecisionEngine # Assuming this exists
from .engines.risk_assessment_engine import RiskAssessmentEngine # Assuming this exists

# Removed TextGenerator and QualityValidator as they are not directly used here

__version__ = "1.0.0"
__author__ = "AlgoTrading System"

__all__ = [
    'EnhancedReasoningOrchestrator',
    'BaseReasoningEngine',
    'HistoricalContextManager',
    'HistoricalPatternEngine',
    'ContextAnalysisEngine',
    'PsychologyAssessmentEngine',
    'ExecutionDecisionEngine',
    'RiskAssessmentEngine'
]