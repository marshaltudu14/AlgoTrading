#!/usr/bin/env python3
"""
Reasoning Orchestrator
=====================

Main orchestrator that coordinates all reasoning engines to generate
comprehensive trading reasoning for each data point.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

from ..context.historical_context_manager import HistoricalContextManager
from ..engines.pattern_recognition_engine import PatternRecognitionEngine
from ..engines.context_analysis_engine import ContextAnalysisEngine
from ..engines.psychology_assessment_engine import PsychologyAssessmentEngine
from ..engines.execution_decision_engine import ExecutionDecisionEngine
from ..engines.risk_assessment_engine import RiskAssessmentEngine
from ..generators.text_generator import TextGenerator
from ..generators.quality_validator import QualityValidator

logger = logging.getLogger(__name__)


class ReasoningOrchestrator:
    """
    Main orchestrator for the reasoning generation system.
    
    Coordinates all reasoning engines and manages the complete pipeline
    from feature data to final reasoning columns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the reasoning orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.context_manager = HistoricalContextManager(
            window_size=self.config.get('context_window_size', 100)
        )
        
        # Initialize reasoning engines
        self.engines = {
            'pattern_recognition': PatternRecognitionEngine(config),
            'context_analysis': ContextAnalysisEngine(config),
            'psychology_assessment': PsychologyAssessmentEngine(config),
            'execution_decision': ExecutionDecisionEngine(config),
            'risk_assessment': RiskAssessmentEngine(config)
        }
        
        # Initialize text processing components
        self.text_generator = TextGenerator(config)
        self.quality_validator = QualityValidator(config)
        
        # Reasoning column names
        self.reasoning_columns = [
            'pattern_recognition_text',
            'context_analysis_text',
            'psychology_assessment_text',
            'execution_decision_text',
            'confidence_score',
            'risk_assessment_text',
            'alternative_scenarios_text'
        ]
        
        logger.info("ReasoningOrchestrator initialized with all engines")
    
    def process_file(self, input_file_path: str, output_file_path: str) -> Dict[str, Any]:
        """
        Process a single feature file and add reasoning columns.
        
        Args:
            input_file_path: Path to input CSV file with features
            output_file_path: Path to save output CSV with reasoning
            
        Returns:
            Processing summary dictionary
        """
        logger.info(f"Processing file: {input_file_path}")
        
        try:
            # Load feature data
            df = pd.read_csv(input_file_path)
            logger.info(f"Loaded {len(df)} rows from {input_file_path}")
            
            # Validate input data
            if not self._validate_input_data(df):
                raise ValueError("Input data validation failed")
            
            # Generate reasoning for all rows
            reasoning_df = self._generate_reasoning_for_dataframe(df)
            
            # Combine original data with reasoning
            result_df = pd.concat([df, reasoning_df], axis=1)
            
            # Validate output quality
            quality_report = self.quality_validator.validate_reasoning_dataframe(reasoning_df)
            
            # Save result
            result_df.to_csv(output_file_path, index=False)
            logger.info(f"Saved processed data to {output_file_path}")
            
            return {
                'status': 'success',
                'input_rows': len(df),
                'output_rows': len(result_df),
                'reasoning_columns_added': len(self.reasoning_columns),
                'quality_score': quality_report.get('overall_score', 0),
                'output_file': output_file_path
            }
            
        except Exception as e:
            logger.error(f"Error processing file {input_file_path}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'input_file': input_file_path
            }
    
    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Process all feature files in a directory.
        
        Args:
            input_dir: Directory containing feature CSV files
            output_dir: Directory to save reasoning-enhanced files
            
        Returns:
            Processing summary for all files
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all feature CSV files
        feature_files = list(input_path.glob("features_*.csv"))
        
        if not feature_files:
            logger.warning(f"No feature files found in {input_dir}")
            return {'status': 'no_files', 'processed_files': []}
        
        results = []
        
        for feature_file in feature_files:
            # Generate output filename
            output_filename = f"reasoning_{feature_file.name}"
            output_file_path = output_path / output_filename
            
            # Process file
            result = self.process_file(str(feature_file), str(output_file_path))
            result['input_file'] = feature_file.name
            result['output_file'] = output_filename
            results.append(result)
        
        # Generate summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'error']
        
        summary = {
            'total_files': len(feature_files),
            'successful': len(successful),
            'failed': len(failed),
            'results': results
        }
        
        logger.info(f"Processed {len(feature_files)} files: {len(successful)} successful, {len(failed)} failed")
        return summary
    
    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame has required columns and data quality.
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            True if validation passes
        """
        # Check for required base columns
        required_base_columns = ['datetime', 'open', 'high', 'low', 'close']
        missing_base = [col for col in required_base_columns if col not in df.columns]
        
        if missing_base:
            logger.error(f"Missing required base columns: {missing_base}")
            return False
        
        # Check for signal column
        if 'signal' not in df.columns:
            logger.error("Missing signal column")
            return False
        
        # Validate each engine's requirements
        for engine_name, engine in self.engines.items():
            required_cols = engine.get_required_columns()
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"{engine_name} missing columns: {missing_cols}")
                # Continue processing but log warnings
        
        logger.info("Input data validation passed")
        return True
    
    def _generate_reasoning_for_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate reasoning columns for entire DataFrame.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with reasoning columns
        """
        reasoning_data = []
        
        for idx in range(len(df)):
            if idx % 1000 == 0:
                logger.info(f"Processing row {idx}/{len(df)}")
            
            # Generate reasoning for current row
            reasoning_row = self._generate_reasoning_for_row(df, idx)
            reasoning_data.append(reasoning_row)
        
        # Create DataFrame with reasoning columns
        reasoning_df = pd.DataFrame(reasoning_data, columns=self.reasoning_columns)
        
        logger.info(f"Generated reasoning for {len(reasoning_df)} rows")
        return reasoning_df
    
    def _generate_reasoning_for_row(self, df: pd.DataFrame, row_idx: int) -> List[Any]:
        """
        Generate reasoning for a single row.
        
        Args:
            df: Full DataFrame
            row_idx: Index of current row
            
        Returns:
            List of reasoning values for this row
        """
        current_data = df.iloc[row_idx]
        
        # Get historical context
        context = self.context_manager.get_historical_context(df, row_idx)
        
        # Generate reasoning from each engine
        reasoning_texts = {}
        
        try:
            # Pattern Recognition
            reasoning_texts['pattern_recognition'] = self.engines['pattern_recognition'].generate_reasoning(
                current_data, context
            )
            
            # Context Analysis
            reasoning_texts['context_analysis'] = self.engines['context_analysis'].generate_reasoning(
                current_data, context
            )
            
            # Psychology Assessment
            reasoning_texts['psychology_assessment'] = self.engines['psychology_assessment'].generate_reasoning(
                current_data, context
            )
            
            # Execution Decision
            reasoning_texts['execution_decision'] = self.engines['execution_decision'].generate_reasoning(
                current_data, context
            )
            
            # Risk Assessment
            reasoning_texts['risk_assessment'] = self.engines['risk_assessment'].generate_reasoning(
                current_data, context
            )
            
            # Generate confidence score
            confidence_score = self._calculate_confidence_score(current_data, context, reasoning_texts)
            
            # Generate alternative scenarios
            alternative_scenarios = self._generate_alternative_scenarios(current_data, context, reasoning_texts)
            
        except Exception as e:
            logger.error(f"Error generating reasoning for row {row_idx}: {str(e)}")
            # Return default reasoning
            return self._get_default_reasoning()
        
        return [
            reasoning_texts['pattern_recognition'],
            reasoning_texts['context_analysis'],
            reasoning_texts['psychology_assessment'],
            reasoning_texts['execution_decision'],
            confidence_score,
            reasoning_texts['risk_assessment'],
            alternative_scenarios
        ]
    
    def _calculate_confidence_score(self, current_data: pd.Series, context: Dict, 
                                  reasoning_texts: Dict[str, str]) -> int:
        """
        Calculate overall confidence score based on technical analysis.
        
        Args:
            current_data: Current row data
            context: Historical context
            reasoning_texts: Generated reasoning texts
            
        Returns:
            Confidence score (0-100)
        """
        confidence_factors = []
        
        # Technical confluence
        if 'sma_5_20_cross' in current_data.index and 'sma_10_50_cross' in current_data.index:
            if current_data['sma_5_20_cross'] == current_data['sma_10_50_cross']:
                confidence_factors.append(20)  # MA alignment
        
        # Trend strength
        if 'trend_strength' in current_data.index:
            trend_strength = current_data['trend_strength']
            if not pd.isna(trend_strength):
                confidence_factors.append(min(30, trend_strength * 50))
        
        # Pattern presence
        pattern_cols = ['doji', 'hammer', 'bullish_engulfing', 'bearish_engulfing']
        pattern_count = sum(current_data.get(col, 0) for col in pattern_cols if col in current_data.index)
        if pattern_count > 0:
            confidence_factors.append(15)
        
        # Support/resistance proximity
        if 'support_distance' in current_data.index:
            support_dist = current_data['support_distance']
            if not pd.isna(support_dist) and support_dist < 1.0:
                confidence_factors.append(10)
        
        if 'resistance_distance' in current_data.index:
            resistance_dist = current_data['resistance_distance']
            if not pd.isna(resistance_dist) and resistance_dist < 1.0:
                confidence_factors.append(10)
        
        # Calculate final score
        base_score = 40  # Base confidence
        bonus_score = sum(confidence_factors)
        final_score = min(100, base_score + bonus_score)
        
        return int(final_score)
    
    def _generate_alternative_scenarios(self, current_data: pd.Series, context: Dict,
                                      reasoning_texts: Dict[str, str]) -> str:
        """
        Generate alternative scenario analysis.
        
        Args:
            current_data: Current row data
            context: Historical context
            reasoning_texts: Generated reasoning texts
            
        Returns:
            Alternative scenarios text
        """
        scenarios = []
        
        # Trend reversal scenario
        if 'trend_direction' in current_data.index:
            current_trend = "bullish" if current_data['trend_direction'] == 1 else "bearish"
            opposite_trend = "bearish" if current_trend == "bullish" else "bullish"
            scenarios.append(f"Trend reversal scenario suggests potential {opposite_trend} momentum if key levels are breached")
        
        # Volatility scenario
        vol_level = context.get('volatility_analysis', {}).get('level', 'normal')
        if vol_level == 'low':
            scenarios.append("Volatility expansion could trigger rapid directional movement beyond current expectations")
        elif vol_level == 'high':
            scenarios.append("Volatility compression may lead to consolidation phase with reduced directional bias")
        
        # Market regime change
        current_regime = context.get('market_regime', 'consolidation')
        if current_regime == 'trending':
            scenarios.append("Regime shift to consolidation would require adjustment of momentum-based strategies")
        elif current_regime == 'consolidation':
            scenarios.append("Breakout scenario could establish new trending phase with sustained directional movement")
        
        if not scenarios:
            scenarios.append("Alternative interpretations remain limited given current technical setup and market structure")
        
        return ". ".join(scenarios) + "."
    
    def _get_default_reasoning(self) -> List[str]:
        """
        Get default reasoning when generation fails.
        
        Returns:
            List of default reasoning texts
        """
        return [
            "Current market structure shows standard price action without distinct pattern characteristics.",
            "Technical analysis reveals neutral market conditions with balanced momentum indicators.",
            "Market participants appear to be in wait-and-see mode with limited directional conviction.",
            "Current setup suggests cautious approach with focus on risk management over aggressive positioning.",
            50,  # Default confidence score
            "Risk environment remains manageable with standard volatility characteristics and defined support levels.",
            "Alternative scenarios include continuation of current range-bound behavior or potential breakout on volume expansion."
        ]
