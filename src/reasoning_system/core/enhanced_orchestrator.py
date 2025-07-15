import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any

from src.reasoning_system.engines.context_analysis_engine import ContextAnalysisEngine
from src.reasoning_system.engines.historical_pattern_engine import HistoricalPatternEngine
from src.reasoning_system.engines.market_condition_detector import MarketConditionDetector
from src.reasoning_system.engines.psychology_assessment_engine import PsychologyAssessmentEngine
from src.reasoning_system.engines.feature_relationship_engine import FeatureRelationshipEngine
from src.reasoning_system.context.historical_context_manager import HistoricalContextManager
from src.reasoning_system.core.base_engine import BaseReasoningEngine # Import BaseReasoningEngine

logger = logging.getLogger(__name__)

class EnhancedReasoningOrchestrator(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config) # Initialize BaseReasoningEngine
        reasoning_specific_config = config.get('reasoning', {})
        self.context_analyzer = ContextAnalysisEngine(reasoning_specific_config)
        self.pattern_engine = HistoricalPatternEngine(reasoning_specific_config)
        self.market_detector = MarketConditionDetector(reasoning_specific_config)
        self.psychology_engine = PsychologyAssessmentEngine(reasoning_specific_config)
        self.feature_relationship_engine = FeatureRelationshipEngine(reasoning_specific_config)
        self.historical_context_manager = HistoricalContextManager(reasoning_specific_config)

    def process_file(self, input_filepath: str, output_filepath: str) -> Dict[str, Any]:
        logger.info(f"Starting processing for {input_filepath}")
        try:
            df = pd.read_csv(input_filepath)
            logger.info(f"Loaded {len(df)} rows from {input_filepath}")

            # Ensure 'signal' column exists for training guidance
            if 'signal' not in df.columns:
                raise ValueError(f"'signal' column not found in {input_filepath}. This column is required for guiding decision generation.")

            # Add new columns for reasoning and decision
            df['reasoning'] = ''
            df['decision'] = ''

            # Process row by row to generate reasoning and decision
            for i in range(len(df)):
                current_row_features = df.iloc[i].to_dict()
                desired_signal = current_row_features.get('signal')

                # Get historical data for context
                historical_data = self.historical_context_manager.get_historical_context(df, i)

                reasoning, decision = self._generate_reasoning_for_row(
                    current_row_features, desired_signal, historical_data
                )
                df.at[i, 'reasoning'] = reasoning
                df.at[i, 'decision'] = decision

                if (i + 1) % self.config['reasoning']['processing']['progress_reporting_interval'] == 0:
                    logger.info(f"Processed {i + 1}/{len(df)} rows for {Path(input_filepath).name}")

            df.to_csv(output_filepath, index=False)
            logger.info(f"Successfully processed and saved to {output_filepath}")

            # Placeholder for quality score calculation
            quality_score = self._calculate_quality_score(df)

            return {
                'status': 'success',
                'input_file': Path(input_filepath).name,
                'output_file': Path(output_filepath).name,
                'input_rows': len(df),
                'output_rows': len(df),
                'reasoning_columns_added': True,
                'quality_score': quality_score
            }

        except Exception as e:
            logger.error(f"Error processing {input_filepath}: {e}", exc_info=True)
            return {
                'status': 'error',
                'input_file': Path(input_filepath).name,
                'error': str(e)
            }

    def _generate_reasoning_for_row(self, current_row_features: Dict[str, Any], desired_signal: int, historical_data: pd.DataFrame):
        # 1. Analyze market conditions
        market_conditions = self.market_detector.analyze(current_row_features)

        # 2. Identify historical patterns
        historical_patterns = self.pattern_engine.identify_patterns(historical_data, current_row_features)

        # 3. Assess psychological factors
        psychological_factors = self.psychology_engine.assess(current_row_features, market_conditions)

        # 4. Analyze feature relationships
        feature_relationships = self.feature_relationship_engine.analyze(current_row_features)

        # 5. Infer the decision based on desired_signal and contextual factors
        decision = self._infer_decision(desired_signal, market_conditions, historical_patterns, psychological_factors)

        # 6. Generate rule-based natural language reasoning
        reasoning_parts = []

        # Start with market conditions summary
        market_summary = self.context_analyzer.generate_market_summary(market_conditions)
        if market_summary: # Ensure it's not empty
            reasoning_parts.append(market_summary)

        # Incorporate historical context and patterns
        pattern_explanation = self.pattern_engine.explain_patterns(historical_patterns)
        if pattern_explanation:
            reasoning_parts.append(pattern_explanation)

        # Add psychological insights
        psychology_assessment = self.psychology_engine.explain_assessment(psychological_factors)
        if psychology_assessment:
            reasoning_parts.append(psychology_assessment)

        # Justify the decision based on all gathered context
        decision_justification = self._justify_decision(decision, current_row_features, market_conditions, historical_patterns, psychological_factors, feature_relationships)
        if decision_justification:
            reasoning_parts.append(decision_justification)

        reasoning = " ".join(reasoning_parts).strip()
        if not reasoning: # Fallback if no reasoning is generated
            reasoning = f"The market is currently exhibiting {market_conditions.get('overall_sentiment', 'neutral')} sentiment. A decision to {decision} is made based on available indicators."

        return reasoning, decision

    def _infer_decision(self, desired_signal: int, market_conditions: Dict[str, Any], historical_patterns: Dict[str, Any], psychological_factors: Dict[str, Any]) -> str:
        # This method enforces the decision based on the desired_signal for training purposes,
        # but the reasoning will justify it using the rich context.
        if desired_signal == 1:
            return "Long"
        elif desired_signal == 2:
            return "Short"
        else: # desired_signal == 0
            return "Hold"

    def _justify_decision(self, decision: str, current_row_features: Dict[str, Any], market_conditions: Dict[str, Any], historical_patterns: Dict[str, Any], psychological_factors: Dict[str, Any], feature_relationships: Dict[str, Any]) -> str:
        justification_parts = []

        # Start with a general statement
        justification_parts.append(f"Based on this analysis, the decision is to {decision}.")

        if decision == "Long":
            # Market Conditions
            if market_conditions.get('trend') == 'uptrend':
                justification_parts.append("The prevailing uptrend provides a strong foundation for a long position.")
            if market_conditions.get('momentum') == 'strong_positive':
                justification_parts.append("Strong positive momentum indicates continued upward pressure.")
            elif market_conditions.get('momentum') == 'positive':
                justification_parts.append("Positive momentum supports a bullish outlook.")

            # Historical Patterns
            if historical_patterns.get('golden_cross'):
                justification_parts.append("A Golden Cross formation signals potential long-term bullish continuation.")
            if historical_patterns.get('breakout_pattern'):
                justification_parts.append("A significant breakout confirms the bullish sentiment.")
            if historical_patterns.get('bullish_engulfing_detected'):
                justification_parts.append("The presence of a bullish engulfing pattern suggests a strong buying interest.")
            if historical_patterns.get('potential_double_bottom'):
                justification_parts.append("A potential Double Bottom formation indicates a strong reversal point.")
            if historical_patterns.get('trend_continuation') == 'uptrend':
                justification_parts.append("The existing uptrend is showing clear signs of continuation.")

            # Psychological Factors
            if psychological_factors.get('sentiment') == 'greedy' or psychological_factors.get('emotional_state') == 'euphoria':
                justification_parts.append("Trader sentiment is leaning towards greed, often a precursor to continued rallies.")
            elif psychological_factors.get('sentiment') == 'optimistic':
                justification_parts.append("Optimistic trader sentiment provides a favorable backdrop.")

            # Feature Relationships
            if feature_relationships.get('volume_confirmation') == 'strong_bullish':
                justification_parts.append("Strong bullish volume confirms the price action.")
            if feature_relationships.get('price_action_strength') == 'strong':
                justification_parts.append("Strong price action indicates conviction among buyers.")

        elif decision == "Short":
            # Market Conditions
            if market_conditions.get('trend') == 'downtrend':
                justification_parts.append("The prevailing downtrend provides a strong foundation for a short position.")
            if market_conditions.get('momentum') == 'strong_negative':
                justification_parts.append("Strong negative momentum indicates continued downward pressure.")
            elif market_conditions.get('momentum') == 'negative':
                justification_parts.append("Negative momentum supports a bearish outlook.")

            # Historical Patterns
            if historical_patterns.get('death_cross'):
                justification_parts.append("A Death Cross formation signals potential long-term bearish continuation.")
            if historical_patterns.get('breakdown_pattern'):
                justification_parts.append("A significant breakdown confirms the bearish sentiment.")
            if historical_patterns.get('bearish_engulfing_detected'):
                justification_parts.append("The presence of a bearish engulfing pattern suggests strong selling pressure.")
            if historical_patterns.get('potential_double_top'):
                justification_parts.append("A potential Double Top formation indicates a strong reversal point.")
            if historical_patterns.get('trend_continuation') == 'downtrend':
                justification_parts.append("The existing downtrend appears to be continuing.")

            # Psychological Factors
            if psychological_factors.get('sentiment') == 'fearful' or psychological_factors.get('emotional_state') == 'panic':
                justification_parts.append("Fear is gripping the market, often leading to sharp sell-offs.")
            elif psychological_factors.get('sentiment') == 'pessimistic':
                justification_parts.append("Pessimistic trader sentiment provides a favorable backdrop for shorts.")

            # Feature Relationships
            if feature_relationships.get('volume_confirmation') == 'strong_bearish':
                justification_parts.append("Strong bearish volume confirms the price action.")
            if feature_relationships.get('price_action_strength') == 'strong':
                justification_parts.append("Strong price action indicates conviction among sellers.")

        else: # Hold
            # Market Conditions
            if market_conditions.get('trend') == 'sideways':
                justification_parts.append("The market is currently in a sideways trend, indicating a lack of clear direction.")
            if market_conditions.get('momentum') == 'neutral':
                justification_parts.append("Neutral momentum suggests balanced buying and selling pressure.")
            if market_conditions.get('volatility') == 'low':
                justification_parts.append("Low volatility points to a period of calm and indecision.")

            # Historical Patterns
            if historical_patterns.get('consolidation_pattern'):
                justification_parts.append("A consolidation pattern is observed, indicating a lack of conviction among traders.")
            if historical_patterns.get('doji_detected'):
                justification_parts.append("A Doji candlestick pattern signals market indecision and potential trend reversal.")

            # Psychological Factors
            if psychological_factors.get('sentiment') == 'indecisive' or psychological_factors.get('emotional_state') == 'uncertain':
                justification_parts.append("Trader sentiment remains indecisive, awaiting a clearer catalyst.")

            # Feature Relationships
            if feature_relationships.get('volume_confirmation') == 'neutral':
                justification_parts.append("Volume remains neutral, not confirming any strong directional bias.")
            if feature_relationships.get('price_action_strength') == 'weak':
                justification_parts.append("Weak price action suggests a lack of strong conviction from either buyers or sellers.")

        return " ".join(justification_parts).strip()

    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        # Rule-based quality score calculation based on the rubric.
        # This is a simplified implementation for demonstration.
        total_score = 0
        num_rows = len(df)

        if num_rows == 0:
            return 0.0

        for index, row in df.iterrows():
            reasoning = row['reasoning']
            decision = row['decision']
            signal = row['signal']

            row_score = 0

            # 1. Contextual Richness (0-5 points)
            # Check for presence of keywords from different engines
            context_keywords = ["market is currently exhibiting", "volatility", "historical patterns", "trader sentiment"]
            if any(keyword in reasoning.lower() for keyword in context_keywords):
                row_score += 2 # Basic presence
            if "breakout pattern" in reasoning.lower() or "breakdown pattern" in reasoning.lower():
                row_score += 1
            if "fear" in reasoning.lower() or "greed" in reasoning.lower():
                row_score += 1
            if len(reasoning.split()) > 20: # Longer reasoning implies more context
                row_score += 1
            total_score += min(row_score, 5) # Cap at 5
            row_score = 0 # Reset for next criterion

            # 2. Coherence & Justification (0-5 points)
            # Check if decision is explicitly justified and aligns with reasoning
            if f"decision is to {decision.lower()}" in reasoning.lower():
                row_score += 3
            # Simple check for consistency (e.g., bullish reasoning for Long decision)
            if (decision == "Long" and ("bullish" in reasoning.lower() or "uptrend" in reasoning.lower())) or \
               (decision == "Short" and ("bearish" in reasoning.lower() or "downtrend" in reasoning.lower())) or \
               (decision == "Hold" and ("indecision" in reasoning.lower() or "sideways" in reasoning.lower())):
                row_score += 2
            total_score += min(row_score, 5)
            row_score = 0

            # 3. Natural Flow & Readability (0-5 points)
            # Simple checks: sentence length, presence of conjunctions, etc.
            # This is hard to do purely rule-based, so a simplified approach.
            if len(reasoning.split('.')) > 1: # More than one sentence
                row_score += 2
            if len(reasoning) > 50: # Minimum length
                row_score += 1
            total_score += min(row_score, 5)
            row_score = 0

            # 4. Signal Abstraction (0-5 points)
            # Ensure 'signal' keyword is not present
            if "signal" not in reasoning.lower():
                row_score += 5
            else:
                row_score += 0 # Penalize if 'signal' is mentioned
            total_score += min(row_score, 5)
            row_score = 0

            # 5. Historical Robustness (0-5 points)
            # Check for explicit mentions of historical patterns or context
            if "historical patterns" in reasoning.lower() or "past trends" in reasoning.lower() or "similar setups" in reasoning.lower():
                row_score += 3
            if "consolidation phase" in reasoning.lower() or "breakout pattern" in reasoning.lower() or "breakdown pattern" in reasoning.lower():
                row_score += 2
            total_score += min(row_score, 5)

        # Normalize score to a 0-100 scale (max possible score per row is 25, so 25 * num_rows)
        max_possible_score = 25 * num_rows
        if max_possible_score == 0:
            return 0.0
        return (total_score / max_possible_score) * 100.0