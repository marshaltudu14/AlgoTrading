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
            # if 'signal' not in df.columns:
            #     raise ValueError(f"'signal' column not found in {input_filepath}. This column is required for guiding decision generation.")

            # Add new columns for reasoning and decision
            df['reasoning'] = ''
            df['decision'] = ''
            df['signal'] = 0 # Initialize signal column

            # Process row by row to generate reasoning and decision
            for i in range(len(df)):
                current_row_features = df.iloc[i].to_dict()
                # desired_signal = current_row_features.get('signal') # No longer using future-looking signal

                # Get historical data for context
                historical_data = self.historical_context_manager.get_historical_context(df, i)

                # Calculate historical decision
                decision = self._calculate_historical_decision(current_row_features)
                
                # Generate reasoning based purely on historical context
                reasoning = self._generate_reasoning_for_row(
                    current_row_features, historical_data
                )
                
                df.at[i, 'reasoning'] = reasoning
                df.at[i, 'decision'] = decision
                df.at[i, 'signal'] = self._map_decision_to_signal(decision) # Map decision to signal

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

    def _calculate_historical_decision(self, features: Dict[str, Any]) -> str:
        """
        Calculates a historical decision (Long, Short, Hold) based on current features.
        This method uses only historical data and does not look into the future.
        """
        sma_5 = features.get('sma_5')
        sma_20 = features.get('sma_20')
        rsi_14 = features.get('rsi_14')
        close_price = features.get('close')
        bb_middle = features.get('bb_middle')

        # Ensure all necessary features are available
        if any(x is None for x in [sma_5, sma_20, rsi_14, close_price, bb_middle]):
            return "Hold" # Cannot make a decision if features are missing

        # Long conditions
        long_condition_1 = sma_5 > sma_20
        long_condition_2 = rsi_14 < 40
        long_condition_3 = close_price > bb_middle

        if long_condition_1 and long_condition_2 and long_condition_3:
            return "Long"

        # Short conditions
        short_condition_1 = sma_5 < sma_20
        short_condition_2 = rsi_14 > 60
        short_condition_3 = close_price < bb_middle

        if short_condition_1 and short_condition_2 and short_condition_3:
            return "Short"

        return "Hold"

    def _map_decision_to_signal(self, decision: str) -> int:
        """
        Maps a string decision (Long, Short, Hold) to an integer signal (1, 2, 0).
        """
        if decision == "Long":
            return 1
        elif decision == "Short":
            return 2
        else:
            return 0

    def _generate_reasoning_for_row(self, current_row_features: Dict[str, Any], historical_data: pd.DataFrame):
        # 1. Analyze market conditions based on historical features
        market_conditions = self.market_detector.analyze(current_row_features)

        # 2. Identify historical patterns
        historical_patterns = self.pattern_engine.identify_patterns(historical_data, current_row_features)

        # 3. Assess psychological factors
        psychological_factors = self.psychology_engine.assess(current_row_features, market_conditions)

        # 4. Analyze feature relationships
        feature_relationships = self.feature_relationship_engine.analyze(current_row_features)

        # Generate rule-based natural language reasoning based on historical context
        reasoning_parts = []

        # Get key insights from engines
        overall_sentiment = market_conditions.get('overall_sentiment')
        market_volatility = market_conditions.get('volatility')
        trader_sentiment = psychological_factors.get('sentiment')
        emotional_state = psychological_factors.get('emotional_state')
        key_events = market_conditions.get('key_events')
        pattern_explanation = self.pattern_engine.explain_patterns(historical_patterns)

        # --- Constructing the narrative based on historical context ---

        reasoning_parts.append(f"The market is currently exhibiting {overall_sentiment} sentiment with {market_volatility} volatility.")
        reasoning_parts.append(f"Trader sentiment appears {trader_sentiment}, reflecting an overall emotional state of {emotional_state}.")

        # Add key events if present
        if key_events != "none":
            reasoning_parts.append(f"Key events detected: {key_events}.")

        # Incorporate historical context and patterns
        if pattern_explanation:
            reasoning_parts.append(pattern_explanation)

        # Add general concluding remark based on historical analysis
        reasoning_parts.append("Based on historical technical indicators and market structure, the current market conditions suggest a specific trading posture.")

        reasoning = " ".join(reasoning_parts).strip()
        if not reasoning: # Fallback if no reasoning is generated
            reasoning = f"The market is currently exhibiting {market_conditions.get('overall_sentiment', 'neutral')} sentiment based on historical data."

        return reasoning

    def _infer_decision(self, desired_signal: int, market_conditions: Dict[str, Any], historical_patterns: Dict[str, Any], psychological_factors: Dict[str, Any]) -> str:
        # This method is no longer used for generating the decision column.
        # The decision is now calculated historically by _calculate_historical_decision.
        # This method is kept for compatibility if other parts of the system still call it,
        # but its output is not used for the decision column in process_file.
        if desired_signal == 1:
            return "Long"
        elif desired_signal == 2:
            return "Short"
        else: # desired_signal == 0
            return "Hold"

    def _justify_decision(self, decision: str, current_row_features: Dict[str, Any], market_conditions: Dict[str, Any], historical_patterns: Dict[str, Any], psychological_factors: Dict[str, Any], feature_relationships: Dict[str, Any], desired_signal: int) -> str:
        # This method is no longer used for generating the reasoning text.
        # The reasoning is now generated historically by _generate_reasoning_for_row.
        # This method is kept for compatibility if other parts of the system still call it.
        justification_parts = []

        # General opening statement, acknowledging the decision
        justification_parts.append(f"The decision is to {decision}.")

        # Incorporate market conditions
        market_trend = market_conditions.get('trend')
        market_momentum = market_conditions.get('momentum')
        market_volatility = market_conditions.get('volatility')
        overall_sentiment = market_conditions.get('overall_sentiment')

        # Incorporate psychological factors
        trader_sentiment = psychological_factors.get('sentiment')
        emotional_state = psychological_factors.get('emotional_state')

        # Incorporate historical patterns
        has_breakout = historical_patterns.get('breakout_pattern')
        has_breakdown = historical_patterns.get('breakdown_pattern')
        has_consolidation = historical_patterns.get('consolidation_pattern')
        trend_continuation = historical_patterns.get('trend_continuation')
        has_golden_cross = historical_patterns.get('golden_cross')
        has_death_cross = historical_patterns.get('death_cross')
        has_bullish_engulfing = historical_patterns.get('bullish_engulfing_detected')
        has_bearish_engulfing = historical_patterns.get('bearish_engulfing_detected')
        has_doji = historical_patterns.get('doji_detected')
        has_hammer = historical_patterns.get('hammer_detected')
        has_double_top = historical_patterns.get('potential_double_top')
        has_double_bottom = historical_patterns.get('potential_double_bottom')

        # Incorporate feature relationships
        volume_confirmation = feature_relationships.get('volume_confirmation')
        indicator_divergence = feature_relationships.get('indicator_divergence')
        price_action_strength = feature_relationships.get('price_action_strength')

        # --- Justification Logic based on Decision and Context ---

        if decision == "Long":
            if overall_sentiment == "strongly bullish" or overall_sentiment == "bullish":
                justification_parts.append(f"The market exhibits a {overall_sentiment} outlook, supported by strong underlying indicators.")
            elif overall_sentiment == "mildly bullish":
                justification_parts.append("Despite a mildly bullish market, specific factors suggest an upward move.")
            else: # Neutral or bearish market, but desired_signal is Long (gut feeling/strategic)
                justification_parts.append("While the broader market remains somewhat subdued, a strategic opportunity for a long position is identified.")

            if has_golden_cross: justification_parts.append("A recent Golden Cross reinforces the long-term bullish conviction.")
            if has_breakout: justification_parts.append("A decisive breakout from resistance confirms the upward momentum.")
            if has_bullish_engulfing: justification_parts.append("A bullish engulfing pattern signals strong buying interest.")
            if has_double_bottom: justification_parts.append("The formation of a potential Double Bottom suggests a strong reversal point.")
            if trend_continuation == "uptrend": justification_parts.append("The existing uptrend shows clear signs of continuation.")

            if trader_sentiment == "optimistic" or trader_sentiment == "greedy":
                justification_parts.append(f"Trader sentiment is {trader_sentiment}, providing a favorable backdrop for this move.")
            if volume_confirmation == "strong_bullish":
                justification_parts.append("Volume confirms the bullish price action, indicating strong conviction.")

        elif decision == "Short":
            if overall_sentiment == "strongly bearish" or overall_sentiment == "bearish":
                justification_parts.append(f"The market presents a {overall_sentiment} outlook, with indicators pointing to a downturn.")
            elif overall_sentiment == "mildly bearish":
                justification_parts.append("Even with a mildly bearish market, specific factors suggest a downward move.")
            else: # Neutral or bullish market, but desired_signal is Short (gut feeling/strategic)
                justification_parts.append("Despite a mixed market sentiment, a strategic opportunity for a short position is identified.")

            if has_death_cross: justification_parts.append("A recent Death Cross reinforces the long-term bearish conviction.")
            if has_breakdown: justification_parts.append("A decisive breakdown from support confirms the downward momentum.")
            if has_bearish_engulfing: justification_parts.append("A bearish engulfing pattern signals strong selling pressure.")
            if has_double_top: justification_parts.append("A potential Double Top formation indicates a strong reversal point.")
            if trend_continuation == "downtrend": justification_parts.append("The existing downtrend appears to be continuing.")

            if trader_sentiment == "pessimistic" or trader_sentiment == "fearful":
                justification_parts.append(f"Trader sentiment is {trader_sentiment}, contributing to the bearish pressure.")
            if volume_confirmation == "strong_bearish":
                justification_parts.append("Volume confirms the bearish price action, indicating strong conviction.")

        else: # Hold
            justification_parts.append("The market is currently in a state of indecision.")
            if has_consolidation: justification_parts.append("A tight consolidation phase suggests a lack of clear direction.")
            if has_doji: justification_parts.append("The presence of a Doji candlestick signals market indecision.")
            if market_volatility == "low": justification_parts.append("Low volatility points to a period of calm and indecision.")
            if overall_sentiment == "neutral": justification_parts.append("Overall market sentiment remains neutral, with balanced forces.")
            if trader_sentiment == "indecisive": justification_parts.append("Trader sentiment is indecisive, awaiting a clearer catalyst.")

            # Add a strategic note for Hold
            justification_parts.append("It is prudent to observe further price action before committing to a directional bias.")

        # Add a concluding remark about the strategic nature of the signal
        if desired_signal == 1: # Long signal
            justification_parts.append("This decision aligns with the system's long-term bullish strategy, targeting upward movements based on ATR-derived risk-reward.")
        elif desired_signal == 2: # Short signal
            justification_parts.append("This decision aligns with the system's long-term bearish strategy, targeting downward movements based on ATR-derived risk-reward.")
        else: # Hold signal
            justification_parts.append("This decision aligns with the system's risk management strategy, prioritizing capital preservation during uncertain market conditions.")

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