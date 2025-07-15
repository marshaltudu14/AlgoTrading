from typing import Dict, Any
from src.reasoning_system.core.base_engine import BaseReasoningEngine

class PsychologyAssessmentEngine(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def assess(self, current_row_features: Dict[str, Any], market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        sentiment = "neutral"
        emotional_state = "calm"

        # Get relevant market conditions
        overall_market_sentiment = market_conditions.get('overall_sentiment')
        market_volatility = market_conditions.get('volatility')
        market_momentum = market_conditions.get('momentum')

        # Prioritize overall market sentiment for psychological assessment
        if overall_market_sentiment == "bullish":
            sentiment = "optimistic"
            emotional_state = "confident"
            if market_momentum == "strong_positive":
                sentiment = "greedy"
                emotional_state = "euphoria"
        elif overall_market_sentiment == "bearish":
            sentiment = "pessimistic"
            emotional_state = "anxious"
            if market_momentum == "strong_negative":
                sentiment = "fearful"
                emotional_state = "panic"
        else: # neutral market sentiment
            sentiment = "indecisive"
            emotional_state = "uncertain"

        # Adjust based on volatility
        if market_volatility == "high":
            if sentiment == "greedy": emotional_state = "irrational exuberance"
            elif sentiment == "fearful": emotional_state = "capitulation"
            elif sentiment == "optimistic": emotional_state = "excitement"
            elif sentiment == "pessimistic": emotional_state = "distress"
            elif sentiment == "indecisive": emotional_state = "nervousness"
        elif market_volatility == "low":
            if sentiment == "indecisive": emotional_state = "apathy"
            else: emotional_state = "calm"

        # Incorporate specific indicator values, e.g., extreme RSI values
        rsi = current_row_features.get('RSI')
        if rsi is not None:
            if rsi > 80: # Extremely overbought
                sentiment = "overly greedy"
                emotional_state = "irrational exuberance"
            elif rsi < 20: # Extremely oversold
                sentiment = "extreme fear"
                emotional_state = "capitulation"

        return {
            "sentiment": sentiment,
            "emotional_state": emotional_state
        }

    def explain_assessment(self, psychological_factors: Dict[str, Any]) -> str:
        sentiment = psychological_factors.get("sentiment", "neutral")
        emotional_state = psychological_factors.get("emotional_state", "calm")
        return f"Trader sentiment appears {sentiment}, reflecting an overall emotional state of {emotional_state}."