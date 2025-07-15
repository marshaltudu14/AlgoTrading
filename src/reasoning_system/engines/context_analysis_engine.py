from typing import Dict, Any
from src.reasoning_system.core.base_engine import BaseReasoningEngine

class ContextAnalysisEngine(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def analyze(self, current_row_features: Dict[str, Any]) -> Dict[str, Any]:
        # Rule-based context analysis
        overall_sentiment = "neutral"
        volatility = "moderate"
        key_events = "none"

        # Get relevant features
        rsi = current_row_features.get('RSI')
        macd_hist = current_row_features.get('MACD_hist')
        atr = current_row_features.get('ATR')
        close = current_row_features.get('close')
        open_price = current_row_features.get('open')

        # Determine sentiment based on multiple indicators
        bullish_signals = 0
        bearish_signals = 0

        if rsi is not None:
            if rsi > 60: bullish_signals += 1
            if rsi < 40: bearish_signals += 1

        if macd_hist is not None:
            if macd_hist > 0: bullish_signals += 1
            if macd_hist < 0: bearish_signals += 1

        # Simple price action sentiment
        if close is not None and open_price is not None:
            if close > open_price: bullish_signals += 1
            elif close < open_price: bearish_signals += 1

        if bullish_signals > bearish_signals:
            overall_sentiment = "bullish"
        elif bearish_signals > bullish_signals:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        # Volatility based on ATR
        if atr is not None:
            if atr > self.config.get('volatility_threshold_high', 2.0):
                volatility = "high"
            elif atr < self.config.get('volatility_threshold_low', 0.5):
                volatility = "low"
            else:
                volatility = "moderate"

        # Placeholder for key events (would typically come from external data or specific patterns)
        # For now, keeping it simple.

        return {
            "overall_sentiment": overall_sentiment,
            "volatility": volatility,
            "key_events": key_events
        }

    def generate_market_summary(self, market_conditions: Dict[str, Any]) -> str:
        sentiment = market_conditions.get("overall_sentiment", "neutral")
        volatility = market_conditions.get("volatility", "moderate")
        summary = f"The market is currently exhibiting {sentiment} sentiment with {volatility} volatility."
        
        if market_conditions.get("key_events") != "none":
            summary += f" Key events detected: {market_conditions.get("key_events")}."
            
        return summary