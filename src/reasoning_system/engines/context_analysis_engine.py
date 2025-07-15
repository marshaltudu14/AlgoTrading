from typing import Dict, Any
from src.reasoning_system.core.base_engine import BaseReasoningEngine

class ContextAnalysisEngine(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def analyze(self, current_row_features: Dict[str, Any]) -> Dict[str, Any]:
        overall_sentiment = "neutral"
        volatility = "moderate"
        key_events = "none"

        rsi = current_row_features.get('RSI')
        macd_hist = current_row_features.get('MACD_hist')
        close = current_row_features.get('close')
        open_price = current_row_features.get('open')
        atr = current_row_features.get('ATR')

        bullish_score = 0
        bearish_score = 0

        # Stronger weighting for RSI and MACD
        if rsi is not None:
            if rsi > 75: bullish_score += 3 # Very strong bullish
            elif rsi > 60: bullish_score += 2
            elif rsi > 50: bullish_score += 1
            if rsi < 25: bearish_score += 3 # Very strong bearish
            elif rsi < 40: bearish_score += 2
            elif rsi < 50: bearish_score += 1

        if macd_hist is not None:
            if macd_hist > 1.0: bullish_score += 3 # Very strong bullish MACD
            elif macd_hist > 0.2: bullish_score += 2
            elif macd_hist > 0: bullish_score += 1
            if macd_hist < -1.0: bearish_score += 3 # Very strong bearish MACD
            elif macd_hist < -0.2: bearish_score += 2
            elif macd_hist < 0: bearish_score += 1

        # Price action sentiment
        if close is not None and open_price is not None:
            if close > open_price * 1.005: bullish_score += 1 # Significant bullish candle
            elif close < open_price * 0.995: bearish_score += 1 # Significant bearish candle

        # Determine overall sentiment based on scores
        if bullish_score >= 4 and bullish_score > bearish_score + 1: 
            overall_sentiment = "strongly bullish"
        elif bullish_score >= 2 and bullish_score > bearish_score: 
            overall_sentiment = "bullish"
        elif bearish_score >= 4 and bearish_score > bullish_score + 1: 
            overall_sentiment = "strongly bearish"
        elif bearish_score >= 2 and bearish_score > bullish_score: 
            overall_sentiment = "bearish"
        elif bullish_score > 0 and bearish_score == 0:
            overall_sentiment = "mildly bullish"
        elif bearish_score > 0 and bearish_score == 0:
            overall_sentiment = "mildly bearish"
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