from typing import Dict, Any
from src.reasoning_system.core.base_engine import BaseReasoningEngine

class MarketConditionDetector(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def analyze(self, current_row_features: Dict[str, Any]) -> Dict[str, Any]:
        # Rule-based market condition analysis
        trend = "sideways"
        momentum = "neutral"
        volatility = "moderate"
        overall_sentiment = "neutral"

        # Example rules (these would be much more sophisticated)
        # Assuming features like 'SMA_50', 'SMA_200', 'RSI', 'MACD_hist', 'ATR' are present

        # Trend detection using Moving Averages
        sma_50 = current_row_features.get('SMA_50')
        sma_200 = current_row_features.get('SMA_200')
        close = current_row_features.get('close')

        if sma_50 and sma_200 and close:
            if sma_50 > sma_200 and close > sma_50:
                trend = "uptrend"
            elif sma_50 < sma_200 and close < sma_50:
                trend = "downtrend"

        # Momentum detection using MACD Histogram or RSI
        macd_hist = current_row_features.get('MACD_hist')
        rsi = current_row_features.get('RSI')

        if macd_hist is not None and rsi is not None:
            if macd_hist > 0 and rsi > 50:
                momentum = "strong_positive"
            elif macd_hist < 0 and rsi < 50:
                momentum = "strong_negative"
            elif macd_hist > 0 or rsi > 50:
                momentum = "positive"
            elif macd_hist < 0 or rsi < 50:
                momentum = "negative"

        # Volatility detection using ATR
        atr = current_row_features.get('ATR')
        if atr is not None:
            if atr > self.config.get('volatility_threshold_high', 2.0):
                volatility = "high"
            elif atr < self.config.get('volatility_threshold_low', 0.5):
                volatility = "low"

        # Overall sentiment (can be derived from trend and momentum)
        if trend == "uptrend" and momentum in ["positive", "strong_positive"]:
            overall_sentiment = "bullish"
        elif trend == "downtrend" and momentum in ["negative", "strong_negative"]:
            overall_sentiment = "bearish"

        return {
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility,
            "overall_sentiment": overall_sentiment
        }
