import pandas as pd
from typing import Dict, Any
from src.reasoning_system.core.base_engine import BaseReasoningEngine

class HistoricalPatternEngine(BaseReasoningEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def identify_patterns(self, historical_data: pd.DataFrame, current_row_features: Dict[str, Any]) -> Dict[str, Any]:
        patterns = {
            "breakout_pattern": False,
            "breakdown_pattern": False,
            "consolidation_pattern": False,
            "trend_continuation": "none", # Changed to string for uptrend/downtrend
            "golden_cross": False,
            "death_cross": False,
            "bullish_engulfing_detected": False,
            "bearish_engulfing_detected": False,
            "doji_detected": False,
            "hammer_detected": False,
            "potential_double_top": False,
            "potential_double_bottom": False,
        }

        if historical_data.empty:
            return patterns

        # Consolidation: low historical volatility and narrow range
        if len(historical_data) > 10: 
            price_range = historical_data['high'].max() - historical_data['low'].min()
            avg_price = historical_data['close'].mean()
            if avg_price > 0 and (price_range / avg_price) < self.config.get('consolidation_threshold', 0.02):
                patterns['consolidation_pattern'] = True

        # Breakout/Breakdown: current price breaking historical range
        current_close = current_row_features.get('close')
        if current_close is not None:
            historical_high = historical_data['high'].max()
            historical_low = historical_data['low'].min()

            if current_close > historical_high * (1 + self.config.get('breakout_threshold', 0.005)):
                patterns['breakout_pattern'] = True
            elif current_close < historical_low * (1 - self.config.get('breakdown_threshold', 0.005)):
                patterns['breakdown_pattern'] = True

        # Trend continuation: check if current price is moving in line with a recent trend
        if len(historical_data) > 5:
            recent_closes = historical_data['close'].tail(5)
            if recent_closes.iloc[0] < recent_closes.iloc[-1] and current_close > recent_closes.iloc[-1]:
                patterns['trend_continuation'] = "uptrend"
            elif recent_closes.iloc[0] > recent_closes.iloc[-1] and current_close < recent_closes.iloc[-1]:
                patterns['trend_continuation'] = "downtrend"

        # Moving Average Crossovers (Golden Cross / Death Cross)
        sma_50 = current_row_features.get('SMA_50')
        sma_200 = current_row_features.get('SMA_200')
        
        prev_sma_50 = None
        prev_sma_200 = None

        if not historical_data.empty:
            if 'SMA_50' in historical_data.columns:
                prev_sma_50 = historical_data['SMA_50'].iloc[-1]
            if 'SMA_200' in historical_data.columns:
                prev_sma_200 = historical_data['SMA_200'].iloc[-1]

        if sma_50 and sma_200 and prev_sma_50 and prev_sma_200:
            if sma_50 > sma_200 and prev_sma_50 <= prev_sma_200:
                patterns['golden_cross'] = True
            elif sma_50 < sma_200 and prev_sma_50 >= prev_sma_200:
                patterns['death_cross'] = True

        # Candlestick Patterns (leveraging existing features)
        if current_row_features.get('bullish_engulfing') == 1:
            patterns['bullish_engulfing_detected'] = True
        if current_row_features.get('bearish_engulfing') == 1:
            patterns['bearish_engulfing_detected'] = True
        if current_row_features.get('doji') == 1:
            patterns['doji_detected'] = True
        if current_row_features.get('hammer') == 1:
            patterns['hammer_detected'] = True

        # Simple Swing High/Low for potential Double Top/Bottom (very basic)
        # This would need more sophisticated peak/trough detection over a longer period
        if len(historical_data) > 20: # Need a longer history for swing points
            recent_highs = historical_data['high'].nlargest(2)
            recent_lows = historical_data['low'].nsmallest(2)

            if len(recent_highs) == 2 and abs(recent_highs.iloc[0] - recent_highs.iloc[1]) / recent_highs.iloc[0] < 0.005: # Within 0.5% of each other
                patterns['potential_double_top'] = True
            if len(recent_lows) == 2 and abs(recent_lows.iloc[0] - recent_lows.iloc[1]) / recent_lows.iloc[0] < 0.005: # Within 0.5% of each other
                patterns['potential_double_bottom'] = True

        return patterns

    def explain_patterns(self, historical_patterns: Dict[str, Any]) -> str:
        explanations = []
        if historical_patterns.get("breakout_pattern"):
            explanations.append("A significant price breakout has occurred, indicating strong directional momentum.")
        if historical_patterns.get("breakdown_pattern"):
            explanations.append("A critical price breakdown is observed, suggesting potential for further decline.")
        if historical_patterns.get("consolidation_pattern"):
            explanations.append("The market is currently in a tight consolidation phase, often preceding a significant move.")
        
        trend_continuation = historical_patterns.get("trend_continuation")
        if trend_continuation == "uptrend":
            explanations.append("The prevailing uptrend is showing clear signs of continuation.")
        elif trend_continuation == "downtrend":
            explanations.append("The established downtrend appears to be continuing.")

        if historical_patterns.get("golden_cross"):
            explanations.append("A bullish Golden Cross (SMA50 above SMA200) has formed, signaling potential long-term upward momentum.")
        if historical_patterns.get("death_cross"):
            explanations.append("A bearish Death Cross (SMA50 below SMA200) is observed, indicating potential long-term downward pressure.")

        if historical_patterns.get("bullish_engulfing_detected"):
            explanations.append("A bullish engulfing candlestick pattern suggests a potential reversal to the upside.")
        if historical_patterns.get("bearish_engulfing_detected"):
            explanations.append("A bearish engulfing candlestick pattern indicates a potential reversal to the downside.")
        if historical_patterns.get("doji_detected"):
            explanations.append("A Doji candlestick pattern signals market indecision and potential trend reversal.")
        if historical_patterns.get("hammer_detected"):
            explanations.append("A Hammer candlestick pattern suggests a potential bullish reversal after a decline.")

        if historical_patterns.get("potential_double_top"):
            explanations.append("A potential Double Top formation is observed, hinting at a possible bearish reversal.")
        if historical_patterns.get("potential_double_bottom"):
            explanations.append("A potential Double Bottom formation is observed, suggesting a possible bullish reversal.")
            
        return " ".join(explanations)