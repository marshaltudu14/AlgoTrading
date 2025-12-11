import { create } from 'zustand';
import { ProcessedCandleData } from './candleStore';

// Types for strategies
export type SignalType = 'BUY' | 'SELL' | 'HOLD';

export interface StrategySignal {
  signal: SignalType;
  confidence: number; // 0 to 1
  timestamp?: number;
  price?: number;
  details?: {
    [key: string]: unknown;
  };
}

// Pure strategy function - takes data and returns signal
export interface StrategyFunction {
  (candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]): StrategySignal;
}

export interface StrategyState {
  // Current state (for UI display)
  currentSignal: StrategySignal | null;
  isCalculating: boolean;

  // Historical signals (optional, for tracking)
  signalHistory: StrategySignal[];

  // Actions
  calculateSignal: (candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]) => void;
  clearHistory: () => void;
  getLatestSignal: () => StrategySignal | null;

  // Pure strategy function for backtest/real trading
  getSignal: StrategyFunction;
}

// Pure strategy implementations
class RuleBasedStrategies {
  /**
   * RSI Mean Reversion Strategy
   * Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)
   */
  static rsiMeanReversion(candle: ProcessedCandleData): StrategySignal {
    const rsi = candle.rsi_14;
    const rsi21 = candle.rsi_21;

    let signal: SignalType = 'HOLD';
    let confidence = 0;

    if (rsi !== undefined) {
      if (rsi < 30) {
        signal = 'BUY';
        confidence = Math.min(0.8, (30 - rsi) / 20);
      } else if (rsi > 70) {
        signal = 'SELL';
        confidence = Math.min(0.8, (rsi - 70) / 20);
      }
    }

    return {
      signal,
      confidence,
      details: {
        rsi: rsi?.toFixed(2),
        rsi21: rsi21?.toFixed(2)
      }
    };
  }

  /**
   * MACD Crossover Strategy
   * Buy on bullish crossover, Sell on bearish crossover
   */
  static macdCrossover(candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]): StrategySignal {
    if (previousCandles.length === 0) {
      return { signal: 'HOLD', confidence: 0 };
    }

    const prevCandle = previousCandles[previousCandles.length - 1];

    const macd = candle.macd_pct;
    const signal = candle.macd_signal_pct;
    const hist = candle.macd_hist_pct;

    const prevMacd = prevCandle.macd_pct;
    const prevSignal = prevCandle.macd_signal_pct;

    if (macd === undefined || signal === undefined || prevMacd === undefined || prevSignal === undefined) {
      return { signal: 'HOLD', confidence: 0 };
    }

    let finalSignal: SignalType = 'HOLD';
    let confidence = 0;

    // Bullish crossover
    if (macd > signal && prevMacd <= prevSignal && hist !== undefined && hist > 0) {
      finalSignal = 'BUY';
      confidence = Math.min(0.9, Math.abs(hist) * 10);
    }
    // Bearish crossover
    else if (macd < signal && prevMacd >= prevSignal && hist !== undefined && hist < 0) {
      finalSignal = 'SELL';
      confidence = Math.min(0.9, Math.abs(hist) * 10);
    }

    return {
      signal: finalSignal,
      confidence,
      details: {
        macd: macd?.toFixed(4),
        signal: signal?.toFixed(4),
        histogram: hist?.toFixed(4)
      }
    };
  }

  /**
   * EMA Crossover Strategy
   * Golden cross and death cross
   */
  static emaCrossover(candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]): StrategySignal {
    if (previousCandles.length === 0) {
      return { signal: 'HOLD', confidence: 0 };
    }

    const prevCandle = previousCandles[previousCandles.length - 1];

    const distEma20 = candle.dist_ema_20;
    const distSma200 = candle.dist_sma_200;

    const prevDistEma20 = prevCandle.dist_ema_20;
    const prevDistSma200 = prevCandle.dist_sma_200;

    if (distEma20 === undefined || distSma200 === undefined ||
        prevDistEma20 === undefined || prevDistSma200 === undefined) {
      return { signal: 'HOLD', confidence: 0 };
    }

    let signal: SignalType = 'HOLD';
    let confidence = 0;

    // Golden cross - price crosses above EMA20 while above SMA200
    if (distEma20 > 0 && prevDistEma20 <= 0 && distSma200 > 0) {
      signal = 'BUY';
      confidence = Math.min(0.8, Math.abs(distEma20) / 2);
    }
    // Death cross - price crosses below EMA20 while below SMA200
    else if (distEma20 < 0 && prevDistEma20 >= 0 && distSma200 < 0) {
      signal = 'SELL';
      confidence = Math.min(0.8, Math.abs(distEma20) / 2);
    }

    return {
      signal,
      confidence,
      details: {
        distEma20: distEma20.toFixed(2),
        distSma200: distSma200.toFixed(2)
      }
    };
  }

  /**
   * Trend Following Strategy
   * Uses trend direction and strength
   */
  static trendFollowing(candle: ProcessedCandleData): StrategySignal {
    const trendStrength = candle.trend_strength;
    const trendDirection = candle.trend_direction;
    const priceChange = candle.price_change_pct;

    if (trendStrength === undefined || trendDirection === undefined ||
        priceChange === undefined) {
      return { signal: 'HOLD', confidence: 0 };
    }

    let signal: SignalType = 'HOLD';
    let confidence = 0;

    // Strong uptrend
    if (trendDirection > 0 && trendStrength > 0.5 && priceChange > 0) {
      signal = 'BUY';
      confidence = Math.min(0.9, trendStrength);
    }
    // Strong downtrend
    else if (trendDirection < 0 && trendStrength > 0.5 && priceChange < 0) {
      signal = 'SELL';
      confidence = Math.min(0.9, trendStrength);
    }

    return {
      signal,
      confidence,
      details: {
        trendStrength: trendStrength.toFixed(3),
        trendDirection: trendDirection,
        priceChange: priceChange.toFixed(2)
      }
    };
  }

  /**
   * Complete Multi-Indicator Combination Strategy (exact replica from Python)
   */
  static multiIndicatorCombo(candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]): StrategySignal {
    // Get current time to check for 3:15 PM cutoff
    const currentDateTime = new Date(candle.timestamp * 1000);
    const currentHour = currentDateTime.getHours();
    const currentMinute = currentDateTime.getMinutes();

    // Skip trades after 3:15 PM (to allow execution before 3:30 close)
    if (currentHour > 15 || (currentHour === 15 && currentMinute >= 15)) {
      return { signal: 'HOLD', confidence: 0 };
    }

    // Get signals from multiple strategies
    const rsiResult = this.rsiMeanReversion(candle);
    const macdResult = this.macdCrossover(candle, previousCandles);
    const trendResult = this.trendFollowing(candle);

    // Additional indicator analysis (matching Python exactly)
    const atr = candle.atr_pct ?? 0.09;
    const priceChange = candle.price_change_pct ?? 0;
    const bbPosition = candle.bb_position ?? 0.5;
    const bbWidth = candle.bb_width_pct ?? 0.19;
    const trendSlope = candle.trend_slope ?? 0;

    // Get RSI values
    const rsi21 = candle.rsi_21 ?? 50;
    const rsi14 = candle.rsi_14 ?? 50;

    // Candlestick analysis
    const bodySize = candle.body_size_pct ?? 0.047;
    const lowerShadow = candle.lower_shadow_pct ?? 0.022;
    const upperShadow = candle.upper_shadow_pct ?? 0.021;
    const hlRange = candle.hl_range_pct ?? 0.09;

    // Multi-timeframe alignment check
    const distSma5 = candle.dist_sma_5 ?? 0;
    const distSma10 = candle.dist_sma_10 ?? 0;
    const distSma20 = candle.dist_sma_20 ?? 0;
    const distSma50 = candle.dist_sma_50 ?? 0;

    const mtfBuy = (distSma5 > distSma10) && (distSma10 > distSma20) && (distSma20 > distSma50);
    const mtfSell = (distSma5 < distSma10) && (distSma10 < distSma20) && (distSma20 < distSma50);

    // Risk filters based on actual data analysis
    // 1. Avoid huge candles (momentum already passed)
    if (hlRange > 0.5) {  // > 0.5% range is large (75th percentile is 0.11%)
      return { signal: 'HOLD', confidence: 0 };
    }

    // 2. Avoid large body candles (already strong momentum)
    if (bodySize > 0.3) {  // > 0.3% body is large
      return { signal: 'HOLD', confidence: 0 };
    }

    // 3. Check for reversal signals using wicks
    const longLowerWick = lowerShadow > 0.1 && lowerShadow > (bodySize * 2);
    const longUpperWick = upperShadow > 0.1 && upperShadow > (bodySize * 2);

    // Dynamic confidence multipliers
    let confidenceBoost = 1.0;

    // Multi-timeframe alignment bonus
    if (mtfBuy && trendResult.signal === 'BUY') {
      confidenceBoost *= 1.3;
    } else if (mtfSell && trendResult.signal === 'SELL') {
      confidenceBoost *= 1.3;
    }

    // Bollinger Band analysis
    if (bbPosition < 0.15 && bbWidth > 0.15) {  // Near lower band with decent volatility
      if (rsi14 < 40) {
        confidenceBoost *= 1.25;
      }
    } else if (bbPosition > 0.85 && bbWidth > 0.15) {  // Near upper band
      if (rsi14 > 60) {
        confidenceBoost *= 1.25;
      }
    }

    // Reversal signals from wicks
    if (longLowerWick && rsi14 < 45) {
      confidenceBoost *= 1.2;  // Bullish reversal potential
    } else if (longUpperWick && rsi14 > 55) {
      confidenceBoost *= 1.2;  // Bearish reversal potential
    }

    // Squeeze play detection
    if (bbWidth < 0.1) {  // Low volatility squeeze
      if (Math.abs(trendSlope) > 0.5) {
        confidenceBoost *= 1.15;
      }
    }

    // Divergence detection
    if (previousCandles.length > 10) {
      const recentCandles = previousCandles.slice(-5);
      recentCandles.push(candle);

      if (recentCandles.length === 6) {
        const recentRsi = recentCandles.map(c => c.rsi_14 ?? 50);
        const recentPrice = recentCandles.map(c => c.close);

        // Bullish divergence
        if (recentPrice[5] < recentPrice[0] && recentRsi[5] > recentRsi[0] && rsi14 < 40) {
          confidenceBoost *= 1.2;
        }
        // Bearish divergence
        else if (recentPrice[5] > recentPrice[0] && recentRsi[5] < recentRsi[0] && rsi14 > 60) {
          confidenceBoost *= 1.2;
        }
      }
    }

    // Dynamic weighting based on volatility
    if (atr > 0.15) {  // High volatility (mean is 0.09)
      trendResult.confidence *= 1.3;
      rsiResult.confidence *= 0.7;
    } else if (atr < 0.07) {  // Low volatility
      rsiResult.confidence *= 1.3;
      trendResult.confidence *= 0.7;
    }

    // Avoid trading during extreme price movements
    if (Math.abs(priceChange) > 0.3) {  // > 0.3% move is significant
      return { signal: 'HOLD', confidence: 0 };
    }

    // Enhanced RSI logic for overbought/oversold
    if (rsi14 > 75 && trendResult.signal === 'SELL') {
      rsiResult.confidence *= 1.3;  // Strengthen sell in overbought
    } else if (rsi14 < 25 && trendResult.signal === 'BUY') {
      rsiResult.confidence *= 1.3;  // Strengthen buy in oversold
    }

    // Weight voting system
    let buyVotes = 0;
    let sellVotes = 0;
    let totalConfidence = 0;

    const signals = [rsiResult, macdResult, trendResult];
    signals.forEach(s => {
      if (s.signal === 'BUY') {
        buyVotes += s.confidence;
        totalConfidence += s.confidence;
      } else if (s.signal === 'SELL') {
        sellVotes += s.confidence;
        totalConfidence += s.confidence;
      }
    });

    // Apply confidence boost
    totalConfidence *= confidenceBoost;

    // Final decision
    if (totalConfidence > 0) {
      const buyRatio = buyVotes / totalConfidence;
      const sellRatio = sellVotes / totalConfidence;

      // Higher threshold for safety
      const threshold = confidenceBoost < 1.2 ? 0.7 : 0.6;

      if (buyRatio > threshold && totalConfidence > 0.5) {
        const finalConfidence = Math.min(0.9, buyRatio * confidenceBoost);
        return {
          signal: 'BUY',
          confidence: finalConfidence,
          details: {
            buyRatio: buyRatio.toFixed(3),
            sellRatio: sellRatio.toFixed(3),
            threshold: threshold.toFixed(2),
            confidenceBoost: confidenceBoost.toFixed(2),
            atr: atr.toFixed(3),
            bbPosition: bbPosition.toFixed(3),
            rsi14: rsi14.toFixed(1),
            rsi21: rsi21.toFixed(1),
            mtfAligned: mtfBuy,
            divergenceDetected: confidenceBoost >= 1.2
          }
        };
      } else if (sellRatio > threshold && totalConfidence > 0.5) {
        const finalConfidence = Math.min(0.9, sellRatio * confidenceBoost);
        return {
          signal: 'SELL',
          confidence: finalConfidence,
          details: {
            buyRatio: buyRatio.toFixed(3),
            sellRatio: sellRatio.toFixed(3),
            threshold: threshold.toFixed(2),
            confidenceBoost: confidenceBoost.toFixed(2),
            atr: atr.toFixed(3),
            bbPosition: bbPosition.toFixed(3),
            rsi14: rsi14.toFixed(1),
            rsi21: rsi21.toFixed(1),
            mtfAligned: mtfSell,
            divergenceDetected: confidenceBoost >= 1.2
          }
        };
      }
    }

    return {
      signal: 'HOLD',
      confidence: 0,
      details: {
        buyRatio: (totalConfidence > 0 ? buyVotes / totalConfidence : 0).toFixed(3),
        sellRatio: (totalConfidence > 0 ? sellVotes / totalConfidence : 0).toFixed(3)
      }
    };
  }
}

export const useStrategyStore = create<StrategyState>((set, get) => ({
  // Initial state
  currentSignal: null,
  isCalculating: false,
  signalHistory: [],

  // Calculate signal using rule-based strategy
  calculateSignal: (candle, previousCandles) => {
    set({ isCalculating: true });

    // Use the complete multi-indicator combo strategy from Python
    const result = RuleBasedStrategies.multiIndicatorCombo(candle, previousCandles);

    const strategySignal: StrategySignal = {
      ...result,
      timestamp: candle.timestamp,
      price: candle.close
    };

    set((state) => ({
      currentSignal: strategySignal,
      signalHistory: [strategySignal, ...state.signalHistory.slice(0, 99)], // Keep last 100 signals
      isCalculating: false
    }));
  },

  // Pure strategy function for backtest/real trading - no state management
  getSignal: (candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]) => {
    return RuleBasedStrategies.multiIndicatorCombo(candle, previousCandles);
  },

  // Clear signal history
  clearHistory: () => {
    set({
      signalHistory: [],
      currentSignal: null
    });
  },

  // Get latest signal
  getLatestSignal: () => {
    return get().currentSignal;
  }
}));