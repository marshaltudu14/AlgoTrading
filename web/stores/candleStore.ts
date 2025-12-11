import { create } from 'zustand';
import * as technicalIndicators from 'technicalindicators';

// Types for candle data
export interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface ProcessedCandleData extends CandleData {
  // Distance from Moving Averages (percentage)
  dist_sma_5?: number;
  dist_sma_10?: number;
  dist_sma_20?: number;
  dist_sma_50?: number;
  dist_sma_100?: number;
  dist_sma_200?: number;
  dist_ema_5?: number;
  dist_ema_10?: number;
  dist_ema_20?: number;
  dist_ema_50?: number;
  dist_ema_100?: number;
  dist_ema_200?: number;

  // MACD (percentage)
  macd_pct?: number;
  macd_signal_pct?: number;
  macd_hist_pct?: number;

  // RSI
  rsi_14?: number;
  rsi_21?: number;

  // ADX and Directional Indicators
  adx?: number;
  di_plus?: number;
  di_minus?: number;

  // ATR (percentage and raw)
  atr_pct?: number;
  atr?: number;

  // Bollinger Bands (width and position)
  bb_width_pct?: number;
  bb_position?: number;

  // Trend Strength
  trend_slope?: number;
  trend_strength?: number;
  trend_direction?: number;

  // Price Action Features
  price_change_pct?: number;
  price_change_abs?: number;
  hl_range_pct?: number;
  body_size_pct?: number;
  upper_shadow_pct?: number;
  lower_shadow_pct?: number;

  // Volatility
  volatility_10?: number;
  volatility_20?: number;
}

export interface CandleState {
  // Raw data
  candles: CandleData[];
  // Processed data with features
  processedCandles: ProcessedCandleData[];
  // Loading states
  isLoading: boolean;
  isProcessing: boolean;
  // Error handling
  error: string | null;

  // Actions
  setCandles: (candles: CandleData[]) => void;
  processFeatures: () => void;
  clearData: () => void;
  getLatestFeatures: () => ProcessedCandleData | null;
  getFeatureNames: () => string[];
  getExpectedFeatureCount: () => number;
}

// Helper function to calculate trend strength (replicated from Python)
const calculateTrendStrength = (close: number[], period: number) => {
  const trendSlope: number[] = [];
  const trendStrength: number[] = [];

  for (let i = period - 1; i < close.length; i++) {
    const window = close.slice(i - period + 1, i + 1);
    const x = Array.from({ length: period }, (_, idx) => idx);

    // Calculate slope using linear regression
    const n = period;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = window.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, idx) => sum + xi * window[idx], 0);
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    trendSlope.push(slope);

    // Calculate R-squared
    const meanY = sumY / n;
    // yMean is not needed for the calculation

    let ssRes = 0;
    let ssTot = 0;
    for (let j = 0; j < period; j++) {
      const yPred = slope * x[j] + (sumY - slope * sumX) / n;
      ssRes += Math.pow(window[j] - yPred, 2);
      ssTot += Math.pow(window[j] - meanY, 2);
    }

    const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
    trendStrength.push(rSquared);
  }

  // Pad the beginning with undefined values
  return {
    trendSlope: Array(period - 1).fill(undefined).concat(trendSlope),
    trendStrength: Array(period - 1).fill(undefined).concat(trendStrength),
    trendDirection: trendSlope.map(s => s > 0 ? 1 : -1)
  };
};

export const useCandleStore = create<CandleState>((set, get) => ({
  // Initial state
  candles: [],
  processedCandles: [],
  isLoading: false,
  isProcessing: false,
  error: null,

  // Set raw candle data
  setCandles: (candles) => {
    set({
      candles,
      processedCandles: [], // Reset processed data
      error: null
    });
  },

  // Process features from raw candle data
  processFeatures: () => {
    const { candles, isProcessing } = get();

    if (candles.length === 0) {
      set({ error: 'No candle data to process' });
      return;
    }

    if (isProcessing) {
      return; // Already processing, don't start again
    }

    set({ isProcessing: true, error: null });

    try {
      // Extract OHLC arrays
      const open = candles.map(c => c.open);
      const high = candles.map(c => c.high);
      const low = candles.map(c => c.low);
      const close = candles.map(c => c.close);

      // Initialize features object
      const features: Partial<ProcessedCandleData>[] = new Array(candles.length);

      // 1. Moving Averages - Calculate distances (matching Python CSV output)
      const smaPeriods = [5, 10, 20, 50, 100, 200];
      const emaPeriods = [5, 10, 20, 50, 100, 200];

      // SMA
      for (const period of smaPeriods) {
        const sma = technicalIndicators.SMA.calculate({
          period,
          values: close
        });

        // Pad with undefined at the beginning
        const smaPadded = Array(period - 1).fill(undefined).concat(sma);

        for (let i = 0; i < candles.length; i++) {
          if (smaPadded[i] !== undefined) {
            const dist = ((close[i] - smaPadded[i]) / smaPadded[i]) * 100;
            features[i] = features[i] || {};
            const key = `dist_sma_${period}` as keyof ProcessedCandleData;
            features[i][key] = dist;
          }
        }
      }

      // EMA
      for (const period of emaPeriods) {
        const ema = technicalIndicators.EMA.calculate({
          period,
          values: close
        });

        const emaPadded = Array(period - 1).fill(undefined).concat(ema);

        for (let i = 0; i < candles.length; i++) {
          if (emaPadded[i] !== undefined) {
            const dist = ((close[i] - emaPadded[i]) / emaPadded[i]) * 100;
            features[i] = features[i] || {};
            const key = `dist_ema_${period}` as keyof ProcessedCandleData;
            features[i][key] = dist;
          }
        }
      }

      // 2. MACD (percentage)
      const macdData = technicalIndicators.MACD.calculate({
        values: close,
        fastPeriod: 12,
        slowPeriod: 26,
        signalPeriod: 9,
        SimpleMAOscillator: false,
        SimpleMASignal: false
      });

      const macdPadded = Array(25).fill(undefined).concat(macdData);
      for (let i = 0; i < candles.length; i++) {
        if (macdPadded[i] !== undefined) {
          const macd = macdPadded[i] as { MACD: number; signal: number; histogram: number };
          features[i] = features[i] || {};
          features[i].macd_pct = (macd.MACD / close[i]) * 100;
          features[i].macd_signal_pct = (macd.signal / close[i]) * 100;
          features[i].macd_hist_pct = (macd.histogram / close[i]) * 100;
        }
      }

      // 3. RSI
      const rsiPeriods = [14, 21];
      for (const period of rsiPeriods) {
        const rsi = technicalIndicators.RSI.calculate({
          period,
          values: close
        });

        const rsiPadded = Array(period - 1).fill(undefined).concat(rsi);
        for (let i = 0; i < candles.length; i++) {
          if (rsiPadded[i] !== undefined) {
            features[i] = features[i] || {};
            const key = `rsi_${period}` as keyof ProcessedCandleData;
            features[i][key] = rsiPadded[i];
          }
        }
      }

      // 4. ADX and Directional Indicators
      const adxData = technicalIndicators.ADX.calculate({
        period: 14,
        high,
        low,
        close
      });

      const adxPadded = Array(13).fill(undefined).concat(adxData);
      for (let i = 0; i < candles.length; i++) {
        if (adxPadded[i] !== undefined) {
          const adx = adxPadded[i] as { ADX: number; DI_Plus: number; DI_Minus: number };
          features[i] = features[i] || {};
          features[i].adx = adx.ADX;
          features[i].di_plus = adx.DI_Plus;
          features[i].di_minus = adx.DI_Minus;
        }
      }

      // 5. ATR (percentage)
      const atrData = technicalIndicators.ATR.calculate({
        period: 14,
        high,
        low,
        close
      });

      const atrPadded = Array(13).fill(undefined).concat(atrData);
      for (let i = 0; i < candles.length; i++) {
        if (atrPadded[i] !== undefined) {
          features[i] = features[i] || {};
          features[i].atr = atrPadded[i];
          features[i].atr_pct = (atrPadded[i] / close[i]) * 100;
        }
      }

      // 6. Bollinger Bands (width and position)
      const bbData = technicalIndicators.BollingerBands.calculate({
        period: 20,
        stdDev: 2,
        values: close
      });

      const bbPadded = Array(19).fill(undefined).concat(bbData);
      for (let i = 0; i < candles.length; i++) {
        if (bbPadded[i] !== undefined) {
          const bb = bbPadded[i] as { upper: number; middle: number; lower: number };
          features[i] = features[i] || {};
          features[i].bb_width_pct = ((bb.upper - bb.lower) / bb.middle) * 100;
          features[i].bb_position = (close[i] - bb.lower) / (bb.upper - bb.lower);
        }
      }

      // 7. Trend Strength
      const trendData = calculateTrendStrength(close, 20);
      for (let i = 0; i < candles.length; i++) {
        if (trendData.trendSlope[i] !== undefined) {
          features[i] = features[i] || {};
          features[i].trend_slope = trendData.trendSlope[i];
          features[i].trend_strength = trendData.trendStrength[i];
          features[i].trend_direction = trendData.trendDirection[i];
        }
      }

      // 8. Price Action Features
      for (let i = 0; i < candles.length; i++) {
        features[i] = features[i] || {};

        // Price change
        const priceChange = i > 0 ? ((close[i] - close[i - 1]) / close[i - 1]) * 100 : 0;
        features[i].price_change_pct = priceChange;
        features[i].price_change_abs = Math.abs(priceChange);

        // High-Low range
        features[i].hl_range_pct = ((high[i] - low[i]) / close[i]) * 100;

        // Body size
        features[i].body_size_pct = (Math.abs(close[i] - open[i]) / close[i]) * 100;

        // Shadows
        const upperShadow = high[i] - Math.max(open[i], close[i]);
        const lowerShadow = Math.min(open[i], close[i]) - low[i];
        features[i].upper_shadow_pct = (upperShadow / close[i]) * 100;
        features[i].lower_shadow_pct = (lowerShadow / close[i]) * 100;
      }

      // 9. Volatility
      for (const period of [10, 20]) {
        for (let i = period - 1; i < candles.length; i++) {
          const window = close.slice(i - period + 1, i + 1);
          const mean = window.reduce((a, b) => a + b, 0) / period;
          const std = Math.sqrt(
            window.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period
          );

          if (!features[i]) features[i] = {};
          const key = `volatility_${period}` as keyof ProcessedCandleData;
          features[i][key] = (std / mean) * 100;
        }
      }

      // Combine candles with features
      const processedCandles: ProcessedCandleData[] = candles.map((candle, index) => ({
        ...candle,
        ...features[index]
      }));

      // Filter out candles with undefined features (initial periods)
      // Only require basic OHLCV data - some indicators need initial periods
      const validCandles = processedCandles.filter(
        candle =>
          candle.open !== undefined &&
          candle.high !== undefined &&
          candle.low !== undefined &&
          candle.close !== undefined
      );

      set({
        processedCandles: validCandles,
        isProcessing: false
      });

    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Failed to process features',
        isProcessing: false
      });
    }
  },

  // Clear all data
  clearData: () => {
    set({
      candles: [],
      processedCandles: [],
      isLoading: false,
      isProcessing: false,
      error: null
    });
  },

  // Get the latest processed features
  getLatestFeatures: () => {
    const { processedCandles } = get();
    return processedCandles.length > 0
      ? processedCandles[processedCandles.length - 1]
      : null;
  },

  // Get all feature names
  getFeatureNames: () => {
    const { processedCandles } = get();
    if (processedCandles.length === 0) return [];

    // Check which features are actually present in any candle
    const allFeatureKeys = new Set<string>();
    processedCandles.forEach(candle => {
      Object.keys(candle).forEach(key => {
        if (!['timestamp', 'open', 'high', 'low', 'close', 'volume'].includes(key)) {
          allFeatureKeys.add(key);
        }
      });
    });

    return Array.from(allFeatureKeys);
  },

  // Get expected feature count (should match Python implementation)
  getExpectedFeatureCount: () => {
    return 36; // Total features found in Python CSV output
  }
}));