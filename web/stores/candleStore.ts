import { create } from 'zustand';
import { SMA, EMA, RSI, MACD, BollingerBands, ATR } from 'trading-signals';

// Types for candle data
export interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
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

  // Volume
  volume?: number;
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


export const useCandleStore = create<CandleState>((set, get) => ({
  // Initial state
  candles: [],
  processedCandles: [],
  isLoading: false,
  isProcessing: false,
  error: null,

  // Set raw candle data
  setCandles: (candles) => {
    // Filter out all rows with any NaN/null values before processing
    const cleanCandles = candles.filter(candle =>
      candle.timestamp &&
      !isNaN(candle.timestamp) &&
      !isNaN(candle.open) &&
      !isNaN(candle.high) &&
      !isNaN(candle.low) &&
      !isNaN(candle.close) &&
      candle.open > 0 &&
      candle.high > 0 &&
      candle.low > 0 &&
      candle.close > 0 &&
      candle.high >= candle.low
    );

    
    set({
      candles: cleanCandles,
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
      // Filter out invalid candles before processing
      const validCandles = candles.filter(candle =>
        candle.timestamp &&
        !isNaN(candle.timestamp) &&
        !isNaN(candle.open) &&
        !isNaN(candle.high) &&
        !isNaN(candle.low) &&
        !isNaN(candle.close) &&
        candle.open > 0 &&
        candle.high > 0 &&
        candle.low > 0 &&
        candle.close > 0 &&
        candle.high >= candle.low
      );

            
      // Extract OHLC arrays from valid candles only
      const open = validCandles.map(c => c.open);
      const high = validCandles.map(c => c.high);
      const low = validCandles.map(c => c.low);
      const close = validCandles.map(c => c.close);

      // Initialize features object with same length as valid candles
      const features: Partial<ProcessedCandleData>[] = new Array(validCandles.length);

      // 1. Moving Averages - Calculate distances (matching Python CSV output)
      // Backend uses [5, 10, 20, 50, 100, 200] for both SMA and EMA (from settings.yaml)
      const smaPeriods = [5, 10, 20, 50, 100, 200];
      const emaPeriods = [5, 10, 20, 50, 100, 200];

      // SMA
      for (const period of smaPeriods) {
        const sma = new SMA(period);
        const smaValues: (number | null)[] = [];

        // Calculate SMA for each candle
        for (const price of close) {
          const result = sma.add(price);
          smaValues.push(result);
        }

        // Pad with undefined at the beginning
        const smaPadded = Array(period - 1).fill(undefined).concat(smaValues);

        for (let i = 0; i < validCandles.length; i++) {
          if (smaPadded[i] !== undefined && smaPadded[i] !== null) {
            const dist = ((close[i] - smaPadded[i]) / smaPadded[i]) * 100;
            features[i] = features[i] || {};
            const key = `dist_sma_${period}` as keyof ProcessedCandleData;
            features[i][key] = dist;
          }
        }
      }

      // EMA
      for (const period of emaPeriods) {
        const ema = new EMA(period);
        const emaValues: (number | null)[] = [];

        // Calculate EMA for each candle
        for (const price of close) {
          const result = ema.add(price);
          emaValues.push(result);
        }

        const emaPadded = Array(period - 1).fill(undefined).concat(emaValues);

        for (let i = 0; i < validCandles.length; i++) {
          if (emaPadded[i] !== undefined && emaPadded[i] !== null) {
            const dist = ((close[i] - emaPadded[i]) / emaPadded[i]) * 100;
            features[i] = features[i] || {};
            const key = `dist_ema_${period}` as keyof ProcessedCandleData;
            features[i][key] = dist;
          }
        }
      }

      // 2. MACD (percentage)
      const macd = new MACD(new EMA(12), new EMA(26), new EMA(9));

      const macdData: Array<{macd?: number, signal?: number, histogram?: number} | null> = [];
      for (const price of close) {
        const result = macd.add(price);
        macdData.push(result);
      }

      // MACD has a warmup period of 26+9 = 35
      const macdPadded = Array(35).fill(undefined).concat(macdData);
      for (let i = 0; i < validCandles.length; i++) {
        if (macdPadded[i] !== undefined && macdPadded[i] !== null) {
          const macdResult = macdPadded[i] as { macd: number; signal: number; histogram: number };
          features[i] = features[i] || {};

          features[i].macd_pct = (macdResult.macd / close[i]) * 100;
          features[i].macd_signal_pct = (macdResult.signal / close[i]) * 100;
          features[i].macd_hist_pct = (macdResult.histogram / close[i]) * 100;
        }
      }

      // 3. RSI
      const rsiPeriods = [14, 21];
      for (const period of rsiPeriods) {
        const rsi = new RSI(period);
        const rsiValues: (number | null)[] = [];

        for (const price of close) {
          const result = rsi.add(price);
          rsiValues.push(result);
        }

        const rsiPadded = Array(period - 1).fill(undefined).concat(rsiValues);
        for (let i = 0; i < validCandles.length; i++) {
          if (rsiPadded[i] !== undefined && rsiPadded[i] !== null) {
            features[i] = features[i] || {};
            const key = `rsi_${period}` as keyof ProcessedCandleData;
            features[i][key] = rsiPadded[i];
          }
        }
      }

  
      // 4. ATR (percentage)
      const atr = new ATR(14);
      const atrValues: (number | null)[] = [];

      for (let i = 0; i < validCandles.length; i++) {
        const result = atr.add({
          high: high[i],
          low: low[i],
          close: close[i]
        });
        atrValues.push(result);
      }

      const atrPadded = Array(14).fill(undefined).concat(atrValues);
      for (let i = 0; i < validCandles.length; i++) {
        if (atrPadded[i] !== undefined && atrPadded[i] !== null) {
          features[i] = features[i] || {};
          features[i].atr = atrPadded[i];
          features[i].atr_pct = (atrPadded[i] / close[i]) * 100;
        }
      }

      // 5. Bollinger Bands (width and position)
      const bb = new BollingerBands(20, 2);
      const bbData: (null | {upper: number, middle: number, lower: number})[] = [];

      for (const price of close) {
        const result = bb.add(price);
        bbData.push(result);
      }

      const bbPadded = Array(20).fill(undefined).concat(bbData);
      for (let i = 0; i < validCandles.length; i++) {
        if (bbPadded[i] !== undefined && bbPadded[i] !== null) {
          const bbResult = bbPadded[i] as { upper: number; middle: number; lower: number };
          features[i] = features[i] || {};
          features[i].bb_width_pct = ((bbResult.upper - bbResult.lower) / bbResult.middle) * 100;
          features[i].bb_position = (close[i] - bbResult.lower) / (bbResult.upper - bbResult.lower);
        }
      }

      // 7. Trend Strength (exactly matching backend calculation)
      const trendPeriod = 20;
      const trendSlope: number[] = [];
      const trendStrength: number[] = [];

      for (let i = trendPeriod - 1; i < close.length; i++) {
        const window = close.slice(i - trendPeriod + 1, i + 1);
        const x = Array.from({ length: trendPeriod }, (_, idx) => idx);

        // Calculate slope using linear regression (exactly like backend)
        const n = trendPeriod;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = window.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, idx) => sum + xi * window[idx], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);

        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        trendSlope.push(slope);

        // Calculate R-squared
        const meanY = sumY / n;
        let ssRes = 0;
        let ssTot = 0;
        for (let j = 0; j < trendPeriod; j++) {
          const yPred = slope * x[j] + (sumY - slope * sumX) / n;
          ssRes += Math.pow(window[j] - yPred, 2);
          ssTot += Math.pow(window[j] - meanY, 2);
        }

        const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
        trendStrength.push(rSquared);
      }

      const trendDirection = trendSlope.map(s => s > 0 ? 1 : -1);

      const trendSlopePadded = Array(trendPeriod - 1).fill(undefined).concat(trendSlope);
      const trendStrengthPadded = Array(trendPeriod - 1).fill(undefined).concat(trendStrength);
      const trendDirectionPadded = Array(trendPeriod - 1).fill(undefined).concat(trendDirection);

      for (let i = 0; i < candles.length; i++) {
        if (trendSlopePadded[i] !== undefined) {
          features[i] = features[i] || {};
          features[i].trend_slope = trendSlopePadded[i];
          features[i].trend_strength = trendStrengthPadded[i];
          features[i].trend_direction = trendDirectionPadded[i];
        }
      }

      // 8. Price Action Features (exactly matching backend)
      for (let i = 0; i < candles.length; i++) {
        features[i] = features[i] || {};

        // Price change (exactly like backend: pct_change() * 100)
        const priceChange = i > 0 ? ((close[i] - close[i - 1]) / close[i - 1]) * 100 : 0;
        features[i].price_change_pct = priceChange;
        features[i].price_change_abs = Math.abs(priceChange);

        // High-Low range (exactly like backend)
        features[i].hl_range_pct = ((high[i] - low[i]) / close[i]) * 100;

        // Body size (exactly like backend)
        features[i].body_size_pct = Math.abs(close[i] - open[i]) / close[i] * 100;

        // Shadows (exactly like backend)
        const upperShadow = high[i] - Math.max(open[i], close[i]);
        const lowerShadow = Math.min(open[i], close[i]) - low[i];
        features[i].upper_shadow_pct = (upperShadow / close[i]) * 100;
        features[i].lower_shadow_pct = (lowerShadow / close[i]) * 100;
      }

      // 9. Volatility (exactly matching backend: std / mean * 100)
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

  
    
      // Combine valid candles with features
      const processedCandles: ProcessedCandleData[] = validCandles.map((candle, index) => {
        const featureData = features[index];
        const processedCandle: ProcessedCandleData = {
          timestamp: candle.timestamp,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          // Include volume if it exists (backend CSV doesn't have volume)
          ...(candle.volume !== undefined ? { volume: candle.volume } : {}),
          // Add all features (some may be undefined)
          ...(featureData || {})
        };

        return processedCandle;
      });

      // Filter out rows where ANY column is null/undefined/NaN (strict filtering like backend.dropna())
      const cleanProcessedCandles = processedCandles.filter(candle => {
        // Check ALL properties in the candle
        for (const key in candle) {
          const value = candle[key as keyof ProcessedCandleData];

          // Skip undefined volume field if not present
          if (key === 'volume' && value === undefined) {
            continue;
          }

          // For numeric fields, check for null/undefined/NaN
          if (typeof value === 'number') {
            if (isNaN(value) || value === null || value === undefined) {
              return false;
            }
          }
          // For non-numeric fields, check for null/undefined
          else if (value === null || value === undefined) {
            return false;
          }
        }

        // Additional OHLC validation
        if (candle.open <= 0 || candle.high <= 0 || candle.low <= 0 || candle.close <= 0) {
          return false;
        }

        // Ensure high >= low
        if (candle.high < candle.low) {
          return false;
        }

        return true;
      });

      set({
        processedCandles: cleanProcessedCandles,
        isProcessing: false
      });

    } catch (error) {
      console.error('Processing failed:', error);
      console.error('Error details:', error instanceof Error ? error.stack : error);
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

    return Array.from(allFeatureKeys).sort();
  },

  // Get expected feature count (actual features being generated)
  getExpectedFeatureCount: () => {
    const featureNames = get().getFeatureNames();
    return featureNames.length;
  }
}));