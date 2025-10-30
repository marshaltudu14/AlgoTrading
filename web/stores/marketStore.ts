import { create } from 'zustand';
import { CandleData, Timeframe, ChartDataPoint } from '@/types/fyers';
import { MarketDataService } from '@/lib/marketData';

interface MarketState {
  // State
  currentSymbol: string;
  currentTimeframe: Timeframe;
  candles: CandleData[];
  chartData: ChartDataPoint[];
  isLoading: boolean;
  error: string | null;
  lastUpdated: number | null;

  // Market data
  currentPrice: number | null;
  priceChange: number | null;
  priceChangePercent: number | null;
  dayHigh: number | null;
  dayLow: number | null;
  volume: number | null;

  // Actions
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: Timeframe) => void;
  fetchHistoricalData: (accessToken: string, symbol?: string, timeframe?: Timeframe, daysBack?: number) => Promise<void>;
  fetchQuote: (accessToken: string, symbol?: string) => Promise<void>;
  setError: (error: string | null) => void;
  clearData: () => void;
  refreshData: (accessToken: string) => Promise<void>;
}

export const useMarketStore = create<MarketState>((set, get) => ({
  // Initial state
  currentSymbol: MarketDataService.SYMBOLS.NIFTY,
  currentTimeframe: '15',
  candles: [],
  chartData: [],
  isLoading: false,
  error: null,
  lastUpdated: null,
  currentPrice: null,
  priceChange: null,
  priceChangePercent: null,
  dayHigh: null,
  dayLow: null,
  volume: null,

  // Actions
  setSymbol: (symbol: string) => {
    set({ currentSymbol: symbol });
  },

  setTimeframe: (timeframe: Timeframe) => {
    set({ currentTimeframe: timeframe });
  },

  fetchHistoricalData: async (accessToken: string, symbol?: string, timeframe?: Timeframe, daysBack?: number) => {
    try {
      const state = get();
      const targetSymbol = symbol || state.currentSymbol;
      const targetTimeframe = timeframe || state.currentTimeframe;

      set({ isLoading: true, error: null });

      const response = await fetch('/api/market/data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          accessToken,
          symbol: targetSymbol,
          timeframe: targetTimeframe,
          daysBack: daysBack || 30,
        }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Failed to fetch historical data');
      }

      // Transform candles to chart data format
      const chartData: ChartDataPoint[] = data.data.candles.map((candle: CandleData) => ({
        time: candle.timestamp,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        value: candle.volume,
      }));

      set({
        candles: data.data.candles,
        chartData,
        currentSymbol: targetSymbol,
        currentTimeframe: targetTimeframe,
        lastUpdated: Date.now(),
        error: null,
      });

      // Calculate current price and change from the last candle
      if (data.data.candles.length > 0) {
        const lastCandle = data.data.candles[data.data.candles.length - 1];
        const previousCandle = data.data.candles[data.data.candles.length - 2];

        const currentPrice = lastCandle.close;
        const priceChange = previousCandle ? currentPrice - previousCandle.close : 0;
        const priceChangePercent = previousCandle ? (priceChange / previousCandle.close) * 100 : 0;

        set({
          currentPrice,
          priceChange,
          priceChangePercent,
          dayHigh: Math.max(...data.data.candles.map((c: CandleData) => c.high)),
          dayLow: Math.min(...data.data.candles.map((c: CandleData) => c.low)),
          volume: data.data.candles.reduce((sum: number, c: CandleData) => sum + c.volume, 0),
        });
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch market data';
      set({ error: errorMessage });
      console.error('Error fetching historical data:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  fetchQuote: async (accessToken: string, symbol?: string) => {
    try {
      const state = get();
      const targetSymbol = symbol || state.currentSymbol;

      const marketService = new MarketDataService();
      const quoteData = await marketService.getQuote(accessToken, targetSymbol);

      if (quoteData && quoteData.length > 0) {
        const quote = quoteData[0];
        const currentPrice = quote.v || quote.ltp || 0;
        const previousClose = quote.pc || 0;
        const priceChange = currentPrice - previousClose;
        const priceChangePercent = previousClose > 0 ? (priceChange / previousClose) * 100 : 0;

        set({
          currentPrice,
          priceChange,
          priceChangePercent,
          dayHigh: quote.h || state.dayHigh,
          dayLow: quote.l || state.dayLow,
          volume: quote.v || state.volume,
        });
      }

    } catch (error) {
      console.error('Error fetching quote:', error);
      // Don't set error state for quote failures as it's not critical
    }
  },

  setError: (error: string | null) => {
    set({ error });
  },

  clearData: () => {
    set({
      candles: [],
      chartData: [],
      currentPrice: null,
      priceChange: null,
      priceChangePercent: null,
      dayHigh: null,
      dayLow: null,
      volume: null,
      error: null,
      lastUpdated: null,
    });
  },

  refreshData: async (accessToken: string) => {
    const { currentSymbol, currentTimeframe } = get();
    await get().fetchHistoricalData(accessToken, currentSymbol, currentTimeframe);
    await get().fetchQuote(accessToken, currentSymbol);
  },
}));

// Selectors for commonly used state combinations
export const useMarketData = () => {
  const store = useMarketStore();

  return {
    // Basic market state
    symbol: store.currentSymbol,
    timeframe: store.currentTimeframe,
    candles: store.candles,
    chartData: store.chartData,
    isLoading: store.isLoading,
    error: store.error,
    lastUpdated: store.lastUpdated,

    // Price information
    currentPrice: store.currentPrice,
    priceChange: store.priceChange,
    priceChangePercent: store.priceChangePercent,
    dayHigh: store.dayHigh,
    dayLow: store.dayLow,
    volume: store.volume,

    // Actions
    setSymbol: store.setSymbol,
    setTimeframe: store.setTimeframe,
    fetchHistoricalData: store.fetchHistoricalData,
    fetchQuote: store.fetchQuote,
    refreshData: store.refreshData,
    setError: store.setError,
    clearData: store.clearData,

    // Computed values
    isPositive: store.priceChange !== null && store.priceChange > 0,
    isNegative: store.priceChange !== null && store.priceChange < 0,
    hasData: store.chartData.length > 0,
  };
};