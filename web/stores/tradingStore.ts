import { create } from 'zustand';
import { INSTRUMENTS, TIMEFRAMES, DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME } from '@/config/instruments';

interface CandleData {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface TradingState {
  // Current selection
  selectedInstrument: typeof INSTRUMENTS[0];
  selectedTimeframe: typeof TIMEFRAMES[0];

  // Data
  candleData: CandleData[];
  isLoading: boolean;
  error: string | null;

  // Actions
  setSelectedInstrument: (instrument: typeof INSTRUMENTS[0]) => void;
  setSelectedTimeframe: (timeframe: typeof TIMEFRAMES[0]) => void;
  fetchCandleData: () => Promise<void>;
  setError: (error: string | null) => void;
  refreshData: () => Promise<void>;
}

export const useTradingStore = create<TradingState>((set, get) => ({
  // Initial state
  selectedInstrument: DEFAULT_INSTRUMENT,
  selectedTimeframe: DEFAULT_TIMEFRAME,
  candleData: [],
  isLoading: false,
  error: null,

  setSelectedInstrument: (instrument) => {
    set({ selectedInstrument: instrument });
    // Auto-fetch data when instrument changes
    get().fetchCandleData();
  },

  setSelectedTimeframe: (timeframe) => {
    set({ selectedTimeframe: timeframe });
    // Auto-fetch data when timeframe changes
    get().fetchCandleData();
  },

  fetchCandleData: async () => {
    const { selectedInstrument, selectedTimeframe } = get();

    set({ isLoading: true, error: null });

    try {
      const response = await fetch(
        `/api/candle-data/${selectedInstrument.exchangeSymbol}/${selectedTimeframe.name}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch candle data: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success && result.data) {
        set({ candleData: result.data, error: null });
      } else {
        throw new Error(result.error || 'No data available');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch candle data';
      set({ error: errorMessage, candleData: [] });
    } finally {
      set({ isLoading: false });
    }
  },

  setError: (error) => {
    set({ error });
  },

  refreshData: async () => {
    await get().fetchCandleData();
  },
}));