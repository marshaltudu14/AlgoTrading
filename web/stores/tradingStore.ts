import { create } from 'zustand';
import { INSTRUMENTS, TIMEFRAMES, DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME } from '@/config/instruments';
import { CandleData, useCandleStore } from './candleStore';
import { useAuthStore } from './authStore';

// Remove local CandleData interface since it's imported from candleStore

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
  isLoading: true, // Start with loading state initially to show loading on first load
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
    const candleStore = useCandleStore.getState();

    // Reset error state but maintain loading state
    set({ isLoading: true });

    try {
      const apiUrl = `/api/candle-data/${selectedInstrument.exchangeSymbol}/${selectedTimeframe.name}`;

      const response = await fetch(apiUrl);

      if (!response.ok) {
        throw new Error(`Failed to fetch candle data: ${response.statusText}`);
      }

      const result = await response.json();

      // Let middleware handle auth errors - no redirect logic in stores

      if (result.success && result.data) {
        // Set candles in candleStore
        candleStore.setCandles(result.data);

        // Also keep them in tradingStore for backward compatibility
        set({ candleData: result.data, error: null });
      } else {
        throw new Error(result.error || 'No data available');
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch candle data';

      // Check if it's an authentication error
      if (errorMessage.includes('401') || errorMessage.includes('Unauthorized')) {
        
        // Clear auth state in authStore
        const authStore = useAuthStore.getState();
        authStore.setError('Authentication expired. Please login again.');

        // Don't set the error in tradingStore to avoid UI clutter
        // The AuthGuard will handle the redirect
      } else {
        set({ error: errorMessage });
      }
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