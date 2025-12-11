import { create } from 'zustand';
import { DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME } from '@/config/instruments';

export type DurationUnit = 'days' | 'months' | 'years';

interface BacktestConfig {
  symbol: string;
  timeframe: string;
  duration: number;
  durationUnit: DurationUnit;
  durationInDays: number;
  targetPL: number;
  stopLossPL: number;
  initialCapital: number;
  doubleLotSize: boolean;
  trailSL: boolean;
}

interface BacktestState {
  // Configuration
  config: BacktestConfig;

  // UI State
  isDialogOpen: boolean;
  isBacktestMode: boolean;
  isBacktestLoading: boolean;
  backtestError: string | null;

  // Backtest Results
  backtestResults: {
    trades: unknown[];
    metrics: unknown;
    candleData?: unknown[];
    processedCandles?: unknown[];
  } | null;

  // Actions
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: string) => void;
  setDuration: (duration: number) => void;
  setDurationUnit: (unit: DurationUnit) => void;
  setTargetPL: (pl: number) => void;
  setStopLossPL: (pl: number) => void;
  setInitialCapital: (capital: number) => void;
  setDoubleLotSize: (double: boolean) => void;
  setTrailSL: (trail: boolean) => void;
  openDialog: () => void;
  closeDialog: () => void;
  startBacktest: () => void;
  stopBacktest: () => void;
  setBacktestLoading: (loading: boolean) => void;
  setBacktestError: (error: string | null) => void;
  setBacktestResults: (results: {
    trades: unknown[];
    metrics: {
      totalPL: number;
      totalTrades: number;
      winRate: number;
      maxDrawdown: number;
      sharpeRatio: number;
      equityCurve: unknown[];
    };
    candleData?: unknown[];
    processedCandles?: unknown[];
  }) => void;
  calculateDurationInDays: (duration: number, unit: DurationUnit) => number;
  resetToDefaults: () => void;
}

const DEFAULT_CONFIG: BacktestConfig = {
  symbol: DEFAULT_INSTRUMENT.exchangeSymbol,
  timeframe: DEFAULT_TIMEFRAME.name,
  duration: 1,
  durationUnit: 'months',
  durationInDays: 30,
  targetPL: 500,
  stopLossPL: -250,
  initialCapital: 25000,
  doubleLotSize: true,
  trailSL: true,
};

export const useBacktestStore = create<BacktestState>((set, get) => ({
  // Initial state
  config: { ...DEFAULT_CONFIG },
  isDialogOpen: false,
  isBacktestMode: false,
  isBacktestLoading: false,
  backtestError: null,
  backtestResults: null,

  setSymbol: (symbol: string) => {
    set(state => ({
      config: { ...state.config, symbol }
    }));
  },

  setTimeframe: (timeframe: string) => {
    set(state => ({
      config: { ...state.config, timeframe }
    }));
  },

  setDuration: (duration: number) => {
    set(state => {
      const durationInDays = get().calculateDurationInDays(duration, state.config.durationUnit);
      return {
        config: { ...state.config, duration, durationInDays }
      };
    });
  },

  setDurationUnit: (durationUnit: DurationUnit) => {
    set(state => {
      const durationInDays = get().calculateDurationInDays(state.config.duration, durationUnit);
      return {
        config: { ...state.config, durationUnit, durationInDays }
      };
    });
  },

  setTargetPL: (targetPL: number) => {
    set(state => ({
      config: { ...state.config, targetPL }
    }));
  },

  setStopLossPL: (stopLossPL: number) => {
    set(state => ({
      config: { ...state.config, stopLossPL }
    }));
  },

  setInitialCapital: (initialCapital: number) => {
    set(state => ({
      config: { ...state.config, initialCapital }
    }));
  },

  setDoubleLotSize: (doubleLotSize: boolean) => {
    set(state => ({
      config: { ...state.config, doubleLotSize }
    }));
  },

  setTrailSL: (trailSL: boolean) => {
    set(state => ({
      config: { ...state.config, trailSL }
    }));
  },

  openDialog: () => {
    set({ isDialogOpen: true });
  },

  closeDialog: () => {
    set({ isDialogOpen: false });
  },

  startBacktest: () => {
    set({
      isDialogOpen: false,
      isBacktestMode: true,
      isBacktestLoading: true,
      backtestError: null,
      backtestResults: null
    });
  },

  stopBacktest: () => {
    set({
      isBacktestMode: false,
      isBacktestLoading: false,
      backtestError: null,
      backtestResults: null,
      config: { ...DEFAULT_CONFIG }
    });
  },

  setBacktestLoading: (loading: boolean) => {
    set({ isBacktestLoading: loading });
  },

  setBacktestError: (error: string | null) => {
    set({ backtestError: error, isBacktestLoading: false });
  },

  setBacktestResults: (results: {
    trades: unknown[];
    metrics: {
      totalPL: number;
      totalTrades: number;
      winRate: number;
      maxDrawdown: number;
      sharpeRatio: number;
      equityCurve: unknown[];
    };
    candleData?: unknown[];
    processedCandles?: unknown[];
  }) => {
    set({
      backtestResults: results,
      isBacktestLoading: false,
      backtestError: null
    });
  },

  calculateDurationInDays: (duration: number, unit: DurationUnit): number => {
    switch (unit) {
      case 'days':
        return duration;
      case 'months':
        return duration * 30;
      case 'years':
        return duration * 365;
      default:
        return duration;
    }
  },

  resetToDefaults: () => {
    const durationInDays = get().calculateDurationInDays(
      DEFAULT_CONFIG.duration,
      DEFAULT_CONFIG.durationUnit
    );
    set({
      config: { ...DEFAULT_CONFIG, durationInDays },
      isBacktestMode: false,
      isBacktestLoading: false,
      backtestError: null,
      backtestResults: null
    });
  },
}));