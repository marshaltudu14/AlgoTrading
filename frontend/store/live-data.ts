
import { create } from 'zustand';

interface TickData {
  timestamp: string;
  price: number;
  volume: number;
  open: number;
  high: number;
  low: number;
}

interface Position {
    instrument: string;
    direction: string;
    entryPrice: number;
    quantity: number;
    stopLoss?: number;
    targetPrice?: number;
    currentPnl: number;
    tradeType: string;
    isOpen: boolean;
    exitPrice?: number;
    pnl?: number;
}

interface LiveDataState {
  lastTick: TickData | null;
  activePosition: Position | null;
  setLastTick: (tick: TickData) => void;
  setActivePosition: (position: Position | null) => void;
}

export const useLiveDataStore = create<LiveDataState>((set) => ({
  lastTick: null,
  activePosition: null,
  setLastTick: (tick) => set({ lastTick: tick }),
  setActivePosition: (position) => set({ activePosition: position }),
}));
