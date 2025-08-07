import { create } from 'zustand';

interface TickData {
  timestamp: string;
  price: number;
  volume: number;
  open: number;
  high: number;
  low: number;
}

export interface Position {
    instrument: string;
    direction: 'Long' | 'Short';
    entryPrice: number;
    quantity: number;
    stopLoss?: number;
    targetPrice?: number;
    currentPnl: number;
    tradeType: 'Automated' | 'Manual';
    isOpen: boolean;
    entryTime: string;
    exitPrice?: number;
    exitTime?: string;
    pnl?: number;
}

interface LiveDataState {
  isConnected: boolean;
  lastTick: TickData | null;
  activePosition: Position | null;
  status: any | null;
  setIsConnected: (isConnected: boolean) => void;
  setLastTick: (tick: TickData) => void;
  setActivePosition: (position: Position | null) => void;
  setStatus: (status: any) => void;
}

export const useLiveDataStore = create<LiveDataState>((set) => ({
  isConnected: false,
  lastTick: null,
  activePosition: null,
  status: null,
  setIsConnected: (isConnected) => set({ isConnected }),
  setLastTick: (tick) => set({ lastTick: tick }),
  setActivePosition: (position) => set({ activePosition: position }),
  setStatus: (status) => set({ status }),
}));