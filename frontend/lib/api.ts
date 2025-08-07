import { formatApiError } from './utils';

export { formatApiError };

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export interface Instrument {
  symbol: string;
  name: string;
  instrument_type: string;
}

export interface Config {
  instruments: Instrument[];
  timeframes: string[];
}

export interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Metrics {
  todayPnL: number;
  totalTrades: number;
  winRate: number;
  lastTradeTime: string;
}

async function fetcher(url: string, options: RequestInit = {}) {
  const response = await fetch(url, {
    ...options,
    credentials: 'include',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(`${response.status}: ${errorData.detail || 'An unknown error occurred'}`);
  }

  return response.json();
}

export const apiClient = {
  getConfig: async (): Promise<Config> => {
    return fetcher(`${API_BASE_URL}/config`);
  },

  getHistoricalData: async (instrument: string, timeframe: string): Promise<CandlestickData[]> => {
    if (!instrument || !timeframe) {
      throw new Error('Instrument and timeframe parameters are required');
    }
    return fetcher(`${API_BASE_URL}/historical-data?instrument=${instrument}&timeframe=${timeframe}`);
  },

  getMetrics: async (): Promise<Metrics> => {
    return fetcher(`${API_BASE_URL}/metrics`);
  },

  manualTrade: async (data: any): Promise<any> => {
    return fetcher(`${API_BASE_URL}/manual-trade`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
};