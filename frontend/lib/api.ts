import { formatApiError } from './utils';

export { formatApiError };

const API_BASE_URL = 'http://localhost:8000/api';

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

export interface LoginRequest {
  app_id: string;
  secret_key: string;
  redirect_uri: string;
  fy_id: string;
  pin: string;
  totp_secret: string;
}

export interface LoginResponse {
  success: boolean;
  message: string;
}

export interface ManualTradeRequest {
  instrument: string;
  direction: string;
  quantity: number;
  stopLoss?: number;
  target?: number;
}

export interface ManualTradeResponse {
  success: boolean;
  message: string;
}

export interface BacktestRequest {
  instrument: string;
  timeframe: string;
  duration: number;
  initial_capital: number;
}

export interface BacktestResponse {
  backtest_id: string;
  message: string;
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
  login: async (data: LoginRequest): Promise<LoginResponse> => {
    return fetcher(`${API_BASE_URL}/login`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  getProfile: async (): Promise<{ user_id: string; name: string; capital: number }> => {
    return fetcher(`${API_BASE_URL}/profile`);
  },

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

  manualTrade: async (data: ManualTradeRequest): Promise<ManualTradeResponse> => {
    return fetcher(`${API_BASE_URL}/manual-trade`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  startLiveTrading: async (data: {
    instrument: string;
    timeframe: string;
    option_strategy?: string;
    trading_mode: 'paper' | 'real';
  }): Promise<{ message: string; status: string }> => {
    return fetcher(`${API_BASE_URL}/live/start`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  stopLiveTrading: async (): Promise<{ message: string; status: string }> => {
    return fetcher(`${API_BASE_URL}/live/stop`, {
      method: 'POST',
    });
  },

  startBacktest: async (data: BacktestRequest): Promise<BacktestResponse> => {
    return fetcher(`${API_BASE_URL}/backtest/start`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },

  createBacktestWebSocket: (backtestId: string): WebSocket => {
    const wsUrl = `ws://localhost:8000/ws/backtest/${backtestId}`;
    return new WebSocket(wsUrl);
  },

  createLiveWebSocket: (userId: string): WebSocket => {
    const wsUrl = `ws://localhost:8000/ws/live/${userId}`;
    return new WebSocket(wsUrl);
  },
};