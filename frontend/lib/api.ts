/**
 * API Client for AlgoTrading Backend
 */

// For Vercel deployment, use relative URLs. For local development, use localhost:8000
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ||
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost'
    ? '' // Use relative URLs in production (Vercel)
    : 'http://localhost:8000' // Use localhost in development
  )

interface LoginRequest {
  app_id: string
  secret_key: string
  redirect_uri: string
  fy_id: string
  pin: string
  totp_secret: string
}

interface BacktestRequest {
  instrument: string
  timeframe: string
  duration: number
  initial_capital: number
}

interface LiveTradingRequest {
  instrument: string
  timeframe: string
  option_strategy?: string
}

interface ApiResponse<T = unknown> {
  success?: boolean
  data?: T
  message?: string
  error?: string
}

interface FundLimitItem {
  id: number;
  title: string;
  equityAmount: number;
  commodityAmount: number;
}

interface FundsResponse {
  code: number;
  message: string;
  s: string;
  fund_limit: FundLimitItem[];
  todayPnL: number; // Added for direct access
  totalFunds: number; // Added for direct access
}

interface UserProfile {
  user_id: string
  name: string
  capital: number
  login_time: string
}

class ApiClient {
  private token: string | null = null

  constructor() {
    // Token will be managed by HTTP-only cookies
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`
    console.log('Making API request to:', url, 'with options:', options)

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(options.headers as Record<string, string>),
    }

    // Add authorization header if token exists
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`
    }

    const config: RequestInit = {
      ...options,
      headers,
      credentials: 'include', // Include cookies in requests
    }

    console.log('Request config:', config)

    try {
      const response = await fetch(url, config)
      console.log('Response status:', response.status)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(`${response.status}: ${errorData.detail || response.statusText}`)
      }

      const data = await response.json()
      console.log('Response data:', data)
      return data
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error)
      throw error
    }
  }

  // Authentication
  async login(credentials: LoginRequest): Promise<{ success: boolean; message: string }> {
    const response = await this.request<{ success: boolean; message: string }>('/api/login', {
      method: 'POST',
      body: JSON.stringify(credentials),
    })

    // Token is now set as HTTP-only cookie by the server
    // We'll mark as authenticated if login was successful
    if (response.success) {
      this.token = 'authenticated' // Placeholder since we can't access HTTP-only cookie
    }

    return response
  }

  async logout(): Promise<void> {
    this.token = null
  }

  // Health check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    return this.request('/api/health')
  }

  // Profile
  async getProfile(): Promise<UserProfile> {
    return this.request('/api/profile')
  }

  // Funds
  async getFunds(): Promise<FundsResponse> {
    return this.request('/api/funds')
  }

  // Metrics
  async getMetrics(): Promise<MetricsResponse> {
    return this.request('/api/metrics')
  }

  // Configuration
  async getConfig(): Promise<ConfigResponse> {
    return this.request('/api/config')
  }

  // Historical Data
  async getHistoricalData(instrument: string, timeframe: string): Promise<CandlestickData[]> {
    if (!instrument || !timeframe) {
      throw new Error('Instrument and timeframe parameters are required')
    }
    
    const params = new URLSearchParams({
      instrument,
      timeframe
    })
    
    return this.request(`/api/historical-data?${params}`)
  }

  // Backtesting
  async startBacktest(request: BacktestRequest): Promise<{
    backtest_id: string
    status: string
    message: string
  }> {
    return this.request('/api/backtest', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // Live Trading
  async startLiveTrading(request: LiveTradingRequest): Promise<{
    status: string
    message: string
  }> {
    return this.request('/api/live/start', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  async stopLiveTrading(): Promise<{
    status: string
    message: string
  }> {
    return this.request('/api/live/stop', {
      method: 'POST',
    })
  }

  // WebSocket connections
  createBacktestWebSocket(backtestId: string): WebSocket {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/backtest/${backtestId}`
    console.log('Creating WebSocket connection to:', wsUrl)
    return new WebSocket(wsUrl)
  }

  createLiveWebSocket(userId: string): WebSocket {
    const wsUrl = `${API_BASE_URL.replace('http', 'ws')}/ws/live/${userId}`
    return new WebSocket(wsUrl)
  }

  // Utility methods
}

// Create singleton instance
export const apiClient = new ApiClient()

// Export types
interface MetricsResponse {
  totalTrades: number;
  winRate: number;
  lastTradeTime: string;
}

interface Instrument {
  name: string;
  symbol: string;
  'exchange-symbol': string;
  type: string;
  lot_size: number;
  tick_size: number;
  option_premium_range?: [number, number];
}

interface ConfigResponse {
  instruments: Instrument[];
  timeframes: string[];
}

interface CandlestickData {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type {
  LoginRequest,
  BacktestRequest,
  LiveTradingRequest,
  ApiResponse,
  UserProfile,
  FundsResponse,
  MetricsResponse,
  Instrument,
  ConfigResponse,
  CandlestickData,
}

// Export utility functions
export const formatApiError = (error: unknown): string => {
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message)
  }
  if (typeof error === 'string') {
    return error
  }
  return 'An unexpected error occurred'
}

export const isApiError = (error: unknown): error is Error => {
  return error instanceof Error || (error !== null && typeof error === 'object' && 'message' in error)
}
