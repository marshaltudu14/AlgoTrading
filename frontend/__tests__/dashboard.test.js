/**
 * Component integration tests for the Dashboard page
 * 
 * These tests verify that the dashboard correctly integrates with the
 * TradingChart component when users select instruments and timeframes.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { toast } from 'sonner'
import DashboardPage from '../app/dashboard/page.tsx'

// Mock the API client
jest.mock('../lib/api.ts', () => ({
  apiClient: {
    getProfile: jest.fn(),
    getFunds: jest.fn(),
    getMetrics: jest.fn(),
    getConfig: jest.fn(),
    getHistoricalData: jest.fn()
  },
  formatApiError: jest.fn((error) => error.message || 'Unknown error')
}))

// Mock the TradingChart component
jest.mock('../components/trading-chart.tsx', () => ({
  TradingChart: ({ candlestickData, title, className }) => (
    <div data-testid="trading-chart" className={className}>
      <div data-testid="chart-title">{title}</div>
      <div data-testid="chart-data-length">{candlestickData?.length || 0}</div>
    </div>
  )
}))

// Mock other components
jest.mock('../components/app-layout.tsx', () => ({
  AppLayout: ({ children }) => <div data-testid="app-layout">{children}</div>
}))

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn()
  })
}))

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>
  }
}))

// Mock gsap
jest.mock('gsap', () => ({
  gsap: {
    fromTo: jest.fn()
  }
}))

// Mock sonner
jest.mock('sonner', () => ({
  toast: {
    error: jest.fn()
  }
}))

describe('Dashboard Page - Historical Data Integration', () => {
  const mockApiClient = require('../lib/api.ts').apiClient

  beforeEach(() => {
    // Reset all mocks before each test
    jest.clearAllMocks()

    // Setup default successful API responses
    mockApiClient.getProfile.mockResolvedValue({
      name: 'Test User',
      capital: 100000,
      login_time: '2024-01-15T10:00:00Z'
    })

    mockApiClient.getFunds.mockResolvedValue({
      totalFunds: 100000,
      todayPnL: 5000
    })

    mockApiClient.getMetrics.mockResolvedValue({
      totalTrades: 50,
      winRate: 75,
      lastTradeTime: '2 hours ago'
    })

    mockApiClient.getConfig.mockResolvedValue({
      instruments: [
        {
          name: 'Bank Nifty',
          symbol: 'NSE:NIFTYBANK-INDEX',
          'exchange-symbol': 'NSE:NIFTYBANK-INDEX',
          type: 'index',
          lot_size: 35,
          tick_size: 0.05
        }
      ],
      timeframes: ['15', '30', '60']
    })

    mockApiClient.getHistoricalData.mockResolvedValue([
      {
        time: '2024-01-15',
        open: 100.5,
        high: 105.2,
        low: 99.8,
        close: 104.1,
        volume: 1500000
      },
      {
        time: '2024-01-16',
        open: 104.1,
        high: 108.3,
        low: 103.5,
        close: 107.2,
        volume: 1650000
      }
    ])
  })

  test('should display empty chart state initially', async () => {
    render(<DashboardPage />)

    // Wait for component to load
    await waitFor(() => {
      expect(screen.getByText('Select an instrument and timeframe to load the trading chart')).toBeInTheDocument()
    })

    // Should show placeholder message
    expect(screen.getByText('Select an instrument and timeframe to load the trading chart')).toBeInTheDocument()
  })

  test('should fetch and display historical data when both instrument and timeframe are selected', async () => {
    render(<DashboardPage />)

    // Wait for config to load
    await waitFor(() => {
      expect(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)')).toBeInTheDocument()
    })

    // Select instrument
    fireEvent.click(screen.getByText('Select an instrument'))
    fireEvent.click(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)'))

    // Select timeframe
    fireEvent.click(screen.getByText('Select a timeframe'))
    fireEvent.click(screen.getByText('15 minutes'))

    // Wait for historical data to be fetched
    await waitFor(() => {
      expect(mockApiClient.getHistoricalData).toHaveBeenCalledWith('NSE:NIFTYBANK-INDEX', '15')
    })

    // Should show the trading chart with data
    await waitFor(() => {
      expect(screen.getByTestId('trading-chart')).toBeInTheDocument()
      expect(screen.getByTestId('chart-data-length')).toHaveTextContent('2')
    })
  })

  test('should show loading state while fetching historical data', async () => {
    // Make the API call take time
    mockApiClient.getHistoricalData.mockImplementation(() => 
      new Promise(resolve => setTimeout(() => resolve([]), 1000))
    )

    render(<DashboardPage />)

    // Wait for config to load
    await waitFor(() => {
      expect(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)')).toBeInTheDocument()
    })

    // Select both instrument and timeframe
    fireEvent.click(screen.getByText('Select an instrument'))
    fireEvent.click(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)'))
    
    fireEvent.click(screen.getByText('Select a timeframe'))
    fireEvent.click(screen.getByText('15 minutes'))

    // Should show loading state
    await waitFor(() => {
      expect(screen.getByText('Loading historical data for NSE:NIFTYBANK-INDEX...')).toBeInTheDocument()
    })
  })

  test('should display error toast when historical data fetch fails', async () => {
    // Mock API to return an error
    const errorMessage = 'Failed to fetch historical data'
    mockApiClient.getHistoricalData.mockRejectedValue(new Error(errorMessage))

    render(<DashboardPage />)

    // Wait for config to load
    await waitFor(() => {
      expect(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)')).toBeInTheDocument()
    })

    // Select both instrument and timeframe
    fireEvent.click(screen.getByText('Select an instrument'))
    fireEvent.click(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)'))
    
    fireEvent.click(screen.getByText('Select a timeframe'))
    fireEvent.click(screen.getByText('15 minutes'))

    // Wait for error handling
    await waitFor(() => {
      expect(toast.error).toHaveBeenCalledWith('Failed to load historical data', {
        description: errorMessage
      })
    })

    // Should show error state in chart area
    await waitFor(() => {
      expect(screen.getByText('Failed to load chart data')).toBeInTheDocument()
    })
  })

  test('should update chart when instrument selection changes', async () => {
    // Add another instrument to the config
    mockApiClient.getConfig.mockResolvedValue({
      instruments: [
        {
          name: 'Bank Nifty',
          symbol: 'NSE:NIFTYBANK-INDEX',
          'exchange-symbol': 'NSE:NIFTYBANK-INDEX',
          type: 'index',
          lot_size: 35,
          tick_size: 0.05
        },
        {
          name: 'Nifty 50',
          symbol: 'NSE:NIFTY50-INDEX',
          'exchange-symbol': 'NSE:NIFTY50-INDEX',
          type: 'index',
          lot_size: 50,
          tick_size: 0.05
        }
      ],
      timeframes: ['15', '30', '60']
    })

    render(<DashboardPage />)

    // Wait for config to load
    await waitFor(() => {
      expect(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)')).toBeInTheDocument()
    })

    // First, select timeframe
    fireEvent.click(screen.getByText('Select a timeframe'))
    fireEvent.click(screen.getByText('15 minutes'))

    // Then select first instrument
    fireEvent.click(screen.getByText('Select an instrument'))
    fireEvent.click(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)'))

    await waitFor(() => {
      expect(mockApiClient.getHistoricalData).toHaveBeenCalledWith('NSE:NIFTYBANK-INDEX', '15')
    })

    // Clear the mock to track new calls
    mockApiClient.getHistoricalData.mockClear()

    // Change to second instrument
    fireEvent.click(screen.getByDisplayValue('NSE:NIFTYBANK-INDEX'))
    fireEvent.click(screen.getByText('Nifty 50 (NSE:NIFTY50-INDEX)'))

    // Should fetch data for new instrument
    await waitFor(() => {
      expect(mockApiClient.getHistoricalData).toHaveBeenCalledWith('NSE:NIFTY50-INDEX', '15')
    })
  })

  test('should update chart when timeframe selection changes', async () => {
    render(<DashboardPage />)

    // Wait for config to load
    await waitFor(() => {
      expect(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)')).toBeInTheDocument()
    })

    // First, select instrument
    fireEvent.click(screen.getByText('Select an instrument'))
    fireEvent.click(screen.getByText('Bank Nifty (NSE:NIFTYBANK-INDEX)'))

    // Then select first timeframe
    fireEvent.click(screen.getByText('Select a timeframe'))
    fireEvent.click(screen.getByText('15 minutes'))

    await waitFor(() => {
      expect(mockApiClient.getHistoricalData).toHaveBeenCalledWith('NSE:NIFTYBANK-INDEX', '15')
    })

    // Clear the mock to track new calls
    mockApiClient.getHistoricalData.mockClear()

    // Change to different timeframe
    fireEvent.click(screen.getByDisplayValue('15'))
    fireEvent.click(screen.getByText('30 minutes'))

    // Should fetch data for new timeframe
    await waitFor(() => {
      expect(mockApiClient.getHistoricalData).toHaveBeenCalledWith('NSE:NIFTYBANK-INDEX', '30')
    })
  })
})