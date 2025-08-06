/**
 * Simple integration tests for the frontend API client
 * 
 * These tests verify that the getConfig function works correctly
 * by testing against a mock server response.
 */

// Mock fetch for testing
global.fetch = jest.fn()

// Import the API client (using require since this is a JS file)
const { apiClient } = require('../lib/api.ts')

describe('API Client - getConfig', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    fetch.mockClear()
  })

  afterEach(() => {
    // Clean up after each test
    jest.resetAllMocks()
  })

  test('should successfully fetch configuration data', async () => {
    // Mock the fetch response
    const mockConfigData = {
      instruments: [
        {
          name: 'Bank Nifty',
          symbol: 'Bank_Nifty',
          'exchange-symbol': 'NSE:NIFTYBANK-INDEX',
          type: 'index',
          lot_size: 35,
          tick_size: 0.05,
          option_premium_range: [0.02, 0.03]
        }
      ],
      timeframes: ['1', '5', '15', '30', '60']
    }

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockConfigData,
      status: 200
    })

    // Call the getConfig function
    const result = await apiClient.getConfig()

    // Verify the fetch was called correctly
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/config'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        credentials: 'include'
      })
    )

    // Verify the result matches expected structure
    expect(result).toEqual(mockConfigData)
    expect(result.instruments).toBeInstanceOf(Array)
    expect(result.timeframes).toBeInstanceOf(Array)
    expect(result.instruments[0]).toHaveProperty('name')
    expect(result.instruments[0]).toHaveProperty('symbol')
  })

  test('should handle network errors gracefully', async () => {
    // Mock a network error
    fetch.mockRejectedValueOnce(new Error('Network error'))

    // Call the getConfig function and expect it to throw
    await expect(apiClient.getConfig()).rejects.toThrow('Network error')

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  test('should handle HTTP error responses', async () => {
    // Mock an HTTP error response
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ detail: 'Configuration file not found' })
    })

    // Call the getConfig function and expect it to throw
    await expect(apiClient.getConfig()).rejects.toThrow('500: Configuration file not found')

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  test('should handle malformed JSON responses', async () => {
    // Mock a response that fails JSON parsing
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => { throw new Error('Invalid JSON') }
    })

    // Call the getConfig function and expect it to throw
    await expect(apiClient.getConfig()).rejects.toThrow('500: Internal Server Error')

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})

describe('API Client - getHistoricalData', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    fetch.mockClear()
  })

  afterEach(() => {
    // Clean up after each test
    jest.resetAllMocks()
  })

  test('should successfully fetch historical data', async () => {
    // Mock the fetch response
    const mockHistoricalData = [
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
    ]

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockHistoricalData,
      status: 200
    })

    // Call the getHistoricalData function
    const result = await apiClient.getHistoricalData('NSE:NIFTYBANK-INDEX', '15')

    // Verify the fetch was called correctly
    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/api/historical-data?instrument=NSE%3ANIFTYBANK-INDEX&timeframe=15'),
      expect.objectContaining({
        headers: expect.objectContaining({
          'Content-Type': 'application/json'
        }),
        credentials: 'include'
      })
    )

    // Verify the result matches expected structure
    expect(result).toEqual(mockHistoricalData)
    expect(result).toBeInstanceOf(Array)
    expect(result[0]).toHaveProperty('time')
    expect(result[0]).toHaveProperty('open')
    expect(result[0]).toHaveProperty('high')
    expect(result[0]).toHaveProperty('low')
    expect(result[0]).toHaveProperty('close')
    expect(result[0]).toHaveProperty('volume')
  })

  test('should handle missing instrument parameter', async () => {
    // Call the getHistoricalData function with empty instrument and expect it to throw
    await expect(apiClient.getHistoricalData('', '15')).rejects.toThrow('Instrument and timeframe parameters are required')
    
    // Verify fetch was not called
    expect(fetch).toHaveBeenCalledTimes(0)
  })

  test('should handle missing timeframe parameter', async () => {
    // Call the getHistoricalData function with empty timeframe and expect it to throw
    await expect(apiClient.getHistoricalData('NSE:NIFTYBANK-INDEX', '')).rejects.toThrow('Instrument and timeframe parameters are required')
    
    // Verify fetch was not called
    expect(fetch).toHaveBeenCalledTimes(0)
  })

  test('should handle HTTP error responses', async () => {
    // Mock an HTTP error response
    fetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
      statusText: 'Unprocessable Entity',
      json: async () => ({ detail: 'Instrument parameter is required' })
    })

    // Call the getHistoricalData function and expect it to throw
    await expect(apiClient.getHistoricalData('NSE:NIFTYBANK-INDEX', '15')).rejects.toThrow('422: Instrument parameter is required')

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  test('should handle network errors gracefully', async () => {
    // Mock a network error
    fetch.mockRejectedValueOnce(new Error('Network error'))

    // Call the getHistoricalData function and expect it to throw
    await expect(apiClient.getHistoricalData('NSE:NIFTYBANK-INDEX', '15')).rejects.toThrow('Network error')

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  test('should handle empty data response', async () => {
    // Mock an empty data response
    const mockEmptyData = []

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockEmptyData,
      status: 200
    })

    // Call the getHistoricalData function
    const result = await apiClient.getHistoricalData('INVALID:SYMBOL', '15')

    // Verify the result is an empty array
    expect(result).toEqual([])
    expect(result).toBeInstanceOf(Array)
    expect(result).toHaveLength(0)

    // Verify fetch was called
    expect(fetch).toHaveBeenCalledTimes(1)
  })
})