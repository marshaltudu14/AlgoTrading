/**
 * WebSocket hooks for real-time data
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { apiClient } from '@/lib/api'

export interface WebSocketMessage {
  type: string
  data?: any
  message?: string
  [key: string]: any
}

export interface UseWebSocketOptions {
  onMessage?: (message: WebSocketMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

export function useWebSocket(
  createWebSocket: () => WebSocket | null,
  options: UseWebSocketOptions = {}
) {
  const [isConnected, setIsConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null)
  const [error, setError] = useState<string | null>(null)
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const isConnectingRef = useRef(false)
  const maxReconnectAttempts = options.reconnectAttempts || 5
  const reconnectInterval = options.reconnectInterval || 3000

  const connect = useCallback(() => {
    try {
      // Prevent multiple simultaneous connections
      if (isConnectingRef.current || (wsRef.current && wsRef.current.readyState === WebSocket.CONNECTING)) {
        return
      }
      
      // Close existing connection if any
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.close()
      }
      
      isConnectingRef.current = true
      const ws = createWebSocket()
      if (!ws) {
        isConnectingRef.current = false
        return
      }

      wsRef.current = ws

      ws.onopen = () => {
        isConnectingRef.current = false
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        options.onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          // Handle ping/pong to keep connection alive
          if (message.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }))
            return
          }
          
          setLastMessage(message)
          options.onMessage?.(message)
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onclose = () => {
        isConnectingRef.current = false
        setIsConnected(false)
        options.onDisconnect?.()

        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval * Math.pow(1.5, reconnectAttemptsRef.current)) // Exponential backoff
        } else {
          setError('Maximum reconnection attempts reached')
        }
      }

      ws.onerror = (error) => {
        isConnectingRef.current = false
        setError('WebSocket connection error')
        options.onError?.(error)
      }

    } catch (err) {
      isConnectingRef.current = false
      setError('Failed to create WebSocket connection')
      console.error('WebSocket connection error:', err)
    }
  }, [createWebSocket, maxReconnectAttempts, reconnectInterval]) // Removed options from dependencies to prevent reconnections

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    isConnectingRef.current = false
    setIsConnected(false)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(typeof message === 'string' ? message : JSON.stringify(message))
    }
  }, [isConnected])

  useEffect(() => {
    // Add a small delay to prevent rapid reconnections during React renders
    const timer = setTimeout(() => {
      connect()
    }, 100)

    return () => {
      clearTimeout(timer)
      disconnect()
    }
  }, [connect, disconnect])

  return {
    isConnected,
    lastMessage,
    error,
    sendMessage,
    reconnect: connect,
    disconnect,
  }
}

// Specific hook for backtest WebSocket
export function useBacktestWebSocket(
  backtestId: string | null,
  options: UseWebSocketOptions = {}
) {
  const createWebSocket = useCallback(
    () => {
      if (backtestId) {
        return apiClient.createBacktestWebSocket(backtestId)
      }
      return null
    },
    [backtestId]
  )
  
  return useWebSocket(createWebSocket, options)
}

// Specific hook for live trading WebSocket
export function useLiveWebSocket(
  userId: string | null,
  options: UseWebSocketOptions = {}
) {
  const createWebSocket = useCallback(
    () => userId ? apiClient.createLiveWebSocket(userId) : null,
    [userId]
  )
  
  return useWebSocket(createWebSocket, options)
}

// Hook for managing backtest progress
export function useBacktestProgress(backtestId: string | null) {
  const [progress, setProgress] = useState(0)
  const [status, setStatus] = useState<string>('idle')
  const [currentStep, setCurrentStep] = useState<string>('')
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [chartData, setChartData] = useState<any[]>([])
  const [candlestickData, setCandlestickData] = useState<any[]>([])
  const [currentPrice, setCurrentPrice] = useState<number | undefined>(undefined)
  const [portfolioValue, setPortfolioValue] = useState<number | undefined>(undefined)

  const onMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'connected':
        break
      case 'started':
        setStatus('running')
        setProgress(0)
        setCurrentStep('Initializing...')
        setError(null)
        setChartData([])
        setCandlestickData([])
        setCurrentPrice(undefined)
        setPortfolioValue(undefined)
        break
      case 'progress':
        setProgress(message.progress || 0)
        // Map step names to user-friendly messages
        const stepMessages = {
          'loading_data': 'Loading data...',
          'processing_data': 'Processing data...',
          'starting_backtest': 'Starting backtest...',
          'running_backtest': 'Running backtest...'
        }
        setCurrentStep(stepMessages[message.step as keyof typeof stepMessages] || message.message || 'Processing...')
        break
      case 'data_loaded':
        // Data loading complete - reset candlestick data for real-time updates
        setCandlestickData([])
        break
      case 'candle_update':
        // Real-time candle and action update (TradingView approach)
        if (message.candle) {
          // Add the new candle to the array for real-time updates
          setCandlestickData(prev => [...prev, message.candle])
        }
        if (message.action && message.action.type !== 4) { // Not HOLD action
          const actionData = {
            timestamp: message.action.timestamp,
            price: message.action.price,
            action: message.action.type,
            portfolio_value: message.portfolio_value
          }
          setChartData(prev => [...prev, actionData])
        }
        if (message.portfolio_value) {
          setPortfolioValue(message.portfolio_value)
        }
        if (message.action?.price) {
          setCurrentPrice(message.action.price)
        }

        // Update progress if available
        if (message.progress !== undefined) {
          setProgress(message.progress)
        }
        break
      case 'chart_update':
        if (message.data) {
          setChartData(prev => [...prev, ...message.data])
        }
        if (message.current_price) {
          setCurrentPrice(message.current_price)
        }
        if (message.portfolio_value) {
          setPortfolioValue(message.portfolio_value)
        }
        break
      case 'completed':
        setStatus('completed')
        setProgress(100)
        setCurrentStep('Completed')
        setResults(message.results)
        break
      case 'error':
        setStatus('error')
        setCurrentStep('Error')
        setError(message.message || 'Backtest failed')
        break
    }
  }, [])

  const onError = useCallback(() => {
    setError('WebSocket connection error')
  }, [])

  const { isConnected, lastMessage } = useBacktestWebSocket(backtestId, {
    onMessage,
    onError
  })

  return {
    isConnected,
    progress,
    status,
    currentStep,
    results,
    error,
    lastMessage,
    chartData,
    candlestickData,
    currentPrice,
    portfolioValue,
  }
}

// Hook for managing live trading updates
export function useLiveTrading(userId: string | null) {
  const [isTrading, setIsTrading] = useState(false)
  const [stats, setStats] = useState({
    currentPnL: 0,
    todayTrades: 0,
    winRate: 0,
    currentPrice: 0,
    position: 0,
  })
  const [trades, setTrades] = useState<any[]>([])
  const [error, setError] = useState<string | null>(null)

  const onMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'started':
        setIsTrading(true)
        setError(null)
        break
      case 'stopped':
        setIsTrading(false)
        break
      case 'stats_update':
        setStats({
          currentPnL: message.current_pnl || 0,
          todayTrades: message.today_trades || 0,
          winRate: message.win_rate || 0,
          currentPrice: message.current_price || 0,
          position: message.position || 0,
        })
        break
      case 'trade_executed':
        setTrades(prev => [...prev, {
          timestamp: message.timestamp,
          action: message.action,
          price: message.price,
          pnl: message.pnl,
          id: Date.now() + Math.random(),
        }])
        break
      case 'price_update':
      case 'tick':
        // Handle both price updates and tick data
        if (message.data) {
          // Import and use the store for tick data
          import('@/store/live-data').then(({ useLiveDataStore }) => {
            useLiveDataStore.getState().setLastTick(message.data);
          });
        }
        setStats(prev => ({
          ...prev,
          currentPrice: message.price || message.data?.close || prev.currentPrice,
        }))
        break
      case 'position_update':
        // Handle position updates - you may need to add position state to this hook
        // or emit to a global store
        break
      case 'status':
        // Handle status updates
        if (message.data) {
          import('@/store/live-data').then(({ useLiveDataStore }) => {
            useLiveDataStore.getState().setStatus(message.data);
          });
        }
        break
      case 'error':
        setError(message.message || 'Live trading error')
        break
    }
  }, [])

  const onError = useCallback(() => {
    setError('WebSocket connection error')
  }, [])

  const { isConnected, lastMessage } = useLiveWebSocket(userId, {
    onMessage,
    onError
  })

  return {
    isConnected,
    isTrading,
    stats,
    trades,
    error,
    lastMessage,
  }
}
