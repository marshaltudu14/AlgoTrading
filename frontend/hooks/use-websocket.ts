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
  const maxReconnectAttempts = options.reconnectAttempts || 5
  const reconnectInterval = options.reconnectInterval || 3000

  const connect = useCallback(() => {
    try {
      const ws = createWebSocket()
      if (!ws) return

      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        setError(null)
        reconnectAttemptsRef.current = 0
        options.onConnect?.()
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('WebSocket message received:', message)
          setLastMessage(message)
          options.onMessage?.(message)
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        options.onDisconnect?.()

        // Attempt to reconnect
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          console.log(`Attempting to reconnect (${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        } else {
          setError('Maximum reconnection attempts reached')
        }
      }

      ws.onerror = (error) => {
        setError('WebSocket connection error')
        options.onError?.(error)
      }

    } catch (err) {
      setError('Failed to create WebSocket connection')
      console.error('WebSocket connection error:', err)
    }
  }, [createWebSocket, options, maxReconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    setIsConnected(false)
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && isConnected) {
      wsRef.current.send(typeof message === 'string' ? message : JSON.stringify(message))
    }
  }, [isConnected])

  useEffect(() => {
    connect()

    return () => {
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
  return useWebSocket(
    () => {
      if (backtestId) {
        console.log('Creating WebSocket for backtest ID:', backtestId)
        return apiClient.createBacktestWebSocket(backtestId)
      }
      return null
    },
    options
  )
}

// Specific hook for live trading WebSocket
export function useLiveWebSocket(
  userId: string | null,
  options: UseWebSocketOptions = {}
) {
  return useWebSocket(
    () => userId ? apiClient.createLiveWebSocket(userId) : null,
    options
  )
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

  const { isConnected, lastMessage } = useBacktestWebSocket(backtestId, {
    onMessage: (message) => {
      console.log('Backtest message received:', message)
      switch (message.type) {
        case 'connected':
          console.log('WebSocket connection confirmed:', message.message)
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
          if (message.candlestick_data) {
            setCandlestickData(message.candlestick_data)
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
    },
    onError: () => {
      setError('WebSocket connection error')
    }
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

  const { isConnected, lastMessage } = useLiveWebSocket(userId, {
    onMessage: (message) => {
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
          setStats(prev => ({
            ...prev,
            currentPrice: message.price || prev.currentPrice,
          }))
          break
        case 'error':
          setError(message.message || 'Live trading error')
          break
      }
    },
    onError: () => {
      setError('WebSocket connection error')
    }
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
