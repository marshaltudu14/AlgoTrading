"use client"
/* eslint-disable @typescript-eslint/no-unused-vars */

import * as React from "react"
import {
  Play,
  BarChart3,
  Loader2
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"


import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { TradingViewLayout } from "@/components/trading-view-layout"
import { TradingChart, createTradeMarker } from "@/components/trading-chart"
import { apiClient, formatApiError, type Instrument } from "@/lib/api"
import { useBacktestProgress } from "@/hooks/use-websocket"
import { toast } from "sonner"
import { generateDemoTradeMarkers, generateDemoPortfolioData, getRandomDemoDataset, DemoDataStream } from "@/lib/demo-data"

// Helper function to format timeframe labels
const getTimeframeLabel = (timeframe: string): string => {
  if (timeframe === 'D') return 'Daily'
  const minutes = parseInt(timeframe)
  if (minutes >= 60) {
    const hours = minutes / 60
    return `${hours} Hour${hours > 1 ? 's' : ''}`
  }
  return `${minutes} Minute${minutes > 1 ? 's' : ''}`
}

export default function BacktestPage() {
  const [backtestId, setBacktestId] = React.useState<string | null>(null)
  const [_error, setError] = React.useState<string | null>(null)
  const [formData, setFormData] = React.useState({
    instrument: "",
    timeframe: "",
    duration: "30",
    initialCapital: "100000"
  })

  // Configuration state
  const [instruments, setInstruments] = React.useState<Instrument[]>([])
  const [timeframes, setTimeframes] = React.useState<string[]>([])
  const [isLoadingConfig, setIsLoadingConfig] = React.useState(true)

  // Demo data state
  const [demoData, setDemoData] = React.useState<Array<{time: string | number, open: number, high: number, low: number, close: number}>>([])
  const [demoTradeMarkers, setDemoTradeMarkers] = React.useState<Array<{time: string | number, position: 'aboveBar' | 'belowBar', color: string, shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown', text: string}>>([])
  const [demoPortfolioData, setDemoPortfolioData] = React.useState<Array<{time: string | number, value: number}>>([])
  const [showDemo, setShowDemo] = React.useState(true)

  // Initialize demo data on component mount
  React.useEffect(() => {
    const demoDataset = getRandomDemoDataset()
    const candleData = demoDataset.data
    const tradeMarkers = generateDemoTradeMarkers(candleData, 0.08)
    const portfolioData = generateDemoPortfolioData(candleData, 100000)

    console.log('Demo data loaded:', { candleData: candleData.length, tradeMarkers: tradeMarkers.length, portfolioData: portfolioData.length })
    setDemoData(candleData)
    setDemoTradeMarkers(tradeMarkers)
    setDemoPortfolioData(portfolioData)
  }, [])

  // Load configuration data
  React.useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoadingConfig(true)
        const config = await apiClient.getConfig()
        setInstruments(config.instruments)
        setTimeframes(config.timeframes)
      } catch (err) {
        console.error('Failed to fetch configuration:', err)
        toast.error('Failed to load configuration', {
          description: formatApiError(err)
        })
      } finally {
        setIsLoadingConfig(false)
      }
    }
    fetchConfig()
  }, [])

  React.useEffect(() => {
    // Disabled GSAP animations temporarily to avoid ref issues
    // TODO: Re-enable with proper ref checks
  }, []);


  // Initialize demo data on component mount
  React.useEffect(() => {
    const demoDataset = getRandomDemoDataset()
    const candleData = demoDataset.data
    const tradeMarkers = generateDemoTradeMarkers(candleData, 0.08)
    const portfolioData = generateDemoPortfolioData(candleData, 100000)

    setDemoData(candleData)
    setDemoTradeMarkers(tradeMarkers)
    setDemoPortfolioData(portfolioData)
  }, [])

  React.useEffect(() => {
    // Disabled GSAP animations temporarily to avoid ref issues
    // TODO: Re-enable with proper ref checks
  }, []);

  // Use WebSocket hook for real-time progress and chart data
  const {
    progress,
    status,
    currentStep,
    results,
    error: wsError,
    chartData,
    candlestickData,
    currentPrice,
    portfolioValue
  } = useBacktestProgress(backtestId)

  const isRunning = status === 'running'

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setBacktestId(null)

    try {
      // Start backtest using API client
      const response = await apiClient.startBacktest({
        instrument: formData.instrument,
        timeframe: formData.timeframe,
        duration: parseInt(formData.duration),
        initial_capital: parseFloat(formData.initialCapital)
      })

      if (response.backtest_id) {
        console.log('Backtest started with ID:', response.backtest_id)
        setBacktestId(response.backtest_id)
      } else {
        setError(response.message || "Failed to start backtest")
      }
    } catch (err) {
      setError(formatApiError(err))
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  const isFormValid = formData.instrument && formData.timeframe && formData.duration && formData.initialCapital
  
  // Create header controls for TradingView layout
  const headerControls = (
    <div className="flex items-center gap-3 text-xs overflow-x-auto scrollbar-hide min-w-0 flex-1">
      <div className="flex items-center gap-1">
        <label className="text-muted-foreground font-medium">Symbol:</label>
        <Select
          value={formData.instrument}
          onValueChange={(value) => handleInputChange("instrument", value)}
          disabled={isLoadingConfig}
        >
          <SelectTrigger className="w-32 h-8">
            <SelectValue placeholder={isLoadingConfig ? "Loading..." : "Symbol"} />
          </SelectTrigger>
          <SelectContent>
            {instruments.map((instrument) => (
              <SelectItem key={instrument.symbol} value={instrument.symbol}>
                {instrument.symbol}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-1">
        <label className="text-muted-foreground font-medium">Timeframe:</label>
        <Select
          value={formData.timeframe}
          onValueChange={(value) => handleInputChange("timeframe", value)}
          disabled={isLoadingConfig}
        >
          <SelectTrigger className="w-24 h-8">
            <SelectValue placeholder={isLoadingConfig ? "Loading..." : "TF"} />
          </SelectTrigger>
          <SelectContent>
            {timeframes.map((timeframe) => (
              <SelectItem key={timeframe} value={timeframe}>
                {getTimeframeLabel(timeframe)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="flex items-center gap-1">
        <label className="text-muted-foreground font-medium">Days:</label>
        <Input
          type="number"
          min="1"
          max="365"
          value={formData.duration}
          onChange={(e) => handleInputChange("duration", e.target.value)}
          placeholder="Days"
          className="w-16 h-8"
        />
      </div>

      <div className="flex items-center gap-1">
        <label className="text-muted-foreground font-medium">Capital:</label>
        <Input
          type="number"
          min="10000"
          max="10000000"
          step="1000"
          value={formData.initialCapital}
          onChange={(e) => handleInputChange("initialCapital", e.target.value)}
          placeholder="Capital"
          className="w-24 h-8"
        />
      </div>

      <Button
        onClick={handleSubmit}
        disabled={!isFormValid || isRunning}
        size="sm"
        className="h-8 px-3"
      >
        {isRunning ? (
          <>
            <Loader2 className="h-3 w-3 mr-1 animate-spin" />
            {currentStep || 'Running...'}
          </>
        ) : (
          <>
            <Play className="h-3 w-3 mr-1" />
            Run
          </>
        )}
      </Button>
    </div>
  )

  return (
    <TradingViewLayout headerControls={headerControls}>
      {/* Full-screen chart */}
      <div className="h-full w-full">
        {showDemo && !isRunning && !results ? (
          // Show demo chart initially
          <TradingChart
            candlestickData={demoData}
            portfolioData={demoPortfolioData}
            tradeMarkers={demoTradeMarkers}
            title="Demo Trading Chart"
            showPortfolio={true}
            fullScreen={true}
            windowSize={100}
            enableSlidingWindow={false}
            currentPrice={demoData.length > 0 ? demoData[demoData.length - 1]?.close : undefined}
            portfolioValue={demoPortfolioData.length > 0 ? demoPortfolioData[demoPortfolioData.length - 1]?.value : undefined}
          />
        ) : isRunning || results ? (
          // Show real backtest data with real-time candle updates
          <TradingChart
            candlestickData={candlestickData}
            portfolioData={chartData.map(item => ({
              time: item.timestamp,
              value: item.portfolio_value
            }))}
            tradeMarkers={chartData
              .filter(item => item.action !== 4) // Filter out HOLD actions
              .map(item => {
                const actionNames = ['BUY', 'SELL', 'CLOSE_LONG', 'CLOSE_SHORT', 'HOLD']
                return createTradeMarker(
                  item.timestamp,
                  actionNames[item.action] as 'BUY' | 'SELL' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'HOLD',
                  item.price
                )
              })
            }
            title="Live Backtest"
            showPortfolio={true}
            fullScreen={true}
            windowSize={100}
            enableSlidingWindow={true}
            currentPrice={currentPrice}
            portfolioValue={portfolioValue}
          />
        ) : (
          // Empty state
          <div className="h-full w-full flex items-center justify-center bg-muted/5">
            <div className="text-center">
              <BarChart3 className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">Ready to Backtest</h3>
              <p className="text-muted-foreground">
                Configure your parameters in the header and click Run to start
              </p>
            </div>
          </div>
        )}
      </div>
    </TradingViewLayout>
  )
}