"use client"
/* eslint-disable @typescript-eslint/no-unused-vars */
/* eslint-disable @typescript-eslint/no-explicit-any */

import * as React from "react"
import { motion } from "framer-motion"
import { 
  Play, 
  Square, 
  TrendingUp, 
  Activity,
  Clock,
  DollarSign,
  Target,
  AlertCircle,
  Wifi,
  WifiOff,
  Loader2,
  X
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { TradingViewLayout } from "@/components/trading-view-layout"
import { TradingChart, createTradeMarker } from "@/components/trading-chart"
import { ManualTradeForm } from "@/components/manual-trade-form"
import { apiClient, formatApiError, type Instrument, type CandlestickData } from "@/lib/api"
import { useLiveTrading } from "@/hooks/use-websocket"
import WebSocketService from "@/lib/websocket"
import { useLiveDataStore } from "@/store/live-data"
import { toast } from "sonner"
import { generateDemoData, generateDemoTradeMarkers, generateDemoPortfolioData, getRandomDemoDataset } from "@/lib/demo-data"

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

const optionStrategies = [
  { value: "ITM", label: "In The Money (ITM)" },
  { value: "ATM", label: "At The Money (ATM)" },
  { value: "OTM", label: "Out of The Money (OTM)" }
]

export default function LiveTradePage() {
  const [userId] = React.useState("user123") // TODO: Get from auth context
  const [error, setError] = React.useState<string | null>(null)
  const [formData, setFormData] = React.useState({
    instrument: "",
    timeframe: "",
    optionStrategy: "ITM"
  })

  // Configuration state
  const [instruments, setInstruments] = React.useState<Instrument[]>([])
  const [timeframes, setTimeframes] = React.useState<string[]>([])
  const [isLoadingConfig, setIsLoadingConfig] = React.useState(true)

  // Real-time data state
  const [historicalData, setHistoricalData] = React.useState<CandlestickData[]>([])
  const [isChartLoading, setIsChartLoading] = React.useState(false)
  const [chartError, setChartError] = React.useState<string | null>(null)

  // Demo data state
  const [demoData, setDemoData] = React.useState<any[]>([])
  const [demoTradeMarkers, setDemoTradeMarkers] = React.useState<any[]>([])
  const [showDemo, setShowDemo] = React.useState(true)

  // Live data store
  const { isConnected: wsConnected, lastTick, activePosition, setActivePosition } = useLiveDataStore()

  // State for manual trade panel
  const [showManualTrade, setShowManualTrade] = React.useState(false)

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

  // Initialize demo data on component mount
  React.useEffect(() => {
    const demoDataset = getRandomDemoDataset()
    const candleData = demoDataset.data
    const tradeMarkers = generateDemoTradeMarkers(candleData, 0.05)

    setDemoData(candleData)
    setDemoTradeMarkers(tradeMarkers)
  }, [])

  // WebSocket connection for real-time data
  React.useEffect(() => {
    if (!userId) return;
    
    const webSocketService = new WebSocketService(`ws://localhost:8000/ws/live/${userId}`);
    webSocketService.onMessage((event) => {
      const message = JSON.parse(event.data);
      if (message.type === 'tick') {
        useLiveDataStore.getState().setLastTick(message.data);
      } else if (message.type === 'position_update') {
        setActivePosition(message.data);
      } else if (message.type === 'status') {
        useLiveDataStore.getState().setStatus(message.data);
      }
    });
    webSocketService.connect();

    return () => {
      webSocketService.disconnect();
    };
  }, [userId, setActivePosition])

  // Fetch historical data when instrument and timeframe are selected
  React.useEffect(() => {
    const fetchHistoricalData = async () => {
      if (!formData.instrument || !formData.timeframe) {
        return
      }

      try {
        setIsChartLoading(true)
        setChartError(null)
        
        const data = await apiClient.getHistoricalData(formData.instrument, formData.timeframe)
        
        if (data && data.length > 0) {
          setHistoricalData(data)
          setShowDemo(false) // Switch from demo to real data
        } else {
          setHistoricalData([])
          setShowDemo(true) // Fall back to demo
        }
      } catch (err) {
        const errorMessage = formatApiError(err)
        console.error('Failed to fetch historical data:', err)
        setChartError(errorMessage)
        setHistoricalData([])
        setShowDemo(true) // Fall back to demo on error
        toast.error('Failed to load historical data', {
          description: errorMessage
        })
      } finally {
        setIsChartLoading(false)
      }
    }

    fetchHistoricalData()
  }, [formData.instrument, formData.timeframe])

  // Use WebSocket hook for real-time live trading data
  const {
    isConnected,
    isTrading,
    stats,
    trades,
    error: wsError
  } = useLiveTrading(userId)

  const selectedInstrument = instruments.find(i => i.symbol === formData.instrument)
  const isIndexInstrument = selectedInstrument?.type === "index"

  const handleStart = async () => {
    try {
      setError(null)

      // Start live trading using API client
      const response = await apiClient.startLiveTrading({
        instrument: formData.instrument,
        timeframe: formData.timeframe,
        option_strategy: isIndexInstrument ? formData.optionStrategy : undefined
      })

      if (response.status !== "started") {
        setError(response.message || "Failed to start live trading")
      }
    } catch (err) {
      setError(formatApiError(err))
    }
  }

  const handleStop = async () => {
    try {
      setError(null)

      // Stop live trading using API client
      const response = await apiClient.stopLiveTrading()

      if (response.status !== "stopped") {
        setError(response.message || "Failed to stop live trading")
      }
    } catch (err) {
      setError(formatApiError(err))
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }

  // Manual trade submit handler
  const handleManualTradeSubmit = async (values: {
    instrument: string;
    direction: string;
    quantity: number;
    stopLoss?: number;
    target?: number;
  }) => {
    try {
      await apiClient.manualTrade(values);
      toast.success("Manual trade submitted successfully");
    } catch (err) {
      const errorMessage = formatApiError(err);
      console.error("Failed to submit manual trade:", err);
      toast.error("Failed to submit manual trade", {
        description: errorMessage,
      });
    }
  };

  const isFormValid = formData.instrument && formData.timeframe

  // Create header controls for TradingView layout
  const headerControls = (
    <div className="flex items-center gap-3 text-xs overflow-x-auto scrollbar-hide min-w-0 flex-1">
      <div className="flex items-center gap-1">
        <label className="text-muted-foreground font-medium">Symbol:</label>
        <Select
          value={formData.instrument}
          onValueChange={(value) => setFormData(prev => ({ ...prev, instrument: value }))}
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
          onValueChange={(value) => setFormData(prev => ({ ...prev, timeframe: value }))}
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

      {isIndexInstrument && (
        <div className="flex items-center gap-1">
          <label className="text-muted-foreground font-medium">Strategy:</label>
          <Select
            value={formData.optionStrategy}
            onValueChange={(value) => setFormData(prev => ({ ...prev, optionStrategy: value }))}
          >
            <SelectTrigger className="w-20 h-8">
              <SelectValue placeholder="Strategy" />
            </SelectTrigger>
            <SelectContent>
              {optionStrategies.map((strategy) => (
                <SelectItem key={strategy.value} value={strategy.value}>
                  {strategy.value}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <div className="flex items-center gap-1">
        {isConnected ? (
          <Wifi className="h-3 w-3 text-green-500" />
        ) : (
          <WifiOff className="h-3 w-3 text-red-500" />
        )}
        <span className="text-xs text-muted-foreground">
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {!isTrading ? (
        <Button
          onClick={handleStart}
          disabled={!isFormValid || !isConnected}
          size="sm"
          className="h-8 px-3"
        >
          <Play className="h-3 w-3 mr-1" />
          Start
        </Button>
      ) : (
        <Button
          onClick={handleStop}
          variant="destructive"
          size="sm"
          className="h-8 px-3"
        >
          <Square className="h-3 w-3 mr-1" />
          Stop
        </Button>
      )}

      <div className="border-l pl-2 ml-2">
        <Button
          onClick={() => setShowManualTrade(!showManualTrade)}
          variant={showManualTrade ? "default" : "outline"}
          size="sm"
          className="h-8 px-3"
        >
          Manual Trade
        </Button>
      </div>
    </div>
  )

  return (
    <TradingViewLayout headerControls={headerControls}>
      {/* Full-screen chart */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="h-full w-full"
      >
        {isChartLoading ? (
          // Loading state
          <div className="h-full w-full flex items-center justify-center bg-muted/5">
            <div className="text-center">
              <Loader2 className="h-16 w-16 mx-auto mb-4 text-primary animate-spin" />
              <h3 className="text-lg font-medium mb-2">Loading Chart Data</h3>
              <p className="text-muted-foreground">
                Fetching historical data for {formData.instrument}...
              </p>
            </div>
          </div>
        ) : chartError ? (
          // Error state
          <div className="h-full w-full flex items-center justify-center bg-muted/5">
            <div className="text-center">
              <AlertCircle className="h-16 w-16 mx-auto mb-4 text-destructive" />
              <h3 className="text-lg font-medium mb-2">Failed to Load Chart</h3>
              <p className="text-muted-foreground mb-4">{chartError}</p>
              <p className="text-sm text-muted-foreground">Showing demo data instead</p>
            </div>
          </div>
        ) : historicalData.length > 0 && !showDemo ? (
          // Real historical data with live updates
          <TradingChart
            candlestickData={historicalData}
            tradeMarkers={trades?.map((trade: any) =>
              createTradeMarker(
                trade.timestamp,
                trade.action as 'BUY' | 'SELL' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'HOLD',
                trade.price,
                isTrading ? 'Manual' : 'Automated'
              )
            ) || []}
            title={`${formData.instrument} - ${getTimeframeLabel(formData.timeframe)} (Live)`}
            showPortfolio={false}
            fullScreen={true}
            windowSize={100}
            enableSlidingWindow={false}
            currentPrice={lastTick?.price || stats?.currentPrice}
            activePosition={activePosition}
            stopLoss={activePosition?.stopLoss}
            targetPrice={activePosition?.targetPrice}
          />
        ) : showDemo || demoData.length > 0 ? (
          // Show demo chart as fallback
          <TradingChart
            candlestickData={demoData}
            tradeMarkers={demoTradeMarkers}
            title="Demo Live Trading Chart"
            showPortfolio={false}
            fullScreen={true}
            windowSize={100}
            enableSlidingWindow={false}
            currentPrice={demoData.length > 0 ? demoData[demoData.length - 1]?.close : undefined}
          />
        ) : (
          // Empty state
          <div className="h-full w-full flex items-center justify-center bg-muted/5">
            <div className="text-center">
              <TrendingUp className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">Ready for Live Trading</h3>
              <p className="text-muted-foreground">
                Configure your parameters in the header and click Start to begin
              </p>
              {!isConnected && (
                <p className="text-red-500 text-sm mt-2">
                  Waiting for connection...
                </p>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {/* Manual Trade Panel Overlay */}
      {showManualTrade && (
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 300 }}
          className="fixed right-4 top-16 bottom-4 w-80 bg-background border rounded-lg shadow-lg p-4 z-50 overflow-y-auto"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Manual Trade</h3>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowManualTrade(false)}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
          
          <ManualTradeForm 
            instruments={instruments}
            isDisabled={!!activePosition}
            onSubmit={handleManualTradeSubmit}
          />
        </motion.div>
      )}
    </TradingViewLayout>
  )
}