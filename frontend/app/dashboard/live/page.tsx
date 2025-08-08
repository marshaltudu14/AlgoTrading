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
  X,
  BarChart3,
  FileText,
  Smartphone
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { TradingViewLayout } from "@/components/trading-view-layout"
import { TradingChart, createTradeMarker } from "@/components/trading-chart"
import { ManualTradeForm } from "@/components/manual-trade-form"
import { SymbolSelectorDialog } from "@/components/symbol-selector-dialog"
import { TimeframeSelectorDialog } from "@/components/timeframe-selector-dialog"
import { StrategySelectorDialog } from "@/components/strategy-selector-dialog"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { apiClient, formatApiError, type Instrument, type CandlestickData } from "@/lib/api"
import { useLiveTrading } from "@/hooks/use-websocket"
import { useAuth } from "@/hooks/use-auth"
import { useLiveDataStore } from "@/store/live-data"
import { toast } from "sonner"
// Removed demo data imports - using real data only

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
  const { userId, user, isLoading: authLoading, error: authError } = useAuth()
  const [error, setError] = React.useState<string | null>(null)
  const [formData, setFormData] = React.useState({
    instrument: "",
    timeframe: "",
    optionStrategy: "ITM"
  })
  
  // Trading mode state
  const [tradingMode, setTradingMode] = React.useState<'paper' | 'real'>('paper')
  const [autoStartError, setAutoStartError] = React.useState<string | null>(null)

  // Configuration state
  const [instruments, setInstruments] = React.useState<Instrument[]>([])
  const [timeframes, setTimeframes] = React.useState<string[]>([])
  const [isLoadingConfig, setIsLoadingConfig] = React.useState(true)

  // Real-time data state
  const [historicalData, setHistoricalData] = React.useState<CandlestickData[]>([])
  const [isChartLoading, setIsChartLoading] = React.useState(false)
  const [chartError, setChartError] = React.useState<string | null>(null)

  // Removed demo data state - using real data only

  // Live data store
  const { isConnected: wsConnected, lastTick, activePosition, setActivePosition, setIsConnected } = useLiveDataStore()

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

        // Set default values: Nifty instrument and 5-minute timeframe
        if (config.instruments.length > 0 && !formData.instrument) {
          const niftyInstrument = config.instruments.find(i => i.symbol === 'Nifty')
          const defaultInstrument = niftyInstrument || config.instruments[0]
          setFormData(prev => ({
            ...prev,
            instrument: defaultInstrument.symbol
          }))
        }

        if (config.timeframes.length > 0 && !formData.timeframe) {
          const defaultTimeframe = config.timeframes.includes('5') ? '5' : config.timeframes[0]
          setFormData(prev => ({
            ...prev,
            timeframe: defaultTimeframe
          }))
        }
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
  }, [formData.instrument, formData.timeframe])

  // Removed demo data initialization - using real data only

  // WebSocket connection handled by useLiveTrading hook

  // Use WebSocket hook for real-time live trading data
  const {
    isConnected,
    isTrading,
    stats,
    trades,
    error: wsError
  } = useLiveTrading(userId)

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
        } else {
          setHistoricalData([])
        }
      } catch (err) {
        const errorMessage = formatApiError(err)
        console.error('Failed to fetch historical data:', err)
        setChartError(errorMessage)
        setHistoricalData([])
        toast.error('Failed to load historical data', {
          description: errorMessage
        })
      } finally {
        setIsChartLoading(false)
      }
    }

    fetchHistoricalData()
  }, [formData.instrument, formData.timeframe])

  const selectedInstrument = instruments.find(i => i.symbol === formData.instrument)
  const isIndexInstrument = selectedInstrument?.instrument_type === "index"

  const handleStart = React.useCallback(async (mode?: 'paper' | 'real') => {
    try {
      setError(null)
      setAutoStartError(null)

      // Use provided mode or current trading mode
      const currentMode = mode || tradingMode

      // Start live trading using API client
      const response = await apiClient.startLiveTrading({
        instrument: formData.instrument,
        timeframe: formData.timeframe,
        option_strategy: isIndexInstrument ? formData.optionStrategy : undefined,
        trading_mode: currentMode
      })

      if (response.status !== "started") {
        const errorMsg = response.message || "Failed to start live trading"
        setError(errorMsg)
        setAutoStartError(errorMsg)
      }
    } catch (err) {
      const errorMsg = formatApiError(err)
      setError(errorMsg)
      setAutoStartError(errorMsg)
    }
  }, [formData.instrument, formData.timeframe, formData.optionStrategy, isIndexInstrument, tradingMode])

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

  // Auto-start trading when conditions are met
  React.useEffect(() => {
    const shouldAutoStart = () => {
      return (
        formData.instrument &&
        formData.timeframe &&
        wsConnected &&
        !isTrading &&
        !isLoadingConfig &&
        tradingMode
      )
    }

    if (shouldAutoStart()) {
      // Small delay to ensure all connections are stable
      const timer = setTimeout(() => {
        console.log('Auto-starting trading with conditions:', {
          instrument: formData.instrument,
          timeframe: formData.timeframe,
          wsConnected,
          isTrading,
          tradingMode
        })
        
        toast.info(`Auto-starting ${tradingMode} trading`, {
          description: `${formData.instrument} - ${getTimeframeLabel(formData.timeframe)}`
        })
        
        handleStart(tradingMode)
      }, 1000)

      return () => clearTimeout(timer)
    }
  }, [formData.instrument, formData.timeframe, wsConnected, isTrading, isLoadingConfig, tradingMode, handleStart])

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
    <TooltipProvider>
      <div className="flex items-center gap-2 text-xs overflow-x-auto scrollbar-hide min-w-0 flex-1">
        {/* Symbol Selector */}
        <SymbolSelectorDialog
          instruments={instruments}
          selectedSymbol={formData.instrument}
          onSymbolChange={(value) => setFormData(prev => ({ ...prev, instrument: value }))}
          isLoading={isLoadingConfig}
        />

        {/* Timeframe Selector */}
        <TimeframeSelectorDialog
          timeframes={timeframes}
          selectedTimeframe={formData.timeframe}
          onTimeframeChange={(value) => setFormData(prev => ({ ...prev, timeframe: value }))}
          isLoading={isLoadingConfig}
        />

        {/* Strategy Selector - Only for Index Instruments */}
        <StrategySelectorDialog
          selectedStrategy={formData.optionStrategy}
          onStrategyChange={(value) => setFormData(prev => ({ ...prev, optionStrategy: value }))}
          isVisible={isIndexInstrument}
        />

        {/* Connection Status */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center justify-center h-8 w-8">
              {wsConnected ? (
                <Wifi className="h-4 w-4 text-green-500" />
              ) : (
                <WifiOff className="h-4 w-4 text-red-500" />
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>{wsConnected ? 'Connected' : 'Disconnected'}</p>
          </TooltipContent>
        </Tooltip>

        {/* Trading Mode Toggle */}
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 hover:bg-accent/50"
              onClick={() => setTradingMode(prev => prev === 'paper' ? 'real' : 'paper')}
              disabled={isTrading}
              aria-label={`Switch to ${tradingMode === 'paper' ? 'real' : 'paper'} trading`}
            >
              {tradingMode === 'paper' ? (
                <FileText className="h-4 w-4 text-blue-500" />
              ) : (
                <DollarSign className="h-4 w-4 text-green-500" />
              )}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Mode: {tradingMode === 'paper' ? 'Paper Trading' : 'Real Trading'}</p>
          </TooltipContent>
        </Tooltip>

        {/* Trading Status Indicator */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center justify-center h-8 w-8">
              <div className={`h-3 w-3 rounded-full ${isTrading ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`} />
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>{isTrading ? `Trading Active (${tradingMode})` : 'Trading Inactive'}</p>
          </TooltipContent>
        </Tooltip>

        {/* Stop Trading Button - Only shown when trading */}
        {isTrading && (
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handleStop}
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-red-500/10 text-red-500"
                aria-label="Stop trading"
              >
                <Square className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Stop Trading</p>
            </TooltipContent>
          </Tooltip>
        )}

        {/* Manual Trade Button */}
        <div className="border-l border-border/50 pl-2 ml-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={() => setShowManualTrade(!showManualTrade)}
                variant="ghost"
                size="sm"
                className={`h-8 w-8 p-0 hover:bg-accent/50 ${showManualTrade ? 'bg-accent text-accent-foreground' : ''}`}
                aria-label="Toggle manual trade panel"
              >
                <Smartphone className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">
              <p>Manual Trade</p>
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </TooltipProvider>
  )

  // Show loading state while authenticating
  if (authLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
        <span className="ml-2">Loading user profile...</span>
      </div>
    )
  }

  // Show error if authentication failed
  if (authError || !userId) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Alert className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            {authError || 'Failed to load user profile. Please try refreshing the page.'}
          </AlertDescription>
        </Alert>
      </div>
    )
  }

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
              <p className="text-sm text-muted-foreground">Please check your connection and try again</p>
            </div>
          </div>
        ) : historicalData.length > 0 ? (
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
        ) : (
          // Empty state
          <div className="h-full w-full flex items-center justify-center bg-muted/5">
            <div className="text-center">
              <TrendingUp className="h-16 w-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-medium mb-2">Ready for Live Trading</h3>
              <p className="text-muted-foreground">
                Trading will start automatically when all conditions are met
              </p>
              {!isConnected && (
                <p className="text-red-500 text-sm mt-2">
                  Waiting for connection...
                </p>
              )}
              {autoStartError && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg max-w-md mx-auto">
                  <p className="text-red-700 text-sm font-medium">Auto-start failed:</p>
                  <p className="text-red-600 text-xs mt-1">{autoStartError}</p>
                </div>
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
            defaultInstrument={formData.instrument}
            onSubmit={handleManualTradeSubmit}
          />
        </motion.div>
      )}
    </TradingViewLayout>
  )
}