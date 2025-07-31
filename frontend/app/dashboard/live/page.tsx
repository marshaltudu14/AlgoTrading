"use client"

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
  WifiOff
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AppLayout } from "@/components/app-layout"
import { TradingChart, createTradeMarker } from "@/components/trading-chart"
import { apiClient, formatApiError } from "@/lib/api"
import { useLiveTrading } from "@/hooks/use-websocket"

// Mock instruments data
const instruments = [
  { symbol: "Bank_Nifty", type: "index", name: "Bank Nifty" },
  { symbol: "Nifty", type: "index", name: "Nifty 50" },
  { symbol: "RELIANCE", type: "stock", name: "Reliance Industries" },
  { symbol: "TCS", type: "stock", name: "Tata Consultancy Services" },
  { symbol: "HDFC", type: "stock", name: "HDFC Bank" }
]

const timeframes = [
  { value: "1", label: "1 Minute" },
  { value: "5", label: "5 Minutes" },
  { value: "15", label: "15 Minutes" },
  { value: "30", label: "30 Minutes" },
  { value: "60", label: "1 Hour" }
]

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

  const isFormValid = formData.instrument && formData.timeframe

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
                <TrendingUp className="h-8 w-8" />
                Live Trading
              </h1>
              <p className="text-muted-foreground">
                Monitor and control your automated trading bot
              </p>
            </div>
            <div className="flex items-center gap-2">
              {isConnected ? (
                <div className="flex items-center gap-2 text-green-600">
                  <Wifi className="h-4 w-4" />
                  <span className="text-sm">Connected</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-red-600">
                  <WifiOff className="h-4 w-4" />
                  <span className="text-sm">Disconnected</span>
                </div>
              )}
            </div>
          </div>
        </motion.div>

        {/* Error Display */}
        {(error || wsError) && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                {error || wsError}
              </AlertDescription>
            </Alert>
          </motion.div>
        )}

        {/* Status Alert */}
        {isTrading && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3 }}
          >
            <Alert>
              <Activity className="h-4 w-4" />
              <AlertDescription>
                Live trading is active. The bot is monitoring market conditions and executing trades automatically.
              </AlertDescription>
            </Alert>
          </motion.div>
        )}

        <div className="space-y-6">
          {/* Configuration Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>Trading Configuration</CardTitle>
                <CardDescription>
                  Configure your live trading parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="instrument">Instrument</Label>
                  <Select
                    value={formData.instrument}
                    onValueChange={(value) => handleInputChange("instrument", value)}
                    disabled={isTrading}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select an instrument" />
                    </SelectTrigger>
                    <SelectContent>
                      {instruments.map((instrument) => (
                        <SelectItem key={instrument.symbol} value={instrument.symbol}>
                          <div className="flex items-center gap-2">
                            <span className={`px-2 py-1 text-xs rounded ${
                              instrument.type === 'index' 
                                ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200' 
                                : 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
                            }`}>
                              {instrument.type}
                            </span>
                            {instrument.name}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="timeframe">Timeframe</Label>
                  <Select
                    value={formData.timeframe}
                    onValueChange={(value) => handleInputChange("timeframe", value)}
                    disabled={isTrading}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select timeframe" />
                    </SelectTrigger>
                    <SelectContent>
                      {timeframes.map((tf) => (
                        <SelectItem key={tf.value} value={tf.value}>
                          <div className="flex items-center gap-2">
                            <Clock className="h-4 w-4" />
                            {tf.label}
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {isIndexInstrument && (
                  <div className="space-y-2">
                    <Label htmlFor="optionStrategy">Option Strategy</Label>
                    <Select
                      value={formData.optionStrategy}
                      onValueChange={(value) => handleInputChange("optionStrategy", value)}
                      disabled={isTrading}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {optionStrategies.map((strategy) => (
                          <SelectItem key={strategy.value} value={strategy.value}>
                            {strategy.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                <div className="pt-4 space-y-2">
                  {!isTrading ? (
                    <Button
                      onClick={handleStart}
                      className="w-full"
                      disabled={!isFormValid || !isConnected}
                    >
                      <Play className="mr-2 h-4 w-4" />
                      Start Trading
                    </Button>
                  ) : (
                    <Button
                      onClick={handleStop}
                      variant="destructive"
                      className="w-full"
                    >
                      <Square className="mr-2 h-4 w-4" />
                      Stop Trading
                    </Button>
                  )}
                  
                  {isTrading && (
                    <p className="text-xs text-muted-foreground text-center">
                      Configuration is locked during active trading
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Stats and Chart Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-6"
          >
            {/* Live Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <DollarSign className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">Current P&L</span>
                  </div>
                  <div className={`text-xl font-bold ${stats.currentPnL >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {stats.currentPnL >= 0 ? '+' : ''}₹{stats.currentPnL.toLocaleString()}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium">Today&apos;s Trades</span>
                  </div>
                  <div className="text-xl font-bold">{stats.todayTrades}</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium">Win Rate</span>
                  </div>
                  <div className="text-xl font-bold">{stats.winRate.toFixed(1)}%</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium">Current Price</span>
                  </div>
                  <div className="text-xl font-bold">₹{stats.currentPrice.toLocaleString()}</div>
                </CardContent>
              </Card>
            </div>

            {/* Position and Recent Trades */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Current Position</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center p-4">
                    <div className={`text-2xl font-bold ${stats.position > 0 ? 'text-green-600' : stats.position < 0 ? 'text-red-600' : 'text-gray-600'}`}>
                      {stats.position === 0 ? 'No Position' : stats.position > 0 ? `Long ${stats.position}` : `Short ${Math.abs(stats.position)}`}
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Trades</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-32 overflow-y-auto">
                    {trades.length === 0 ? (
                      <p className="text-sm text-muted-foreground text-center py-4">No trades yet</p>
                    ) : (
                      trades.slice(-5).reverse().map((trade) => (
                        <div key={trade.id} className="flex justify-between items-center text-sm p-2 border rounded">
                          <span className={`font-medium ${trade.action.includes('BUY') ? 'text-green-600' : 'text-red-600'}`}>
                            {trade.action}
                          </span>
                          <span>₹{trade.price}</span>
                          <span className={trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {trade.pnl >= 0 ? '+' : ''}₹{trade.pnl}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Chart Area */}
            {isTrading ? (
              <TradingChart
                candlestickData={[]} // Will be populated with real-time data
                portfolioData={[]} // Will be populated with portfolio updates
                tradeMarkers={trades.map(trade =>
                  createTradeMarker(
                    Date.now(), // Will use actual trade timestamp
                    trade.action as 'BUY' | 'SELL' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'HOLD',
                    trade.price
                  )
                )}
                title="Live Trading Chart"
                showVolume={true}
                showPortfolio={false}
                currentPrice={stats.currentPrice}
                portfolioValue={undefined}
                height={400}
              />
            ) : (
              <Card>
                <CardHeader>
                  <CardTitle>Live Chart</CardTitle>
                  <CardDescription>
                    Real-time price action and trading signals
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-96 border rounded-lg flex items-center justify-center bg-muted/20">
                    <p className="text-muted-foreground">
                      Start trading to see live chart
                    </p>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Activity Log */}
            <Card>
              <CardHeader>
                <CardTitle>Activity Log</CardTitle>
                <CardDescription>
                  Real-time trading activity and system events
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-48 border rounded-lg p-4 bg-muted/20 overflow-y-auto">
                  {isTrading ? (
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-2 text-muted-foreground">
                        <span className="text-xs">12:34:56</span>
                        <span>System initialized and monitoring market...</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-muted-foreground text-center">
                      Activity log will appear here when trading is active
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </AppLayout>
  )
}
