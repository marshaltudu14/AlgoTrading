"use client"

import * as React from "react"
import { motion } from "framer-motion"
import {
  Play,
  BarChart3,
  Clock,
  Loader2
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { AppLayout } from "@/components/app-layout"
import { TradingChart, createTradeMarker } from "@/components/trading-chart"
import { apiClient, formatApiError } from "@/lib/api"
import { useBacktestProgress } from "@/hooks/use-websocket"

// Mock instruments data - replace with actual config/instruments.yaml data
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

export default function BacktestPage() {
  const [backtestId, setBacktestId] = React.useState<string | null>(null)
  const [error, setError] = React.useState<string | null>(null)
  const [formData, setFormData] = React.useState({
    instrument: "",
    timeframe: "",
    duration: "30",
    initialCapital: "100000"
  })

  // Use WebSocket hook for real-time progress and chart data
  const {
    progress,
    status,
    results,
    error: wsError,
    chartData,
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

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <BarChart3 className="h-8 w-8" />
            Backtesting
          </h1>
          <p className="text-muted-foreground">
            Test your trading strategies against historical market data
          </p>
        </motion.div>

        <div className="space-y-6">
          {/* Configuration Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>Backtest Configuration</CardTitle>
                <CardDescription>
                  Configure your backtest parameters
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="instrument">Instrument</Label>
                    <Select
                      value={formData.instrument}
                      onValueChange={(value) => handleInputChange("instrument", value)}
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

                  <div className="space-y-2">
                    <Label htmlFor="duration">Duration (Days)</Label>
                    <Input
                      id="duration"
                      type="number"
                      min="1"
                      max="365"
                      value={formData.duration}
                      onChange={(e) => handleInputChange("duration", e.target.value)}
                      placeholder="Enter number of days"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="initialCapital">Initial Capital (₹)</Label>
                    <Input
                      id="initialCapital"
                      type="number"
                      min="10000"
                      max="10000000"
                      step="1000"
                      value={formData.initialCapital}
                      onChange={(e) => handleInputChange("initialCapital", e.target.value)}
                      placeholder="Enter initial capital amount"
                    />
                  </div>

                  <Button
                    type="submit"
                    className="w-full"
                    disabled={!isFormValid || isRunning}
                  >
                    {isRunning ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Running Backtest...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Run Backtest
                      </>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </motion.div>

          {/* Results Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
                <CardDescription>
                  {results ? "Backtest completed successfully" : "Results will appear here after running a backtest"}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {(error || wsError) && (
                  <div className="p-4 border border-red-200 rounded-lg bg-red-50 text-red-700 mb-4">
                    {error || wsError}
                  </div>
                )}

                {isRunning && (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
                      <p className="text-sm text-muted-foreground mb-2">
                        Running backtest... This may take a few moments.
                      </p>
                      {progress > 0 && (
                        <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                          <div
                            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      )}
                      <p className="text-xs text-muted-foreground">
                        Progress: {progress}%
                      </p>
                    </div>
                  </div>
                )}

                {/* Real-time Chart */}
                {(isRunning || results) && chartData.length > 0 && (
                  <div className="mb-6">
                    <TradingChart
                      candlestickData={chartData.map(item => ({
                        time: item.timestamp,
                        open: item.price,
                        high: item.price * 1.001,
                        low: item.price * 0.999,
                        close: item.price,
                        volume: 1000
                      }))}
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
                      title="Backtest Progress"
                      showVolume={false}
                      showPortfolio={true}
                      currentPrice={currentPrice}
                      portfolioValue={portfolioValue}
                      height={300}
                    />
                  </div>
                )}

                {results && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center p-4 border rounded-lg">
                        <div className={`text-2xl font-bold ${results.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                          {results.total_pnl >= 0 ? '+' : ''}₹{results.total_pnl?.toLocaleString() || 0}
                        </div>
                        <p className="text-sm text-muted-foreground">Total P&L</p>
                      </div>
                      <div className="text-center p-4 border rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {results.win_rate?.toFixed(1) || 0}%
                        </div>
                        <p className="text-sm text-muted-foreground">Win Rate</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                      <div className="text-center p-3 border rounded-lg">
                        <div className="text-lg font-semibold">{results.total_trades || 0}</div>
                        <p className="text-xs text-muted-foreground">Total Trades</p>
                      </div>
                      <div className="text-center p-3 border rounded-lg">
                        <div className="text-lg font-semibold text-red-600">{results.max_drawdown?.toFixed(1) || 0}%</div>
                        <p className="text-xs text-muted-foreground">Max Drawdown</p>
                      </div>
                      <div className="text-center p-3 border rounded-lg">
                        <div className="text-lg font-semibold text-purple-600">{results.sharpe_ratio?.toFixed(2) || 0}</div>
                        <p className="text-xs text-muted-foreground">Sharpe Ratio</p>
                      </div>
                    </div>

                    <div className="h-64 border rounded-lg flex items-center justify-center bg-muted/20">
                      <p className="text-muted-foreground">Chart visualization will be implemented here</p>
                    </div>
                  </div>
                )}

                {!isRunning && !results && (
                  <div className="flex items-center justify-center py-8">
                    <div className="text-center">
                      <BarChart3 className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">
                        Configure and run a backtest to see results
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </AppLayout>
  )
}
