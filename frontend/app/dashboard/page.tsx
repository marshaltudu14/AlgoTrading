"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { gsap } from "gsap"
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  BarChart3,
  Clock,
  Target,
  Loader2
} from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/app-layout"
import { TradingChart } from "@/components/trading-chart"
import { ManualTradeForm } from "@/components/manual-trade-form"
import WebSocketService from "@/lib/websocket";
import { useLiveDataStore } from "@/store/live-data";
import { toast } from "sonner"

import { formatIndianCurrency } from "@/lib/formatters"

interface DashboardUserData {
  name: string;
  capital: number;
}

const AnimatedNumber = ({ value, prefix = "", suffix = "", formatter }: { 
  value: number; 
  prefix?: string; 
  suffix?: string; 
  formatter?: (num: number) => string;
}) => {
  const numberRef = React.useRef<HTMLSpanElement>(null)

  React.useEffect(() => {
    if (numberRef.current) {
      gsap.fromTo(numberRef.current, 
        { textContent: 0 },
        {
          textContent: value,
          duration: 2,
          ease: "power2.out",
          snap: { textContent: 1 },
          onUpdate: function() {
            if (numberRef.current) {
              const currentValue = Number(this.targets()[0].textContent);
              numberRef.current.textContent = formatter ? formatter(currentValue) : `${prefix}${currentValue.toLocaleString()}${suffix}`;
            }
          }
        }
      )
    }
  }, [value, prefix, suffix, formatter])

  return <span ref={numberRef}>{formatter ? formatter(0) : `${prefix}0${suffix}`}</span>
}

export default function DashboardPage() {
  const [userData, setUserData] = React.useState<DashboardUserData | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const [instruments, setInstruments] = React.useState<Instrument[]>([])
  const [timeframes, setTimeframes] = React.useState<string[]>([])
  const [isLoadingConfig, setIsLoadingConfig] = React.useState(true)
  const [configError, setConfigError] = React.useState<string | null>(null)
  const [selectedInstrument, setSelectedInstrument] = React.useState<string>("")
  const [selectedTimeframe, setSelectedTimeframe] = React.useState<string>("")
  const [historicalData, setHistoricalData] = React.useState<CandlestickData[]>([])
  const [isChartLoading, setIsChartLoading] = React.useState(false)
  const [chartError, setChartError] = React.useState<string | null>(null)
  const [metrics, setMetrics] = React.useState<Metrics | null>(null)
  const [isMetricsLoading, setIsMetricsLoading] = React.useState(true)
  const [metricsError, setMetricsError] = React.useState<string | null>(null)
  const [countdown, setCountdown] = React.useState("00:00")
  const [fetchStatus, setFetchStatus] = React.useState("Idle")
  const router = useRouter()

  const { isConnected, lastTick, activePosition, setActivePosition } = useLiveDataStore();

  React.useEffect(() => {
    const webSocketService = new WebSocketService(`ws://localhost:8000/ws/live/${userData.userId}`);
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
  }, [router, setActivePosition, userData.userId]);

  React.useEffect(() => {
    const fetchConfig = async () => {
      try {
        setIsLoadingConfig(true)
        setConfigError(null)
        
        const config = await apiClient.getConfig()
        
        setInstruments(config.instruments)
        setTimeframes(config.timeframes)
      } catch (err) {
        const errorMessage = formatApiError(err)
        console.error('Failed to fetch configuration:', err)
        setConfigError(errorMessage)
        toast.error('Failed to load configuration', {
          description: errorMessage
        })
      } finally {
        setIsLoadingConfig(false)
      }
    }
    fetchConfig()
  }, [])

  React.useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setIsMetricsLoading(true)
        setMetricsError(null)
        const data = await apiClient.getMetrics()
        setMetrics(data)
      } catch (err) {
        const errorMessage = formatApiError(err)
        console.error('Failed to fetch metrics:', err)
        setMetricsError(errorMessage)
        toast.error('Failed to load metrics', {
          description: errorMessage
        })
      } finally {
        setIsMetricsLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000) // Fetch every 10 seconds

    return () => clearInterval(interval)
  }, [])

  React.useEffect(() => {
    let timer: NodeJS.Timeout;

    const updateCountdown = () => {
      const status = useLiveDataStore.getState().status;
      if (status && status.nextFetchTimestamp) {
        const now = new Date().getTime();
        const nextFetch = new Date(status.nextFetchTimestamp).getTime();
        const difference = nextFetch - now;

        if (difference > 0) {
          const minutes = Math.floor((difference % (1000 * 60 * 60)) / (1000 * 60));
          const seconds = Math.floor((difference % (1000 * 60)) / 1000);
          setCountdown(`${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
          setFetchStatus("Waiting");
        } else {
          setCountdown("00:00");
          setFetchStatus("Fetching...");
        }
      }
    };

    const unsubscribe = useLiveDataStore.subscribe((state) => {
      if (state.status?.fetchError) {
        setFetchStatus(`Error: ${state.status.fetchError}`);
      }
    });

    timer = setInterval(updateCountdown, 1000);

    return () => {
      clearInterval(timer);
      unsubscribe();
    };
  }, []);

  const handleSelectionChange = React.useCallback(async (instrument?: string, timeframe?: string) => {
    // Use the current values or the provided ones
    const currentInstrument = instrument || selectedInstrument
    const currentTimeframe = timeframe || selectedTimeframe
    
    // Only fetch data if both selections are made
    if (!currentInstrument || !currentTimeframe) {
      return
    }

    try {
      setIsChartLoading(true)
      setChartError(null)
      
      console.log(`Fetching historical data for ${currentInstrument} with timeframe ${currentTimeframe}`)
      
      const data = await apiClient.getHistoricalData(currentInstrument, currentTimeframe)
      
      if (data && data.length > 0) {
        setHistoricalData(data)
        console.log(`Successfully loaded ${data.length} candles for ${currentInstrument}`)
      } else {
        setHistoricalData([])
        console.warn(`No historical data available for ${currentInstrument}`)
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
  }, [selectedInstrument, selectedTimeframe])

  const handleInstrumentChange = React.useCallback((value: string) => {
    setSelectedInstrument(value)
    handleSelectionChange(value, selectedTimeframe)
  }, [selectedTimeframe, handleSelectionChange])

  const handleManualTradeSubmit = async (values: any) => {
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
    setSelectedTimeframe(value)
    handleSelectionChange(selectedInstrument, value)
  }, [selectedInstrument, handleSelectionChange])

  const stats = [
    {
      title: "Available Capital",
      value: userData?.capital || 0,
      prefix: "₹",
      icon: DollarSign,
      description: "Total trading capital",
      color: "text-blue-600",
      formatter: formatIndianCurrency
    },
    {
      title: "Today's P&L",
      value: metrics?.todayPnL || 0,
      prefix: (metrics?.todayPnL || 0) >= 0 ? "+₹" : "-₹",
      icon: (metrics?.todayPnL || 0) >= 0 ? TrendingUp : TrendingDown,
      description: "Today's profit/loss",
      color: (metrics?.todayPnL || 0) >= 0 ? "text-green-600" : "text-red-600",
      formatter: formatIndianCurrency
    },
    {
      title: "Total Trades",
      value: metrics?.totalTrades || 0,
      icon: BarChart3,
      description: "Trades executed",
      color: "text-purple-600"
    },
    {
      title: "Win Rate",
      value: metrics?.winRate || 0,
      suffix: "%",
      icon: Target,
      description: "Success percentage",
      color: "text-orange-600"
    }
  ]

  return (
    <AppLayout>
      {isLoading ? (
        <div className="flex items-center justify-center min-h-screen">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl font-bold tracking-tight">
              Welcome back, {userData?.name}
            </h1>
            <p className="text-muted-foreground">
              Here&apos;s an overview of your trading performance
            </p>
          </motion.div>

          {/* Stats Grid */}
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {isMetricsLoading ? (
              Array.from({ length: 4 }).map((_, index) => (
                <Card key={index}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Loading...</CardTitle>
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-muted-foreground">-</div>
                    <p className="text-xs text-muted-foreground">Loading data...</p>
                  </CardContent>
                </Card>
              ))
            ) : metricsError ? (
              <div className="col-span-4 text-center text-destructive">
                <p>Failed to load metrics: {metricsError}</p>
              </div>
            ) : (
              stats.map((stat, index) => (
                <motion.div
                  key={stat.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">
                        {stat.title}
                      </CardTitle>
                      <stat.icon className={`h-4 w-4 ${stat.color}`} />
                    </CardHeader>
                    <CardContent>
                      <div className={`text-2xl font-bold ${stat.color}`}>
                        <AnimatedNumber 
                          value={Math.abs(stat.value)} 
                          prefix={stat.prefix} 
                          suffix={stat.suffix} 
                        />
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {stat.description}
                      </p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))
            )}
          </div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Recent Activity
                </CardTitle>
                <CardDescription>
                  Your latest trading activity and system status
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center gap-4 p-4 border rounded-lg">
                    <div className="p-2 bg-green-100 dark:bg-green-900 rounded-full">
                      <TrendingUp className="h-4 w-4 text-green-600" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Last trade executed</p>
                      <p className="text-sm text-muted-foreground">
                        Bank Nifty Call Option - Profit: ₹1,250
                      </p>
                    </div>
                    <div className="flex items-center gap-1 text-sm text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      {userData?.lastTradeTime}
                    </div>
                  </div>

                  <ConnectionStatus />

                  <div className="flex items-center gap-4 p-4 border rounded-lg">
                    <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-full">
                      <Clock className="h-4 w-4 text-blue-600" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Next Data Fetch</p>
                      <p className="text-sm text-muted-foreground">
                        {fetchStatus}
                      </p>
                    </div>
                    <div className="text-2xl font-bold text-blue-600">
                      {countdown}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* System Configuration */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>System Configuration</CardTitle>
                <CardDescription>
                  Available trading instruments and timeframes
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoadingConfig ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-primary mr-2" />
                    <span className="text-muted-foreground">Loading configuration...</span>
                  </div>
                ) : configError ? (
                  <div className="text-center py-8 text-destructive">
                    <p>Failed to load configuration</p>
                    <p className="text-sm text-muted-foreground mt-1">{configError}</p>
                  </div>
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Available Instruments</label>
                      <Select value={selectedInstrument} onValueChange={handleInstrumentChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select an instrument" />
                        </SelectTrigger>
                        <SelectContent>
                          {instruments.map((instrument) => (
                            <SelectItem key={instrument.symbol} value={instrument.symbol}>
                              {instrument.name} ({instrument.symbol})
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {instruments.length} instruments available
                      </p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-medium">Available Timeframes</label>
                      <Select value={selectedTimeframe} onValueChange={handleTimeframeChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a timeframe" />
                        </SelectTrigger>
                        <SelectContent>
                          {timeframes.map((timeframe) => (
                            <SelectItem key={timeframe} value={timeframe}>
                              {timeframe === 'D' ? 'Daily' : `${timeframe} minute${parseInt(timeframe) > 1 ? 's' : ''}`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {timeframes.length} timeframes available
                      </p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Trading Chart */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
          >
            <Card>
              <CardHeader>
                <CardTitle>Trading Chart</CardTitle>
                <CardDescription>
                  {selectedInstrument && selectedTimeframe 
                    ? `${selectedInstrument} - ${selectedTimeframe === 'D' ? 'Daily' : `${selectedTimeframe} minute${parseInt(selectedTimeframe) > 1 ? 's' : ''}`} chart`
                    : 'Select an instrument and timeframe to view the chart'
                  }
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!selectedInstrument || !selectedTimeframe ? (
                  <div className="h-96 flex items-center justify-center border-2 border-dashed border-muted-foreground/25 rounded-lg">
                    <div className="text-center">
                      <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        Select an instrument and timeframe to load the trading chart
                      </p>
                    </div>
                  </div>
                ) : isChartLoading ? (
                  <div className="h-96 flex items-center justify-center">
                    <div className="text-center">
                      <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
                      <p className="text-muted-foreground">
                        Loading historical data for {selectedInstrument}...
                      </p>
                    </div>
                  </div>
                ) : chartError ? (
                  <div className="h-96 flex items-center justify-center">
                    <div className="text-center text-destructive">
                      <p className="font-medium">Failed to load chart data</p>
                      <p className="text-sm text-muted-foreground mt-1">{chartError}</p>
                    </div>
                  </div>
                ) : (
                  <div className="h-96">
                    <TradingChart
                      candlestickData={historicalData}
                      title={`${selectedInstrument} - ${selectedTimeframe === 'D' ? 'Daily' : `${selectedTimeframe}m`}`}
                      className="h-full"
                      activePosition={activePosition}
                      stopLoss={activePosition?.stopLoss}
                      targetPrice={activePosition?.targetPrice}
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
          >
            </Card>
          </motion.div>

          {/* Manual Trade Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.9 }}
          >
            <ManualTradeForm 
              instruments={instruments}
              isDisabled={!!activePosition}
              onSubmit={handleManualTradeSubmit}
            />
          </motion.div>
        </div>
      )}
    </AppLayout>
  )
}
