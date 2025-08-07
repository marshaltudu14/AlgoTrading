"use client"

import * as React from "react"
import {
  createChart,
  ColorType,
  IChartApi,
  createSeriesMarkers,
  UTCTimestamp,
  CandlestickSeries,
  LineSeries
} from "lightweight-charts"
import { useTheme } from "next-themes"
import { ZoomIn, ZoomOut, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

interface CandlestickData {
  time: string | number | UTCTimestamp
  open: number
  high: number
  low: number
  close: number
}

interface TradeMarker {
  time: string | number | UTCTimestamp
  position: 'aboveBar' | 'belowBar'
  color: string
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown'
  text: string
  size?: number
}

interface PortfolioData {
  time: string | number | UTCTimestamp
  value: number
}

interface TradingChartProps {
  candlestickData?: CandlestickData[]
  portfolioData?: PortfolioData[]
  tradeMarkers?: TradeMarker[]
  title?: string
  showPortfolio?: boolean
  currentPrice?: number
  portfolioValue?: number
  fullScreen?: boolean
  className?: string
  windowSize?: number // For sliding window functionality
  enableSlidingWindow?: boolean // Whether to enable sliding window (only for backtests)
  stopLoss?: number
  targetPrice?: number
}

import { useLiveDataStore } from "@/store/live-data";

export function TradingChart({
  candlestickData = [],
  portfolioData = [],
  tradeMarkers = [],
  title = "Trading Chart",
  showPortfolio = false,
  fullScreen = false,
  className = "",
  windowSize: _windowSize = 100, // eslint-disable-line @typescript-eslint/no-unused-vars
  enableSlidingWindow = false,
  stopLoss,
  targetPrice,
}: TradingChartProps) {
  const chartContainerRef = React.useRef<HTMLDivElement>(null)
  const chartRef = React.useRef<IChartApi | null>(null)
  const candlestickSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const portfolioSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const slPriceLineRef = React.useRef<any>(null)
  const tpPriceLineRef = React.useRef<any>(null)
  const { theme } = useTheme()
  const [chartHeight, setChartHeight] = React.useState(400) // eslint-disable-line @typescript-eslint/no-unused-vars


  // Zoom control functions
  const handleZoomIn = React.useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale()
      const logicalRange = timeScale.getVisibleLogicalRange()
      if (logicalRange) {
        const newLogicalRange = {
          from: logicalRange.from + (logicalRange.to - logicalRange.from) * 0.1,
          to: logicalRange.to - (logicalRange.to - logicalRange.from) * 0.1,
        }
        timeScale.setVisibleLogicalRange(newLogicalRange)
      }
    }
  }, [])

  const handleZoomOut = React.useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale()
      const logicalRange = timeScale.getVisibleLogicalRange()
      if (logicalRange) {
        const newLogicalRange = {
          from: logicalRange.from - (logicalRange.to - logicalRange.from) * 0.1,
          to: logicalRange.to + (logicalRange.to - logicalRange.from) * 0.1,
        }
        timeScale.setVisibleLogicalRange(newLogicalRange)
      }
    }
  }, [])

  const handleResetView = React.useCallback(() => {
    if (chartRef.current) {
      chartRef.current.timeScale().resetTimeScale()
      chartRef.current.timeScale().fitContent()
    }
  }, [])



  // Initialize chart and handle updates
  React.useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: chartContainerRef.current.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: theme === 'dark' ? '#ffffff' : '#000000',
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: theme === 'dark' ? '#333333' : '#e5e5e5',
      },
      timeScale: {
        borderColor: theme === 'dark' ? '#333333' : '#e5e5e5',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    })
    chartRef.current = chart

    const candlestickSeries = chart.addSeries(CandlestickSeries, {})
    candlestickSeriesRef.current = candlestickSeries

    if (showPortfolio) {
      const portfolioSeries = chart.addSeries(LineSeries, {
        priceScaleId: 'portfolio',
        lineWidth: 2,
      })
      portfolioSeriesRef.current = portfolioSeries
      chart.priceScale('portfolio').applyOptions({
        scaleMargins: { top: 0.1, bottom: 0.1 },
      })
    }

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.resize(
          chartContainerRef.current.clientWidth,
          chartContainerRef.current.clientHeight
        )
      }
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(chartContainerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
    }
  }, [showPortfolio])

  // Update theme
  React.useEffect(() => {
    if (!chartRef.current) return

    const isDark = theme === 'dark'
    const backgroundColor = isDark ? '#000000' : '#ffffff'
    const textColor = isDark ? '#ffffff' : '#000000'
    const borderColor = isDark ? '#333333' : '#e5e5e5'
    const mutedForegroundColor = isDark ? '#888888' : '#666666'
    const upColor = isDark ? '#10b981' : '#059669'
    const downColor = isDark ? '#f87171' : '#dc2626'
    const portfolioColor = isDark ? '#60a5fa' : '#2563eb'

    chartRef.current.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: backgroundColor },
        textColor: textColor,
      },
      crosshair: {
        vertLine: { color: mutedForegroundColor },
        horzLine: { color: mutedForegroundColor },
      },
      rightPriceScale: {
        borderColor: borderColor,
        textColor: mutedForegroundColor,
      },
      timeScale: {
        borderColor: borderColor,
      },
    })

    if (candlestickSeriesRef.current) {
      candlestickSeriesRef.current.applyOptions({
        upColor: upColor,
        downColor: downColor,
        borderDownColor: downColor,
        borderUpColor: upColor,
        wickDownColor: downColor,
        wickUpColor: upColor,
      })
    }
    if (portfolioSeriesRef.current) {
      portfolioSeriesRef.current.applyOptions({
        color: portfolioColor,
      })
    }
  }, [theme])

  // Update height
  React.useEffect(() => {
    const updateHeight = () => {
      if (chartContainerRef.current) {
        if (fullScreen) {
          chartContainerRef.current.style.height = `${window.innerHeight - 40}px`
        } else {
          const parentHeight = chartContainerRef.current.parentElement?.clientHeight || 400
          chartContainerRef.current.style.height = `${Math.max(parentHeight - 100, 300)}px`
        }
        if (chartRef.current) {
          chartRef.current.resize(
            chartContainerRef.current.clientWidth,
            chartContainerRef.current.clientHeight
          )
        }
      }
    }
    updateHeight()

    if (fullScreen) {
      window.addEventListener('resize', updateHeight)
      return () => window.removeEventListener('resize', updateHeight)
    }
  }, [fullScreen])

  const { lastTick } = useLiveDataStore();

  // Real-time data updates using TradingView's update() method
  React.useEffect(() => {
    if (candlestickSeriesRef.current && lastTick) {
      const formattedCandle = {
        time: lastTick.timestamp as UTCTimestamp,
        open: lastTick.open,
        high: lastTick.high,
        low: lastTick.low,
        close: lastTick.price,
      };

      candlestickSeriesRef.current.update(formattedCandle);

      if (chartRef.current) {
        chartRef.current.timeScale().scrollToRealTime();
      }
    }
  }, [lastTick]);

  // Initial data loading for demo/static data
  React.useEffect(() => {
    if (candlestickSeriesRef.current && candlestickData.length > 0 && !enableSlidingWindow) {
      const formattedData = candlestickData.map(item => ({
        time: typeof item.time === 'string' ?
          item.time.includes('T') ? item.time.split('T')[0] : item.time :
          (item.time as UTCTimestamp),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))

      candlestickSeriesRef.current.setData(formattedData)

      // Auto-fit content
      setTimeout(() => {
        chartRef.current?.timeScale().fitContent()
      }, 100)
    }
  }, [candlestickData, enableSlidingWindow])



  // Update portfolio data
  React.useEffect(() => {
    if (portfolioSeriesRef.current && portfolioData.length > 0 && showPortfolio) {
      const formattedData = portfolioData.map(item => ({
        time: typeof item.time === 'string' ?
          item.time.includes('T') ? item.time.split('T')[0] : item.time :
          (Math.floor(item.time as number / 1000) as UTCTimestamp),
        value: item.value,
      }))
      portfolioSeriesRef.current.setData(formattedData)
    }
  }, [portfolioData, showPortfolio])

  // Update trade markers
  React.useEffect(() => {
    if (candlestickSeriesRef.current && tradeMarkers.length > 0) {
      const formattedMarkers = tradeMarkers.map(marker => ({
        time: typeof marker.time === 'string' ?
          marker.time.includes('T') ? marker.time.split('T')[0] : marker.time :
          (Math.floor(marker.time as number / 1000) as UTCTimestamp),
        position: marker.position,
        color: marker.color,
        shape: marker.shape,
        text: marker.text,
      }))

      // Use the new v5 markers API
      createSeriesMarkers(candlestickSeriesRef.current, formattedMarkers)
    }
  }, [tradeMarkers])

  // Update SL/TP lines
  React.useEffect(() => {
    if (!candlestickSeriesRef.current) return;

    // Remove existing lines before adding new ones
    if (slPriceLineRef.current) {
      candlestickSeriesRef.current.removePriceLine(slPriceLineRef.current);
      slPriceLineRef.current = null;
    }
    if (tpPriceLineRef.current) {
      candlestickSeriesRef.current.removePriceLine(tpPriceLineRef.current);
      tpPriceLineRef.current = null;
    }

    if (stopLoss) {
      slPriceLineRef.current = candlestickSeriesRef.current.createPriceLine({
        price: stopLoss,
        color: '#ef4444',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'SL',
      });
    }

    if (targetPrice) {
      tpPriceLineRef.current = candlestickSeriesRef.current.createPriceLine({
        price: targetPrice,
        color: '#22c55e',
        lineWidth: 2,
        lineStyle: 2, // Dashed
        axisLabelVisible: true,
        title: 'TP',
      });
    }
  }, [stopLoss, targetPrice]);



  if (fullScreen) {
    return (
      <div className={`h-full w-full bg-background relative ${className}`}>
        {/* Floating zoom controls */}
        <div className="absolute bottom-10 left-0 right-0 z-10 flex justify-center items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleZoomIn}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleZoomOut}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleResetView}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>

        <div
          ref={chartContainerRef}
          style={{ height: `${chartHeight}px` }}
          className="w-full"
        />
      </div>
    )
  }

  return (
    <div className={`bg-background border rounded-lg relative ${className}`}>
      {/* Compact header for non-full-screen mode */}
      <div className="flex justify-between items-center p-3 border-b">
        <h3 className="text-sm font-medium">{title}</h3>
      </div>
      <div className="p-3 relative">
        <div
          ref={chartContainerRef}
          style={{ height: `${chartHeight}px` }}
          className="w-full"
        />
        {/* Floating zoom controls for non-fullscreen */}
        <div className="absolute bottom-10 left-0 right-0 z-10 flex justify-center items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleZoomIn}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleZoomOut}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleResetView}
            className="h-8 w-8 p-0 bg-background/80 backdrop-blur-sm"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}

// Helper function to create trade markers
export function createTradeMarker(
  time: string | number | UTCTimestamp,
  action: 'BUY' | 'SELL' | 'CLOSE_LONG' | 'CLOSE_SHORT' | 'HOLD',
  price?: number
): TradeMarker {
  const markerConfig = {
    BUY: {
      position: 'belowBar' as const,
      color: '#22c55e',
      shape: 'arrowUp' as const,
      text: 'B',
    },
    SELL: {
      position: 'aboveBar' as const,
      color: '#ef4444',
      shape: 'arrowDown' as const,
      text: 'S',
    },
    CLOSE_LONG: {
      position: 'aboveBar' as const,
      color: '#3b82f6',
      shape: 'circle' as const,
      text: 'CL',
    },
    CLOSE_SHORT: {
      position: 'belowBar' as const,
      color: '#f59e0b',
      shape: 'circle' as const,
      text: 'CS',
    },
    HOLD: {
      position: 'aboveBar' as const,
      color: '#6b7280',
      shape: 'square' as const,
      text: 'H',
    },
  }

  const config = markerConfig[action]
  return {
    time,
    position: config.position,
    color: config.color,
    shape: config.shape,
    text: price ? `${config.text}\n${price}` : config.text,
  }
}
