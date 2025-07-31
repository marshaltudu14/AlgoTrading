"use client"

import * as React from "react"
import {
  createChart,
  ColorType,
  IChartApi,
  createSeriesMarkers,
  UTCTimestamp,
  CandlestickSeries,
  HistogramSeries,
  LineSeries
} from "lightweight-charts"
import { useTheme } from "next-themes"
import { gsap } from "gsap"

interface CandlestickData {
  time: string | number | UTCTimestamp
  open: number
  high: number
  low: number
  close: number
  volume?: number
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
  showVolume?: boolean
  showPortfolio?: boolean
  currentPrice?: number
  portfolioValue?: number
  fullScreen?: boolean
  className?: string
  windowSize?: number // For sliding window functionality
  enableSlidingWindow?: boolean // Whether to enable sliding window (only for backtests)
}

export function TradingChart({
  candlestickData = [],
  portfolioData = [],
  tradeMarkers = [],
  title = "Trading Chart",
  showVolume = true,
  showPortfolio = false,
  currentPrice,
  portfolioValue,
  fullScreen = false,
  className = "",
  windowSize = 100,
  enableSlidingWindow = false
}: TradingChartProps) {
  const chartContainerRef = React.useRef<HTMLDivElement>(null)
  const chartRef = React.useRef<IChartApi | null>(null)
  const candlestickSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const volumeSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const portfolioSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const { theme } = useTheme()
  const [chartHeight, setChartHeight] = React.useState(400)
  const [visibleData, setVisibleData] = React.useState<CandlestickData[]>([])
  const [dataIndex, setDataIndex] = React.useState(0)



  // Sliding window functionality (only for backtests)
  React.useEffect(() => {
    if (candlestickData.length === 0) return

    if (enableSlidingWindow && candlestickData.length > windowSize) {
      const startIndex = Math.max(0, candlestickData.length - windowSize)
      setVisibleData(candlestickData.slice(startIndex))
    } else {
      setVisibleData(candlestickData)
    }
  }, [candlestickData, windowSize, enableSlidingWindow])

  // Update chart height based on container size and mode with ResizeObserver
  React.useEffect(() => {
    const updateHeight = () => {
      if (chartContainerRef.current) {
        if (fullScreen) {
          setChartHeight(window.innerHeight - 40) // Subtract header height (2.5rem = 40px)
        } else {
          const containerHeight = chartContainerRef.current.parentElement?.clientHeight || 400
          setChartHeight(Math.max(containerHeight - 100, 300))
        }
      }
    }

    updateHeight()

    // Use ResizeObserver for better responsiveness
    let resizeObserver: ResizeObserver | null = null

    if (chartContainerRef.current?.parentElement) {
      resizeObserver = new ResizeObserver(() => {
        updateHeight()
      })
      resizeObserver.observe(chartContainerRef.current.parentElement)
    }

    // Fallback to window resize for fullscreen mode
    const handleWindowResize = () => {
      if (fullScreen) updateHeight()
    }

    window.addEventListener('resize', handleWindowResize)

    return () => {
      window.removeEventListener('resize', handleWindowResize)
      if (resizeObserver) {
        resizeObserver.disconnect()
      }
    }
  }, [fullScreen])

  // Initialize chart
  React.useEffect(() => {
    if (!chartContainerRef.current) return

    // Use simple theme-based colors instead of parsing CSS variables
    const isDark = theme === 'dark'
    const backgroundColor = isDark ? '#000000' : '#ffffff'
    const textColor = isDark ? '#ffffff' : '#000000'
    const borderColor = isDark ? '#333333' : '#e5e5e5'
    const mutedForegroundColor = isDark ? '#888888' : '#666666'

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: backgroundColor },
        textColor: textColor,
      },
      width: chartContainerRef.current.clientWidth,
      height: chartHeight,
      grid: {
        vertLines: { color: borderColor },
        horzLines: { color: borderColor },
      },
      crosshair: {
        mode: 1,
        vertLine: { color: mutedForegroundColor, labelBackgroundColor: mutedForegroundColor },
        horzLine: { color: mutedForegroundColor, labelBackgroundColor: mutedForegroundColor },
      },
      rightPriceScale: {
        borderColor: borderColor,
        textColor: mutedForegroundColor,
      },
      timeScale: {
        borderColor: borderColor,
        timeVisible: true,
        secondsVisible: false,
      },
      // Enable chart interactivity
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
        horzTouchDrag: true,
        vertTouchDrag: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    })

    chartRef.current = chart

    // Add candlestick series with theme-aware colors
    const upColor = isDark ? '#10b981' : '#059669'  // Emerald 500/600
    const downColor = isDark ? '#f87171' : '#dc2626'  // Red 400/600

    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: upColor,
      downColor: downColor,
      borderDownColor: downColor,
      borderUpColor: upColor,
      wickDownColor: downColor,
      wickUpColor: upColor,
    })
    candlestickSeriesRef.current = candlestickSeries

    // Add volume series if enabled
    if (showVolume) {
      const volumeColor = isDark ? '#14b8a6' : '#0d9488'  // Teal 500/600
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: volumeColor,
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: 'volume',
      })
      volumeSeriesRef.current = volumeSeries

      // Set volume scale to right
      chart.priceScale('volume').applyOptions({
        scaleMargins: {
          top: 0.8,
          bottom: 0,
        },
      })
    }

    // Add portfolio series if enabled
    if (showPortfolio) {
      const portfolioColor = isDark ? '#60a5fa' : '#2563eb'  // Blue 400/600
      const portfolioSeries = chart.addSeries(LineSeries, {
        color: portfolioColor,
        lineWidth: 2,
        priceScaleId: 'portfolio',
      })
      portfolioSeriesRef.current = portfolioSeries

      // Set portfolio scale to left
      chart.priceScale('portfolio').applyOptions({
        scaleMargins: {
          top: 0.1,
          bottom: 0.1,
        },
      })
    }

    // Handle resize with ResizeObserver for better responsiveness
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartHeight,
        })
      }
    }

    // Use ResizeObserver for chart container
    let chartResizeObserver: ResizeObserver | null = null

    if (chartContainerRef.current) {
      chartResizeObserver = new ResizeObserver(() => {
        handleResize()
      })
      chartResizeObserver.observe(chartContainerRef.current)
    }

    return () => {
      if (chartResizeObserver) {
        chartResizeObserver.disconnect()
      }
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [chartHeight, showVolume, showPortfolio, theme])

  // Update candlestick data with sliding window
  React.useEffect(() => {
    if (candlestickSeriesRef.current && visibleData.length > 0) {
      const formattedData = visibleData.map(item => ({
        time: typeof item.time === 'string' ?
          item.time.includes('T') ? item.time.split('T')[0] : item.time :
          (Math.floor(item.time as number / 1000) as UTCTimestamp),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))

      // Animate chart container for smooth data transitions
      if (chartContainerRef.current) {
        gsap.fromTo(chartContainerRef.current,
          { opacity: 0.8 },
          { opacity: 1, duration: 0.3, ease: "power2.out" }
        )
      }

      candlestickSeriesRef.current.setData(formattedData)

      // Auto-fit content with smooth animation
      setTimeout(() => {
        chartRef.current?.timeScale().fitContent()
      }, 100)
    }
  }, [visibleData])

  // Update volume data with sliding window
  React.useEffect(() => {
    if (volumeSeriesRef.current && visibleData.length > 0 && showVolume) {
      const volumeData = visibleData
        .filter(item => item.volume !== undefined)
        .map(item => ({
          time: typeof item.time === 'string' ?
            item.time.includes('T') ? item.time.split('T')[0] : item.time :
            (Math.floor(item.time as number / 1000) as UTCTimestamp),
          value: item.volume!,
          color: item.close >= item.open ? '#22c55e' : '#ef4444',
        }))

      if (volumeData.length > 0) {
        volumeSeriesRef.current.setData(volumeData)
      }
    }
  }, [visibleData, showVolume])

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



  if (fullScreen) {
    return (
      <div className={`h-full w-full bg-background ${className}`}>

        <div
          ref={chartContainerRef}
          style={{ height: `${chartHeight}px` }}
          className="w-full"
        />
      </div>
    )
  }

  return (
    <div className={`bg-background border rounded-lg ${className}`}>
      {/* Compact header for non-full-screen mode */}
      <div className="flex justify-between items-center p-3 border-b">
        <h3 className="text-sm font-medium">{title}</h3>

      </div>
      <div className="p-3">
        <div
          ref={chartContainerRef}
          style={{ height: `${chartHeight}px` }}
          className="w-full"
        />
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
