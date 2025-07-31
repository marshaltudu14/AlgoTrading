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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

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
  height?: number
  title?: string
  showVolume?: boolean
  showPortfolio?: boolean
  currentPrice?: number
  portfolioValue?: number
}

export function TradingChart({
  candlestickData = [],
  portfolioData = [],
  tradeMarkers = [],
  height = 400,
  title = "Trading Chart",
  showVolume = true,
  showPortfolio = false,
  currentPrice,
  portfolioValue
}: TradingChartProps) {
  const chartContainerRef = React.useRef<HTMLDivElement>(null)
  const chartRef = React.useRef<IChartApi | null>(null)
  const candlestickSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const volumeSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)
  const portfolioSeriesRef = React.useRef<ReturnType<IChartApi['addSeries']> | null>(null)

  // Initialize chart
  React.useEffect(() => {
    if (!chartContainerRef.current) return

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#333',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#e1e5e9' },
        horzLines: { color: '#e1e5e9' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#cccccc',
      },
      timeScale: {
        borderColor: '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
    })

    chartRef.current = chart

    // Add candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    })
    candlestickSeriesRef.current = candlestickSeries

    // Add volume series if enabled
    if (showVolume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#26a69a',
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
      const portfolioSeries = chart.addSeries(LineSeries, {
        color: '#3b82f6',
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

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [height, showVolume, showPortfolio])

  // Update candlestick data
  React.useEffect(() => {
    if (candlestickSeriesRef.current && candlestickData.length > 0) {
      const formattedData = candlestickData.map(item => ({
        time: typeof item.time === 'string' ?
          item.time.includes('T') ? item.time.split('T')[0] : item.time :
          (Math.floor(item.time as number / 1000) as UTCTimestamp),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      }))
      candlestickSeriesRef.current.setData(formattedData)
    }
  }, [candlestickData])

  // Update volume data
  React.useEffect(() => {
    if (volumeSeriesRef.current && candlestickData.length > 0 && showVolume) {
      const volumeData = candlestickData
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
  }, [candlestickData, showVolume])

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

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <CardTitle>{title}</CardTitle>
          <div className="flex gap-4 text-sm">
            {currentPrice && (
              <div className="text-blue-600 font-medium">
                Price: {formatCurrency(currentPrice)}
              </div>
            )}
            {portfolioValue && (
              <div className="text-green-600 font-medium">
                Portfolio: {formatCurrency(portfolioValue)}
              </div>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div 
          ref={chartContainerRef} 
          style={{ height: `${height}px` }}
          className="w-full"
        />
      </CardContent>
    </Card>
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
