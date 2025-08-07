import {
  ISeriesPrimitive,
  Time
} from 'lightweight-charts'
import {
  TrendlineCalculator,
  Trendline as CalculatedTrendline,
  defaultTrendlineCalculator
} from '../utils/trendline-calculator'
import { CandlestickData as PeakCandlestickData } from '../utils/peak-detection'
import { BreakoutDetector, BreakoutSignal } from '../utils/breakout-detector'

interface TrendlineConfig {
  recentPeriod: number // Number of recent candles to analyze (default: 25)
  historicalPeriod: number // Number of historical candles to search (default: 150)
  breakoutThreshold: number // Price threshold for breakout detection (default: 0.001)
  updateFrequency: number // How often to recalculate (in candles, default: 5)
}

// Helper function to convert Time to number
function timeToNumber(time: Time): number {
  if (typeof time === 'number') {
    return time
  }
  // For string dates, convert to timestamp
  if (typeof time === 'string') {
    return new Date(time).getTime() / 1000
  }
  // For BusinessDay, create a date and convert
  if (typeof time === 'object' && 'year' in time) {
    return new Date(time.year, time.month - 1, time.day).getTime() / 1000
  }
  return 0
}

export class TrendlinePrimitive implements ISeriesPrimitive {
  private _data: PeakCandlestickData[] = []
  private _upperTrendline: CalculatedTrendline | null = null
  private _lowerTrendline: CalculatedTrendline | null = null
  private _config: TrendlineConfig
  private _calculator: TrendlineCalculator
  private _breakoutDetector: BreakoutDetector
  private _lastUpdateTime: Time | null = null
  private _updateCounter: number = 0
  private _breakoutSignals: BreakoutSignal[] = []

  constructor(config: Partial<TrendlineConfig> = {}) {
    this._config = {
      recentPeriod: 25,
      historicalPeriod: 150,
      breakoutThreshold: 0.001,
      updateFrequency: 1, // Calculate trendlines on every update - FORCE REBUILD
      ...config
    }

    // Use very lenient parameters for better detection
    this._calculator = new TrendlineCalculator({
      maxSlope: 0.5,           // Allow very steep slopes
      minSlope: 0.000001,      // Allow extremely gentle slopes
      touchTolerance: 0.05,    // Very tolerant touching (5%)
      minTouchingPoints: 0,    // Don't require any additional touching points
      dynamicSelection: true,
      maxPointDistance: 0      // No distance limit
    })

    this._breakoutDetector = new BreakoutDetector({
      priceThreshold: this._config.breakoutThreshold,
      confirmationCandles: 1,  // Faster confirmation
      lookbackPeriod: 20
    })
  }

  // Update data and recalculate trendlines
  updateData(data: PeakCandlestickData[]): void {
    this._data = [...data]
    this._updateCounter++

    console.log('TrendlinePrimitive updateData:', {
      dataLength: this._data.length,
      historicalPeriod: this._config.historicalPeriod,
      updateCounter: this._updateCounter,
      updateFrequency: this._config.updateFrequency,
      hasEnoughData: this._data.length >= this._config.historicalPeriod
    })

    // Only recalculate if we have enough data - ALWAYS UPDATE FOR TESTING
    if (this._data.length >= 50) {
      const latestTime = this._data[this._data.length - 1]?.time
      const shouldUpdate = true // Force update every time for testing

      console.log('TrendlinePrimitive update check:', {
        latestTime,
        lastUpdateTime: this._lastUpdateTime,
        shouldUpdate,
        forceUpdate: true
      })

      if (shouldUpdate) {
        console.log('TrendlinePrimitive calling _calculateTrendlines')
        this._calculateTrendlines()
        this._detectBreakouts()
        this._lastUpdateTime = latestTime
      }
    }
  }

  // Get current breakout signals
  getBreakoutSignals(): BreakoutSignal[] {
    return [...this._breakoutSignals]
  }

  // Clear breakout signals (call after processing)
  clearBreakoutSignals(): void {
    this._breakoutSignals = []
  }

  // Get current trendlines for debugging
  getTrendlines(): { upper: CalculatedTrendline | null, lower: CalculatedTrendline | null } {
    return {
      upper: this._upperTrendline,
      lower: this._lowerTrendline
    }
  }

  // Get debug information
  getDebugInfo(): object {
    // Get peak detection info for debugging
    const allPeaks = this._data.length > 0 ? this._calculator.peakDetector.findPeaksAndLows(this._data) : []
    const peaks = allPeaks.filter(p => p.type === 'peak')
    const lows = allPeaks.filter(p => p.type === 'low')

    return {
      dataLength: this._data.length,
      hasUpperTrendline: !!this._upperTrendline,
      hasLowerTrendline: !!this._lowerTrendline,
      upperTrendlineScore: this._upperTrendline?.score || 0,
      lowerTrendlineScore: this._lowerTrendline?.score || 0,
      breakoutSignalsCount: this._breakoutSignals.length,
      lastUpdateTime: this._lastUpdateTime,
      peaksDetected: peaks.length,
      lowsDetected: lows.length,
      totalPointsDetected: allPeaks.length,
      samplePeaks: peaks.slice(0, 3).map(p => ({ time: p.time, price: p.price, significance: p.significance })),
      sampleLows: lows.slice(0, 3).map(p => ({ time: p.time, price: p.price, significance: p.significance }))
    }
  }

  // ISeriesPrimitive interface implementation
  paneViews() {
    return [new TrendlinePaneView(this._upperTrendline, this._lowerTrendline, this._data)]
  }

  // Calculate trendlines based on current data
  private _calculateTrendlines(): void {
    if (this._data.length < 50) return // Need sufficient data

    console.log('_calculateTrendlines called with data length:', this._data.length)

    // Find significant peaks and valleys for proper trendlines
    const peaks = this._findPeaks()
    const valleys = this._findValleys()

    console.log('Found peaks:', peaks.length, 'valleys:', valleys.length)

    // Create resistance line from recent peaks
    if (peaks.length >= 2) {
      const recentPeaks = peaks.slice(-3) // Use last 3 peaks
      const firstPeak = recentPeaks[0]
      const lastPeak = recentPeaks[recentPeaks.length - 1]

      this._upperTrendline = {
        point1: {
          time: firstPeak.time,
          price: firstPeak.price,
          significance: 1.0,
          index: firstPeak.index
        },
        point2: {
          time: lastPeak.time,
          price: lastPeak.price,
          significance: 1.0,
          index: lastPeak.index
        },
        slope: (lastPeak.price - firstPeak.price) / (lastPeak.index - firstPeak.index),
        strength: 1.0,
        score: 1.0,
        touchingPoints: recentPeaks.length,
        type: 'resistance'
      }
    }

    // Create support line from recent valleys
    if (valleys.length >= 2) {
      const recentValleys = valleys.slice(-3) // Use last 3 valleys
      const firstValley = recentValleys[0]
      const lastValley = recentValleys[recentValleys.length - 1]

      this._lowerTrendline = {
        point1: {
          time: firstValley.time,
          price: firstValley.price,
          significance: 1.0,
          index: firstValley.index
        },
        point2: {
          time: lastValley.time,
          price: lastValley.price,
          significance: 1.0,
          index: lastValley.index
        },
        slope: (lastValley.price - firstValley.price) / (lastValley.index - firstValley.index),
        strength: 1.0,
        score: 1.0,
        touchingPoints: recentValleys.length,
        type: 'support'
      }
    }

    console.log('Trendlines created:', {
      hasUpper: !!this._upperTrendline,
      hasLower: !!this._lowerTrendline,
      upperPoints: this._upperTrendline ? [this._upperTrendline.point1.price, this._upperTrendline.point2.price] : null,
      lowerPoints: this._lowerTrendline ? [this._lowerTrendline.point1.price, this._lowerTrendline.point2.price] : null
    })
  }

  /**
   * Find significant peaks in the data
   */
  private _findPeaks(): Array<{time: Time, price: number, index: number}> {
    const peaks: Array<{time: Time, price: number, index: number}> = []
    const lookback = 5 // Look 5 candles back and forward

    for (let i = lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isPeak = true

      // Check if current candle high is higher than surrounding candles
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && this._data[j].high >= current.high) {
          isPeak = false
          break
        }
      }

      if (isPeak) {
        peaks.push({
          time: current.time,
          price: current.high,
          index: i
        })
      }
    }

    return peaks
  }

  /**
   * Find significant valleys in the data
   */
  private _findValleys(): Array<{time: Time, price: number, index: number}> {
    const valleys: Array<{time: Time, price: number, index: number}> = []
    const lookback = 5 // Look 5 candles back and forward

    for (let i = lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isValley = true

      // Check if current candle low is lower than surrounding candles
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i && this._data[j].low <= current.low) {
          isValley = false
          break
        }
      }

      if (isValley) {
        valleys.push({
          time: current.time,
          price: current.low,
          index: i
        })
      }
    }

    return valleys
  }

  /**
   * Simple resistance line: connect highest point in recent data to highest point in historical data
   */
  private _calculateSimpleResistance(): CalculatedTrendline | null {
    if (this._data.length < 50) return null

    const recentStart = Math.floor(this._data.length * 0.75) // Last 25% of data
    const historicalEnd = Math.floor(this._data.length * 0.75) // First 75% of data

    // Find highest point in recent data
    let recentHigh = this._data[recentStart]
    let recentIndex = recentStart
    for (let i = recentStart; i < this._data.length; i++) {
      if (this._data[i].high > recentHigh.high) {
        recentHigh = this._data[i]
        recentIndex = i
      }
    }

    // Find highest point in historical data
    let historicalHigh = this._data[0]
    let historicalIndex = 0
    for (let i = 0; i < historicalEnd; i++) {
      if (this._data[i].high > historicalHigh.high) {
        historicalHigh = this._data[i]
        historicalIndex = i
      }
    }

    // Create trendline
    const timeSpan = timeToNumber(recentHigh.time) - timeToNumber(historicalHigh.time)
    const priceSpan = recentHigh.high - historicalHigh.high
    const slope = timeSpan !== 0 ? priceSpan / timeSpan : 0

    return {
      point1: {
        time: historicalHigh.time,
        price: historicalHigh.high,
        significance: 1.0,
        index: historicalIndex
      },
      point2: {
        time: recentHigh.time,
        price: recentHigh.high,
        significance: 1.0,
        index: recentIndex
      },
      slope,
      strength: 1.0,
      score: 1.0,
      touchingPoints: 2,
      type: 'resistance'
    }
  }

  /**
   * Simple support line: connect lowest point in recent data to lowest point in historical data
   */
  private _calculateSimpleSupport(): CalculatedTrendline | null {
    if (this._data.length < 50) return null

    const recentStart = Math.floor(this._data.length * 0.75) // Last 25% of data
    const historicalEnd = Math.floor(this._data.length * 0.75) // First 75% of data

    // Find lowest point in recent data
    let recentLow = this._data[recentStart]
    for (let i = recentStart; i < this._data.length; i++) {
      if (this._data[i].low < recentLow.low) {
        recentLow = this._data[i]
      }
    }

    // Find lowest point in historical data
    let historicalLow = this._data[0]
    for (let i = 0; i < historicalEnd; i++) {
      if (this._data[i].low < historicalLow.low) {
        historicalLow = this._data[i]
      }
    }

    // Create trendline
    const timeSpan = timeToNumber(recentLow.time) - timeToNumber(historicalLow.time)
    const priceSpan = recentLow.low - historicalLow.low
    const slope = timeSpan !== 0 ? priceSpan / timeSpan : 0

    return {
      point1: {
        time: historicalLow.time,
        price: historicalLow.low,
        significance: 1.0,
        index: 0
      },
      point2: {
        time: recentLow.time,
        price: recentLow.low,
        significance: 1.0,
        index: recentStart
      },
      slope,
      strength: 1.0,
      score: 1.0,
      touchingPoints: 2,
      type: 'support'
    }
  }

  // Detect breakouts from current trendlines
  private _detectBreakouts(): void {
    if (this._data.length === 0) return

    // Use the breakout detector to find breakouts
    const breakouts = this._breakoutDetector.detectBreakouts(
      this._data,
      this._upperTrendline,
      this._lowerTrendline
    )

    // Add new breakouts to our signals
    this._breakoutSignals.push(...breakouts)
  }
}

// Pane view for rendering trendlines
class TrendlinePaneView {
  constructor(
    private _upperTrendline: CalculatedTrendline | null,
    private _lowerTrendline: CalculatedTrendline | null,
    private _data: PeakCandlestickData[]
  ) {}

  zOrder(): any {
    return 'top' // Draw on top of other elements
  }

  renderer() {
    return new TrendlineRenderer(this._upperTrendline, this._lowerTrendline, this._data)
  }
}

// Renderer for drawing trendlines
class TrendlineRenderer {
  constructor(
    private _upperTrendline: CalculatedTrendline | null,
    private _lowerTrendline: CalculatedTrendline | null,
    private _data: PeakCandlestickData[]
  ) {}

  draw(target: any): void {
    // Use media coordinate space for simpler drawing
    target.useMediaCoordinateSpace((scope: any) => {
      const ctx = scope.context

      // Save context state
      ctx.save()

      try {
        // Draw upper trendline (resistance)
        if (this._upperTrendline) {
          this._drawTrendlineInMediaSpace(ctx, this._upperTrendline, '#ff4444', 2, scope)
        }

        // Draw lower trendline (support)
        if (this._lowerTrendline) {
          this._drawTrendlineInMediaSpace(ctx, this._lowerTrendline, '#44ff44', 2, scope)
        }
      } finally {
        // Restore context state
        ctx.restore()
      }
    })
  }

  private _drawTrendlineInMediaSpace(
    ctx: CanvasRenderingContext2D,
    trendline: CalculatedTrendline,
    color: string,
    width: number,
    scope: { mediaSize: { width: number; height: number } }
  ): void {
    if (this._data.length === 0) return

    ctx.strokeStyle = color
    ctx.lineWidth = width
    ctx.setLineDash([5, 5]) // Dashed line

    // Get the time range of visible data
    const firstTime = timeToNumber(this._data[0].time)
    const lastTime = timeToNumber(this._data[this._data.length - 1].time)

    // Calculate trendline prices at start and end
    const startPrice = this._getTrendlinePrice(trendline, firstTime)
    const endPrice = this._getTrendlinePrice(trendline, lastTime)

    if (startPrice !== null && endPrice !== null) {
      // Use media coordinate space for drawing
      const mediaWidth = scope.mediaSize.width
      const mediaHeight = scope.mediaSize.height

      // Simple linear mapping across the full width
      const x1 = 0
      const x2 = mediaWidth

      // Map prices to canvas Y coordinates (inverted because canvas Y increases downward)
      const priceRange = Math.max(...this._data.map(d => d.high)) - Math.min(...this._data.map(d => d.low))
      const minPrice = Math.min(...this._data.map(d => d.low))

      const y1 = mediaHeight - ((startPrice - minPrice) / priceRange) * mediaHeight
      const y2 = mediaHeight - ((endPrice - minPrice) / priceRange) * mediaHeight

      ctx.beginPath()
      ctx.moveTo(x1, y1)
      ctx.lineTo(x2, y2)
      ctx.stroke()
    }

    ctx.setLineDash([]) // Reset line dash
  }

  private _getTrendlinePrice(trendline: CalculatedTrendline, time: number): number | null {
    const time1 = timeToNumber(trendline.point1.time)
    const timeDiff = time - time1
    return trendline.point1.price + (trendline.slope * timeDiff)
  }
}