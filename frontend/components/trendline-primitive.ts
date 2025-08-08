import {
  ISeriesPrimitive,
  Time
} from 'lightweight-charts'
import {
  TrendlineCalculator,
  Trendline as CalculatedTrendline
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
  private _chart: any = null
  private _series: any = null

  constructor(config: Partial<TrendlineConfig> = {}) {
    this._config = {
      recentPeriod: 25,
      historicalPeriod: 150,
      breakoutThreshold: 0.001,
      updateFrequency: 1, // Calculate trendlines on every update
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

  // Series primitive lifecycle methods
  attached(param: any): void {
    this._chart = param.chart
    this._series = param.series
  }

  detached(): void {
    this._chart = null
    this._series = null
  }

  // Update data and recalculate trendlines - focus on latest data
  updateData(data: PeakCandlestickData[]): void {
    this._data = [...data]
    this._updateCounter++

    // Recalculate trendlines more frequently for latest data
    if (this._data.length >= 20) {
      const latestTime = this._data[this._data.length - 1]?.time
      
      // Always recalculate to get the latest trendlines
      this._calculateTrendlines()
      this._detectBreakouts()
      this._lastUpdateTime = latestTime
      
      // Log update for debugging
      console.log('🔄 Trendlines recalculated for latest data:', {
        totalCandles: this._data.length,
        latestPrice: this._data[this._data.length - 1]?.close,
        updateCounter: this._updateCounter
      })
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
    const peaks = this._findSignificantPeaks()
    const valleys = this._findSignificantValleys()

    return {
      dataLength: this._data.length,
      hasUpperTrendline: !!this._upperTrendline,
      hasLowerTrendline: !!this._lowerTrendline,
      upperTrendlineScore: this._upperTrendline?.score || 0,
      lowerTrendlineScore: this._lowerTrendline?.score || 0,
      peaksDetected: peaks.length,
      valleysDetected: valleys.length
    }
  }

  // ISeriesPrimitive interface implementation
  paneViews() {
    return [new TrendlinePaneView(this._upperTrendline, this._lowerTrendline, this._data, this._chart, this._series)]
  }

  // Calculate trendlines based on current data - focus on recent price action
  private _calculateTrendlines(): void {
    if (this._data.length < 20) return // Need minimum data

    // Focus on recent price action for more relevant trendlines
    this._upperTrendline = this._findLatestResistanceLine()
    this._lowerTrendline = this._findLatestSupportLine()

    // Only log if we actually have trendlines
    if (this._upperTrendline || this._lowerTrendline) {
      console.log('📊 Latest Trendlines Updated:', {
        resistance: this._upperTrendline ? '✅' : '❌',
        support: this._lowerTrendline ? '✅' : '❌',
        resistanceSlope: this._upperTrendline?.slope.toFixed(6),
        supportSlope: this._lowerTrendline?.slope.toFixed(6),
        dataLength: this._data.length
      })
    }
  }

  /**
   * Find latest resistance line by connecting the most recent significant high with a previous high
   */
  private _findLatestResistanceLine(): CalculatedTrendline | null {
    if (this._data.length < 20) return null

    // Get recent peaks with a more aggressive detection for current price action
    const recentPeaks = this._findRecentPeaks()
    if (recentPeaks.length < 2) return null

    // Sort peaks by time (most recent first)
    const sortedPeaks = recentPeaks.sort((a, b) => b.index - a.index)
    
    // Find the most recent significant high
    const mostRecentPeak = sortedPeaks[0]
    
    // Find the best historical peak to connect to
    let bestTrendline: CalculatedTrendline | null = null
    let minTimeDiff = Infinity
    
    // Look for peaks that are at least 10 candles away from the most recent
    for (let i = 1; i < sortedPeaks.length; i++) {
      const historicalPeak = sortedPeaks[i]
      const timeDiff = mostRecentPeak.index - historicalPeak.index
      
      // Ensure minimum separation and prioritize closer-in-time peaks for relevance
      if (timeDiff >= 10 && timeDiff < minTimeDiff) {
        const trendline = this._createTrendlineFromPoints(
          mostRecentPeak, historicalPeak, 'resistance'
        )
        
        if (trendline && this._isRecentTrendlineValid(trendline)) {
          minTimeDiff = timeDiff
          bestTrendline = trendline
        }
      }
    }

    return bestTrendline
  }

  /**
   * Find latest support line by connecting the most recent significant low with a previous low
   */
  private _findLatestSupportLine(): CalculatedTrendline | null {
    if (this._data.length < 20) return null

    // Get recent valleys with a more aggressive detection for current price action
    const recentValleys = this._findRecentValleys()
    if (recentValleys.length < 2) return null

    // Sort valleys by time (most recent first)
    const sortedValleys = recentValleys.sort((a, b) => b.index - a.index)
    
    // Find the most recent significant low
    const mostRecentValley = sortedValleys[0]
    
    // Find the best historical valley to connect to
    let bestTrendline: CalculatedTrendline | null = null
    let minTimeDiff = Infinity
    
    // Look for valleys that are at least 10 candles away from the most recent
    for (let i = 1; i < sortedValleys.length; i++) {
      const historicalValley = sortedValleys[i]
      const timeDiff = mostRecentValley.index - historicalValley.index
      
      // Ensure minimum separation and prioritize closer-in-time valleys for relevance
      if (timeDiff >= 10 && timeDiff < minTimeDiff) {
        const trendline = this._createTrendlineFromPoints(
          mostRecentValley, historicalValley, 'support'
        )
        
        if (trendline && this._isRecentTrendlineValid(trendline)) {
          minTimeDiff = timeDiff
          bestTrendline = trendline
        }
      }
    }

    return bestTrendline
  }

  /**
   * Find recent peaks focusing on the latest price action (last 100 candles)
   */
  private _findRecentPeaks(): Array<{time: Time, price: number, index: number}> {
    const peaks: Array<{time: Time, price: number, index: number}> = []
    const lookback = 5 // Look 5 candles back and forward
    const recentDataStart = Math.max(0, this._data.length - 100) // Focus on last 100 candles
    
    for (let i = recentDataStart + lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isPeak = true
      
      // Check if current candle HIGH is higher than surrounding candles
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
    
    // Filter out peaks that are too close to each other - keep the highest one in each group
    const filteredPeaks = this._filterClosePeaks(peaks)
    return filteredPeaks.sort((a, b) => timeToNumber(a.time) - timeToNumber(b.time))
  }

  /**
   * Find recent valleys focusing on the latest price action (last 100 candles)
   */
  private _findRecentValleys(): Array<{time: Time, price: number, index: number}> {
    const valleys: Array<{time: Time, price: number, index: number}> = []
    const lookback = 5 // Look 5 candles back and forward
    const recentDataStart = Math.max(0, this._data.length - 100) // Focus on last 100 candles
    
    for (let i = recentDataStart + lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isValley = true
      
      // Check if current candle LOW is lower than surrounding candles
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
    
    // Filter out valleys that are too close to each other - keep the lowest one in each group
    const filteredValleys = this._filterCloseValleys(valleys)
    return filteredValleys.sort((a, b) => timeToNumber(a.time) - timeToNumber(b.time))
  }

  /**
   * Filter out peaks that are too close together, keeping the highest
   */
  private _filterClosePeaks(peaks: Array<{time: Time, price: number, index: number}>): Array<{time: Time, price: number, index: number}> {
    if (peaks.length <= 1) return peaks
    
    const filtered: Array<{time: Time, price: number, index: number}> = []
    const minDistance = 8 // Minimum 8 candles apart
    
    for (let i = 0; i < peaks.length; i++) {
      const current = peaks[i]
      let shouldAdd = true
      
      // Check if there's a higher peak within minDistance
      for (let j = 0; j < peaks.length; j++) {
        if (i !== j) {
          const other = peaks[j]
          const distance = Math.abs(current.index - other.index)
          
          if (distance < minDistance && other.price > current.price) {
            shouldAdd = false
            break
          }
        }
      }
      
      if (shouldAdd) {
        filtered.push(current)
      }
    }
    
    return filtered
  }

  /**
   * Filter out valleys that are too close together, keeping the lowest
   */
  private _filterCloseValleys(valleys: Array<{time: Time, price: number, index: number}>): Array<{time: Time, price: number, index: number}> {
    if (valleys.length <= 1) return valleys
    
    const filtered: Array<{time: Time, price: number, index: number}> = []
    const minDistance = 8 // Minimum 8 candles apart
    
    for (let i = 0; i < valleys.length; i++) {
      const current = valleys[i]
      let shouldAdd = true
      
      // Check if there's a lower valley within minDistance
      for (let j = 0; j < valleys.length; j++) {
        if (i !== j) {
          const other = valleys[j]
          const distance = Math.abs(current.index - other.index)
          
          if (distance < minDistance && other.price < current.price) {
            shouldAdd = false
            break
          }
        }
      }
      
      if (shouldAdd) {
        filtered.push(current)
      }
    }
    
    return filtered
  }

  /**
   * Validate trendlines for recent price action
   */
  private _isRecentTrendlineValid(trendline: CalculatedTrendline): boolean {
    // More lenient validation for recent trendlines
    const absSlope = Math.abs(trendline.slope)
    return absSlope >= 0.000001 && absSlope <= 0.05 // Allow steeper slopes for recent action
  }

  /**
   * Find significant peaks using high prices with better filtering (legacy method)
   */
  private _findSignificantPeaks(): Array<{time: Time, price: number, index: number}> {
    const peaks: Array<{time: Time, price: number, index: number}> = []
    const lookback = 8 // Look 8 candles back and forward for more significance
    const minHeightThreshold = 0.0005 // Minimum relative height (0.05%) for peak significance

    for (let i = lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isPeak = true
      let maxSurroundingHigh = 0

      // Check if current candle HIGH is higher than surrounding candles
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i) {
          if (this._data[j].high >= current.high) {
            isPeak = false
            break
          }
          maxSurroundingHigh = Math.max(maxSurroundingHigh, this._data[j].high)
        }
      }

      // Additional check for peak significance - must be higher than surrounding by threshold
      if (isPeak && maxSurroundingHigh > 0) {
        const relativeHeight = (current.high - maxSurroundingHigh) / maxSurroundingHigh
        if (relativeHeight < minHeightThreshold) {
          isPeak = false
        }
      }

      if (isPeak) {
        peaks.push({
          time: current.time,
          price: current.high, // Use HIGH for resistance - snaps to actual high price
          index: i
        })
      }
    }

    // Sort by time to ensure proper ordering
    return peaks.sort((a, b) => timeToNumber(a.time) - timeToNumber(b.time))
  }

  /**
   * Find significant valleys using low prices with better filtering
   */
  private _findSignificantValleys(): Array<{time: Time, price: number, index: number}> {
    const valleys: Array<{time: Time, price: number, index: number}> = []
    const lookback = 8 // Look 8 candles back and forward for more significance
    const minDepthThreshold = 0.0005 // Minimum relative depth (0.05%) for valley significance

    for (let i = lookback; i < this._data.length - lookback; i++) {
      const current = this._data[i]
      let isValley = true
      let minSurroundingLow = Infinity

      // Check if current candle LOW is lower than surrounding candles
      for (let j = i - lookback; j <= i + lookback; j++) {
        if (j !== i) {
          if (this._data[j].low <= current.low) {
            isValley = false
            break
          }
          minSurroundingLow = Math.min(minSurroundingLow, this._data[j].low)
        }
      }

      // Additional check for valley significance - must be lower than surrounding by threshold
      if (isValley && minSurroundingLow < Infinity) {
        const relativeDepth = (minSurroundingLow - current.low) / current.low
        if (relativeDepth < minDepthThreshold) {
          isValley = false
        }
      }

      if (isValley) {
        valleys.push({
          time: current.time,
          price: current.low, // Use LOW for support - snaps to actual low price
          index: i
        })
      }
    }

    // Sort by time to ensure proper ordering
    return valleys.sort((a, b) => timeToNumber(a.time) - timeToNumber(b.time))
  }

  /**
   * Find best trendline from a set of peaks/valleys
   */
  private _findBestTrendlineFromPeaks(
    points: Array<{time: Time, price: number, index: number}>, 
    type: 'resistance' | 'support'
  ): CalculatedTrendline | null {
    let bestTrendline: CalculatedTrendline | null = null
    let bestScore = 0

    // Try all combinations of points to find the best trendline
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const trendline = this._createTrendlineFromPoints(points[i], points[j], type)
        
        if (trendline && this._isValidTrendline(trendline)) {
          const score = this._scoreTrendline(trendline)
          if (score > bestScore) {
            bestScore = score
            bestTrendline = trendline
          }
        }
      }
    }

    return bestTrendline
  }

  /**
   * Create trendline from two points
   */
  private _createTrendlineFromPoints(
    point1: {time: Time, price: number, index: number},
    point2: {time: Time, price: number, index: number},
    type: 'resistance' | 'support'
  ): CalculatedTrendline | null {
    const time1 = timeToNumber(point1.time)
    const time2 = timeToNumber(point2.time)
    const timeSpan = Math.abs(time1 - time2)
    
    if (timeSpan <= 0) return null

    const slope = (point1.price - point2.price) / timeSpan
    
    // Ensure point1 is the more recent point (larger time)
    const [recentPoint, historicalPoint] = time1 > time2 ? [point1, point2] : [point2, point1]
    
    return {
      point1: {
        time: recentPoint.time,
        price: recentPoint.price,
        significance: 1.0,
        index: recentPoint.index
      },
      point2: {
        time: historicalPoint.time,
        price: historicalPoint.price,
        significance: 1.0,
        index: historicalPoint.index
      },
      slope,
      strength: 1.0,
      score: 0,
      touchingPoints: 0,
      type
    }
  }

  /**
   * Validate trendline
   */
  private _isValidTrendline(trendline: CalculatedTrendline): boolean {
    const absSlope = Math.abs(trendline.slope)
    return absSlope >= 0.000001 && absSlope <= 0.01 // Reasonable slope constraints
  }

  /**
   * Get trendline price at specific time
   */
  private _getTrendlinePrice(trendline: CalculatedTrendline, time: number): number | null {
    const time1 = timeToNumber(trendline.point1.time)
    const timeDiff = time - time1
    return trendline.point1.price + (trendline.slope * timeDiff)
  }

  /**
   * Score a trendline based on how well it fits the data
   */
  private _scoreTrendline(trendline: CalculatedTrendline): number {
    let touchingPoints = 0
    const tolerance = 0.015 // 1.5% tolerance
    
    const startIndex = Math.min(trendline.point1.index, trendline.point2.index)
    const endIndex = Math.max(trendline.point1.index, trendline.point2.index)
    
    for (let i = startIndex; i <= endIndex; i++) {
      const expectedPrice = this._getTrendlinePrice(trendline, timeToNumber(this._data[i].time))
      if (expectedPrice === null) continue
      
      const actualPrice = trendline.type === 'resistance' ? this._data[i].high : this._data[i].low
      const diff = Math.abs(actualPrice - expectedPrice) / expectedPrice
      
      if (diff <= tolerance) {
        touchingPoints++
      }
    }
    
    // Score based on touching points and time span
    const timeSpan = Math.abs(trendline.point1.index - trendline.point2.index)
    return touchingPoints * 10 + timeSpan * 0.1
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
    private _data: PeakCandlestickData[],
    private _chart: any,
    private _series: any
  ) {}

  zOrder(): 'bottom' | 'normal' | 'top' {
    return 'top' // Draw on top of other elements
  }

  renderer() {
    return new TrendlineRenderer(this._upperTrendline, this._lowerTrendline, this._data, this._chart, this._series)
  }
}

// Renderer for drawing trendlines
class TrendlineRenderer {
  constructor(
    private _upperTrendline: CalculatedTrendline | null,
    private _lowerTrendline: CalculatedTrendline | null,
    private _data: PeakCandlestickData[],
    private _chart: any,
    private _series: any
  ) {}

  // Helper method to calculate trendline price at specific time
  private _getTrendlinePrice(trendline: CalculatedTrendline, time: number): number | null {
    const time1 = timeToNumber(trendline.point1.time)
    const timeDiff = time - time1
    return trendline.point1.price + (trendline.slope * timeDiff)
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  draw(target: any): void {
    if (!this._series || !this._chart) return

    // Use the media coordinate space with proper chart integration
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    target.useMediaCoordinateSpace((scope: any) => {
      const ctx = scope.context

      // Save context state
      ctx.save()

      try {
        // Draw upper trendline (resistance) - Red for resistance
        if (this._upperTrendline) {
          this._drawTrendlineWithCoordinateSystem(ctx, this._upperTrendline, '#ff6b6b', 2)
        }

        // Draw lower trendline (support) - Teal for support
        if (this._lowerTrendline) {
          this._drawTrendlineWithCoordinateSystem(ctx, this._lowerTrendline, '#4ecdc4', 2)
        }
      } finally {
        // Restore context state
        ctx.restore()
      }
    })
  }

  private _drawTrendlineWithCoordinateSystem(
    ctx: CanvasRenderingContext2D,
    trendline: CalculatedTrendline,
    color: string,
    width: number
  ): void {
    try {
      // Convert time and price to coordinates using the chart's coordinate system
      const point1Time = timeToNumber(trendline.point1.time)
      const point2Time = timeToNumber(trendline.point2.time)

      // Use the series methods to convert to pixel coordinates
      const x1 = this._chart.timeScale().timeToCoordinate(point1Time)
      const x2 = this._chart.timeScale().timeToCoordinate(point2Time)
      const y1 = this._series.priceToCoordinate(trendline.point1.price)
      const y2 = this._series.priceToCoordinate(trendline.point2.price)

      // Only draw if coordinates are valid
      if (x1 !== null && x2 !== null && y1 !== null && y2 !== null) {
        ctx.strokeStyle = color
        ctx.lineWidth = width
        ctx.setLineDash([])

        // Draw line connecting the two detected peaks/valleys
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()

        // Draw small circles at the exact peak/valley points to show detection
        ctx.fillStyle = color
        ctx.beginPath()
        ctx.arc(x1, y1, 3, 0, 2 * Math.PI)
        ctx.fill()
        ctx.beginPath()
        ctx.arc(x2, y2, 3, 0, 2 * Math.PI)
        ctx.fill()

        // Optional: Add text labels to show what type of trendline this is
        ctx.fillStyle = color
        ctx.font = '10px Arial'
        const label = trendline.type === 'resistance' ? 'R' : 'S'
        const labelX = (x1 + x2) / 2
        const labelY = (y1 + y2) / 2 - 10
        ctx.fillText(label, labelX, labelY)
      }
    } catch (error) {
      console.warn('Error drawing trendline with coordinate system:', error)
    }
  }

}