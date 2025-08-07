import { Time } from 'lightweight-charts'

export interface CandlestickData {
  time: Time
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

export interface PeakLowPoint {
  time: Time
  price: number
  significance: number
  type: 'peak' | 'low'
  index: number
}

export interface PeakDetectionConfig {
  lookbackWindow: number // Number of candles to look back for local extremes (default: 2)
  minSignificance: number // Minimum significance threshold (default: 0.1)
  volumeWeight: number // Weight for volume in significance calculation (default: 0.2)
  priceWeight: number // Weight for price difference in significance calculation (default: 0.8)
  adaptiveWindow: boolean // Use adaptive window based on data volatility (default: true)
  maxPeaksPercentage: number // Maximum percentage of data points that can be peaks (default: 0.05)
}

export class PeakDetectionAlgorithm {
  private config: PeakDetectionConfig

  constructor(config: Partial<PeakDetectionConfig> = {}) {
    this.config = {
      lookbackWindow: 2,
      minSignificance: 0.01, // Much more lenient - 1% instead of 10%
      volumeWeight: 0.2,
      priceWeight: 0.8,
      adaptiveWindow: true,
      maxPeaksPercentage: 0.1, // Allow more peaks - 10% instead of 5%
      ...config
    }
  }

  /**
   * Find all significant peaks and lows in the given data using dynamic detection
   */
  findPeaksAndLows(data: CandlestickData[]): PeakLowPoint[] {
    if (data.length < 10) {
      return []
    }

    // Use dynamic detection for better results
    return this.findDynamicPeaksAndLows(data)
  }

  /**
   * Dynamic peak and low detection that adapts to data characteristics
   */
  findDynamicPeaksAndLows(data: CandlestickData[]): PeakLowPoint[] {
    const points: PeakLowPoint[] = []

    // Calculate adaptive window based on data volatility
    const adaptiveWindow = this.config.adaptiveWindow ?
      this.calculateAdaptiveWindow(data) :
      this.config.lookbackWindow

    // Find all potential peaks and lows with adaptive window
    for (let i = adaptiveWindow; i < data.length - adaptiveWindow; i++) {
      const current = data[i]

      // Check for peak with adaptive window
      if (this.isLocalMaximum(data, i, adaptiveWindow)) {
        const significance = this.calculateSignificance(data, i, 'peak')
        points.push({
          time: current.time,
          price: current.high,
          significance,
          type: 'peak',
          index: i
        })
      }

      // Check for low with adaptive window
      if (this.isLocalMinimum(data, i, adaptiveWindow)) {
        const significance = this.calculateSignificance(data, i, 'low')
        points.push({
          time: current.time,
          price: current.low,
          significance,
          type: 'low',
          index: i
        })
      }
    }

    // Filter and rank points dynamically
    return this.filterAndRankPoints(points, data.length)
  }

  /**
   * Calculate adaptive window size based on data volatility
   */
  private calculateAdaptiveWindow(data: CandlestickData[]): number {
    if (data.length < 20) return 2

    // Calculate price volatility over recent data
    const recentData = data.slice(-Math.min(100, data.length))
    let totalVolatility = 0

    for (let i = 1; i < recentData.length; i++) {
      const priceChange = Math.abs(recentData[i].close - recentData[i-1].close) / recentData[i-1].close
      totalVolatility += priceChange
    }

    const avgVolatility = totalVolatility / (recentData.length - 1)

    // Adaptive window: higher volatility = smaller window, lower volatility = larger window
    if (avgVolatility > 0.02) return 2      // High volatility: 2 candles
    else if (avgVolatility > 0.01) return 3 // Medium volatility: 3 candles
    else if (avgVolatility > 0.005) return 4 // Low volatility: 4 candles
    else return 5                           // Very low volatility: 5 candles
  }

  /**
   * Filter and rank points based on significance and distribution
   */
  private filterAndRankPoints(points: PeakLowPoint[], dataLength: number): PeakLowPoint[] {
    // Sort by significance first
    points.sort((a, b) => b.significance - a.significance)

    // Limit the number of points to avoid over-detection
    const maxPoints = Math.floor(dataLength * this.config.maxPeaksPercentage)
    const limitedPoints = points.slice(0, maxPoints)

    // Apply minimum significance filter
    const filteredPoints = limitedPoints.filter(p => p.significance >= this.config.minSignificance)

    // Ensure good distribution - avoid clustering
    return this.ensureGoodDistribution(filteredPoints)
  }

  /**
   * Ensure good distribution of points to avoid clustering
   */
  private ensureGoodDistribution(points: PeakLowPoint[]): PeakLowPoint[] {
    if (points.length <= 2) return points

    const distributed: PeakLowPoint[] = []
    const minDistance = 5 // Minimum distance between points

    for (const point of points) {
      const tooClose = distributed.some(existing =>
        Math.abs(existing.index - point.index) < minDistance
      )

      if (!tooClose) {
        distributed.push(point)
      }
    }

    return distributed.sort((a, b) => a.index - b.index)
  }

  /**
   * Find recent peaks (last N candles)
   */
  findRecentPeaks(data: CandlestickData[], recentPeriod: number = 25): PeakLowPoint[] {
    if (data.length < recentPeriod) {
      return this.findPeaksAndLows(data).filter(p => p.type === 'peak')
    }

    const recentData = data.slice(-recentPeriod)
    const recentPoints = this.findPeaksAndLows(recentData)
    
    // Adjust indices to match original data
    const adjustedPoints = recentPoints
      .filter(p => p.type === 'peak')
      .map(p => ({
        ...p,
        index: p.index + (data.length - recentPeriod)
      }))

    return adjustedPoints
  }

  /**
   * Find recent lows (last N candles)
   */
  findRecentLows(data: CandlestickData[], recentPeriod: number = 25): PeakLowPoint[] {
    if (data.length < recentPeriod) {
      return this.findPeaksAndLows(data).filter(p => p.type === 'low')
    }

    const recentData = data.slice(-recentPeriod)
    const recentPoints = this.findPeaksAndLows(recentData)
    
    // Adjust indices to match original data
    const adjustedPoints = recentPoints
      .filter(p => p.type === 'low')
      .map(p => ({
        ...p,
        index: p.index + (data.length - recentPeriod)
      }))

    return adjustedPoints
  }

  /**
   * Find historical peaks/lows that could connect to recent points
   */
  findHistoricalPoints(
    data: CandlestickData[], 
    type: 'peak' | 'low', 
    excludeRecentPeriod: number = 25
  ): PeakLowPoint[] {
    if (data.length <= excludeRecentPeriod) {
      return []
    }

    const historicalData = data.slice(0, -excludeRecentPeriod)
    const historicalPoints = this.findPeaksAndLows(historicalData)
    
    return historicalPoints
      .filter(p => p.type === type)
      .sort((a, b) => b.significance - a.significance)
      .slice(0, 10) // Limit to top 10 most significant points
  }

  /**
   * Check if a point is a local maximum
   */
  private isLocalMaximum(data: CandlestickData[], index: number, window?: number): boolean {
    const current = data[index].high
    const lookbackWindow = window || this.config.lookbackWindow

    for (let i = Math.max(0, index - lookbackWindow); i <= Math.min(data.length - 1, index + lookbackWindow); i++) {
      if (i !== index && data[i].high >= current) {
        return false
      }
    }
    return true
  }

  /**
   * Check if a point is a local minimum
   */
  private isLocalMinimum(data: CandlestickData[], index: number, window?: number): boolean {
    const current = data[index].low
    const lookbackWindow = window || this.config.lookbackWindow

    for (let i = Math.max(0, index - lookbackWindow); i <= Math.min(data.length - 1, index + lookbackWindow); i++) {
      if (i !== index && data[i].low <= current) {
        return false
      }
    }
    return true
  }

  /**
   * Calculate significance of a peak or low
   */
  private calculateSignificance(data: CandlestickData[], index: number, type: 'peak' | 'low'): number {
    const current = data[index]
    const price = type === 'peak' ? current.high : current.low
    const window = Math.min(5, this.config.lookbackWindow * 2) // Use larger window for significance
    
    // Calculate price-based significance
    let priceDifference = 0
    let count = 0
    
    for (let i = Math.max(0, index - window); i <= Math.min(data.length - 1, index + window); i++) {
      if (i !== index) {
        const comparePrice = type === 'peak' ? data[i].high : data[i].low
        const difference = type === 'peak' ? 
          Math.max(0, (price - comparePrice) / price) : 
          Math.max(0, (comparePrice - price) / price)
        priceDifference += difference
        count++
      }
    }
    
    const priceSignificance = count > 0 ? priceDifference / count : 0
    
    // Calculate volume-based significance (if volume data is available)
    let volumeSignificance = 0
    if (current.volume !== undefined) {
      let avgVolume = 0
      let volumeCount = 0
      
      for (let i = Math.max(0, index - window); i <= Math.min(data.length - 1, index + window); i++) {
        if (data[i].volume !== undefined) {
          avgVolume += data[i].volume!
          volumeCount++
        }
      }
      
      if (volumeCount > 0) {
        avgVolume /= volumeCount
        volumeSignificance = current.volume > avgVolume ? 
          Math.min(1, (current.volume - avgVolume) / avgVolume) : 0
      }
    }
    
    // Combine price and volume significance
    const totalSignificance = 
      (priceSignificance * this.config.priceWeight) + 
      (volumeSignificance * this.config.volumeWeight)
    
    return Math.min(1, totalSignificance) // Cap at 1.0
  }

  /**
   * Filter points by minimum distance to avoid clustering
   */
  filterByDistance(points: PeakLowPoint[], minDistance: number = 5): PeakLowPoint[] {
    if (points.length <= 1) return points

    const filtered: PeakLowPoint[] = []
    const sortedPoints = [...points].sort((a, b) => b.significance - a.significance)

    for (const point of sortedPoints) {
      const tooClose = filtered.some(existing => 
        Math.abs(existing.index - point.index) < minDistance
      )
      
      if (!tooClose) {
        filtered.push(point)
      }
    }

    return filtered.sort((a, b) => a.index - b.index)
  }

  /**
   * Get the most significant recent peak
   */
  getMostSignificantRecentPeak(data: CandlestickData[], recentPeriod: number = 25): PeakLowPoint | null {
    const recentPeaks = this.findRecentPeaks(data, recentPeriod)
    return recentPeaks.length > 0 ? recentPeaks[0] : null
  }

  /**
   * Get the most significant recent low
   */
  getMostSignificantRecentLow(data: CandlestickData[], recentPeriod: number = 25): PeakLowPoint | null {
    const recentLows = this.findRecentLows(data, recentPeriod)
    return recentLows.length > 0 ? recentLows[0] : null
  }

  /**
   * Debug function to analyze peak/low detection
   */
  analyzeData(data: CandlestickData[], recentPeriod: number = 25): {
    recentPeaks: PeakLowPoint[]
    recentLows: PeakLowPoint[]
    historicalPeaks: PeakLowPoint[]
    historicalLows: PeakLowPoint[]
    totalPoints: number
  } {
    const recentPeaks = this.findRecentPeaks(data, recentPeriod)
    const recentLows = this.findRecentLows(data, recentPeriod)
    const historicalPeaks = this.findHistoricalPoints(data, 'peak', recentPeriod)
    const historicalLows = this.findHistoricalPoints(data, 'low', recentPeriod)

    return {
      recentPeaks: this.filterByDistance(recentPeaks, 3),
      recentLows: this.filterByDistance(recentLows, 3),
      historicalPeaks: this.filterByDistance(historicalPeaks, 5),
      historicalLows: this.filterByDistance(historicalLows, 5),
      totalPoints: data.length
    }
  }
}

// Export a default instance with standard configuration
export const defaultPeakDetector = new PeakDetectionAlgorithm()

// Export factory function for custom configurations
export function createPeakDetector(config: Partial<PeakDetectionConfig> = {}): PeakDetectionAlgorithm {
  return new PeakDetectionAlgorithm(config)
}
