import { Time } from 'lightweight-charts'
import { PeakLowPoint, CandlestickData, defaultPeakDetector, PeakDetectionAlgorithm } from './peak-detection'

export interface TrendlinePoint {
  time: Time
  price: number
  significance: number
  index: number
}

export interface Trendline {
  point1: TrendlinePoint
  point2: TrendlinePoint
  slope: number
  strength: number
  type: 'resistance' | 'support'
  score: number
  touchingPoints: number
}

export interface TrendlineConfig {
  maxSlope: number // Maximum allowed slope (default: 0.05)
  minSlope: number // Minimum allowed slope (default: 0.0001)
  touchTolerance: number // Tolerance for considering a point "touching" the line (default: 0.005)
  minTouchingPoints: number // Minimum points that should touch the line (default: 1)
  slopeWeight: number // Weight for slope in scoring (default: 0.3)
  strengthWeight: number // Weight for strength in scoring (default: 0.4)
  significanceWeight: number // Weight for point significance in scoring (default: 0.3)
  dynamicSelection: boolean // Use dynamic point selection (default: true)
  maxPointDistance: number // Maximum distance between points in candles (default: 0 = no limit)
}

// Helper function to convert Time to number
function timeToNumber(time: Time): number {
  if (typeof time === 'number') {
    return time
  }
  if (typeof time === 'string') {
    return new Date(time).getTime() / 1000
  }
  if (typeof time === 'object' && 'year' in time) {
    return new Date(time.year, time.month - 1, time.day).getTime() / 1000
  }
  return 0
}

export class TrendlineCalculator {
  private config: TrendlineConfig
  public peakDetector: PeakDetectionAlgorithm

  constructor(config: Partial<TrendlineConfig> = {}) {
    this.config = {
      maxSlope: 0.2,        // Allow steeper slopes
      minSlope: 0.00001,    // Allow very gentle slopes
      touchTolerance: 0.02, // Much more tolerant - 2% instead of 0.5%
      minTouchingPoints: 0, // Don't require any additional touching points
      slopeWeight: 0.3,
      strengthWeight: 0.4,
      significanceWeight: 0.3,
      dynamicSelection: true,
      maxPointDistance: 0,
      ...config
    }

    this.peakDetector = defaultPeakDetector
  }

  /**
   * Calculate the best upper trendline (resistance) using dynamic detection
   */
  calculateUpperTrendline(data: CandlestickData[]): Trendline | null {
    if (data.length < 20) return null

    // Get all peaks dynamically
    const allPeaks = this.peakDetector.findPeaksAndLows(data).filter(p => p.type === 'peak')

    if (allPeaks.length < 2) return null

    return this.findBestDynamicTrendline(allPeaks, data, 'resistance')
  }

  /**
   * Calculate the best lower trendline (support) using dynamic detection
   */
  calculateLowerTrendline(data: CandlestickData[]): Trendline | null {
    if (data.length < 20) return null

    // Get all lows dynamically
    const allLows = this.peakDetector.findPeaksAndLows(data).filter(p => p.type === 'low')

    if (allLows.length < 2) return null

    return this.findBestDynamicTrendline(allLows, data, 'support')
  }

  /**
   * Calculate both upper and lower trendlines
   */
  calculateTrendlines(data: CandlestickData[]): { upper: Trendline | null, lower: Trendline | null } {
    return {
      upper: this.calculateUpperTrendline(data),
      lower: this.calculateLowerTrendline(data)
    }
  }

  /**
   * Find the best trendline dynamically from all available points
   */
  private findBestDynamicTrendline(
    points: PeakLowPoint[],
    data: CandlestickData[],
    type: 'resistance' | 'support'
  ): Trendline | null {
    if (points.length < 2) return null

    let bestTrendline: Trendline | null = null
    let bestScore = 0

    // Try all combinations of points to find the best trendline
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const point1 = points[i]
        const point2 = points[j]

        // Skip if points are too close together (unless maxPointDistance is 0)
        if (this.config.maxPointDistance > 0) {
          const distance = Math.abs(point1.index - point2.index)
          if (distance < 5 || distance > this.config.maxPointDistance) continue
        }

        const trendline = this.createTrendline(point1, point2, type)
        if (trendline && this.isValidTrendline(trendline)) {
          const touchingPoints = this.countTouchingPoints(trendline, data)
          trendline.touchingPoints = touchingPoints

          if (touchingPoints >= this.config.minTouchingPoints) {
            const score = this.scoreTrendline(trendline)
            trendline.score = score

            if (score > bestScore) {
              bestScore = score
              bestTrendline = trendline
            }
          }
        }
      }
    }

    return bestTrendline
  }

  /**
   * Create a trendline from two points
   */
  private createTrendline(point1: PeakLowPoint, point2: PeakLowPoint, type: 'resistance' | 'support'): Trendline | null {
    const time1 = timeToNumber(point1.time)
    const time2 = timeToNumber(point2.time)
    
    const timeDiff = time1 - time2
    if (timeDiff <= 0) return null
    
    const priceDiff = point1.price - point2.price
    const slope = priceDiff / timeDiff
    
    // Ensure point1 is the more recent point
    const [recentPoint, historicalPoint] = time1 > time2 ? [point1, point2] : [point2, point1]
    
    return {
      point1: {
        time: recentPoint.time,
        price: recentPoint.price,
        significance: recentPoint.significance,
        index: recentPoint.index
      },
      point2: {
        time: historicalPoint.time,
        price: historicalPoint.price,
        significance: historicalPoint.significance,
        index: historicalPoint.index
      },
      slope,
      strength: 2, // Base strength for 2 points
      type,
      score: 0,
      touchingPoints: 0
    }
  }

  /**
   * Validate if trendline meets basic criteria
   */
  private isValidTrendline(trendline: Trendline): boolean {
    const absSlope = Math.abs(trendline.slope)
    
    // Check slope constraints
    if (absSlope < this.config.minSlope || absSlope > this.config.maxSlope) {
      return false
    }
    
    // Check that the slope direction makes sense for the trendline type
    if (trendline.type === 'resistance' && trendline.slope > 0.05) {
      return false // Resistance lines shouldn't be too steep upward
    }
    
    if (trendline.type === 'support' && trendline.slope < -0.05) {
      return false // Support lines shouldn't be too steep downward
    }
    
    return true
  }

  /**
   * Count how many points touch or are close to the trendline
   */
  private countTouchingPoints(trendline: Trendline, data: CandlestickData[]): number {
    let count = 0
    const tolerance = this.config.touchTolerance
    
    for (let i = 0; i < data.length; i++) {
      const candle = data[i]
      const timeValue = timeToNumber(candle.time)
      const expectedPrice = this.getTrendlinePrice(trendline, timeValue)
      
      if (expectedPrice !== null) {
        const actualPrice = trendline.type === 'resistance' ? candle.high : candle.low
        const difference = Math.abs(actualPrice - expectedPrice) / expectedPrice
        
        if (difference <= tolerance) {
          count++
        }
      }
    }
    
    return count
  }

  /**
   * Get trendline price at specific time
   */
  getTrendlinePrice(trendline: Trendline, time: number): number | null {
    const time1 = timeToNumber(trendline.point1.time)
    const timeDiff = time - time1
    return trendline.point1.price + (trendline.slope * timeDiff)
  }

  /**
   * Score trendline based on various factors
   */
  private scoreTrendline(trendline: Trendline): number {
    let score = 0
    
    // Significance score (how significant are the anchor points)
    const significanceScore = (trendline.point1.significance + trendline.point2.significance) / 2
    score += significanceScore * this.config.significanceWeight
    
    // Strength score (how many points touch the line)
    const strengthScore = Math.min(1, trendline.touchingPoints / 5) // Normalize to 0-1
    score += strengthScore * this.config.strengthWeight
    
    // Slope score (prefer moderate slopes)
    const absSlope = Math.abs(trendline.slope)
    const idealSlope = (this.config.minSlope + this.config.maxSlope) / 2
    const slopeScore = 1 - Math.abs(absSlope - idealSlope) / idealSlope
    score += Math.max(0, slopeScore) * this.config.slopeWeight
    
    return score
  }

  /**
   * Check if current price breaks the trendline
   */
  checkBreakout(trendline: Trendline, currentCandle: CandlestickData, threshold: number = 0.001): boolean {
    const currentTime = timeToNumber(currentCandle.time)
    const trendlinePrice = this.getTrendlinePrice(trendline, currentTime)
    
    if (trendlinePrice === null) return false
    
    if (trendline.type === 'resistance') {
      // Breakout above resistance
      return currentCandle.close > trendlinePrice * (1 + threshold)
    } else {
      // Breakout below support
      return currentCandle.close < trendlinePrice * (1 - threshold)
    }
  }

  /**
   * Get trendline projection for future time
   */
  projectTrendline(trendline: Trendline, futureTime: Time): number | null {
    const timeValue = timeToNumber(futureTime)
    return this.getTrendlinePrice(trendline, timeValue)
  }

  /**
   * Analyze trendline quality
   */
  analyzeTrendline(trendline: Trendline, data: CandlestickData[]): {
    score: number
    touchingPoints: number
    strength: number
    slope: number
    isValid: boolean
    type: string
  } {
    return {
      score: trendline.score,
      touchingPoints: trendline.touchingPoints,
      strength: trendline.strength,
      slope: trendline.slope,
      isValid: this.isValidTrendline(trendline),
      type: trendline.type
    }
  }
}

// Export a default instance
export const defaultTrendlineCalculator = new TrendlineCalculator()

// Export factory function for custom configurations
export function createTrendlineCalculator(config: Partial<TrendlineConfig> = {}): TrendlineCalculator {
  return new TrendlineCalculator(config)
}
