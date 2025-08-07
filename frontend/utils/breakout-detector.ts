import { Time } from 'lightweight-charts'
import { CandlestickData } from './peak-detection'
import { Trendline, TrendlineCalculator } from './trendline-calculator'

export interface BreakoutSignal {
  type: 'resistance_break' | 'support_break'
  time: Time
  price: number
  trendlinePrice: number
  strength: number // 0-1, how strong the breakout is
  volume?: number
  confidence: number // 0-1, confidence in the breakout
}

export interface BreakoutConfig {
  priceThreshold: number // Minimum price movement beyond trendline (default: 0.001 = 0.1%)
  volumeThreshold: number // Minimum volume multiplier vs average (default: 1.2)
  confirmationCandles: number // Number of candles to confirm breakout (default: 2)
  lookbackPeriod: number // Period for calculating average volume (default: 20)
  minTrendlineAge: number // Minimum age of trendline in candles (default: 10)
  maxFalseBreakouts: number // Max false breakouts before reducing confidence (default: 3)
}

export class BreakoutDetector {
  private config: BreakoutConfig
  private calculator: TrendlineCalculator
  private recentBreakouts: BreakoutSignal[] = []
  private falseBreakoutCount: number = 0

  constructor(config: Partial<BreakoutConfig> = {}) {
    this.config = {
      priceThreshold: 0.001,
      volumeThreshold: 1.2,
      confirmationCandles: 2,
      lookbackPeriod: 20,
      minTrendlineAge: 10,
      maxFalseBreakouts: 3,
      ...config
    }
    
    this.calculator = new TrendlineCalculator()
  }

  /**
   * Detect breakouts from given trendlines
   */
  detectBreakouts(
    data: CandlestickData[], 
    upperTrendline: Trendline | null, 
    lowerTrendline: Trendline | null
  ): BreakoutSignal[] {
    const signals: BreakoutSignal[] = []
    
    if (data.length < this.config.confirmationCandles + 1) return signals

    // Check resistance breakout
    if (upperTrendline && this.isTrendlineValid(upperTrendline, data)) {
      const resistanceBreakout = this.checkResistanceBreakout(data, upperTrendline)
      if (resistanceBreakout) {
        signals.push(resistanceBreakout)
      }
    }

    // Check support breakout
    if (lowerTrendline && this.isTrendlineValid(lowerTrendline, data)) {
      const supportBreakout = this.checkSupportBreakout(data, lowerTrendline)
      if (supportBreakout) {
        signals.push(supportBreakout)
      }
    }

    // Update recent breakouts
    this.recentBreakouts.push(...signals)
    this.cleanupOldBreakouts()

    return signals
  }

  /**
   * Check for resistance (upper trendline) breakout
   */
  private checkResistanceBreakout(data: CandlestickData[], trendline: Trendline): BreakoutSignal | null {
    const recentCandles = data.slice(-this.config.confirmationCandles - 1)
    const latestCandle = recentCandles[recentCandles.length - 1]
    
    // Get trendline price at current time
    const trendlinePrice = this.calculator.getTrendlinePrice(trendline, this.timeToNumber(latestCandle.time))
    if (!trendlinePrice) return null

    // Check if price breaks above trendline with sufficient margin
    const breakoutPrice = trendlinePrice * (1 + this.config.priceThreshold)
    const hasBreakout = latestCandle.close > breakoutPrice

    if (!hasBreakout) return null

    // Calculate breakout strength
    const strength = this.calculateBreakoutStrength(latestCandle, trendlinePrice, 'resistance')
    
    // Check volume confirmation
    const volumeConfirmation = this.checkVolumeConfirmation(data, latestCandle)
    
    // Check for confirmation candles
    const confirmationScore = this.checkConfirmationCandles(recentCandles, trendlinePrice, 'resistance')
    
    // Calculate overall confidence
    const confidence = this.calculateConfidence(strength, volumeConfirmation, confirmationScore)

    // Only return signal if confidence is above threshold
    if (confidence < 0.5) return null

    return {
      type: 'resistance_break',
      time: latestCandle.time,
      price: latestCandle.close,
      trendlinePrice,
      strength,
      volume: latestCandle.volume,
      confidence
    }
  }

  /**
   * Check for support (lower trendline) breakout
   */
  private checkSupportBreakout(data: CandlestickData[], trendline: Trendline): BreakoutSignal | null {
    const recentCandles = data.slice(-this.config.confirmationCandles - 1)
    const latestCandle = recentCandles[recentCandles.length - 1]
    
    // Get trendline price at current time
    const trendlinePrice = this.calculator.getTrendlinePrice(trendline, this.timeToNumber(latestCandle.time))
    if (!trendlinePrice) return null

    // Check if price breaks below trendline with sufficient margin
    const breakoutPrice = trendlinePrice * (1 - this.config.priceThreshold)
    const hasBreakout = latestCandle.close < breakoutPrice

    if (!hasBreakout) return null

    // Calculate breakout strength
    const strength = this.calculateBreakoutStrength(latestCandle, trendlinePrice, 'support')
    
    // Check volume confirmation
    const volumeConfirmation = this.checkVolumeConfirmation(data, latestCandle)
    
    // Check for confirmation candles
    const confirmationScore = this.checkConfirmationCandles(recentCandles, trendlinePrice, 'support')
    
    // Calculate overall confidence
    const confidence = this.calculateConfidence(strength, volumeConfirmation, confirmationScore)

    // Only return signal if confidence is above threshold
    if (confidence < 0.5) return null

    return {
      type: 'support_break',
      time: latestCandle.time,
      price: latestCandle.close,
      trendlinePrice,
      strength,
      volume: latestCandle.volume,
      confidence
    }
  }

  /**
   * Calculate breakout strength based on price movement
   */
  private calculateBreakoutStrength(candle: CandlestickData, trendlinePrice: number, type: 'resistance' | 'support'): number {
    if (type === 'resistance') {
      const breakoutAmount = (candle.close - trendlinePrice) / trendlinePrice
      return Math.min(1, breakoutAmount / 0.02) // Normalize to 0-1, max at 2% breakout
    } else {
      const breakoutAmount = (trendlinePrice - candle.close) / trendlinePrice
      return Math.min(1, breakoutAmount / 0.02) // Normalize to 0-1, max at 2% breakout
    }
  }

  /**
   * Check volume confirmation for breakout
   */
  private checkVolumeConfirmation(data: CandlestickData[], currentCandle: CandlestickData): number {
    if (!currentCandle.volume) return 0.5 // Neutral if no volume data

    // Calculate average volume over lookback period
    const lookbackData = data.slice(-this.config.lookbackPeriod)
    const volumeSum = lookbackData.reduce((sum, candle) => sum + (candle.volume || 0), 0)
    const avgVolume = volumeSum / lookbackData.length

    if (avgVolume === 0) return 0.5 // Neutral if no volume data

    const volumeRatio = currentCandle.volume / avgVolume
    
    // Return score based on volume ratio
    if (volumeRatio >= this.config.volumeThreshold) {
      return Math.min(1, volumeRatio / (this.config.volumeThreshold * 2)) // Max score at 2x threshold
    }
    
    return volumeRatio / this.config.volumeThreshold // Proportional score below threshold
  }

  /**
   * Check confirmation candles
   */
  private checkConfirmationCandles(candles: CandlestickData[], trendlinePrice: number, type: 'resistance' | 'support'): number {
    if (candles.length < this.config.confirmationCandles + 1) return 0

    let confirmationCount = 0
    const confirmationCandles = candles.slice(-this.config.confirmationCandles)

    for (const candle of confirmationCandles) {
      if (type === 'resistance') {
        if (candle.close > trendlinePrice) confirmationCount++
      } else {
        if (candle.close < trendlinePrice) confirmationCount++
      }
    }

    return confirmationCount / this.config.confirmationCandles
  }

  /**
   * Calculate overall confidence score
   */
  private calculateConfidence(strength: number, volumeConfirmation: number, confirmationScore: number): number {
    // Weight the different factors
    const strengthWeight = 0.4
    const volumeWeight = 0.3
    const confirmationWeight = 0.3

    let confidence = (strength * strengthWeight) + 
                    (volumeConfirmation * volumeWeight) + 
                    (confirmationScore * confirmationWeight)

    // Reduce confidence based on recent false breakouts
    const falseBreakoutPenalty = Math.min(0.3, this.falseBreakoutCount / this.config.maxFalseBreakouts * 0.3)
    confidence -= falseBreakoutPenalty

    return Math.max(0, Math.min(1, confidence))
  }

  /**
   * Check if trendline is valid for breakout detection
   */
  private isTrendlineValid(trendline: Trendline, data: CandlestickData[]): boolean {
    // Check if trendline is old enough
    const latestTime = this.timeToNumber(data[data.length - 1].time)
    const trendlineTime = this.timeToNumber(trendline.point1.time)
    const ageInCandles = (latestTime - trendlineTime) / (5 * 60) // Assuming 5-minute candles
    
    return ageInCandles >= this.config.minTrendlineAge
  }

  /**
   * Mark a breakout as false (for learning)
   */
  markFalseBreakout(signal: BreakoutSignal): void {
    this.falseBreakoutCount++
  }

  /**
   * Get recent breakout signals
   */
  getRecentBreakouts(maxAge: number = 3600): BreakoutSignal[] { // Default 1 hour
    const cutoffTime = Date.now() / 1000 - maxAge
    return this.recentBreakouts.filter(signal => {
      const signalTime = this.timeToNumber(signal.time)
      return signalTime >= cutoffTime
    })
  }

  /**
   * Clean up old breakout signals
   */
  private cleanupOldBreakouts(): void {
    const cutoffTime = Date.now() / 1000 - 3600 // Keep 1 hour of history
    this.recentBreakouts = this.recentBreakouts.filter(signal => {
      const signalTime = this.timeToNumber(signal.time)
      return signalTime >= cutoffTime
    })
  }

  /**
   * Helper function to convert Time to number
   */
  private timeToNumber(time: Time): number {
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

  /**
   * Reset false breakout counter
   */
  resetFalseBreakoutCounter(): void {
    this.falseBreakoutCount = 0
  }

  /**
   * Get breakout statistics
   */
  getStatistics(): {
    totalBreakouts: number
    falseBreakouts: number
    accuracy: number
    recentBreakouts: number
  } {
    const recentBreakouts = this.getRecentBreakouts()
    return {
      totalBreakouts: this.recentBreakouts.length,
      falseBreakouts: this.falseBreakoutCount,
      accuracy: this.recentBreakouts.length > 0 ? 
        1 - (this.falseBreakoutCount / this.recentBreakouts.length) : 0,
      recentBreakouts: recentBreakouts.length
    }
  }
}

// Export default instance
export const defaultBreakoutDetector = new BreakoutDetector()

// Export factory function
export function createBreakoutDetector(config: Partial<BreakoutConfig> = {}): BreakoutDetector {
  return new BreakoutDetector(config)
}
