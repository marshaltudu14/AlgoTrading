// Demo data generator for TradingView-like charts

export interface DemoCandle {
  time: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface DemoTradeMarker {
  time: string
  position: 'aboveBar' | 'belowBar'
  color: string
  shape: 'circle' | 'square' | 'arrowUp' | 'arrowDown'
  text: string
  size?: number
}

// Generate realistic demo candlestick data
export function generateDemoData(
  days: number = 100,
  basePrice: number = 100,
  volatility: number = 0.02
): DemoCandle[] {
  const data: DemoCandle[] = []
  let currentPrice = basePrice
  const startDate = new Date()
  startDate.setDate(startDate.getDate() - days)

  for (let i = 0; i < days; i++) {
    const date = new Date(startDate)
    date.setDate(date.getDate() + i)
    
    // Generate realistic price movement
    const priceChange = (Math.random() - 0.5) * volatility * currentPrice
    const open = currentPrice
    const close = Math.max(0.1, currentPrice + priceChange)
    
    // Generate high and low based on open and close
    const minPrice = Math.min(open, close)
    const maxPrice = Math.max(open, close)
    const range = maxPrice - minPrice
    
    const high = maxPrice + (Math.random() * range * 0.5)
    const low = minPrice - (Math.random() * range * 0.5)
    
    // Generate volume (higher volume on bigger price moves)
    const priceMovement = Math.abs(close - open) / open
    const baseVolume = 1000000
    const volume = Math.floor(baseVolume * (1 + priceMovement * 5) * (0.5 + Math.random()))

    data.push({
      time: date.toISOString().split('T')[0],
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(Math.max(0.1, low).toFixed(2)),
      close: Number(close.toFixed(2)),
      volume
    })

    currentPrice = close
  }

  return data
}

// Generate demo trade markers
export function generateDemoTradeMarkers(
  candleData: DemoCandle[],
  tradeFrequency: number = 0.1
): DemoTradeMarker[] {
  const markers: DemoTradeMarker[] = []
  
  for (let i = 0; i < candleData.length; i++) {
    if (Math.random() < tradeFrequency) {
      const candle = candleData[i]
      const isBuy = Math.random() > 0.5
      
      markers.push({
        time: candle.time,
        position: isBuy ? 'belowBar' : 'aboveBar',
        color: isBuy ? '#22c55e' : '#ef4444',
        shape: isBuy ? 'arrowUp' : 'arrowDown',
        text: isBuy ? 'BUY' : 'SELL',
        size: 1
      })
    }
  }
  
  return markers
}

// Generate demo portfolio data
export function generateDemoPortfolioData(
  candleData: DemoCandle[],
  initialValue: number = 100000
): Array<{ time: string; value: number }> {
  const portfolioData: Array<{ time: string; value: number }> = []
  let currentValue = initialValue
  
  for (let i = 0; i < candleData.length; i++) {
    const candle = candleData[i]
    
    // Simulate portfolio growth/decline based on market movement
    if (i > 0) {
      const prevCandle = candleData[i - 1]
      const marketChange = (candle.close - prevCandle.close) / prevCandle.close
      
      // Add some randomness to portfolio performance
      const portfolioChange = marketChange * (0.8 + Math.random() * 0.4)
      currentValue *= (1 + portfolioChange)
    }
    
    portfolioData.push({
      time: candle.time,
      value: Number(currentValue.toFixed(2))
    })
  }
  
  return portfolioData
}

// Predefined demo datasets for different market conditions
export const demoDatasets = {
  bullish: {
    name: "Bullish Market",
    data: generateDemoData(100, 100, 0.015),
    description: "Strong upward trend with low volatility"
  },
  bearish: {
    name: "Bearish Market", 
    data: generateDemoData(100, 100, 0.025).map((candle, i, _arr) => ({ // eslint-disable-line @typescript-eslint/no-unused-vars
      ...candle,
      close: candle.close * (1 - i * 0.005) // Gradual decline
    })),
    description: "Downward trend with increased volatility"
  },
  sideways: {
    name: "Sideways Market",
    data: generateDemoData(100, 100, 0.01).map(candle => ({
      ...candle,
      close: 100 + (candle.close - 100) * 0.3 // Reduce price movement
    })),
    description: "Range-bound market with low volatility"
  },
  volatile: {
    name: "Volatile Market",
    data: generateDemoData(100, 100, 0.04),
    description: "High volatility with frequent price swings"
  }
}

// Get random demo dataset
export function getRandomDemoDataset() {
  const datasets = Object.values(demoDatasets)
  return datasets[Math.floor(Math.random() * datasets.length)]
}

// Real-time data simulator for live updates
export class DemoDataStream {
  private data: DemoCandle[]
  private currentIndex: number = 0
  private intervalId: NodeJS.Timeout | null = null
  private callbacks: Array<(candle: DemoCandle, isComplete: boolean) => void> = []

  constructor(data: DemoCandle[]) {
    this.data = data
  }

  start(intervalMs: number = 1000) {
    if (this.intervalId) return

    this.intervalId = setInterval(() => {
      if (this.currentIndex < this.data.length) {
        const candle = this.data[this.currentIndex]
        const isComplete = this.currentIndex === this.data.length - 1
        
        this.callbacks.forEach(callback => callback(candle, isComplete))
        this.currentIndex++
        
        if (isComplete) {
          this.stop()
        }
      }
    }, intervalMs)
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId)
      this.intervalId = null
    }
  }

  onData(callback: (candle: DemoCandle, isComplete: boolean) => void) {
    this.callbacks.push(callback)
  }

  reset() {
    this.stop()
    this.currentIndex = 0
  }

  getCurrentProgress() {
    return {
      current: this.currentIndex,
      total: this.data.length,
      percentage: (this.currentIndex / this.data.length) * 100
    }
  }
}
