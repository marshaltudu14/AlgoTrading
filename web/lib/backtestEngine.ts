import { ProcessedCandleData } from '../stores/candleStore';
import { StrategySignal } from '../stores/strategyStore';

// Trade configuration matching Python implementation
export interface BacktestConfig {
  targetPnL: number;        // Target profit in Rs (per position) - Default: 500
  stopLossPnL: number;      // Stop loss in Rs (per position) - Default: -250
  initialCapital: number;   // Starting capital - Default: 25000
  brokerageEntry: number;   // Brokerage fee per entry - Default: 25
  brokerageExit: number;    // Brokerage fee per exit - Default: 25
  lotSize: number;          // Lot size for position sizing
  minConfidence: number;    // Minimum confidence threshold - Default: 0.5
  maxDailyLosses: number;   // Maximum losses per day - Default: 2
}

// Trade record interface
export interface Trade {
  entryTime: Date;
  exitTime: Date;
  position: 'BUY' | 'SELL';
  entryPrice: number;
  exitPrice: number;
  targetPrice?: number;
  stopLossPrice?: number;
  lotSize: number;
  pnlPoints: number;        // P&L in price points
  pnlCurrency: number;      // P&L in Rs (after brokerage)
  barsHeld: number;
  exitReason: string;
  confidence: number;
  capital: number;          // Capital after this trade
}

// Performance metrics
export interface BacktestMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  totalPnLPct: number;
  maxDrawdown: number;
  maxDrawdownPct: number;
  sharpeRatio: number;
  profitFactor: number;
  avgTradePnL: number;
  maxWinningStreak: number;
  maxLosingStreak: number;
  highestDailyProfit: number;
  highestDailyLoss: number;
  maxTradesPerDay: number;
  minTradesPerDay: number;
}

// Equity curve point
export interface EquityPoint {
  timestamp: Date;
  capital: number;
}

// Complete backtest results
export interface BacktestResults {
  trades: Trade[];
  equityCurve: EquityPoint[];
  metrics: BacktestMetrics;
  config: BacktestConfig;
}

// Main backtest engine
export class BacktestEngine {
  private config: BacktestConfig;
  private trades: Trade[] = [];
  private equityCurve: EquityPoint[] = [];
  private peakCapital: number;

  constructor(config: Partial<BacktestConfig> = {}) {
    // Default configuration matching trading.py
    this.config = {
      targetPnL: 500,
      stopLossPnL: -250,
      initialCapital: 25000,
      brokerageEntry: 25,
      brokerageExit: 25,
      lotSize: 75, // Default for Nifty
      minConfidence: 0.5,
      maxDailyLosses: 2,
      ...config
    };
    this.peakCapital = this.config.initialCapital;
  }

  /**
   * Simulate a single trade with trailing stop loss (exact replica from Python)
   */
  private simulateTrade(
    entryPrice: number,
    direction: 'BUY' | 'SELL',
    candles: ProcessedCandleData[],
    entryIndex: number,
    lotSize: number,
    targetPnL?: number,
    stopLossPnL?: number
  ): { isWin: boolean; pnlPoints: number; barsHeld: number; exitDetails: unknown } {
    let barsHeld = 0;
    const exitDetails: unknown = {
      exitPrice: entryPrice,
      targetPrice: null,
      stopLossPrice: null,
      exitReason: null
    };

    // Use provided P&L targets or fall back to defaults
    const currentTargetPnL = targetPnL ?? this.config.targetPnL;
    const currentStopLossPnL = stopLossPnL ?? this.config.stopLossPnL;

    // Calculate points needed based on P&L targets
    // P&L targets are per lot, so we divide by lot_size to get price movement
    const targetPoints = Math.abs(currentTargetPnL) / lotSize;
    const stopLossPoints = Math.abs(currentStopLossPnL) / lotSize;

    // Initialize variables for trailing
    let hitInitialTarget = false;
    let targetsHit = 0;

    if (direction === 'BUY') {
      // Initial target and stop loss
      const initialTarget = entryPrice + targetPoints;
      let currentTarget = initialTarget;
      let currentSl = entryPrice - stopLossPoints;

      exitDetails.targetPrice = Math.round(initialTarget * 100) / 100;
      exitDetails.stopLossPrice = Math.round(currentSl * 100) / 100;

      // Process future candles
      for (let i = entryIndex + 1; i < candles.length; i++) {
        const candle = candles[i];
        barsHeld++;

        if (!hitInitialTarget) {
          // Before hitting initial target
          if (candle.close >= currentTarget) {
            // Hit initial target - start trailing
            hitInitialTarget = true;
            targetsHit = 1;
            // Move SL to target price (protect profit at this level)
            const previousTarget = currentTarget;
            currentSl = previousTarget;
            // Set next target (add same target_points from the target we just hit)
            currentTarget = previousTarget + targetPoints;
          } else if (candle.low <= currentSl) {
            // SL breached during the candle - exit immediately at SL
            exitDetails.exitPrice = Math.round(currentSl * 100) / 100;
            exitDetails.exitReason = 'STOP_LOSS';
            const pnlPoints = -(entryPrice - currentSl);
            return { isWin: false, pnlPoints, barsHeld, exitDetails };
          }
        } else {
          // After hitting initial target - trailing mode
          // IMPORTANT: Check SL breach FIRST to ensure we exit if SL is hit
          if (candle.close <= currentSl) {
            // For trailing SL, wait for close (as per user requirement)
            // HYBRID APPROACH: Weighted average of SL level and close price
            const slLevel = currentSl;
            // Weight 70% towards SL level (instant exit advantage), 30% towards close (realism)
            const exitPrice = (slLevel * 0.7 + candle.close * 0.3);
            exitDetails.exitPrice = Math.round(exitPrice * 100) / 100;
            exitDetails.exitReason = `TRAILING_STOP_${targetsHit}`;
            // For BUY, profit = exit_price - entry_price
            const pnlPoints = exitPrice - entryPrice;
            return { isWin: pnlPoints > 0, pnlPoints, barsHeld, exitDetails };
          } else if (candle.close >= currentTarget) {
            // Hit another target - trail further
            targetsHit++;
            // The price we just hit becomes the defended level
            const previousTarget = currentTarget;
            // Move SL to defend the profit at this level
            currentSl = previousTarget;
            // Set new target (go up by target_points from the target we just hit)
            currentTarget = previousTarget + targetPoints;
          }
        }
      }
    } else { // SELL
      // Initial target and stop loss
      const initialTarget = entryPrice - targetPoints;
      let currentTarget = initialTarget;
      let currentSl = entryPrice + stopLossPoints;

      exitDetails.targetPrice = Math.round(initialTarget * 100) / 100;
      exitDetails.stopLossPrice = Math.round(currentSl * 100) / 100;

      // Process future candles
      for (let i = entryIndex + 1; i < candles.length; i++) {
        const candle = candles[i];
        barsHeld++;

        if (!hitInitialTarget) {
          // Before hitting initial target
          if (candle.close <= currentTarget) {
            // Hit initial target - start trailing
            hitInitialTarget = true;
            targetsHit = 1;
            // Move SL to target price (protect profit at this level)
            const previousTarget = currentTarget;
            currentSl = previousTarget;
            // Set next target (subtract same target_points again from the target we just hit)
            currentTarget = previousTarget - targetPoints;
          } else if (candle.high >= currentSl) {
            // SL breached during the candle - exit immediately at SL
            exitDetails.exitPrice = Math.round(currentSl * 100) / 100;
            exitDetails.exitReason = 'STOP_LOSS';
            const pnlPoints = -(currentSl - entryPrice); // For SELL, loss = SL - entry
            return { isWin: false, pnlPoints, barsHeld, exitDetails };
          }
        } else {
          // After hitting initial target - trailing mode
          // IMPORTANT: Check SL breach FIRST to ensure we exit if SL is hit
          if (candle.close >= currentSl) {
            // For trailing SL, wait for close (as per user requirement)
            // HYBRID APPROACH: Weighted average of SL level and close price
            const slLevel = currentSl;
            // Weight 70% towards SL level (instant exit advantage), 30% towards close (realism)
            const exitPrice = (slLevel * 0.7 + candle.close * 0.3);
            exitDetails.exitPrice = Math.round(exitPrice * 100) / 100;
            exitDetails.exitReason = `TRAILING_STOP_${targetsHit}`;
            // For SELL, profit = entry_price - exit_price
            const pnlPoints = entryPrice - exitPrice;
            return { isWin: pnlPoints > 0, pnlPoints, barsHeld, exitDetails };
          } else if (candle.close <= currentTarget) {
            // Hit another target - trail further
            targetsHit++;
            // The price we just hit becomes the defended level
            const previousTarget = currentTarget;
            // Move SL to defend the profit at this level
            currentSl = previousTarget;
            // Set new target (go down by target_points from the target we just hit)
            currentTarget = previousTarget - targetPoints;
          }
        }
      }
    }

    // If we reach here, trade didn't hit target or SL
    if (barsHeld > 0) {
      const lastCandle = candles[Math.min(entryIndex + barsHeld, candles.length - 1)];
      const lastPrice = direction === 'BUY' ? lastCandle.close : lastCandle.close;
      const pnlPoints = direction === 'BUY' ? lastPrice - entryPrice : entryPrice - lastPrice;
      exitDetails.exitPrice = Math.round(lastPrice * 100) / 100;
      exitDetails.exitReason = 'EXPIRY';
      return { isWin: pnlPoints > 0, pnlPoints, barsHeld, exitDetails };
    }

    return { isWin: false, pnlPoints: 0, barsHeld, exitDetails };
  }

  /**
   * Run backtest on historical data
   */
  async runBacktest(
    candles: ProcessedCandleData[],
    getSignal: (candle: ProcessedCandleData, previousCandles: ProcessedCandleData[]) => StrategySignal
  ): Promise<BacktestResults> {
    // Reset state
    this.trades = [];
    this.equityCurve = [];
    this.peakCapital = this.config.initialCapital;

    // Initialize tracking variables
    let capital = this.config.initialCapital;
    let position: 'BUY' | 'SELL' | null = null;
    let entryPrice: number | null = null;
    let entryTime: Date | null = null;
    let entryConfidence = 0;
    let totalTrades = 0;
    let winningTrades = 0;
    let losingTrades = 0;

    // Track daily losses
    const dailyLosses = new Map<string, number>();

    // Dynamic lot size based on capital (matching Python logic)
    const lotMultiplier = 1;
    const lotSize = this.config.lotSize * lotMultiplier;

    // Process candles (skip last 50 bars for trade simulation)
    for (let i = 0; i < candles.length - 50; i++) {
      const candle = candles[i];
      const currentSignal = getSignal(candle, candles.slice(0, i));

      // If not in position, check for entry signal
      if (position === null && currentSignal.signal !== 'HOLD' && currentSignal.confidence >= this.config.minConfidence) {
        // Check daily loss limit
        const tradeDate = new Date(candle.timestamp * 1000).toDateString();
        const dailyLossCount = dailyLosses.get(tradeDate) ?? 0;

        if (dailyLossCount >= this.config.maxDailyLosses) {
          continue;
        }

        // Enter position
        position = currentSignal.signal;
        entryPrice = candle.close;
        entryTime = new Date(candle.timestamp * 1000);
        entryConfidence = currentSignal.confidence;

        // Calculate scaled lot size and P&L
        let currentLotMultiplier = 1;
        let tempCapital = capital;
        while (tempCapital >= 50000) {
          tempCapital /= 2;
          currentLotMultiplier *= 2;
        }
        const currentLotSize = this.config.lotSize * currentLotMultiplier;
        const scaledTargetPnL = this.config.targetPnL * currentLotMultiplier;
        const scaledStopLossPnL = this.config.stopLossPnL * currentLotMultiplier;

        // Simulate the trade
        const result = this.simulateTrade(
          entryPrice,
          position,
          candles,
          i,
          currentLotSize,
          scaledTargetPnL,
          scaledStopLossPnL
        );

        // Calculate P&L
        const pnlCurrency = result.pnlPoints * currentLotSize;
        const totalBrokerage = this.config.brokerageEntry + this.config.brokerageExit;
        const netPnL = pnlCurrency - totalBrokerage;
        capital += netPnL;

        // Update trade counts
        totalTrades++;
        if (result.isWin) {
          winningTrades++;
        } else {
          losingTrades++;
          if (entryTime) {
            const dateStr = entryTime.toDateString();
            dailyLosses.set(dateStr, (dailyLosses.get(dateStr) ?? 0) + 1);
          }
        }

        // Create trade record
        const exitTime = new Date(candles[Math.min(i + result.barsHeld, candles.length - 1)].timestamp * 1000);
        const trade: Trade = {
          entryTime,
          exitTime,
          position,
          entryPrice: Math.round(entryPrice * 100) / 100,
          exitPrice: Math.round(result.exitDetails.exitPrice * 100) / 100,
          targetPrice: result.exitDetails.targetPrice,
          stopLossPrice: result.exitDetails.stopLossPrice,
          lotSize: currentLotSize,
          pnlPoints: Math.round(result.pnlPoints * 100) / 100,
          pnlCurrency: Math.round(netPnL * 100) / 100,
          barsHeld: result.barsHeld,
          exitReason: result.exitDetails.exitReason,
          confidence: Math.round(entryConfidence * 1000) / 10,
          capital
        };
        this.trades.push(trade);

        // Update equity curve
        this.equityCurve.push({
          timestamp: exitTime,
          capital
        });

        // Update peak capital
        this.peakCapital = Math.max(this.peakCapital, capital);

        // Reset position
        position = null;
        entryPrice = null;
        entryTime = null;
        entryConfidence = 0;

        // Skip ahead by bars_held to avoid overlapping trades
        i += result.barsHeld;
      }
    }

    // Calculate final metrics
    const metrics = this.calculateMetrics();

    return {
      trades: this.trades,
      equityCurve: this.equityCurve,
      metrics,
      config: { ...this.config }
    };
  }

  /**
   * Calculate performance metrics
   */
  private calculateMetrics(): BacktestMetrics {
    if (this.trades.length === 0) {
      return {
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
        winRate: 0,
        totalPnL: 0,
        totalPnLPct: 0,
        maxDrawdown: 0,
        maxDrawdownPct: 0,
        sharpeRatio: 0,
        profitFactor: 0,
        avgTradePnL: 0,
        maxWinningStreak: 0,
        maxLosingStreak: 0,
        highestDailyProfit: 0,
        highestDailyLoss: 0,
        maxTradesPerDay: 0,
        minTradesPerDay: 0
      };
    }

    // Total P&L
    const totalPnL = this.trades.reduce((sum, trade) => sum + trade.pnlCurrency, 0);
    const totalPnLPct = (totalPnL / this.config.initialCapital) * 100;

    // Win rate
    const winningTrades = this.trades.filter(t => t.pnlCurrency > 0).length;
    const losingTrades = this.trades.filter(t => t.pnlCurrency < 0).length;
    const winRate = winningTrades / this.trades.length;

    // Average trade P&L
    const avgTradePnL = totalPnL / this.trades.length;

    // Maximum drawdown
    let maxDrawdown = 0;
    let maxDrawdownPct = 0;

    if (this.equityCurve.length > 0) {
      let peak = this.equityCurve[0].capital;
      for (const point of this.equityCurve) {
        if (point.capital > peak) {
          peak = point.capital;
        }
        const drawdown = peak - point.capital;
        const drawdownPct = (drawdown / peak) * 100;
        if (drawdown > maxDrawdown) {
          maxDrawdown = drawdown;
          maxDrawdownPct = drawdownPct;
        }
      }
    }

    // Sharpe ratio
    let sharpeRatio = 0;
    if (this.equityCurve.length > 1) {
      const returns: number[] = [];
      for (let i = 1; i < this.equityCurve.length; i++) {
        const prevCapital = this.equityCurve[i - 1].capital;
        const currCapital = this.equityCurve[i].capital;
        returns.push((currCapital - prevCapital) / prevCapital);
      }
      const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
      const stdDev = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length);
      sharpeRatio = stdDev !== 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized
    }

    // Profit factor
    const profits = this.trades.filter(t => t.pnlCurrency > 0).reduce((sum, t) => sum + t.pnlCurrency, 0);
    const losses = Math.abs(this.trades.filter(t => t.pnlCurrency < 0).reduce((sum, t) => sum + t.pnlCurrency, 0));
    const profitFactor = losses !== 0 ? profits / losses : Infinity;

    // Daily statistics
    const dailyPnL = new Map<string, number>();
    const dailyTradeCount = new Map<string, number>();

    for (const trade of this.trades) {
      const dateStr = trade.entryTime.toDateString();
      dailyPnL.set(dateStr, (dailyPnL.get(dateStr) ?? 0) + trade.pnlCurrency);
      dailyTradeCount.set(dateStr, (dailyTradeCount.get(dateStr) ?? 0) + 1);
    }

    const highestDailyProfit = Math.max(...dailyPnL.values(), 0);
    const highestDailyLoss = Math.min(...dailyPnL.values(), 0);
    const maxTradesPerDay = Math.max(...dailyTradeCount.values(), 0);
    const minTradesPerDay = Math.min(...dailyTradeCount.values(), 0);

    // Winning and losing streaks
    let maxWinningStreak = 0;
    let maxLosingStreak = 0;
    let currentWinningStreak = 0;
    let currentLosingStreak = 0;

    for (const trade of this.trades) {
      if (trade.pnlCurrency > 0) {
        currentWinningStreak++;
        maxWinningStreak = Math.max(maxWinningStreak, currentWinningStreak);
        currentLosingStreak = 0;
      } else {
        currentLosingStreak++;
        maxLosingStreak = Math.max(maxLosingStreak, currentLosingStreak);
        currentWinningStreak = 0;
      }
    }

    return {
      totalTrades: this.trades.length,
      winningTrades,
      losingTrades,
      winRate,
      totalPnL,
      totalPnLPct,
      maxDrawdown,
      maxDrawdownPct,
      sharpeRatio,
      profitFactor,
      avgTradePnL,
      maxWinningStreak,
      maxLosingStreak,
      highestDailyProfit,
      highestDailyLoss,
      maxTradesPerDay,
      minTradesPerDay
    };
  }

  /**
   * Get trades
   */
  getTrades(): Trade[] {
    return this.trades;
  }

  /**
   * Get equity curve
   */
  getEquityCurve(): EquityPoint[] {
    return this.equityCurve;
  }
}