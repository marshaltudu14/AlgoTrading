"use client";

import { useState, useEffect } from "react";
import { useBacktestStore, type DurationUnit } from "@/stores/backtestStore";
import { useTradingContext } from "@/components/TradingProvider";
import { useStrategyStore } from "@/stores/strategyStore";
import { useCandleStore } from "@/stores/candleStore";
import { INSTRUMENTS, TIMEFRAMES } from "@/config/instruments";
import { BacktestEngine, BacktestResults } from '@/lib/backtestEngine';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Switch } from "@/components/ui/switch";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Play, BarChart3, ChevronDown, TrendingUp, Clock, Loader2 } from "lucide-react";

export default function BacktestDialog() {
  const { symbol, timeframe } = useTradingContext();
  const {
    config,
    isDialogOpen,
    isBacktestLoading,
    setSymbol,
    setTimeframe,
    setDuration,
    setDurationUnit,
    setTargetPL,
    setStopLossPL,
    setInitialCapital,
    setDoubleLotSize,
    setTrailSL,
    openDialog,
    closeDialog,
    startBacktest,
    setBacktestError,
    setBacktestResults,
  } = useBacktestStore();

  const [errors, setErrors] = useState<Record<string, string>>({});

  // Sync backtest config with current trading selection when dialog opens
  useEffect(() => {
    if (isDialogOpen) {
      // Only update if the current symbol/timeframe differs from the backtest config
      if (config.symbol !== symbol) {
        setSymbol(symbol);
      }
      if (config.timeframe !== timeframe) {
        setTimeframe(timeframe);
      }
    }
  }, [isDialogOpen, symbol, timeframe, config.symbol, config.timeframe, setSymbol, setTimeframe]);

  // Types for backtest data
  interface CandleData {
    timestamp: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }

  
  const runBacktestWithEngine = async (candleData: CandleData[]) => {
    console.log("🎯 Starting backtest with proper feature processing");

    try {
      // Use candleStore to process features
      const { setCandles, processFeatures } = useCandleStore.getState();

      // Set raw candles and process them
      setCandles(candleData);
      processFeatures();

      // Wait a moment for processing to complete
      await new Promise(resolve => setTimeout(resolve, 500));

      // Get processed candles from the store
      const finalProcessedCandles = useCandleStore.getState().processedCandles;
      console.log(`✅ Processed ${finalProcessedCandles.length} candles with features`);

      // Get the strategy function
      const { getSignal } = useStrategyStore.getState();

      // Create backtest engine with target configuration
      const engine = new BacktestEngine({
        targetPnL: config.targetPL,
        stopLossPnL: config.stopLossPL,
        initialCapital: config.initialCapital,
        lotSize: 75,
        minConfidence: 0.5
      });

      // Run the backtest
      console.log("🚀 Running backtest engine...");
      const results: BacktestResults = await engine.runBacktest(finalProcessedCandles, getSignal);

      console.log(`✅ Backtest complete: ${results.trades.length} trades, Total P&L: ₹${results.metrics.totalPnL.toFixed(0)}`);

      return results;

    } catch (error) {
      console.error("❌ Backtest failed:", error);
      throw error;
    }
  };

      
  const validateForm = () => {
    const newErrors: Record<string, string> = {};

    if (config.duration <= 0) {
      newErrors.duration = "Duration must be greater than 0";
    }

    if (config.targetPL <= 0) {
      newErrors.targetPL = "Target P&L must be greater than 0";
    }

    if (config.stopLossPL >= 0) {
      newErrors.stopLossPL = "Stop Loss must be negative";
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleStartBacktest = async () => {
    if (validateForm()) {
      console.log("🚀 Starting backtest with config:", config);
      startBacktest();

      try {
        // Fetch historical data
        console.log(`📊 Fetching ${config.durationInDays} days of data for ${config.symbol} (${config.timeframe}M)`);
        const response = await fetch(
          `/api/candle-data/${config.symbol}/${config.timeframe}?days=${config.durationInDays}`,
          {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
            },
          }
        );

        console.log("📥 Response status:", response.status);

        if (!response.ok) {
          throw new Error(`Failed to fetch historical data: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("📊 Fetched data:", data);

        if (!data.success) {
          throw new Error(data.error || 'Failed to fetch historical data');
        }

        const candleData: CandleData[] = data.data;
        console.log(`✅ Got ${candleData.length} candles`);

        // Execute backtest
        console.log("⚙️ Running backtest...");
        console.log("📊 Sample candle data:", candleData.slice(0, 2));

        // Note: The strategy expects processed candles with features, but we have raw candles
        // We need to process them first or the backtest should handle this internally
        console.log("📝 Note: Strategy requires processed candles with features");

        // Run backtest using the engine
        const results = await runBacktestWithEngine(candleData);
        console.log("📈 Backtest results:", results);

        // Get processed candles for saving
        const processedCandles = useCandleStore.getState().processedCandles;

        // Save processed data to CSV for comparison
        try {
          const saveResponse = await fetch('/api/save-processed-data', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              processedCandles,
              symbol: config.symbol.replace(':', '-'), // Format for filename
              timeframe: config.timeframe
            })
          });

          if (saveResponse.ok) {
            const saveResult = await saveResponse.json();
            console.log("✅ Saved processed data:", saveResult.message);
          }
        } catch (error) {
          console.error("⚠️ Failed to save processed data:", error);
        }

        // Store results with both raw and processed candle data
        setBacktestResults({
          trades: results.trades,
          metrics: {
            totalPL: results.metrics.totalPnL,
            totalTrades: results.metrics.totalTrades,
                        winRate: results.metrics.winRate,
            maxDrawdown: results.metrics.maxDrawdown,
            sharpeRatio: results.metrics.sharpeRatio,
            equityCurve: results.equityCurve.map(point => ({
              time: point.timestamp.getTime() / 1000,
              value: point.capital
            }))
          },
          candleData,
          processedCandles
        });
        console.log("✅ Backtest completed successfully!");

      } catch (error) {
        console.error('❌ Backtest error:', error);
        setBacktestError(error instanceof Error ? error.message : 'Failed to execute backtest');
      }
    }
  };

  const handleDurationChange = (value: string) => {
    const numValue = parseInt(value) || 0;
    if (numValue > 0) {
      setDuration(numValue);
      setErrors(prev => ({ ...prev, duration: "" }));
    }
  };

  const handleTargetPLChange = (value: string) => {
    const numValue = parseFloat(value) || 0;
    if (numValue > 0) {
      setTargetPL(numValue);
      setErrors(prev => ({ ...prev, targetPL: "" }));
    }
  };

  const handleStopLossPLChange = (value: string) => {
    const numValue = parseFloat(value) || 0;
    if (numValue < 0) {
      setStopLossPL(numValue);
      setErrors(prev => ({ ...prev, stopLossPL: "" }));
    }
  };

  const currentInstrument = INSTRUMENTS.find(inst => inst.exchangeSymbol === config.symbol);
  const currentTimeframe = TIMEFRAMES.find(tf => tf.name === config.timeframe);

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        className="h-8 w-8 p-0 cursor-pointer"
        onClick={openDialog}
      >
        <BarChart3 className="h-4 w-4" />
      </Button>
      <Dialog open={isDialogOpen} onOpenChange={(open) => !open ? closeDialog() : {}}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Backtest Configuration</DialogTitle>
            <DialogDescription>
              Configure the parameters for your backtest simulation.
            </DialogDescription>
          </DialogHeader>
        <div className="grid gap-4 py-4">
          {/* Instrument Selector */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="instrument" className="text-right">
              Instrument
            </Label>
            <div className="col-span-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="w-full justify-between">
                    <div className="flex items-center gap-2">
                      <TrendingUp className="h-4 w-4" />
                      <span>{currentInstrument?.name || config.symbol}</span>
                    </div>
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-full">
                  {INSTRUMENTS.map((instrument) => (
                    <DropdownMenuItem
                      key={instrument.id}
                      onClick={() => setSymbol(instrument.exchangeSymbol)}
                      className={config.symbol === instrument.exchangeSymbol ? "bg-accent" : ""}
                    >
                      <div className="font-medium">{instrument.name}</div>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Timeframe Selector */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="timeframe" className="text-right">
              Timeframe
            </Label>
            <div className="col-span-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="w-full justify-between">
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      <span>{currentTimeframe?.name}M</span>
                    </div>
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-full">
                  {TIMEFRAMES.map((tf) => (
                    <DropdownMenuItem
                      key={tf.id}
                      onClick={() => setTimeframe(tf.name)}
                      className={config.timeframe === tf.name ? "bg-accent" : ""}
                    >
                      <div className="font-medium">{tf.name}M - {tf.description}</div>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Duration */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="duration" className="text-right">
              Duration
            </Label>
            <div className="col-span-3 space-y-2">
              <Input
                id="duration"
                type="number"
                value={config.duration}
                onChange={(e) => handleDurationChange(e.target.value)}
                placeholder="Enter duration"
                className={errors.duration ? "border-red-500" : ""}
                min="1"
              />
              {errors.duration && (
                <p className="text-sm text-red-500">{errors.duration}</p>
              )}
              <RadioGroup
                value={config.durationUnit}
                onValueChange={(value: DurationUnit) => setDurationUnit(value)}
                className="flex gap-4"
              >
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="days" id="days" />
                  <Label htmlFor="days">Days</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="months" id="months" />
                  <Label htmlFor="months">Months</Label>
                </div>
                <div className="flex items-center space-x-2">
                  <RadioGroupItem value="years" id="years" />
                  <Label htmlFor="years">Years</Label>
                </div>
              </RadioGroup>
              <p className="text-sm text-muted-foreground">
                Total: {config.durationInDays} days
              </p>
            </div>
          </div>

          {/* Target P&L */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="targetPL" className="text-right">
              Target P&L
            </Label>
            <div className="col-span-3 space-y-1">
              <Input
                id="targetPL"
                type="number"
                value={config.targetPL}
                onChange={(e) => handleTargetPLChange(e.target.value)}
                placeholder="Enter target P&L"
                className={errors.targetPL ? "border-red-500" : ""}
                min="0.01"
                step="0.01"
              />
              {errors.targetPL && (
                <p className="text-sm text-red-500">{errors.targetPL}</p>
              )}
            </div>
          </div>

          {/* Stop Loss P&L */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="stopLossPL" className="text-right">
              Stop Loss
            </Label>
            <div className="col-span-3 space-y-1">
              <Input
                id="stopLossPL"
                type="number"
                value={config.stopLossPL}
                onChange={(e) => handleStopLossPLChange(e.target.value)}
                placeholder="Enter stop loss P&L"
                className={errors.stopLossPL ? "border-red-500" : ""}
                max="-0.01"
                step="0.01"
              />
              {errors.stopLossPL && (
                <p className="text-sm text-red-500">{errors.stopLossPL}</p>
              )}
            </div>
          </div>

          {/* Initial Capital */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="initialCapital" className="text-right">
              Initial Capital
            </Label>
            <div className="col-span-3 space-y-1">
              <Input
                id="initialCapital"
                type="number"
                value={config.initialCapital}
                onChange={(e) => {
                  const numValue = parseFloat(e.target.value) || 0;
                  if (numValue > 0) {
                    setInitialCapital(numValue);
                  }
                }}
                placeholder="Enter initial capital"
                min="1000"
                step="1000"
              />
            </div>
          </div>

          {/* Double Lot Size & Trail SL */}
          <div className="grid grid-cols-4 items-center gap-4">
            <Label className="text-right">
              Options
            </Label>
            <div className="col-span-3 flex items-center gap-6">
              <div className="flex items-center gap-2">
                <Label className="text-sm">Double Lot Size</Label>
                <Switch
                  checked={config.doubleLotSize}
                  onCheckedChange={setDoubleLotSize}
                />
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-sm">Trail SL</Label>
                <Switch
                  checked={config.trailSL}
                  onCheckedChange={setTrailSL}
                />
              </div>
            </div>
          </div>
        </div>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={closeDialog} disabled={isBacktestLoading}>
            Cancel
          </Button>
          <Button type="button" onClick={handleStartBacktest} disabled={isBacktestLoading}>
            {isBacktestLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Running Backtest...
              </>
            ) : (
              <>
                <Play className="h-4 w-4 mr-2" />
                Start Backtest
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
    </>
  );
}