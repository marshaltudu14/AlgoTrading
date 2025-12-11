"use client";

import { useEffect, useRef, useCallback } from "react";
import { createChart, IChartApi, UTCTimestamp, ColorType, CandlestickSeries } from "lightweight-charts";
import { useTheme } from "next-themes";
import { ZoomIn, ZoomOut, RefreshCw, Brain, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTradingStore } from "@/stores/tradingStore";
import { useCandleStore } from "@/stores/candleStore";

interface ChartData {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
}

export default function TradingChart() {
  
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<unknown | null>(null);
  const { theme } = useTheme();
  const {
    candleData,
    isLoading,
    error,
    refreshData,
    fetchCandleData
  } = useTradingStore();

  // Memoize fetchCandleData to prevent unnecessary re-renders
  const memoizedFetchCandleData = useCallback(() => {
    fetchCandleData();
  }, [fetchCandleData]);

  const {
    processedCandles,
    isProcessing,
    error: processingError,
    getFeatureNames,
    getExpectedFeatureCount
  } = useCandleStore();

  
  // Perform the initial fetch on mount
  useEffect(() => {
        memoizedFetchCandleData();
  }, [memoizedFetchCandleData]);

  // Process features when raw data is available and not already processing
  useEffect(() => {
    
    // Only process if we have raw data but no processed data, and we're not currently processing
    if (candleData.length > 0 && processedCandles.length === 0 && !isProcessing && !processingError) {
      // Get the fresh state each time
      const state = useCandleStore.getState();

      if (!state.isProcessing && state.candles.length > 0 && state.processedCandles.length === 0) {
        state.processFeatures();
      }
    }
  }, [candleData.length, isProcessing, processedCandles.length, processingError]); // Include all dependencies

  // Show loading states - only show chart when processing is complete
  const isLoadingData = isLoading;
  const isProcessingData = isProcessing;
  const shouldShowLoading = isLoadingData || isProcessingData;
  // Only show no data state if not loading AND not processing AND no data exists
  const shouldShowNoData = !isLoadingData && !isProcessingData && candleData.length === 0 && !error;
  // Only show chart when processing is complete
  const shouldShowChart = !isLoadingData && !isProcessingData && processedCandles.length > 0 && !error;

  const handleZoomIn = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const visibleRange = timeScale.getVisibleRange();
      if (visibleRange) {
        const range = Number(visibleRange.to) - Number(visibleRange.from);
        const center = Number(visibleRange.from) + range / 2;
        const newRange = range * 0.8; // Zoom in by 20%
        timeScale.setVisibleRange({
          from: center - newRange / 2 as UTCTimestamp,
          to: center + newRange / 2 as UTCTimestamp,
        });
      }
    }
  };

  const handleZoomOut = () => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale();
      const visibleRange = timeScale.getVisibleRange();
      if (visibleRange) {
        const range = Number(visibleRange.to) - Number(visibleRange.from);
        const center = Number(visibleRange.from) + range / 2;
        const newRange = range * 1.25; // Zoom out by 25%
        timeScale.setVisibleRange({
          from: center - newRange / 2 as UTCTimestamp,
          to: center + newRange / 2 as UTCTimestamp,
        });
      }
    }
  };

  const handleReset = () => {
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent();
    }
  };

  // Use processed candles for chart display when processing is complete
  // Timestamps from Fyers API are in seconds and already have the correct time
  // No timezone conversion needed - use timestamps as received from API
  const dataToDisplay = shouldShowChart ? processedCandles : [];
  const chartData: ChartData[] = dataToDisplay.map((candle) => ({
    // Use timestamp as-is (in seconds) for lightweight-charts
    time: candle.timestamp as UTCTimestamp,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
  }));

  // Get feature count for display
  const featureCount = processedCandles.length > 0 ? getFeatureNames().length : 0;
  const expectedFeatureCount = getExpectedFeatureCount();

  
  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || !shouldShowChart || chartData.length === 0) return;

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: {
          type: ColorType.Solid,
          color: theme === "dark" ? "#000000" : "#ffffff",
        },
        textColor: theme === "dark" ? "#ffffff" : "#000000",
      },
      grid: {
        vertLines: {
          color: theme === "dark" ? "#333333" : "#e5e5e5",
        },
        horzLines: {
          color: theme === "dark" ? "#333333" : "#e5e5e5",
        },
      },
      rightPriceScale: {
        borderColor: theme === "dark" ? "#333333" : "#e5e5e5",
        textColor: theme === "dark" ? "#ffffff" : "#000000",
      },
      timeScale: {
        borderColor: theme === "dark" ? "#333333" : "#e5e5e5",
        timeVisible: true,
        secondsVisible: false,
        // Ensure gaps in time are preserved (e.g., overnight, weekends, market closure)
        // This tells the chart to respect actual time intervals rather than interpolating
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: theme === "dark" ? "#444444" : "#cccccc",
          width: 1,
          style: 3,
        },
        horzLine: {
          color: theme === "dark" ? "#444444" : "#cccccc",
          width: 1,
          style: 3,
        },
      },
      // Explicitly set timezone for the chart if possible
      // Note: lightweight-charts doesn't have a direct timezone option in createChart
      // The timestamps should be handled properly by the library
      });

    // Add candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: theme === "dark" ? "#00d084" : "#10b981",
      downColor: theme === "dark" ? "#f44336" : "#ef4444",
      borderVisible: false,
      wickUpColor: theme === "dark" ? "#00d084" : "#10b981",
      wickDownColor: theme === "dark" ? "#f44336" : "#ef4444",
    });

    // Set data
    candlestickSeries.setData(chartData);

    // Configure time scale to respect actual time gaps and not interpolate
    const timeScale = chart.timeScale();
    timeScale.applyOptions({
      // Ensure that the time scale respects actual time gaps rather than interpolating
      visible: true,
      timeVisible: true,
      secondsVisible: false,
    });

    // Fit content
    timeScale.fitContent();

    // Store references
    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    return () => {
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [chartData, theme, isLoading]);

  // Update chart theme when theme changes
  useEffect(() => {
    if (chartRef.current && seriesRef.current) {
      // Update chart colors for theme
      chartRef.current.applyOptions({
        layout: {
          background: {
            type: ColorType.Solid,
            color: theme === "dark" ? "#000000" : "#ffffff",
          },
          textColor: theme === "dark" ? "#ffffff" : "#000000",
        },
        grid: {
          vertLines: {
            color: theme === "dark" ? "#333333" : "#e5e5e5",
          },
          horzLines: {
            color: theme === "dark" ? "#333333" : "#e5e5e5",
          },
        },
        rightPriceScale: {
          borderColor: theme === "dark" ? "#333333" : "#e5e5e5",
          textColor: theme === "dark" ? "#ffffff" : "#000000",
        },
        timeScale: {
          borderColor: theme === "dark" ? "#333333" : "#e5e5e5",
        },
        });

      // Update series colors for theme
      if (seriesRef.current && typeof seriesRef.current === 'object' && 'applyOptions' in seriesRef.current) {
        (seriesRef.current as { applyOptions: (options: unknown) => void }).applyOptions({
        upColor: theme === "dark" ? "#00d084" : "#10b981",
        downColor: theme === "dark" ? "#f44336" : "#ef4444",
        borderDownColor: theme === "dark" ? "#f44336" : "#ef4444",
        borderUpColor: theme === "dark" ? "#00d084" : "#10b981",
        wickDownColor: theme === "dark" ? "#f44336" : "#ef4444",
        wickUpColor: theme === "dark" ? "#00d084" : "#10b981",
      });
      }
    }
  }, [theme]);

  if (shouldShowLoading) {
    return (
      <div className="flex items-center justify-center w-full h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">
            {isLoadingData ? 'Fetching chart data...' : 'Processing technical indicators...'}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center w-full h-full">
        <div className="text-center">
          <div className="text-destructive mb-4">
            <RefreshCw className="h-8 w-8 mx-auto" />
          </div>
          <p className="text-destructive font-medium mb-2">Failed to load chart data</p>
          <p className="text-muted-foreground text-sm mb-4">{error}</p>
          <Button onClick={refreshData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  if (shouldShowNoData) {
    return (
      <div className="flex items-center justify-center w-full h-full">
        <div className="text-center">
          <p className="text-muted-foreground">No data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full">
      {/* Only show chart when processing is complete */}
      {shouldShowChart && (
        <div
          ref={chartContainerRef}
          className="w-full h-full"
          style={{ width: "100%", height: "100%" }}
        />
      )}

      {/* Processing Status and Feature Info */}
      <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm border border-border rounded-lg p-2 text-xs space-y-1 z-50">
        {/* Processing Status */}
        <div className="flex items-center gap-2">
          {isProcessing ? (
            <>
              <Activity className="h-3 w-3 animate-spin" />
              <span className="text-muted-foreground">Processing features...</span>
            </>
          ) : processedCandles.length > 0 ? (
            <>
              <Brain className="h-3 w-3 text-green-500" />
              <span className="text-green-500">Features ready</span>
            </>
          ) : candleData.length > 0 ? (
            <>
              <Brain className="h-3 w-3 text-yellow-500" />
              <span className="text-yellow-500">Ready to process</span>
            </>
          ) : null}
        </div>

        {/* Feature Count */}
        {featureCount > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Features:</span>
            <span className={`font-medium ${featureCount === expectedFeatureCount ? 'text-green-500' : 'text-yellow-500'}`}>
              {featureCount}/{expectedFeatureCount}
            </span>
          </div>
        )}

        {/* Processing Error */}
        {processingError && (
          <div className="text-destructive text-xs">
            Error: {processingError}
          </div>
        )}
      </div>

      {/* Chart Controls - Only show when chart is visible */}
      {shouldShowChart && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-background/80 backdrop-blur-sm border border-border rounded-lg p-1 flex items-center gap-1 z-50">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomOut}
            className="h-8 w-8 p-0 hover:bg-muted cursor-pointer"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="h-8 w-8 p-0 hover:bg-muted cursor-pointer"
          >
            <RefreshCw className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleZoomIn}
            className="h-8 w-8 p-0 hover:bg-muted cursor-pointer"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      )}
    </div>
  );
}