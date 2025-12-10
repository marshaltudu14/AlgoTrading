"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi, UTCTimestamp, ColorType, CandlestickSeries } from "lightweight-charts";
import { useTheme } from "next-themes";
import { ZoomIn, ZoomOut, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTradingStore } from "@/stores/tradingStore";

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
    refreshData
  } = useTradingStore();

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

  // Convert store data to chart format
  const chartData: ChartData[] = candleData.map((candle) => ({
    time: candle.timestamp as UTCTimestamp,
    open: candle.open,
    high: candle.high,
    low: candle.low,
    close: candle.close,
  }));

  
  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || isLoading || chartData.length === 0) return;

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

    // Fit content
    chart.timeScale().fitContent();

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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center w-full h-full">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading chart data...</p>
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

  if (chartData.length === 0) {
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
      <div
        ref={chartContainerRef}
        className="w-full h-full"
        style={{ width: "100%", height: "100%" }}
      />

      
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
    </div>
  );
}