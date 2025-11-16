"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { createChart, IChartApi, UTCTimestamp, ColorType, CandlestickSeries } from "lightweight-charts";
import { useTheme } from "next-themes";
import { ZoomIn, ZoomOut, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { TIMEFRAMES } from "@/config/instruments";
import { useTradingState } from "@/components/TradingProvider";

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
  const { symbol: contextSymbol, timeframe: contextTimeframe, dataRefreshTrigger } = useTradingState();
  const [data, setData] = useState<ChartData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Always use context values, never fall back to props or defaults
  const currentSymbol = contextSymbol || "NSE:NIFTY50-INDEX";
  const currentTimeframe = contextTimeframe || "5";

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

  // Convert epoch timestamp to format lightweight-charts accepts
  const convertTimestampForChart = (epochTime: number): UTCTimestamp => {
    // Backend returns epoch in seconds, convert to seconds for lightweight-charts
    // If timestamp is in milliseconds, convert to seconds
    const timestampSeconds = epochTime > 10000000000 ? Math.floor(epochTime / 1000) : epochTime;

    return timestampSeconds as UTCTimestamp;
  };

  // Fetch candle data from API
  const fetchCandleData = useCallback(async () => {
    if (!currentSymbol || !currentTimeframe) return;

    setIsLoading(true);
    setError(null);

    try {
      // Get access token and app_id from localStorage
      const access_token = localStorage.getItem('access_token');
      const app_id = localStorage.getItem('app_id');

      // Find timeframe configuration to get days parameter
      const timeframeConfig = TIMEFRAMES.find(tf => tf.name === currentTimeframe);
      const days = timeframeConfig?.days || 15; // Default to 15 days if not found

      const params = new URLSearchParams();
      if (access_token) params.append('access_token', access_token);
      if (app_id) params.append('app_id', app_id);
      params.append('days', days.toString());

      const response = await fetch(`/api/candle-data/${currentSymbol}/${currentTimeframe}?${params.toString()}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch candle data: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.success && result.data) {
        // Convert API data to ChartData format
        const chartData: ChartData[] = result.data.map((item: {
          timestamp?: number;
          time?: number;
          open: string | number;
          high: string | number;
          low: string | number;
          close: string | number;
        }) => ({
          time: convertTimestampForChart(item.timestamp || item.time || 0), // Backend uses 'timestamp' field
          open: parseFloat(item.open.toString()),
          high: parseFloat(item.high.toString()),
          low: parseFloat(item.low.toString()),
          close: parseFloat(item.close.toString()),
        }));

        setData(chartData);
      } else {
        throw new Error(result.error || 'No data available');
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch candle data';
      setError(errorMessage);
      console.error('Error fetching candle data:', err);
    } finally {
      setIsLoading(false);
    }
  }, [currentSymbol, currentTimeframe]);

  // Fetch data when component mounts, symbol/timeframe changes, or refresh is triggered
  useEffect(() => {
    fetchCandleData();
  }, [fetchCandleData, dataRefreshTrigger]);

  
  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current || isLoading || data.length === 0) return;

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
    candlestickSeries.setData(data);

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
  }, [data, theme, isLoading]);

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
          <Button onClick={fetchCandleData} variant="outline" size="sm">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
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