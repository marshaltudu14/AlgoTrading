"use client";

import { useEffect, useRef } from "react";
import { createChart, IChartApi, UTCTimestamp, ColorType } from "lightweight-charts";
import { useTheme } from "next-themes";

interface ChartData {
  time: string | UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface TradingChartProps {
  data?: ChartData[];
  symbol?: string;
  interval?: string;
}

// Generate demo data with realistic candlestick patterns
function generateDemoData(): ChartData[] {
  const now = new Date();
  const data: ChartData[] = [];
  let lastClose = 19500;

  // Generate 500 data points (about 2 years of daily data)
  for (let i = 500; i >= 0; i--) {
    const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);

    // Add some randomness to create realistic patterns
    const volatility = 0.02; // 2% daily volatility
    const trend = Math.sin(i * 0.05) * 0.01; // Sinusoidal trend

    const randomWalk = (Math.random() - 0.5) * volatility;
    const change = trend + randomWalk;

    const open = lastClose;
    const close = open * (1 + change);

    // Ensure high >= max(open, close) and low <= min(open, close)
    const range = Math.abs(close - open) * (0.5 + Math.random() * 0.5);
    const high = Math.max(open, close) + range * 0.5;
    const low = Math.min(open, close) - range * 0.5;

    // Add some gaps (weekends) for realism
    if (Math.random() < 0.05) {
      const gap = (Math.random() - 0.5) * 0.03;
      lastClose = close * (1 + gap);
    } else {
      lastClose = close;
    }

    data.push({
      time: (date.getTime() / 1000) as UTCTimestamp,
      open: Math.round(open * 100) / 100,
      high: Math.round(high * 100) / 100,
      low: Math.round(low * 100) / 100,
      close: Math.round(close * 100) / 100,
    });
  }

  return data;
}

export default function TradingChart({
  data = generateDemoData(),
  symbol = "NIFTY",
  interval = "1D"
}: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<any>(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (!chartContainerRef.current) return;

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
    const candlestickSeries = (chart as any).addSeries({
      type: "candlestick",
      upColor: theme === "dark" ? "#00d084" : "#10b981",
      downColor: theme === "dark" ? "#f44336" : "#ef4444",
      borderDownColor: theme === "dark" ? "#f44336" : "#ef4444",
      borderUpColor: theme === "dark" ? "#00d084" : "#10b981",
      wickDownColor: theme === "dark" ? "#f44336" : "#ef4444",
      wickUpColor: theme === "dark" ? "#00d084" : "#10b981",
    });

    // Set data
    candlestickSeries.setData(data);

    // Fit content
    chart.timeScale().fitContent();

    // Store references
    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    // Handle resize
    const handleResize = () => {
      if (chartRef.current && chartContainerRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: chartContainerRef.current.clientHeight,
        });
      }
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
      }
    };
  }, [data]);

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
      seriesRef.current.applyOptions({
        upColor: theme === "dark" ? "#00d084" : "#10b981",
        downColor: theme === "dark" ? "#f44336" : "#ef4444",
        borderDownColor: theme === "dark" ? "#f44336" : "#ef4444",
        borderUpColor: theme === "dark" ? "#00d084" : "#10b981",
        wickDownColor: theme === "dark" ? "#f44336" : "#ef4444",
        wickUpColor: theme === "dark" ? "#00d084" : "#10b981",
      });
    }
  }, [theme, symbol]);

  return (
    <div className="relative w-full h-full">
      <div
        ref={chartContainerRef}
        className="w-full h-full"
        style={{ width: "100%", height: "100%" }}
      />
      <div className="absolute top-4 left-4 bg-background/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-border">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm font-medium">{symbol}</span>
          <span className="text-xs text-muted-foreground">{interval}</span>
        </div>
      </div>
    </div>
  );
}