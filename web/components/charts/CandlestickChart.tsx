'use client';

import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, CandlestickData, Time, ColorType, CandlestickSeries, HistogramSeries } from 'lightweight-charts';
import { Timeframe } from '@/types/fyers';
import { useMarketData } from '@/stores/marketStore';

interface CandlestickChartProps {
  width?: number;
  height?: number;
  onTimeframeChange?: (timeframe: Timeframe) => void;
  showVolume?: boolean;
  showToolbar?: boolean;
}

export function CandlestickChart({
  width = 800,
  height = 600,
  onTimeframeChange,
  showVolume = true,
  showToolbar = true,
}: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);

  const {
    chartData,
    symbol,
    timeframe,
    isLoading,
    currentPrice,
    priceChange,
    priceChangePercent,
    dayHigh,
    dayLow,
    volume,
    setTimeframe,
    isPositive,
    isNegative,
  } = useMarketData();

  const [chartWidth, setChartWidth] = useState(width);

  // Timeframe options
  const timeframeOptions: { value: Timeframe; label: string }[] = [
    { value: '1', label: '1m' },
    { value: '3', label: '3m' },
    { value: '5', label: '5m' },
    { value: '10', label: '10m' },
    { value: '15', label: '15m' },
    { value: '30', label: '30m' },
    { value: '60', label: '1h' },
    { value: '120', label: '2h' },
    { value: '240', label: '4h' },
    { value: 'D', label: '1D' },
  ];

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (chartContainerRef.current) {
        const containerWidth = chartContainerRef.current.clientWidth;
        setChartWidth(containerWidth);
      }
    };

    window.addEventListener('resize', handleResize);
    handleResize(); // Initial sizing

    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize and update chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Create chart if it doesn't exist
    if (!chartRef.current) {
      const chart = createChart(chartContainerRef.current, {
        width: chartWidth,
        height,
        layout: {
          background: { type: ColorType.Solid, color: '#1a1a1a' },
          textColor: '#d1d5db',
        },
        grid: {
          vertLines: { color: '#2a2a2a' },
          horzLines: { color: '#2a2a2a' },
        },
        crosshair: {
          mode: 1,
          vertLine: {
            color: '#4a5568',
            width: 1,
            style: 3,
          },
          horzLine: {
            color: '#4a5568',
            width: 1,
            style: 3,
          },
        },
        rightPriceScale: {
          borderColor: '#2a2a2a',
          textColor: '#d1d5db',
        },
        timeScale: {
          borderColor: '#2a2a2a',
          timeVisible: true,
          secondsVisible: false,
        },
        overlayPriceScales: {
          ticksVisible: false,
          borderVisible: false,
        },
        localization: {
          priceFormatter: (price: number) => {
            return price.toFixed(2);
          },
        },
      });

      // Add candlestick series using correct API
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
        priceFormat: {
          type: 'price',
          precision: 2,
          minMove: 0.01,
        },
      });

      // Add volume series if enabled
      let volumeSeries: ISeriesApi<'Histogram'> | null = null;
      if (showVolume) {
        volumeSeries = chart.addSeries(HistogramSeries, {
          color: '#4a5568',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        });

        chart.priceScale('volume').applyOptions({
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        });
      }

      chartRef.current = chart;
      candlestickSeriesRef.current = candlestickSeries;
      volumeSeriesRef.current = volumeSeries;

      // Cleanup function
      return () => {
        chart.remove();
        chartRef.current = null;
        candlestickSeriesRef.current = null;
        volumeSeriesRef.current = null;
      };
    }

    // Update chart size
    chartRef.current.applyOptions({
      width: chartWidth,
      height,
    });
  }, [chartWidth, height, showVolume]);

  // Update chart data
  useEffect(() => {
    if (!candlestickSeriesRef.current || !chartData.length) return;

    // Transform data for lightweight-charts
    const candlestickData: CandlestickData[] = chartData.map((point) => ({
      time: point.time as Time,
      open: point.open,
      high: point.high,
      low: point.low,
      close: point.close,
    }));

    candlestickSeriesRef.current.setData(candlestickData);

    // Update volume data if enabled
    if (showVolume && volumeSeriesRef.current) {
      const volumeData = chartData.map((point) => ({
        time: point.time as Time,
        value: point.value || 0,
        color: point.close >= point.open ? '#10b98130' : '#ef444430',
      }));

      volumeSeriesRef.current.setData(volumeData);
    }

    // Fit content to show all data
    chartRef.current?.timeScale().fitContent();
  }, [chartData, showVolume]);

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe: Timeframe) => {
    setTimeframe(newTimeframe);
    onTimeframeChange?.(newTimeframe);
  };

  // Format large numbers
  const formatNumber = (num: number | null) => {
    if (num === null) return '0';
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(2);
  };

  return (
    <div className="w-full bg-gray-900 rounded-lg overflow-hidden">
      {/* Header with price info */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-white">{symbol}</h2>
            <div className="flex items-baseline space-x-2 mt-1">
              <span className="text-2xl font-bold text-white">
                {currentPrice ? currentPrice.toFixed(2) : '0.00'}
              </span>
              <span className={`text-sm font-medium ${isPositive ? 'text-green-400' : isNegative ? 'text-red-400' : 'text-gray-400'}`}>
                {priceChange !== null && (
                  <>
                    {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)} ({priceChangePercent?.toFixed(2)}%)
                  </>
                )}
              </span>
            </div>
            <div className="flex items-center space-x-4 mt-1 text-xs text-gray-400">
              <span>H: {dayHigh ? dayHigh.toFixed(2) : '0.00'}</span>
              <span>L: {dayLow ? dayLow.toFixed(2) : '0.00'}</span>
              <span>V: {formatNumber(volume)}</span>
            </div>
          </div>

          {/* Timeframe selector */}
          {showToolbar && (
            <div className="flex items-center space-x-1 bg-gray-800 rounded-lg p-1">
              {timeframeOptions.map((tf) => (
                <button
                  key={tf.value}
                  onClick={() => handleTimeframeChange(tf.value)}
                  className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                    timeframe === tf.value
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {tf.label}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Chart container */}
      <div className="relative">
        {isLoading && (
          <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-10">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
              <p className="text-gray-400 text-sm">Loading chart data...</p>
            </div>
          </div>
        )}

        <div ref={chartContainerRef} className="w-full" style={{ height: chartHeight }} />
      </div>

      {/* Footer with additional info */}
      <div className="p-3 border-t border-gray-800 bg-gray-900 bg-opacity-50">
        <div className="flex items-center justify-between text-xs text-gray-400">
          <span>Last updated: {new Date().toLocaleTimeString()}</span>
          <span>Timeframe: {timeframeOptions.find(tf => tf.value === timeframe)?.label}</span>
        </div>
      </div>
    </div>
  );
}