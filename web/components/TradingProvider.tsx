"use client";

import React, { useState, createContext, useContext, useEffect, useMemo } from "react";
import { DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME } from "@/config/instruments";

interface TradingState {
  symbol: string;
  timeframe: string;
  currentTime: Date | null;
  nextUpdateTime: Date | null;
  countdown: number;
  dataRefreshTrigger: number;
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: string) => void;
  setCurrentTime: (time: Date) => void;
  triggerDataRefresh: () => void;
}

// Create context for trading state
const TradingContext = createContext<TradingState>({
  symbol: DEFAULT_INSTRUMENT.exchangeSymbol,
  timeframe: DEFAULT_TIMEFRAME.name,
  currentTime: null,
  nextUpdateTime: null,
  countdown: 0,
  dataRefreshTrigger: 0,
  setSymbol: () => {},
  setTimeframe: () => {},
  setCurrentTime: () => {},
  triggerDataRefresh: () => {},
});

export const useTradingState = () => useContext(TradingContext);
export const useTradingContext = useTradingState;

export default function TradingProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [symbol, setSymbol] = useState(DEFAULT_INSTRUMENT.exchangeSymbol);
  const [timeframe, setTimeframe] = useState(DEFAULT_TIMEFRAME.name);
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [nextUpdateTime, setNextUpdateTime] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [dataRefreshTrigger, setDataRefreshTrigger] = useState(0);

  // Calculate next rounded interval based on timeframe (with 1-second delay for latest data)
  const calculateNextInterval = (tf: string, now: Date): Date => {
    const timeframeMinutes = parseInt(tf);

    if (timeframeMinutes < 60) {
      // For minute-based timeframes
      const currentMinutes = now.getMinutes();
      const nextIntervalMinutes = Math.ceil((currentMinutes + 1) / timeframeMinutes) * timeframeMinutes;
      const nextTime = new Date(now);
      nextTime.setMinutes(nextIntervalMinutes, 0, 1); // Add 1-second delay for latest data

      // If next time is in the past, add the timeframe interval
      if (nextTime <= now) {
        nextTime.setMinutes(nextTime.getMinutes() + timeframeMinutes);
      }

      return nextTime;
    } else {
      // For hour-based timeframes (60, 120, 180, 240 minutes)
      const hours = timeframeMinutes / 60;
      const currentHours = now.getHours();
      const nextIntervalHours = Math.ceil((currentHours + 1) / hours) * hours;
      const nextTime = new Date(now);
      nextTime.setHours(nextIntervalHours, 0, 0, 1); // Add 1-second delay for latest data

      // If next time is in the past, add the timeframe interval
      if (nextTime <= now) {
        nextTime.setHours(nextTime.getHours() + hours);
      }

      return nextTime;
    }
  };

  // Update current time and countdown every second
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(now);

      // Update countdown based on current timeframe and time
      if (timeframe) {
        const nextInterval = calculateNextInterval(timeframe, now);
        const diff = nextInterval.getTime() - now.getTime();
        const seconds = Math.max(0, Math.floor(diff / 1000));
        setCountdown(seconds);
        setNextUpdateTime(nextInterval);

        // Trigger data refresh when countdown reaches 0 (with 1-second buffer for latest data)
        if (seconds === 0) {
          setDataRefreshTrigger(prev => prev + 1);
        }
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [timeframe]);

  const triggerDataRefresh = () => {
    setDataRefreshTrigger(prev => prev + 1);
  };

  // Memoize state to prevent unnecessary re-renders
  const state = useMemo(() => ({
    symbol,
    timeframe,
    currentTime,
    nextUpdateTime,
    countdown,
    dataRefreshTrigger,
    setSymbol,
    setTimeframe,
    setCurrentTime,
    triggerDataRefresh,
  }), [symbol, timeframe, currentTime, nextUpdateTime, countdown, dataRefreshTrigger]);

  return (
    <TradingContext.Provider value={state}>
      {children}
    </TradingContext.Provider>
  );
}