"use client";

import React, { useState, createContext, useContext, useEffect, useMemo, useCallback } from "react";
import { DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME, INSTRUMENTS, TIMEFRAMES } from "@/config/instruments";
import { useTradingStore } from "@/stores/tradingStore";

interface TradingState {
  symbol: string;
  timeframe: string;
  currentTime: Date | null;
  nextUpdateTime: Date | null;
  countdown: number;
  selectedInstrument: typeof INSTRUMENTS[0];
  selectedTimeframe: typeof TIMEFRAMES[0];
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: string) => void;
  setSelectedInstrument: (instrument: typeof INSTRUMENTS[0]) => void;
  setSelectedTimeframe: (timeframe: typeof TIMEFRAMES[0]) => void;
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
  selectedInstrument: DEFAULT_INSTRUMENT,
  selectedTimeframe: DEFAULT_TIMEFRAME,
  setSymbol: () => {},
  setTimeframe: () => {},
  setSelectedInstrument: () => {},
  setSelectedTimeframe: () => {},
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
  const {
    selectedInstrument,
    selectedTimeframe,
    setSelectedInstrument,
    setSelectedTimeframe,
  } = useTradingStore();

  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [nextUpdateTime, setNextUpdateTime] = useState<Date | null>(null);
  const [countdown, setCountdown] = useState(0);

  const symbol = selectedInstrument.exchangeSymbol;
  const timeframe = selectedTimeframe.name;

  const setSymbol = useCallback((newSymbol: string) => {
    const instrument = INSTRUMENTS.find(i => i.exchangeSymbol === newSymbol);
    if (instrument) {
      setSelectedInstrument(instrument);
    }
  }, [setSelectedInstrument]);

  const setTimeframe = useCallback((newTimeframe: string) => {
    const tf = TIMEFRAMES.find(t => t.name === newTimeframe);
    if (tf) {
      setSelectedTimeframe(tf);
    }
  }, [setSelectedTimeframe]);

  // Calculate next rounded interval based on timeframe
  const calculateNextInterval = (tf: string, now: Date): Date => {
    const timeframeMinutes = parseInt(tf);

    if (timeframeMinutes < 60) {
      // For minute-based timeframes
      const currentMinutes = now.getMinutes();
      const nextIntervalMinutes = Math.ceil((currentMinutes + 1) / timeframeMinutes) * timeframeMinutes;
      const nextTime = new Date(now);
      nextTime.setMinutes(nextIntervalMinutes, 0, 0); // Exact interval time

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
      nextTime.setHours(nextIntervalHours, 0, 0, 0); // Exact interval time

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
        const rawSeconds = Math.floor(diff / 1000);
        const seconds = Math.max(0, rawSeconds);
        setCountdown(seconds);
        setNextUpdateTime(nextInterval);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [timeframe]);

  const triggerDataRefresh = useCallback(() => {
    // Refresh data through the trading store
    useTradingStore.getState().refreshData();
  }, []);

  // Memoize state to prevent unnecessary re-renders
  const state = useMemo(() => ({
    symbol,
    timeframe,
    currentTime,
    nextUpdateTime,
    countdown,
    selectedInstrument,
    selectedTimeframe,
    setSymbol,
    setTimeframe,
    setSelectedInstrument,
    setSelectedTimeframe,
    setCurrentTime,
    triggerDataRefresh,
  }), [symbol, timeframe, currentTime, nextUpdateTime, countdown, selectedInstrument, selectedTimeframe, setSymbol, setTimeframe, setSelectedInstrument, setSelectedTimeframe, triggerDataRefresh]);

  return (
    <TradingContext.Provider value={state}>
      {children}
    </TradingContext.Provider>
  );
}