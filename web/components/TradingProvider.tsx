"use client";

import React, { useState, createContext, useContext } from "react";
import { DEFAULT_INSTRUMENT, DEFAULT_TIMEFRAME } from "@/config/instruments";

// Create context for trading controls
const TradingContext = createContext<{
  symbol: string;
  timeframe: string;
  setSymbol: (symbol: string) => void;
  setTimeframe: (timeframe: string) => void;
}>({
  symbol: DEFAULT_INSTRUMENT.exchangeSymbol,
  timeframe: DEFAULT_TIMEFRAME.name,
  setSymbol: () => {},
  setTimeframe: () => {},
});

export const useTradingContext = () => useContext(TradingContext);

export default function TradingProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [symbol, setSymbol] = useState(DEFAULT_INSTRUMENT.exchangeSymbol);
  const [timeframe, setTimeframe] = useState(DEFAULT_TIMEFRAME.name);

  return (
    <TradingContext.Provider value={{ symbol, timeframe, setSymbol, setTimeframe }}>
      {children}
    </TradingContext.Provider>
  );
}