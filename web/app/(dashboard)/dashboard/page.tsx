"use client";

import TradingChart from "@/components/TradingChart";
import { useTradingContext } from "@/components/TradingProvider";

export default function Dashboard() {
  const { symbol, timeframe } = useTradingContext();

  return (
    <div className="w-full h-full">
      <TradingChart symbol={symbol} timeframe={timeframe} />
    </div>
  );
}