"use client";

import { useState } from "react";
import { useTradingContext } from "./TradingProvider";
import { useBacktestStore } from "@/stores/backtestStore";
import { TIMEFRAMES } from "@/config/instruments";
import { Button } from "@/components/ui/button";
import { Clock } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function TimeframeSelector() {
  const { timeframe, setTimeframe } = useTradingContext();
  const { isBacktestMode, setTimeframe: setBacktestTimeframe } = useBacktestStore();
  const [isOpen, setIsOpen] = useState(false);

  // Show backtest timeframe when in backtest mode
  const displayTimeframe = isBacktestMode ? useBacktestStore.getState().config.timeframe : timeframe;

  const handleTimeframeSelect = (timeframeName: string) => {
    setTimeframe(timeframeName);
    // Also update backtest timeframe if in backtest mode
    if (isBacktestMode) {
      setBacktestTimeframe(timeframeName);
    }
    setIsOpen(false);
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0 cursor-pointer">
          <Clock className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-40">
        {TIMEFRAMES.map((tf) => (
          <DropdownMenuItem
            key={tf.id}
            onClick={() => handleTimeframeSelect(tf.name)}
            className={displayTimeframe === tf.name ? "bg-accent" : ""}
          >
            <div className="font-medium">{tf.name}M</div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}