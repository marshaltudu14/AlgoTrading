"use client";

import { useState } from "react";
import { useTradingContext } from "./TradingProvider";
import { useBacktestStore } from "@/stores/backtestStore";
import { INSTRUMENTS } from "@/config/instruments";
import { Button } from "@/components/ui/button";
import { TrendingUp } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function InstrumentSelector() {
  const { symbol, setSymbol } = useTradingContext();
  const { isBacktestMode, setSymbol: setBacktestSymbol } = useBacktestStore();
  const [isOpen, setIsOpen] = useState(false);

  // Show backtest symbol when in backtest mode
  const displaySymbol = isBacktestMode ? useBacktestStore.getState().config.symbol : symbol;

  const handleInstrumentSelect = (exchangeSymbol: string) => {
    setSymbol(exchangeSymbol);
    // Also update backtest symbol if in backtest mode
    if (isBacktestMode) {
      setBacktestSymbol(exchangeSymbol);
    }
    setIsOpen(false);
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 w-8 p-0 cursor-pointer">
          <TrendingUp className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-48">
        {INSTRUMENTS.map((instrument) => (
          <DropdownMenuItem
            key={instrument.id}
            onClick={() => handleInstrumentSelect(instrument.exchangeSymbol)}
            className={displaySymbol === instrument.exchangeSymbol ? "bg-accent" : ""}
          >
            <div className="font-medium">{instrument.name}</div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}