"use client";

import { useState } from "react";
import { useTradingContext } from "./TradingProvider";
import { INSTRUMENTS } from "@/config/instruments";
import { Button } from "@/components/ui/button";
import { ChevronDown, TrendingUp } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function InstrumentSelector() {
  const { symbol, setSymbol } = useTradingContext();
  const [isOpen, setIsOpen] = useState(false);

  const currentInstrument = INSTRUMENTS.find(inst => inst.exchangeSymbol === symbol);

  const handleInstrumentSelect = (exchangeSymbol: string) => {
    setSymbol(exchangeSymbol);
    setIsOpen(false);
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 gap-1">
          <TrendingUp className="h-4 w-4" />
          <span>{currentInstrument?.name || symbol}</span>
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-48">
        {INSTRUMENTS.map((instrument) => (
          <DropdownMenuItem
            key={instrument.id}
            onClick={() => handleInstrumentSelect(instrument.exchangeSymbol)}
            className={symbol === instrument.exchangeSymbol ? "bg-accent" : ""}
          >
            <div>
              <div className="font-medium">{instrument.name}</div>
              <div className="text-xs text-muted-foreground">{instrument.symbol}</div>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}