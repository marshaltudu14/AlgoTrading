"use client";

import { useState } from "react";
import { useTradingContext } from "./TradingProvider";
import { TIMEFRAMES } from "@/config/instruments";
import { Button } from "@/components/ui/button";
import { ChevronDown, Clock } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function TimeframeSelector() {
  const { timeframe, setTimeframe } = useTradingContext();
  const [isOpen, setIsOpen] = useState(false);

  const currentTimeframe = TIMEFRAMES.find(tf => tf.name === timeframe);

  const handleTimeframeSelect = (timeframeName: string) => {
    setTimeframe(timeframeName);
    setIsOpen(false);
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost" size="sm" className="h-8 gap-1">
          <Clock className="h-4 w-4" />
          <span>{currentTimeframe?.name}M</span>
          <ChevronDown className="h-4 w-4" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-40">
        {TIMEFRAMES.map((tf) => (
          <DropdownMenuItem
            key={tf.id}
            onClick={() => handleTimeframeSelect(tf.name)}
            className={timeframe === tf.name ? "bg-accent" : ""}
          >
            <div>
              <div className="font-medium">{tf.name}M</div>
              <div className="text-xs text-muted-foreground">{tf.description}</div>
            </div>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}