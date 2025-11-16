"use client";

import { useTradingState } from "@/components/TradingProvider";

interface TimeDisplayProps {
  className?: string;
}

export default function TimeDisplay({ className = "" }: TimeDisplayProps) {
  const { currentTime, countdown } = useTradingState();

  const formatTime = (date: Date | null) => {
    if (!date) return "12:00:00 AM";
    return date.toLocaleTimeString("en-US", {
      hour12: true,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit"
    });
  };

  const formatCountdown = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={`flex flex-col items-end ${className}`}>
      <div className="text-sm font-mono text-foreground">
        {formatTime(currentTime)}
      </div>
      {countdown > 0 && (
        <div className="text-xs text-muted-foreground font-mono">
          {formatCountdown(countdown)}s left
        </div>
      )}
    </div>
  );
}