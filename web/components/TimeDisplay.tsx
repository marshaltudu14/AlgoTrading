"use client";

import { useEffect, useCallback } from "react";
import { useTradingState } from "@/components/TradingProvider";
import { useBacktestStore } from "@/stores/backtestStore";

interface TimeDisplayProps {
  className?: string;
}

export default function TimeDisplay({ className = "" }: TimeDisplayProps) {
  const { currentTime, countdown, triggerDataRefresh } = useTradingState();
  const { isBacktestMode } = useBacktestStore();

  // Check if market is closed based on IST (Indian Standard Time)
  // Market is open from 9:15 AM to 3:30 PM IST
  const isMarketClosed = useCallback((): boolean => {
    if (!currentTime) return false;

    // Convert to IST (UTC+5:30)
    const utcTime = currentTime.getTime() + (currentTime.getTimezoneOffset() * 60000);
    const istTime = new Date(utcTime + (5.5 * 3600000));

    const hours = istTime.getHours();
    const minutes = istTime.getMinutes();

    // Before 9:15 AM or after 3:30 PM (15:30)
    const isBeforeMarketOpen = hours < 9 || (hours === 9 && minutes < 15);
    const isAfterMarketClose = hours > 15 || (hours === 15 && minutes > 30);

    return isBeforeMarketOpen || isAfterMarketClose;
  }, [currentTime]);

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

  // Trigger data refresh when countdown reaches 0 and not in backtest mode
  useEffect(() => {
    if (countdown === 0 && !isBacktestMode && !isMarketClosed()) {
      triggerDataRefresh();
    }
  }, [countdown, isBacktestMode, currentTime, isMarketClosed, triggerDataRefresh]);

  return (
    <div className={`flex flex-col items-end ${className}`}>
      <div className="text-sm font-mono text-foreground">
        {formatTime(currentTime)}
      </div>
      {isBacktestMode ? (
        <div className="text-xs text-blue-500 font-mono">
          Backtesting
        </div>
      ) : isMarketClosed() ? (
        <div className="text-xs text-red-500 font-mono">
          Market Closed
        </div>
      ) : (
        countdown >= 0 && (
          <div className="text-xs text-muted-foreground font-mono">
            {formatCountdown(countdown)}s left
          </div>
        )
      )}
    </div>
  );
}