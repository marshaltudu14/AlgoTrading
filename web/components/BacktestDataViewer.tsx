"use client";

import { useState, forwardRef, useImperativeHandle, useMemo } from "react";
import { useBacktestStore } from "@/stores/backtestStore";
import { Trade } from "@/lib/backtestEngine";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown } from "lucide-react";

const ROWS_PER_PAGE = 100;

export interface BacktestDataViewerRef {
  open: () => void;
}

export const BacktestDataViewer = forwardRef<BacktestDataViewerRef>((props, ref) => {
  const [open, setOpen] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const { backtestResults } = useBacktestStore();

  // Expose open function to parent
  useImperativeHandle(
    ref,
    () => ({
      open: () => setOpen(true),
    }),
    []
  );

  // Get trades from backtest results
  const trades = useMemo(() => {
    if (!backtestResults?.trades) return [];
    return backtestResults.trades as Trade[];
  }, [backtestResults]);

  const tradeCount = trades.length;

  // Calculate pagination
  const totalPages = Math.ceil(tradeCount / ROWS_PER_PAGE);

  // Sort trades by entry time (oldest first)
  const sortedTrades = useMemo(() => {
    if (trades.length === 0) return [];

    // Sort by entry time ascending (oldest first)
    return [...trades].sort((a, b) =>
      new Date(a.entryTime).getTime() - new Date(b.entryTime).getTime()
    );
  }, [trades]);

  // Get trades for current page
  const paginatedTrades = useMemo(() => {
    if (tradeCount === 0) return [];

    // Calculate start and end indices for current page
    const startIndex = (currentPage - 1) * ROWS_PER_PAGE;
    const endIndex = Math.min(startIndex + ROWS_PER_PAGE, tradeCount);

    return sortedTrades.slice(startIndex, endIndex);
  }, [sortedTrades, currentPage, tradeCount]);

  // Format value for display
  const formatValue = (value: unknown, key: string): string => {
    // Format Date as readable datetime
    if ((key === 'entryTime' || key === 'exitTime') && value instanceof Date) {
      return value.toLocaleString();
    }

    if (typeof value === 'number') {
      if (isNaN(value)) return "NaN";
      if (value === null || value === undefined) return "-";

      // Format currency values
      if (key.includes('pnlCurrency') || key.includes('capital') || key.includes('Price')) {
        return `₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      }

      // Format percentage
      if (key === 'confidence') {
        return `${value.toFixed(1)}%`;
      }

      // Default number formatting
      return value.toFixed(2);
    }

    return String(value || "-");
  };

  // Get exit reason badge variant
  const getExitReasonVariant = (reason: string) => {
    switch (reason) {
      case 'TARGET':
      case 'TRAILING_STOP_1':
      case 'TRAILING_STOP_2':
      case 'TRAILING_STOP_3':
        return 'default';
      case 'STOP_LOSS':
        return 'destructive';
      case 'EXPIRY':
        return 'secondary';
      default:
        return 'outline';
    }
  };

  if (!backtestResults || !backtestResults.trades || tradeCount === 0) {
    return (
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent side="bottom" className="h-[90vh] pt-8">
          <SheetHeader>
            <SheetTitle>No Backtest Data Available</SheetTitle>
          </SheetHeader>
          <div className="flex items-center justify-center h-[calc(90vh-180px)]">
            <p className="text-muted-foreground">Run a backtest to see trade details here.</p>
          </div>
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="bottom" className="h-[90vh] pt-8">
        <SheetHeader>
          <div className="flex items-center justify-between">
            <SheetTitle className="flex items-center gap-4">
              <span>{tradeCount.toLocaleString()} Total Trades</span>
              <span className="text-muted-foreground text-sm font-normal">
                Showing {Math.min(ROWS_PER_PAGE, tradeCount)} of {tradeCount.toLocaleString()} trades
              </span>
            </SheetTitle>
          </div>
        </SheetHeader>

        <div className="overflow-auto h-[calc(90vh-180px)]">
          <Table>
            <TableHeader className="sticky top-0 bg-background border-b z-10">
              <TableRow>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[120px]">
                  Entry Time
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[120px]">
                  Exit Time
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[80px]">
                  Position
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]">
                  Entry Price
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]">
                  Exit Price
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]">
                  Lot Size
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]">
                  P&L (₹)
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[80px]">
                  Bars
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]">
                  Exit Reason
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[80px]">
                  Conf.
                </TableHead>
                <TableHead className="whitespace-nowrap text-xs font-medium p-2 min-w-[120px]">
                  Capital
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedTrades.map((trade, index) => (
                <TableRow
                  key={index}
                  className={
                    index === 0 ? "bg-muted/30" : ""
                  }
                >
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono">
                    {formatValue(trade.entryTime, 'entryTime')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono">
                    {formatValue(trade.exitTime, 'exitTime')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2">
                    <Badge
                      variant={trade.position === 'BUY' ? 'default' : 'destructive'}
                      className="flex items-center gap-1 w-12 justify-center"
                    >
                      {trade.position === 'BUY' ? (
                        <TrendingUp className="h-3 w-3" />
                      ) : (
                        <TrendingDown className="h-3 w-3" />
                      )}
                      {trade.position}
                    </Badge>
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono">
                    {formatValue(trade.entryPrice, 'entryPrice')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono">
                    {formatValue(trade.exitPrice, 'exitPrice')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono text-center">
                    {trade.lotSize}
                  </TableCell>
                  <TableCell className={`whitespace-nowrap text-xs p-2 font-mono font-medium ${
                    trade.pnlCurrency >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {formatValue(trade.pnlCurrency, 'pnlCurrency')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono text-center">
                    {trade.barsHeld}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2">
                    <Badge
                      variant={getExitReasonVariant(trade.exitReason)}
                      className="text-xs"
                    >
                      {trade.exitReason.replace('_', ' ')}
                    </Badge>
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono text-center">
                    {formatValue(trade.confidence, 'confidence')}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-xs p-2 font-mono">
                    {formatValue(trade.capital, 'capital')}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex justify-center items-center gap-2 pt-4 border-t">
            <Pagination>
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    className={
                      currentPage === 1
                        ? "pointer-events-none opacity-50"
                        : "cursor-pointer"
                    }
                  />
                </PaginationItem>

                <PaginationItem>
                  <PaginationLink>
                    Page {currentPage} of {totalPages}
                  </PaginationLink>
                </PaginationItem>

                <PaginationItem>
                  <PaginationNext
                    onClick={() =>
                      setCurrentPage((p) => Math.min(totalPages, p + 1))
                    }
                    className={
                      currentPage === totalPages
                        ? "pointer-events-none opacity-50"
                        : "cursor-pointer"
                    }
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
});

BacktestDataViewer.displayName = "BacktestDataViewer";