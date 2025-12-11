"use client";

import { useState, forwardRef, useImperativeHandle, useMemo } from "react";
import { useCandleStore } from "@/stores/candleStore";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
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

type DataType = "raw" | "processed";
const ROWS_PER_PAGE = 100;

export interface DataViewerRef {
  open: () => void;
}

export const DataViewer = forwardRef<DataViewerRef>((props, ref) => {
  const [open, setOpen] = useState(false);
  const [dataType, setDataType] = useState<DataType>("processed");
  const [currentPage, setCurrentPage] = useState(1);
  const { candles, processedCandles, getFeatureNames } = useCandleStore();

  // Expose open function to parent
  useImperativeHandle(
    ref,
    () => ({
      open: () => setOpen(true),
    }),
    []
  );

  // Get current data based on selection
  const currentData = dataType === "raw" ? candles : processedCandles;
  const rowCount = currentData.length;

  // Calculate pagination
  const totalPages = Math.ceil(rowCount / ROWS_PER_PAGE);

  // Get data for current page (show last page first, most recent data)
  const paginatedData = useMemo(() => {
    if (rowCount === 0) return [];

    // Start from the end (most recent data)
    const startIndex = Math.max(0, rowCount - currentPage * ROWS_PER_PAGE);
    const endIndex = rowCount - (currentPage - 1) * ROWS_PER_PAGE;

    return currentData.slice(startIndex, endIndex).reverse();
  }, [currentData, currentPage, rowCount]);

  // Get ALL column names including OHLC, datetime, volume, and all features
  const getColumns = () => {
    if (currentData.length === 0) return [];

    // Get all keys from the first row to ensure we show everything
    const allColumns = Object.keys(currentData[0]);

    // If it's processed data, ensure we're showing all features
    if (dataType === "processed") {
      const featureNames = getFeatureNames();
      // Combine base columns with features and ensure all are unique
      const baseColumns = ["timestamp", "open", "high", "low", "close"];
      if (currentData[0]?.volume !== undefined) {
        baseColumns.push("volume");
      }
      const allUniqueColumns = [
        ...new Set([...baseColumns, ...featureNames, ...allColumns]),
      ];
      return allUniqueColumns;
    }

    return allColumns;
  };

  const columns = getColumns();

  // Format value for display
  const formatValue = (value: unknown, key: string): string => {
    // Format timestamp as readable datetime
    if (key === "timestamp" && typeof value === "number") {
      return new Date(value * 1000).toLocaleString();
    }

    if (typeof value === "number") {
      if (isNaN(value)) return "NaN";
      if (value === null || value === undefined) return "-";
      // Format to appropriate decimal places
      if (key.includes("pct") || key.includes("rsi") || key.includes("adx")) {
        return value.toFixed(2);
      }
      return value.toFixed(4);
    }

    return String(value || "-");
  };

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent side="bottom" className="h-[90vh] pt-8">
        <SheetHeader>
          <div className="flex items-center justify-between">
            <SheetTitle className="flex items-center gap-4">
              <span>{rowCount.toLocaleString()} total rows</span>
              <span className="text-muted-foreground text-sm font-normal">
                Showing {Math.min(ROWS_PER_PAGE, rowCount)} of{" "}
                {rowCount.toLocaleString()} (
                {dataType === "raw" ? "Raw" : "Processed"})
              </span>
            </SheetTitle>
            <div className="flex items-center space-x-2">
              <Label htmlFor="data-type-switch" className="text-sm">
                Raw
              </Label>
              <Switch
                id="data-type-switch"
                checked={dataType === "processed"}
                onCheckedChange={(checked) => {
                  setDataType(checked ? "processed" : "raw");
                  setCurrentPage(1); // Reset to first page when switching data type
                }}
              />
              <Label htmlFor="data-type-switch" className="text-sm">
                Processed
              </Label>
            </div>
          </div>
        </SheetHeader>

        <div className="overflow-auto h-[calc(90vh-180px)]">
          <Table>
            <TableHeader className="sticky top-0 bg-background border-b z-10">
              <TableRow>
                {columns.map((column) => (
                  <TableHead
                    key={column}
                    className="whitespace-nowrap text-xs font-medium p-2 min-w-[100px]"
                  >
                    {column}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {paginatedData.map((row, index) => (
                <TableRow
                  key={index}
                  className={
                    index === 0 ? "bg-muted/30" : "" // Highlight most recent row
                  }
                >
                  {columns.map((column) => (
                    <TableCell
                      key={column}
                      className="whitespace-nowrap text-xs p-2 font-mono"
                    >
                      {formatValue(
                        (row as unknown as Record<string, unknown>)[column],
                        column
                      )}
                    </TableCell>
                  ))}
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

DataViewer.displayName = "DataViewer";
