"use client";

import { useState, useEffect } from "react";
import { useCandleStore } from "@/stores/candleStore";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { X } from "lucide-react";

type DataType = 'raw' | 'processed';

export function DataViewer() {
  const [open, setOpen] = useState(false);
  const [dataType, setDataType] = useState<DataType>('processed');
  const { candles, processedCandles, getFeatureNames } = useCandleStore();

  // Get current data based on selection
  const currentData = dataType === 'raw' ? candles : processedCandles;
  const rowCount = currentData.length;

  // Get ALL column names including OHLC, datetime, volume, and all features
  const getColumns = () => {
    if (currentData.length === 0) return [];

    // Get all keys from the first row to ensure we show everything
    const allColumns = Object.keys(currentData[0]);

    // If it's processed data, ensure we're showing all features
    if (dataType === 'processed') {
      const featureNames = getFeatureNames();
      // Combine base columns with features and ensure all are unique
      const baseColumns = ['timestamp', 'open', 'high', 'low', 'close'];
      if (currentData[0]?.volume !== undefined) {
        baseColumns.push('volume');
      }
      const allUniqueColumns = [...new Set([...baseColumns, ...featureNames, ...allColumns])];
      return allUniqueColumns.sort();
    }

    return allColumns.sort();
  };

  const columns = getColumns();

  // Format value for display
  const formatValue = (value: any, key: string) => {
    // Format timestamp as readable datetime
    if (key === 'timestamp' && typeof value === 'number') {
      return new Date(value * 1000).toLocaleString();
    }

    if (typeof value === 'number') {
      if (isNaN(value)) return 'NaN';
      if (value === null || value === undefined) return '-';
      // Format to appropriate decimal places
      if (key.includes('pct') || key.includes('rsi') || key.includes('adx')) {
        return value.toFixed(2);
      }
      return value.toFixed(4);
    }

    return value || '-';
  };

  if (!open) {
    return (
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs hover:bg-muted"
        onClick={() => setOpen(true)}
      >
        View
      </Button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-background">
      {/* Header with controls */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-4">
          <span className="font-medium">{rowCount.toLocaleString()} rows</span>
          <span className="text-muted-foreground text-sm">
            {dataType === 'raw' ? 'Raw data' : 'Processed data'}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center space-x-2">
            <Label htmlFor="data-type-switch" className="text-sm">
              Raw
            </Label>
            <Switch
              id="data-type-switch"
              checked={dataType === 'processed'}
              onCheckedChange={(checked) =>
                setDataType(checked ? 'processed' : 'raw')
              }
            />
            <Label htmlFor="data-type-switch" className="text-sm">
              Processed
            </Label>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setOpen(false)}
            className="h-8 w-8 p-0"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Table container */}
      <div className="overflow-auto" style={{ height: 'calc(100vh - 80px)' }}>
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
            {currentData.map((row, index) => (
              <TableRow
                key={index}
                className={index === currentData.length - 1 ? "bg-muted/30" : ""}
              >
                {columns.map((column) => (
                  <TableCell
                    key={column}
                    className="whitespace-nowrap text-xs p-2 font-mono"
                  >
                    {formatValue(row[column as keyof typeof row], column)}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}