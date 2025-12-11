"use client";

import { useRef } from "react";
import TradingChart from "@/components/TradingChart";
import { DataViewer, DataViewerRef } from "@/components/DataViewer";
import { BacktestDataViewer, BacktestDataViewerRef } from "@/components/BacktestDataViewer";

export default function Dashboard() {
  const dataViewerRef = useRef<DataViewerRef>(null);
  const backtestDataViewerRef = useRef<BacktestDataViewerRef>(null);

  return (
    <div className="w-full h-full relative">
      <DataViewer ref={dataViewerRef} />
      <BacktestDataViewer ref={backtestDataViewerRef} />
      <TradingChart dataViewerRef={dataViewerRef} backtestDataViewerRef={backtestDataViewerRef} />
    </div>
  );
}