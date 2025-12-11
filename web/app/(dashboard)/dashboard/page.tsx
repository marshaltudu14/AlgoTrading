"use client";

import { useRef } from "react";
import TradingChart from "@/components/TradingChart";
import { DataViewer, DataViewerRef } from "@/components/DataViewer";

export default function Dashboard() {
  const dataViewerRef = useRef<DataViewerRef>(null);

  return (
    <div className="w-full h-full relative">
      <DataViewer ref={dataViewerRef} />
      <TradingChart dataViewerRef={dataViewerRef} />
    </div>
  );
}