import { ModeToggle } from "@/components/mode-toggle";
import LogoutButton from "@/components/LogoutButton";
import TradingProvider from "@/components/TradingProvider";
import InstrumentSelector from "@/components/InstrumentSelector";
import TimeframeSelector from "@/components/TimeframeSelector";
import TimeDisplay from "@/components/TimeDisplay";
import BacktestDialog from "@/components/BacktestDialog";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <TradingProvider>
      <div className="flex flex-col h-screen">
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <div className="flex items-center">
              <div className="h-6 w-px bg-border mx-1" />
              <InstrumentSelector />
              <div className="h-6 w-px bg-border mx-1" />
              <TimeframeSelector />
              <div className="h-6 w-px bg-border mx-1" />
              <BacktestDialog />
              <div className="h-6 w-px bg-border mx-1" />
          </div>
          <div className="flex-1" />
          <TimeDisplay />
          <ModeToggle />
          <LogoutButton />
        </header>
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </TradingProvider>
  );
}