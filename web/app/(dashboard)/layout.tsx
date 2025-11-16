import { SidebarProvider, Sidebar, SidebarContent, SidebarHeader, SidebarFooter, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarTrigger, SidebarInset } from "@/components/ui/sidebar";
import { ModeToggle } from "@/components/mode-toggle";
import { Home, TrendingUp, Settings } from "lucide-react";
import Link from "next/link";
import LogoutButton from "@/components/LogoutButton";
import TradingProvider from "@/components/TradingProvider";
import InstrumentSelector from "@/components/InstrumentSelector";
import TimeframeSelector from "@/components/TimeframeSelector";

const items = [
  {
    title: "Dashboard",
    url: "/dashboard",
    icon: Home,
  },
  {
    title: "Trading",
    url: "/dashboard/trading",
    icon: TrendingUp,
  },
  {
    title: "Settings",
    url: "/dashboard/settings",
    icon: Settings,
  },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <TradingProvider>
      <SidebarProvider>
        <Sidebar variant="inset" collapsible="icon">
          <SidebarHeader>
            <div className="flex items-center p-4">
              <h1 className="text-xl font-bold text-foreground">AlgoTrading</h1>
            </div>
          </SidebarHeader>
          <SidebarContent>
            <SidebarMenu>
              {items.map((item) => (
                <SidebarMenuItem key={item.title}>
                  <SidebarMenuButton asChild>
                    <Link href={item.url}>
                      <item.icon className="h-4 w-4" />
                      <span>{item.title}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarContent>
          <SidebarFooter>
            <div className="p-4">
              <LogoutButton />
            </div>
          </SidebarFooter>
        </Sidebar>
        <SidebarInset>
          <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
            <SidebarTrigger className="md:hidden" />
            <div className="flex items-center gap-2">
              <InstrumentSelector />
              <TimeframeSelector />
            </div>
            <div className="flex-1" />
            <ModeToggle />
          </header>
          <main className="flex-1 overflow-hidden">
            {children}
          </main>
        </SidebarInset>
      </SidebarProvider>
    </TradingProvider>
  );
}