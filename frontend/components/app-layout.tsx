"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { motion } from "framer-motion"
import {
  BarChart3,
  TrendingUp,
  Activity,
  Home,
  Menu
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { ThemeToggle } from "@/components/theme-toggle"
import { useMobile } from "@/hooks/use-mobile"

const navigation = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: Home,
  },
  {
    name: "Backtest",
    href: "/dashboard/backtest",
    icon: BarChart3,
  },
  {
    name: "Live Trade",
    href: "/dashboard/live",
    icon: TrendingUp,
  },
]

interface AppLayoutProps {
  children: React.ReactNode
}

export function AppLayout({ children }: AppLayoutProps) {
  const pathname = usePathname()
  const isMobile = useMobile()
  const [sidebarOpen, setSidebarOpen] = React.useState(false)

  const NavItems = ({ mobile = false }: { mobile?: boolean }) => (
    <>
      {navigation.map((item) => {
        const isActive = pathname === item.href
        return (
          <Link
            key={item.name}
            href={item.href}
            className={`
              flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all hover:bg-accent
              ${isActive ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:text-foreground'}
              ${mobile ? 'justify-center flex-col gap-1' : ''}
            `}
            onClick={() => mobile && setSidebarOpen(false)}
          >
            <item.icon className={mobile ? "h-5 w-5" : "h-4 w-4"} />
            <span className={mobile ? "text-xs" : ""}>{item.name}</span>
          </Link>
        )
      })}
    </>
  )

  if (isMobile) {
    return (
      <div className="flex h-screen flex-col">
        {/* Mobile Header */}
        <header className="flex h-14 items-center justify-between border-b bg-background px-4">
          <div className="flex items-center gap-2">
            <Activity className="h-6 w-6" />
            <span className="font-semibold">AlgoTrading</span>
          </div>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon">
                  <Menu className="h-4 w-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64">
                <div className="flex items-center gap-2 pb-4">
                  <Activity className="h-6 w-6" />
                  <span className="font-semibold">AlgoTrading</span>
                </div>
                <nav className="space-y-2">
                  <NavItems />
                </nav>
              </SheetContent>
            </Sheet>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto p-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {children}
          </motion.div>
        </main>

        {/* Bottom Navigation - Conditionally rendered */}
        {pathname.startsWith("/dashboard") && (
          <nav className="border-t bg-background">
            <div className="flex h-16 items-center justify-around px-4">
              <NavItems mobile />
            </div>
          </nav>
        )}
      </div>
    )
  }

  return (
    <div className="flex h-screen">
      {/* Desktop Sidebar */}
      <aside className="w-64 border-r bg-background">
        <div className="flex h-full flex-col">
          <div className="flex h-14 items-center gap-2 border-b px-4">
            <Activity className="h-6 w-6" />
            <span className="font-semibold">AlgoTrading</span>
          </div>
          <nav className="flex-1 space-y-2 p-4">
            <NavItems />
          </nav>
          <div className="border-t p-4">
            <ThemeToggle />
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex flex-1 flex-col">
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex h-14 items-center justify-end border-b bg-background px-6"
        >
          <ThemeToggle />
        </motion.header>
        <main className="flex-1 overflow-auto p-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  )
}
