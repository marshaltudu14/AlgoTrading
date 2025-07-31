"use client"

import * as React from "react"
import { motion, AnimatePresence } from "framer-motion"
import { gsap } from "gsap"
import {
  Activity,
  Menu,
  X,
  ChevronLeft,
  ChevronRight,
  Settings,
  BarChart3,
  TrendingUp,
  Home,
  Play,
  Zap
} from "lucide-react"
import { usePathname } from "next/navigation"
import Link from "next/link"

import { Button } from "@/components/ui/button"
import { ThemeToggle } from "@/components/theme-toggle"
import { cn } from "@/lib/utils"

interface TradingViewLayoutProps {
  children: React.ReactNode
  headerControls?: React.ReactNode
  sidebarContent?: React.ReactNode
  showSidebar?: boolean
}

const navigationItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: Home,
  },
  {
    title: "Backtest",
    href: "/dashboard/backtest",
    icon: Play,
  },
  {
    title: "Live Trading",
    href: "/dashboard/live",
    icon: Zap,
  },
  {
    title: "Analytics",
    href: "/dashboard/analytics",
    icon: BarChart3,
  },
  {
    title: "Performance",
    href: "/dashboard/performance",
    icon: TrendingUp,
  },
]

export function TradingViewLayout({
  children,
  headerControls,
  sidebarContent,
  showSidebar = true
}: TradingViewLayoutProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = React.useState(false)
  const [sidebarHidden, setSidebarHidden] = React.useState(false)
  const [isMobile, setIsMobile] = React.useState(false)
  const pathname = usePathname()
  const sidebarRef = React.useRef<HTMLDivElement>(null)
  const contentRef = React.useRef<HTMLDivElement>(null)

  // Check for mobile screen
  React.useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
      if (window.innerWidth < 768) {
        setSidebarHidden(true)
      }
    }
    
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  // GSAP animations for sidebar
  const toggleSidebar = React.useCallback(() => {
    if (isMobile) {
      setSidebarHidden(!sidebarHidden)
    } else {
      if (sidebarCollapsed) {
        setSidebarCollapsed(false)
      } else {
        setSidebarCollapsed(true)
      }
    }
  }, [isMobile, sidebarCollapsed, sidebarHidden])

  // Use CSS transitions instead of GSAP for now to avoid ref issues
  // The animations will be handled by CSS classes

  const NavItems = ({ collapsed = false }: { collapsed?: boolean }) => (
    <>
      {navigationItems.map((item) => {
        const Icon = item.icon
        const isActive = pathname === item.href
        
        return (
          <Link key={item.href} href={item.href}>
            <motion.div
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-all hover:bg-accent",
                isActive ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground",
                collapsed && "justify-center px-2"
              )}
            >
              <Icon className={cn("h-4 w-4", collapsed ? "h-5 w-5" : "")} />
              {!collapsed && <span className="font-medium">{item.title}</span>}
            </motion.div>
          </Link>
        )
      })}
    </>
  )

  return (
    <div className="h-screen w-screen overflow-hidden bg-background">
      {/* Sidebar */}
      {showSidebar && (
        <motion.aside
          ref={sidebarRef}
          className={cn(
            "fixed left-0 top-0 z-40 h-full border-r bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 transition-all duration-300",
            isMobile ? "w-64" : sidebarCollapsed ? "w-16" : "w-64",
            isMobile && sidebarHidden && "-translate-x-full"
          )}
          initial={false}
        >
          <div className="flex h-full flex-col">
            {/* Sidebar Header */}
            <div className={cn(
              "flex h-12 items-center border-b px-3",
              sidebarCollapsed && !isMobile && "justify-center px-2"
            )}>
              {(!sidebarCollapsed || isMobile) && (
                <div className="flex items-center gap-2">
                  <Activity className="h-6 w-6" />
                  <span className="font-semibold text-sm">AlgoTrading</span>
                </div>
              )}
              {sidebarCollapsed && !isMobile && (
                <Activity className="h-6 w-6" />
              )}
            </div>

            {/* Navigation */}
            <nav className={cn(
              "flex-1 space-y-1 p-2",
              sidebarCollapsed && !isMobile && "px-1"
            )}>
              <NavItems collapsed={sidebarCollapsed && !isMobile} />
            </nav>

            {/* Sidebar Footer */}
            <div className={cn(
              "border-t p-4",
              sidebarCollapsed && !isMobile && "p-2"
            )}>
              {(!sidebarCollapsed || isMobile) && <ThemeToggle />}
              {sidebarCollapsed && !isMobile && (
                <div className="flex justify-center">
                  <ThemeToggle />
                </div>
              )}
            </div>
          </div>
        </motion.aside>
      )}

      {/* Main Content Area */}
      <div
        ref={contentRef}
        className={cn(
          "h-full transition-all duration-300",
          showSidebar && !isMobile && (sidebarCollapsed ? "ml-16" : "ml-64"),
          isMobile && "ml-0"
        )}
      >
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex h-10 items-center justify-between border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-3"
        >
          <div className="flex items-center gap-2">
            {showSidebar && (
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleSidebar}
                className="h-8 w-8 p-0"
              >
                {isMobile ? (
                  sidebarHidden ? <Menu className="h-4 w-4" /> : <X className="h-4 w-4" />
                ) : (
                  sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />
                )}
              </Button>
            )}
            
            {/* Header Controls */}
            <motion.div
              className="flex items-center gap-2 min-w-0 flex-1 overflow-hidden"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              {headerControls}
            </motion.div>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
              <Settings className="h-4 w-4" />
            </Button>
            <ThemeToggle />
          </div>
        </motion.header>

        {/* Chart Content Area */}
        <main className="h-[calc(100vh-2.5rem)] w-full overflow-hidden">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
            className="h-full w-full"
          >
            {children}
          </motion.div>
        </main>
      </div>

      {/* Mobile Overlay */}
      <AnimatePresence>
        {isMobile && !sidebarHidden && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-30 bg-black/50"
            onClick={() => setSidebarHidden(true)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
