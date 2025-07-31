"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { gsap } from "gsap"
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  BarChart3,
  Clock,
  Target
} from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { AppLayout } from "@/components/app-layout"
import { apiClient } from "@/lib/api"

// Mock data - replace with actual API calls
const mockUserData = {
  name: "John Doe",
  capital: 500000,
  todayPnL: 2500,
  totalTrades: 45,
  winRate: 68.9,
  lastTradeTime: "2 hours ago"
}

const AnimatedNumber = ({ value, prefix = "", suffix = "" }: { 
  value: number; 
  prefix?: string; 
  suffix?: string; 
}) => {
  const numberRef = React.useRef<HTMLSpanElement>(null)

  React.useEffect(() => {
    if (numberRef.current) {
      gsap.fromTo(numberRef.current, 
        { textContent: 0 },
        {
          textContent: value,
          duration: 2,
          ease: "power2.out",
          snap: { textContent: 1 },
          onUpdate: function() {
            if (numberRef.current) {
              const currentValue = Math.round(Number(this.targets()[0].textContent))
              numberRef.current.textContent = `${prefix}${currentValue.toLocaleString()}${suffix}`
            }
          }
        }
      )
    }
  }, [value, prefix, suffix])

  return <span ref={numberRef}>{prefix}0{suffix}</span>
}

export default function DashboardPage() {
  const [userData, setUserData] = React.useState(mockUserData)

  React.useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        const profile = await apiClient.getProfile()

        setUserData({
          name: profile.name,
          capital: profile.capital,
          todayPnL: 0, // TODO: Get from trading stats
          totalTrades: 0, // TODO: Get from trading stats
          winRate: 0, // TODO: Get from trading stats
          lastTradeTime: "No trades yet"
        })
      } catch (err) {
        console.error('Failed to fetch profile:', err)
        // Keep using mock data on error
      }
    }

    if (apiClient.isAuthenticated()) {
      fetchUserProfile()
    }
  }, [])

  const stats = [
    {
      title: "Available Capital",
      value: userData.capital,
      prefix: "₹",
      icon: DollarSign,
      description: "Total trading capital",
      color: "text-blue-600"
    },
    {
      title: "Today's P&L",
      value: userData.todayPnL,
      prefix: userData.todayPnL >= 0 ? "+₹" : "-₹",
      icon: userData.todayPnL >= 0 ? TrendingUp : TrendingDown,
      description: "Today's profit/loss",
      color: userData.todayPnL >= 0 ? "text-green-600" : "text-red-600"
    },
    {
      title: "Total Trades",
      value: userData.totalTrades,
      icon: BarChart3,
      description: "Trades executed",
      color: "text-purple-600"
    },
    {
      title: "Win Rate",
      value: userData.winRate,
      suffix: "%",
      icon: Target,
      description: "Success percentage",
      color: "text-orange-600"
    }
  ]

  return (
    <AppLayout>
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome back, {userData.name}
          </h1>
          <p className="text-muted-foreground">
            Here&apos;s an overview of your trading performance
          </p>
        </motion.div>

        {/* Stats Grid */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card>
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    {stat.title}
                  </CardTitle>
                  <stat.icon className={`h-4 w-4 ${stat.color}`} />
                </CardHeader>
                <CardContent>
                  <div className={`text-2xl font-bold ${stat.color}`}>
                    <AnimatedNumber 
                      value={Math.abs(stat.value)} 
                      prefix={stat.prefix} 
                      suffix={stat.suffix} 
                    />
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {stat.description}
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Recent Activity
              </CardTitle>
              <CardDescription>
                Your latest trading activity and system status
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center gap-4 p-4 border rounded-lg">
                  <div className="p-2 bg-green-100 dark:bg-green-900 rounded-full">
                    <TrendingUp className="h-4 w-4 text-green-600" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Last trade executed</p>
                    <p className="text-sm text-muted-foreground">
                      Bank Nifty Call Option - Profit: ₹1,250
                    </p>
                  </div>
                  <div className="flex items-center gap-1 text-sm text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    {userData.lastTradeTime}
                  </div>
                </div>

                <div className="flex items-center gap-4 p-4 border rounded-lg">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-full">
                    <Activity className="h-4 w-4 text-blue-600" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">System Status</p>
                    <p className="text-sm text-muted-foreground">
                      All systems operational - API connected
                    </p>
                  </div>
                  <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
        >
          <Card>
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
              <CardDescription>
                Jump to your most used features
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4 md:grid-cols-2">
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="p-4 border rounded-lg cursor-pointer hover:bg-accent transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <BarChart3 className="h-8 w-8 text-blue-600" />
                    <div>
                      <h3 className="font-medium">Run Backtest</h3>
                      <p className="text-sm text-muted-foreground">
                        Test strategies on historical data
                      </p>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  className="p-4 border rounded-lg cursor-pointer hover:bg-accent transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <TrendingUp className="h-8 w-8 text-green-600" />
                    <div>
                      <h3 className="font-medium">Start Live Trading</h3>
                      <p className="text-sm text-muted-foreground">
                        Begin automated trading session
                      </p>
                    </div>
                  </div>
                </motion.div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </AppLayout>
  )
}
