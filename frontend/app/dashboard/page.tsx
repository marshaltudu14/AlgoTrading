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
  Target,
  Loader2
} from "lucide-react"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/app-layout"
import { apiClient } from "@/lib/api"

import { formatIndianCurrency } from "@/lib/formatters"

interface DashboardUserData {
  name: string;
  capital: number;
  todayPnL: number;
  totalTrades: number;
  winRate: number;
  lastTradeTime: string;
}

const AnimatedNumber = ({ value, prefix = "", suffix = "", formatter }: { 
  value: number; 
  prefix?: string; 
  suffix?: string; 
  formatter?: (num: number) => string;
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
              const currentValue = Number(this.targets()[0].textContent);
              numberRef.current.textContent = formatter ? formatter(currentValue) : `${prefix}${currentValue.toLocaleString()}${suffix}`;
            }
          }
        }
      )
    }
  }, [value, prefix, suffix, formatter])

  return <span ref={numberRef}>{formatter ? formatter(0) : `${prefix}0${suffix}`}</span>
}

export default function DashboardPage() {
  const [userData, setUserData] = React.useState<DashboardUserData | null>(null)
  const [isLoading, setIsLoading] = React.useState(true)
  const router = useRouter()

  React.useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        const [profile, funds, metrics] = await Promise.all([
          apiClient.getProfile(),
          apiClient.getFunds(),
          apiClient.getMetrics()
        ])

        setUserData({
          name: profile.name,
          capital: funds.totalFunds,
          todayPnL: funds.todayPnL, 
          totalTrades: metrics.totalTrades, 
          winRate: metrics.winRate, 
          lastTradeTime: metrics.lastTradeTime
        })
      } catch (err) {
        console.error('Failed to fetch profile:', err)
        if (err instanceof Error && err.message.includes('401')) {
          router.push('/login')
        }
      } finally {
        setIsLoading(false)
      }
    }
    fetchUserProfile()
  }, [router])

  const stats = [
    {
      title: "Available Capital",
      value: userData?.capital || 0,
      prefix: "₹",
      icon: DollarSign,
      description: "Total trading capital",
      color: "text-blue-600",
      formatter: formatIndianCurrency
    },
    {
      title: "Today's P&L",
      value: userData?.todayPnL || 0,
      prefix: (userData?.todayPnL || 0) >= 0 ? "+₹" : "-₹",
      icon: (userData?.todayPnL || 0) >= 0 ? TrendingUp : TrendingDown,
      description: "Today's profit/loss",
      color: (userData?.todayPnL || 0) >= 0 ? "text-green-600" : "text-red-600",
      formatter: formatIndianCurrency
    },
    {
      title: "Total Trades",
      value: userData?.totalTrades || 0,
      icon: BarChart3,
      description: "Trades executed",
      color: "text-purple-600"
    },
    {
      title: "Win Rate",
      value: userData?.winRate || 0,
      suffix: "%",
      icon: Target,
      description: "Success percentage",
      color: "text-orange-600"
    }
  ]

  return (
    <AppLayout>
      {isLoading ? (
        <div className="flex items-center justify-center min-h-screen">
          <Loader2 className="h-12 w-12 animate-spin text-primary" />
        </div>
      ) : (
        <div className="space-y-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <h1 className="text-3xl font-bold tracking-tight">
              Welcome back, {userData?.name}
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
                      {userData?.lastTradeTime}
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
      )}
    </AppLayout>
  )
}
