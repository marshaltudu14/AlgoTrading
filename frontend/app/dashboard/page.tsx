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
import { AppLayout } from "@/components/app-layout"
import ConnectionStatus from "@/components/connection-status";
import { toast } from "sonner"

import { formatIndianCurrency } from "@/lib/formatters"
import { apiClient, formatApiError, Metrics } from "@/lib/api"

interface DashboardUserData {
  name: string;
  capital: number;
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
  const [metrics, setMetrics] = React.useState<Metrics | null>(null)
  const [isMetricsLoading, setIsMetricsLoading] = React.useState(true)
  const [metricsError, setMetricsError] = React.useState<string | null>(null)

  React.useEffect(() => {
    const fetchUserProfile = async () => {
      try {
        setIsLoading(true)
        const response = await fetch('http://localhost:8000/api/profile', {
          method: 'GET',
          credentials: 'include'
        })
        
        if (response.ok) {
          const profileData = await response.json()
          setUserData({
            name: profileData.name,
            capital: profileData.capital
          })
        }
      } catch (err) {
        console.error('Failed to fetch user profile:', err)
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchUserProfile()
  }, [])


  React.useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setIsMetricsLoading(true)
        setMetricsError(null)
        const data = await apiClient.getMetrics()
        setMetrics(data)
      } catch (err) {
        const errorMessage = formatApiError(err)
        console.error('Failed to fetch metrics:', err)
        setMetricsError(errorMessage)
        toast.error('Failed to load metrics', {
          description: errorMessage
        })
      } finally {
        setIsMetricsLoading(false)
      }
    }

    fetchMetrics()
    const interval = setInterval(fetchMetrics, 10000) // Fetch every 10 seconds

    return () => clearInterval(interval)
  }, [])



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
      value: metrics?.todayPnL || 0,
      prefix: (metrics?.todayPnL || 0) >= 0 ? "+₹" : "-₹",
      icon: (metrics?.todayPnL || 0) >= 0 ? TrendingUp : TrendingDown,
      description: "Today's profit/loss",
      color: (metrics?.todayPnL || 0) >= 0 ? "text-green-600" : "text-red-600",
      formatter: formatIndianCurrency
    },
    {
      title: "Total Trades",
      value: metrics?.totalTrades || 0,
      icon: BarChart3,
      description: "Trades executed",
      color: "text-purple-600"
    },
    {
      title: "Win Rate",
      value: metrics?.winRate || 0,
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
            {isMetricsLoading ? (
              Array.from({ length: 4 }).map((_, index) => (
                <Card key={index}>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Loading...</CardTitle>
                    <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-muted-foreground">-</div>
                    <p className="text-xs text-muted-foreground">Loading data...</p>
                  </CardContent>
                </Card>
              ))
            ) : metricsError ? (
              <div className="col-span-4 text-center text-destructive">
                <p>Failed to load metrics: {metricsError}</p>
              </div>
            ) : (
              stats.map((stat, index) => (
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
              ))
            )}
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
                      {metrics?.lastTradeTime || 'No trades yet'}
                    </div>
                  </div>

                  <ConnectionStatus />

                </div>
              </CardContent>
            </Card>
          </motion.div>

        </div>
      )}
    </AppLayout>
  )
}
