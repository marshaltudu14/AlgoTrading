"use client"

import * as React from "react"
import { Target, TrendingUp, Minus, TrendingDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

interface StrategySelectorDialogProps {
  selectedStrategy: string
  onStrategyChange: (strategy: string) => void
  isVisible?: boolean
}

const optionStrategies = [
  {
    value: "ITM",
    label: "In The Money",
    shortLabel: "ITM",
    description: "Options with intrinsic value - strike price favorable to current market price",
    icon: TrendingUp,
    color: "text-green-600",
    bgColor: "bg-green-50 hover:bg-green-100",
    borderColor: "border-green-200"
  },
  {
    value: "ATM",
    label: "At The Money",
    shortLabel: "ATM", 
    description: "Options where strike price equals current market price",
    icon: Minus,
    color: "text-blue-600",
    bgColor: "bg-blue-50 hover:bg-blue-100",
    borderColor: "border-blue-200"
  },
  {
    value: "OTM",
    label: "Out of The Money",
    shortLabel: "OTM",
    description: "Options with no intrinsic value - strike price unfavorable to current market price",
    icon: TrendingDown,
    color: "text-orange-600", 
    bgColor: "bg-orange-50 hover:bg-orange-100",
    borderColor: "border-orange-200"
  }
]

export function StrategySelectorDialog({
  selectedStrategy,
  onStrategyChange,
  isVisible = true
}: StrategySelectorDialogProps) {
  const [open, setOpen] = React.useState(false)

  if (!isVisible) return null

  const selectedStrategyInfo = optionStrategies.find(s => s.value === selectedStrategy)

  const handleStrategySelect = (strategy: string) => {
    onStrategyChange(strategy)
    setOpen(false)
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <Dialog open={open} onOpenChange={setOpen}>
          <TooltipTrigger asChild>
            <DialogTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 relative hover:bg-accent/50"
                aria-label="Select option strategy"
              >
                <Target className="h-4 w-4" />
                {selectedStrategy && (
                  <Badge 
                    variant="secondary" 
                    className="absolute -top-1 -right-1 h-3 w-3 p-0 text-[10px] flex items-center justify-center"
                  >
                    •
                  </Badge>
                )}
              </Button>
            </DialogTrigger>
          </TooltipTrigger>
          
          <TooltipContent side="bottom">
            <p>Strategy: {selectedStrategyInfo?.shortLabel || "None"}</p>
          </TooltipContent>

          <DialogContent className="sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>Select Option Strategy</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="grid gap-3">
                {optionStrategies.map((strategy) => {
                  const Icon = strategy.icon
                  const isSelected = selectedStrategy === strategy.value
                  
                  return (
                    <Card
                      key={strategy.value}
                      className={`cursor-pointer transition-all hover:shadow-md ${
                        isSelected 
                          ? `ring-2 ring-primary ${strategy.bgColor} ${strategy.borderColor}` 
                          : `hover:bg-accent/50 border-border`
                      }`}
                      onClick={() => handleStrategySelect(strategy.value)}
                    >
                      <CardHeader className="pb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-full ${strategy.bgColor}`}>
                            <Icon className={`h-5 w-5 ${strategy.color}`} />
                          </div>
                          <div className="flex-1">
                            <CardTitle className="text-base flex items-center gap-2">
                              {strategy.label}
                              <Badge variant="outline" className="text-xs">
                                {strategy.shortLabel}
                              </Badge>
                            </CardTitle>
                          </div>
                          {isSelected && (
                            <Badge variant="default" className="ml-auto">
                              Selected
                            </Badge>
                          )}
                        </div>
                      </CardHeader>
                      <CardContent className="pt-0">
                        <CardDescription className="text-sm">
                          {strategy.description}
                        </CardDescription>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>

              {selectedStrategyInfo && (
                <div className={`flex items-center gap-3 p-3 rounded-lg ${selectedStrategyInfo.bgColor} ${selectedStrategyInfo.borderColor} border`}>
                  <div className={`p-2 rounded-full bg-background`}>
                    <selectedStrategyInfo.icon className={`h-4 w-4 ${selectedStrategyInfo.color}`} />
                  </div>
                  <div className="flex-1">
                    <div className="font-medium text-sm">
                      {selectedStrategyInfo.label} Strategy
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Active for index options trading
                    </div>
                  </div>
                  <Badge variant="secondary">
                    {selectedStrategyInfo.shortLabel}
                  </Badge>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      </Tooltip>
    </TooltipProvider>
  )
}