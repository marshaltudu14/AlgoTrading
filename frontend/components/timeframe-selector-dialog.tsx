"use client"

import * as React from "react"
import { Clock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

interface TimeframeSelectorDialogProps {
  timeframes: string[]
  selectedTimeframe: string
  onTimeframeChange: (timeframe: string) => void
  isLoading?: boolean
}

// Helper function to format timeframe labels
const getTimeframeLabel = (timeframe: string): string => {
  if (timeframe === 'D') return 'Daily'
  const minutes = parseInt(timeframe)
  if (minutes >= 60) {
    const hours = minutes / 60
    return `${hours}H`
  }
  return `${minutes}M`
}

const getTimeframeFullLabel = (timeframe: string): string => {
  if (timeframe === 'D') return 'Daily'
  const minutes = parseInt(timeframe)
  if (minutes >= 60) {
    const hours = minutes / 60
    return `${hours} Hour${hours > 1 ? 's' : ''}`
  }
  return `${minutes} Minute${minutes > 1 ? 's' : ''}`
}

export function TimeframeSelectorDialog({
  timeframes,
  selectedTimeframe,
  onTimeframeChange,
  isLoading = false
}: TimeframeSelectorDialogProps) {
  const [open, setOpen] = React.useState(false)

  const handleTimeframeSelect = (timeframe: string) => {
    onTimeframeChange(timeframe)
    setOpen(false)
  }

  // Sort timeframes by duration for better UX
  const sortedTimeframes = React.useMemo(() => {
    return [...timeframes].sort((a, b) => {
      if (a === 'D') return 1
      if (b === 'D') return -1
      return parseInt(a) - parseInt(b)
    })
  }, [timeframes])

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
                disabled={isLoading}
                aria-label="Select timeframe"
              >
                <Clock className="h-4 w-4" />
                {selectedTimeframe && (
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
            <p>Timeframe: {selectedTimeframe ? getTimeframeFullLabel(selectedTimeframe) : "None"}</p>
          </TooltipContent>

          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Select Timeframe</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-2">
                {sortedTimeframes.map((timeframe) => {
                  const isSelected = selectedTimeframe === timeframe
                  return (
                    <Button
                      key={timeframe}
                      variant={isSelected ? "default" : "outline"}
                      className="h-16 flex flex-col items-center justify-center gap-1 hover:bg-accent/50"
                      onClick={() => handleTimeframeSelect(timeframe)}
                    >
                      <div className="font-bold text-lg">
                        {getTimeframeLabel(timeframe)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {getTimeframeFullLabel(timeframe)}
                      </div>
                    </Button>
                  )
                })}
              </div>

              {selectedTimeframe && (
                <div className="flex items-center gap-2 p-2 bg-muted/50 rounded-lg">
                  <Clock className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1">
                    <div className="font-medium text-sm">
                      {getTimeframeLabel(selectedTimeframe)} Chart
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {getTimeframeFullLabel(selectedTimeframe)} candlesticks
                    </div>
                  </div>
                  <Badge variant="secondary">
                    {getTimeframeLabel(selectedTimeframe)}
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