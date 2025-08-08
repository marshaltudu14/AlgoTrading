"use client"

import * as React from "react"
import { BarChart3, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import type { Instrument } from "@/lib/api"

interface SymbolSelectorDialogProps {
  instruments: Instrument[]
  selectedSymbol: string
  onSymbolChange: (symbol: string) => void
  isLoading?: boolean
}

export function SymbolSelectorDialog({
  instruments,
  selectedSymbol,
  onSymbolChange,
  isLoading = false
}: SymbolSelectorDialogProps) {
  const [open, setOpen] = React.useState(false)
  const [searchQuery, setSearchQuery] = React.useState("")

  const selectedInstrument = instruments.find(i => i.symbol === selectedSymbol)

  const filteredInstruments = React.useMemo(() => {
    if (!searchQuery) return instruments
    
    return instruments.filter(instrument =>
      instrument.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
      instrument.name?.toLowerCase().includes(searchQuery.toLowerCase())
    )
  }, [instruments, searchQuery])

  const handleSymbolSelect = (symbol: string) => {
    onSymbolChange(symbol)
    setOpen(false)
    setSearchQuery("")
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
                disabled={isLoading}
                aria-label="Select trading symbol"
              >
                <BarChart3 className="h-4 w-4" />
                {selectedSymbol && (
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
            <p>Symbol: {selectedSymbol || "None"}</p>
          </TooltipContent>

          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Select Symbol</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search symbols..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                  autoComplete="off"
                />
              </div>

              <ScrollArea className="h-[300px] rounded border">
                <div className="p-2 space-y-1">
                  {filteredInstruments.length === 0 ? (
                    <div className="text-center py-6 text-muted-foreground">
                      No symbols found
                    </div>
                  ) : (
                    filteredInstruments.map((instrument) => (
                      <Button
                        key={instrument.symbol}
                        variant={selectedSymbol === instrument.symbol ? "secondary" : "ghost"}
                        className="w-full justify-start h-auto p-3"
                        onClick={() => handleSymbolSelect(instrument.symbol)}
                      >
                        <div className="flex flex-col items-start gap-1 w-full">
                          <div className="flex items-center gap-2 w-full">
                            <span className="font-medium">{instrument.symbol}</span>
                            <Badge variant="outline" className="text-xs">
                              {instrument.instrument_type}
                            </Badge>
                          </div>
                          {instrument.name && (
                            <span className="text-sm text-muted-foreground truncate w-full text-left">
                              {instrument.name}
                            </span>
                          )}
                        </div>
                      </Button>
                    ))
                  )}
                </div>
              </ScrollArea>

              {selectedInstrument && (
                <div className="flex items-center gap-2 p-2 bg-muted/50 rounded-lg">
                  <BarChart3 className="h-4 w-4 text-muted-foreground" />
                  <div className="flex-1">
                    <div className="font-medium text-sm">{selectedInstrument.symbol}</div>
                    {selectedInstrument.name && (
                      <div className="text-xs text-muted-foreground">{selectedInstrument.name}</div>
                    )}
                  </div>
                  <Badge variant="secondary">{selectedInstrument.instrument_type}</Badge>
                </div>
              )}
            </div>
          </DialogContent>
        </Dialog>
      </Tooltip>
    </TooltipProvider>
  )
}