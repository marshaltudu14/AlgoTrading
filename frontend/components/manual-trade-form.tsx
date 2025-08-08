"use client"

import * as React from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Instrument } from "@/lib/api"

const formSchema = z.object({
  instrument: z.string().min(1, "Instrument is required"),
  direction: z.enum(["buy", "sell"]),
  quantity: z.number().int().positive("Quantity must be a positive integer"),
  stopLoss: z.number().optional(),
  target: z.number().optional(),
})

interface ManualTradeFormProps {
  instruments: Instrument[];
  isDisabled: boolean;
  defaultInstrument?: string;
  onSubmit: (values: z.infer<typeof formSchema>) => void;
}

export function ManualTradeForm({ instruments, isDisabled, defaultInstrument, onSubmit }: ManualTradeFormProps) {
  const [isConfirmOpen, setIsConfirmOpen] = React.useState(false)
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      instrument: defaultInstrument || "",
      direction: "buy",
      quantity: 1,
    },
  })

  // Update form when defaultInstrument changes
  React.useEffect(() => {
    if (defaultInstrument && defaultInstrument !== form.getValues("instrument")) {
      form.setValue("instrument", defaultInstrument)
    }
  }, [defaultInstrument, form])

  return (
    <Card className="border-0 shadow-lg bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <CardHeader className="space-y-1 pb-4">
        <CardTitle className="text-xl font-semibold flex items-center gap-2">
          <div className="p-1.5 rounded-full bg-primary/10">
            <svg className="h-4 w-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
            </svg>
          </div>
          Manual Trade
        </CardTitle>
        <CardDescription className="text-sm text-muted-foreground">
          Enter a trade manually when no automated position is active.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <TooltipProvider>
          <Tooltip open={isDisabled ? undefined : false}>
            <TooltipTrigger asChild>
              <div className={isDisabled ? "cursor-not-allowed" : ""}>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(() => setIsConfirmOpen(true))} className="space-y-5">
                    <fieldset disabled={isDisabled} className="space-y-5">
                      <FormField
                        control={form.control}
                        name="instrument"
                        render={({ field }) => {
                          const selectedInstrument = instruments.find(i => i.symbol === field.value);
                          return (
                            <FormItem>
                              <FormLabel className="text-sm font-medium">Instrument</FormLabel>
                              <FormControl>
                                <div className="relative w-full">
                                  <Input 
                                    value={selectedInstrument ? `${selectedInstrument.name} (${selectedInstrument.symbol})` : defaultInstrument || ""}
                                    readOnly 
                                    className="h-10 bg-muted/50 cursor-not-allowed w-full"
                                    placeholder="No instrument selected"
                                  />
                                  <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                                    <svg className="h-4 w-4 text-muted-foreground" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 15l-3-3h6l-3 3z" />
                                    </svg>
                                  </div>
                                </div>
                              </FormControl>
                              <p className="text-xs text-muted-foreground mt-1">
                                Instrument is automatically set from the selected chart symbol
                              </p>
                              <FormMessage />
                            </FormItem>
                          );
                        }}
                      />

                      <FormField
                        control={form.control}
                        name="direction"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel className="text-sm font-medium">Direction</FormLabel>
                            <FormControl>
                              <RadioGroup
                                onValueChange={field.onChange}
                                value={field.value}
                                className="grid grid-cols-2 gap-3"
                              >
                                <FormItem className="flex items-center space-x-0 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem value="buy" className="peer sr-only" />
                                  </FormControl>
                                  <FormLabel className="flex flex-1 items-center justify-center rounded-md border-2 border-muted bg-background p-3 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer">
                                    <span className="font-medium text-green-600 dark:text-green-400">Buy</span>
                                  </FormLabel>
                                </FormItem>
                                <FormItem className="flex items-center space-x-0 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem value="sell" className="peer sr-only" />
                                  </FormControl>
                                  <FormLabel className="flex flex-1 items-center justify-center rounded-md border-2 border-muted bg-background p-3 hover:bg-accent hover:text-accent-foreground peer-data-[state=checked]:border-primary [&:has([data-state=checked])]:border-primary cursor-pointer">
                                    <span className="font-medium text-red-600 dark:text-red-400">Sell</span>
                                  </FormLabel>
                                </FormItem>
                              </RadioGroup>
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="quantity"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel className="text-sm font-medium">Quantity</FormLabel>
                            <FormControl>
                              <Input 
                                type="number" 
                                min="1"
                                step="1"
                                className="h-10 bg-background"
                                placeholder="Enter quantity"
                                {...field} 
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                        <FormField
                          control={form.control}
                          name="stopLoss"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-sm font-medium">Stop Loss</FormLabel>
                              <FormControl>
                                <Input 
                                  type="number" 
                                  step="0.01"
                                  className="h-10 bg-background"
                                  placeholder="Optional"
                                  {...field} 
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                        <FormField
                          control={form.control}
                          name="target"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel className="text-sm font-medium">Target Price</FormLabel>
                              <FormControl>
                                <Input 
                                  type="number" 
                                  step="0.01"
                                  className="h-10 bg-background"
                                  placeholder="Optional"
                                  {...field} 
                                />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>

                      <Button 
                        type="submit" 
                        className="w-full h-11 text-base font-medium bg-primary hover:bg-primary/90 shadow-md"
                        size="lg"
                      >
                        <svg className="mr-2 h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Submit Trade
                      </Button>
                    </fieldset>
                  </form>
                </Form>

                <Dialog open={isConfirmOpen} onOpenChange={setIsConfirmOpen}>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Confirm Manual Trade</DialogTitle>
                      <DialogDescription>
                        Please review the details of your trade before confirming.
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4">
                      <div>
                        <p className="font-medium">Instrument</p>
                        <p className="text-muted-foreground">{form.getValues("instrument")}</p>
                      </div>
                      <div>
                        <p className="font-medium">Direction</p>
                        <p className="text-muted-foreground">{form.getValues("direction").toUpperCase()}</p>
                      </div>
                      <div>
                        <p className="font-medium">Quantity</p>
                        <p className="text-muted-foreground">{form.getValues("quantity")}</p>
                      </div>
                      {form.getValues("stopLoss") && (
                        <div>
                          <p className="font-medium">Stop-Loss</p>
                          <p className="text-muted-foreground">{form.getValues("stopLoss")}</p>
                        </div>
                      )}
                      {form.getValues("target") && (
                        <div>
                          <p className="font-medium">Target</p>
                          <p className="text-muted-foreground">{form.getValues("target")}</p>
                        </div>
                      )}
                    </div>
                    <DialogFooter>
                      <Button variant="outline" onClick={() => setIsConfirmOpen(false)}>
                        Cancel
                      </Button>
                      <Button onClick={() => {
                        onSubmit(form.getValues())
                        setIsConfirmOpen(false)
                      }}>
                        Confirm
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </TooltipTrigger>
            {isDisabled && (
              <TooltipContent>
                <p>Manual trading is disabled while an automated trade is active.</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </CardContent>
    </Card>
  )
}
