"use client"

import * as React from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Instrument } from "@/lib/api"

const formSchema = z.object({
  instrument: z.string().min(1, "Instrument is required"),
  direction: z.enum(["buy", "sell"], { required_error: "Direction is required" }),
  quantity: z.coerce.number().int().positive("Quantity must be a positive integer"),
  stopLoss: z.coerce.number().optional(),
  target: z.coerce.number().optional(),
})

interface ManualTradeFormProps {
  instruments: Instrument[];
  isDisabled: boolean;
  onSubmit: (values: z.infer<typeof formSchema>) => void;
}

export function ManualTradeForm({ instruments, isDisabled, onSubmit }: ManualTradeFormProps) {
  const [isConfirmOpen, setIsConfirmOpen] = React.useState(false)
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      instrument: "",
      direction: "buy",
      quantity: 1,
    },
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle>Manual Trade</CardTitle>
        <CardDescription>Manually enter a trade when no automated position is active.</CardDescription>
      </CardHeader>
      <CardContent>
        <TooltipProvider>
          <Tooltip open={isDisabled ? undefined : false}>
            <TooltipTrigger asChild>
              <div className={isDisabled ? "cursor-not-allowed" : ""}>
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(() => setIsConfirmOpen(true))} className="space-y-6">
                    <fieldset disabled={isDisabled} className="space-y-6">
                      <FormField
                        control={form.control}
                        name="instrument"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Instrument</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder="Select an instrument" />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                {instruments.map((instrument) => (
                                  <SelectItem key={instrument.symbol} value={instrument.symbol}>
                                    {instrument.name}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <FormField
                        control={form.control}
                        name="direction"
                        render={({ field }) => (
                          <FormItem className="space-y-3">
                            <FormLabel>Direction</FormLabel>
                            <FormControl>
                              <RadioGroup
                                onValueChange={field.onChange}
                                defaultValue={field.value}
                                className="flex items-center space-x-4"
                              >
                                <FormItem className="flex items-center space-x-2 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem value="buy" />
                                  </FormControl>
                                  <FormLabel className="font-normal">Buy</FormLabel>
                                </FormItem>
                                <FormItem className="flex items-center space-x-2 space-y-0">
                                  <FormControl>
                                    <RadioGroupItem value="sell" />
                                  </FormControl>
                                  <FormLabel className="font-normal">Sell</FormLabel>
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
                            <FormLabel>Quantity</FormLabel>
                            <FormControl>
                              <Input type="number" {...field} />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="grid grid-cols-2 gap-4">
                        <FormField
                          control={form.control}
                          name="stopLoss"
                          render={({ field }) => (
                            <FormItem>
                              <FormLabel>Stop-Loss (Optional)</FormLabel>
                              <FormControl>
                                <Input type="number" {...field} />
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
                              <FormLabel>Target (Optional)</FormLabel>
                              <FormControl>
                                <Input type="number" {...field} />
                              </FormControl>
                              <FormMessage />
                            </FormItem>
                          )}
                        />
                      </div>

                      <Button type="submit" className="w-full">
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
