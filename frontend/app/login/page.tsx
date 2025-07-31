"use client"

import * as React from "react"
import { motion } from "framer-motion"
import { useRouter } from "next/navigation"
import { Activity, Eye, EyeOff, Loader2 } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ThemeToggle } from "@/components/theme-toggle"
import { apiClient, formatApiError } from "@/lib/api"

export default function LoginPage() {
  const router = useRouter()
  const [isLoading, setIsLoading] = React.useState(false)
  const [showPin, setShowPin] = React.useState(false)
  const [error, setError] = React.useState("")
  const [currentStep, setCurrentStep] = React.useState(1) // 1 for Fyers ID, PIN, TOTP; 2 for API Key, Secret Key, Redirect URI
  const [formData, setFormData] = React.useState({
    app_id: "TS79V3NXK1-100",
    secret_key: "KQCPB0FJ74",
    redirect_uri: "https://google.com",
    fy_id: "XM22383",
    pin: "4628",
    totp_secret: "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"
  })

  const handleNext = (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setCurrentStep(2)
  }

  const handleBack = (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setCurrentStep(1)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    console.log("Login form submitted with data:", formData)

    try {
      // Use the API client to login
      const response = await apiClient.login(formData)
      console.log("Login response:", response)

      if (response.success) {
        // Login successful, redirect to dashboard
        router.push("/dashboard")
        setIsLoading(false)
      } else {
        setError(response.message || "Login failed. Please check your credentials.")
        setIsLoading(false)
      }
    } catch (err) {
      setError(formatApiError(err))
      setIsLoading(false)
    }
  }

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    if (error) setError("")
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background to-muted p-4">
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md"
      >
        <Card className="shadow-lg">
          <CardHeader className="text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
              className="flex justify-center mb-4"
            >
              <div className="p-3 bg-primary/10 rounded-full">
                <Activity className="h-8 w-8 text-primary" />
              </div>
            </motion.div>
            <CardTitle className="text-2xl font-bold">Welcome to AlgoTrading</CardTitle>
            <CardDescription>
              Sign in with your Fyers credentials to access your trading dashboard
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={currentStep === 1 ? handleNext : handleSubmit} className="space-y-4">
              {currentStep === 1 && (
                <>
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="fy_id">Fyers ID</Label>
                    <Input
                      id="fy_id"
                      type="text"
                      placeholder="Enter your Fyers ID"
                      value={formData.fy_id}
                      onChange={(e) => handleInputChange("fy_id", e.target.value)}
                      required
                    />
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.25 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="pin">PIN</Label>
                    <div className="relative">
                      <Input
                        id="pin"
                        type={showPin ? "text" : "password"}
                        placeholder="Enter your PIN"
                        value={formData.pin}
                        onChange={(e) => handleInputChange("pin", e.target.value)}
                        required
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent cursor-pointer"
                        onClick={() => setShowPin(!showPin)}
                      >
                        {showPin ? (
                          <EyeOff className="h-4 w-4" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="totp_secret">TOTP Secret</Label>
                    <Input
                      id="totp_secret"
                      type="password"
                      placeholder="Enter your TOTP secret"
                      value={formData.totp_secret}
                      onChange={(e) => handleInputChange("totp_secret", e.target.value)}
                      required
                    />
                  </motion.div>
                </>
              )}

              {currentStep === 2 && (
                <>
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="app_id">API Key</Label>
                    <Input
                      id="app_id"
                      type="text"
                      placeholder="Enter your API Key"
                      value={formData.app_id}
                      onChange={(e) => handleInputChange("app_id", e.target.value)}
                      required
                    />
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.25 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="secret_key">Secret Key</Label>
                    <Input
                      id="secret_key"
                      type="password"
                      placeholder="Enter your Secret Key"
                      value={formData.secret_key}
                      onChange={(e) => handleInputChange("secret_key", e.target.value)}
                      required
                    />
                  </motion.div>

                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 }}
                    className="space-y-2"
                  >
                    <Label htmlFor="redirect_uri">Redirect URI</Label>
                    <Input
                      id="redirect_uri"
                      type="url"
                      placeholder="Enter your Redirect URI"
                      value={formData.redirect_uri}
                      onChange={(e) => handleInputChange("redirect_uri", e.target.value)}
                      required
                    />
                  </motion.div>
                </>
              )}

              {error && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.2 }}
                >
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                </motion.div>
              )}

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
                className="flex gap-1 justify-between"
              >
                {currentStep === 2 && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={handleBack}
                    disabled={isLoading}
                    className="flex-1 w-1/2 cursor-pointer"
                  >
                    Back
                  </Button>
                )}
                <Button
                  type="submit"
                  disabled={isLoading}
                  className="flex-1 w-1/2 cursor-pointer"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {currentStep === 1 ? "Next" : "Signing in..."}
                    </>
                  ) : (
                    currentStep === 1 ? "Next" : "Sign In"
                  )}
                </Button>
              </motion.div>
            </form>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  )
}
