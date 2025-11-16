"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { ModeToggle } from "@/components/mode-toggle";
import { Eye, EyeOff, Key, Link, User, Lock, Shield } from "lucide-react";

export default function LoginForm() {
  const [formData, setFormData] = useState({
    appId: "TS79V3NXK1-100",
    secretKey: "KQCPB0FJ74",
    redirectUrl: "https://google.com",
    fyersId: "XM22383",
    pin: "4628",
    totpSecret: "EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW"
  });

  const [isLoading, setIsLoading] = useState(false);
  const [showSecretKey, setShowSecretKey] = useState(false);
  const [showPin, setShowPin] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await fetch('/api/auth/fyers/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (response.ok) {
        toast.success("Authentication successful!");
        // Store token and app_id in localStorage or secure storage
        if (data.access_token) {
          localStorage.setItem('access_token', data.access_token);
          localStorage.setItem('app_id', formData.appId); // Get app_id from form data
          localStorage.setItem('user_profile', JSON.stringify(data.profile));
        }
        // Redirect to dashboard
        setTimeout(() => {
          window.location.href = '/dashboard';
        }, 1000);
      } else {
        toast.error(data.error || 'Authentication failed');
      }
    } catch {
      toast.error('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4 relative">
      {/* Theme toggle in top right corner */}
      <div className="fixed top-4 right-4 z-50">
        <ModeToggle />
      </div>

      <div className="w-full max-w-md">
        <Card className="border shadow-2xl bg-card backdrop-blur-sm sm:border border-0 sm:border-1">
          <CardHeader className="space-y-3 pb-6">
            <CardTitle className="text-3xl font-bold text-center">Welcome Back</CardTitle>
            <CardDescription className="text-center text-base">
              Sign in to your AlgoTrading account with Fyers
            </CardDescription>
          </CardHeader>
          <CardContent className="px-8 pb-8">
            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-2">
                <Label htmlFor="appId" className="text-sm font-medium">App ID</Label>
                <div className="relative">
                  <Key className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="appId"
                    name="appId"
                    type="text"
                    value={formData.appId}
                    onChange={handleChange}
                    placeholder="Enter your Fyers App ID"
                    required
                    className="h-11 pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="secretKey" className="text-sm font-medium">Secret Key</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="secretKey"
                    name="secretKey"
                    type={showSecretKey ? "text" : "password"}
                    value={formData.secretKey}
                    onChange={handleChange}
                    placeholder="Enter your Secret Key"
                    required
                    className="h-11 pl-10 pr-12"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowSecretKey(!showSecretKey)}
                    className="absolute right-1 top-1 h-9 w-9"
                  >
                    {showSecretKey ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                    <span className="sr-only">
                      {showSecretKey ? "Hide secret key" : "Show secret key"}
                    </span>
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="redirectUrl" className="text-sm font-medium">Redirect URL</Label>
                <div className="relative">
                  <Link className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="redirectUrl"
                    name="redirectUrl"
                    type="url"
                    value={formData.redirectUrl}
                    onChange={handleChange}
                    placeholder="Enter redirect URL"
                    required
                    className="h-11 pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="fyersId" className="text-sm font-medium">Fyers ID</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="fyersId"
                    name="fyersId"
                    type="text"
                    value={formData.fyersId}
                    onChange={handleChange}
                    placeholder="Enter your Fyers ID"
                    required
                    className="h-11 pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="pin" className="text-sm font-medium">PIN</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="pin"
                    name="pin"
                    type={showPin ? "text" : "password"}
                    value={formData.pin}
                    onChange={handleChange}
                    placeholder="Enter your PIN"
                    required
                    className="h-11 pl-10 pr-12"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowPin(!showPin)}
                    className="absolute right-1 top-1 h-9 w-9"
                  >
                    {showPin ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                    <span className="sr-only">
                      {showPin ? "Hide PIN" : "Show PIN"}
                    </span>
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="totpSecret" className="text-sm font-medium">TOTP Secret</Label>
                <div className="relative">
                  <Shield className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="totpSecret"
                    name="totpSecret"
                    type="password"
                    value={formData.totpSecret}
                    onChange={handleChange}
                    placeholder="Enter your TOTP Secret"
                    required
                    className="h-11 pl-10"
                  />
                </div>
              </div>

              <Button
                type="submit"
                className="w-full h-11 text-base font-medium shadow-lg"
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2"></div>
                    Authenticating...
                  </div>
                ) : (
                  'Sign In'
                )}
              </Button>
            </form>

            <div className="mt-8 text-center flex items-center justify-center gap-2">
              <Shield className="h-3 w-3 text-muted-foreground" />
              <p className="text-xs text-muted-foreground">
                Secure authentication powered by Fyers API
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}