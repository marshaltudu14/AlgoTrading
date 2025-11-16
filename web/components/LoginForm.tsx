"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import ThemeToggle from "@/components/ThemeToggle";
import { Eye, EyeOff } from "lucide-react";

export default function LoginForm() {
  const [formData, setFormData] = useState({
    appId: "TS79V3NXK1-100",
    secretKey: "KQCPB0FJ74",
    redirectUrl: "https://google.com",
    fyersId: "XM22383",
    pin: "4628"
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
        // Store token in localStorage or secure storage
        if (data.access_token) {
          localStorage.setItem('fyers_access_token', data.access_token);
          localStorage.setItem('user_profile', JSON.stringify(data.profile));
        }
        // Redirect to dashboard
        setTimeout(() => {
          window.location.href = '/dashboard';
        }, 1000);
      } else {
        toast.error(data.error || 'Authentication failed');
      }
    } catch (err) {
      toast.error('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white dark:bg-black flex items-center justify-center p-4 relative">
      {/* Theme toggle in top right corner */}
      <div className="absolute top-4 right-4">
        <ThemeToggle />
      </div>

      <div className="w-full max-w-md">

        <Card className="shadow-lg dark:shadow-white/5 bg-white dark:bg-black border-gray-200 dark:border-gray-800">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl text-center dark:text-white">Sign in</CardTitle>
            <CardDescription className="text-center dark:text-gray-400">
              Enter your Fyers credentials to access your account
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="appId" className="dark:text-white">App ID</Label>
                <Input
                  id="appId"
                  name="appId"
                  type="text"
                  value={formData.appId}
                  onChange={handleChange}
                  placeholder="Enter your Fyers App ID"
                  required
                  className="bg-background"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="secretKey" className="dark:text-white">Secret Key</Label>
                <div className="relative">
                  <Input
                    id="secretKey"
                    name="secretKey"
                    type={showSecretKey ? "text" : "password"}
                    value={formData.secretKey}
                    onChange={handleChange}
                    placeholder="Enter your Secret Key"
                    required
                    className="bg-background pr-10"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowSecretKey(!showSecretKey)}
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent dark:text-white dark:hover:bg-transparent"
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
                <Label htmlFor="redirectUrl" className="dark:text-white">Redirect URL</Label>
                <Input
                  id="redirectUrl"
                  name="redirectUrl"
                  type="url"
                  value={formData.redirectUrl}
                  onChange={handleChange}
                  placeholder="Enter redirect URL"
                  required
                  className="bg-background"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="fyersId" className="dark:text-white">Fyers ID</Label>
                <Input
                  id="fyersId"
                  name="fyersId"
                  type="text"
                  value={formData.fyersId}
                  onChange={handleChange}
                  placeholder="Enter your Fyers ID"
                  required
                  className="bg-background"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="pin" className="dark:text-white">PIN</Label>
                <div className="relative">
                  <Input
                    id="pin"
                    name="pin"
                    type={showPin ? "text" : "password"}
                    value={formData.pin}
                    onChange={handleChange}
                    placeholder="Enter your PIN"
                    required
                    className="bg-background pr-10"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowPin(!showPin)}
                    className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent dark:text-white dark:hover:bg-transparent"
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

              <Button
                type="submit"
                className="w-full bg-black text-white hover:bg-gray-800 dark:bg-white dark:text-black dark:hover:bg-gray-200"
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

            <div className="mt-6 text-center">
              <p className="text-xs text-muted-foreground dark:text-gray-400">
                Secure authentication powered by Fyers API
              </p>
            </div>
          </CardContent>
        </Card>

        </div>
    </div>
  );
}