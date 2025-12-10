"use client";

import React, { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { ModeToggle } from "@/components/mode-toggle";
import { Eye, EyeOff, Key, Lock } from "lucide-react";
import { useAuthStore } from "@/stores/authStore";

export default function LoginForm() {
  const searchParams = useSearchParams();
  const error = searchParams.get('error');

  const { login, isLoading, error: authError, setError } = useAuthStore();

  const [formData, setFormData] = useState({
    appId: "TS79V3NXK1-100",
    secretKey: "KQCPB0FJ74",
  });

  const [showSecretKey, setShowSecretKey] = useState(false);

  // Show error message if present in URL or from store
  useEffect(() => {
    if (error) {
      toast.error(decodeURIComponent(error || 'Authentication failed'));
    }
    if (authError) {
      toast.error(authError);
    }
  }, [error, authError]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear any existing errors when user types
    setError(null);
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.appId || !formData.secretKey) {
      setError("Please enter both App ID and Secret Key");
      return;
    }

    await login(formData.appId, formData.secretKey);
  };

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4 relative">
      {/* Theme toggle in top right corner */}
      <div className="fixed top-4 right-4 z-50">
        <ModeToggle />
      </div>

      <div className="w-full max-w-md">
        <Card className="border shadow-xl bg-card">
          <CardHeader className="space-y-2 pb-4">
            <CardTitle className="text-2xl font-bold text-center">Login</CardTitle>
            <CardDescription className="text-center">
              Enter your Fyers credentials
            </CardDescription>
          </CardHeader>
          <CardContent className="px-6 pb-6">
            <form onSubmit={handleLogin} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="appId">App ID</Label>
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
                    className="h-10 pl-10"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="secretKey">Secret Key</Label>
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
                    className="h-10 pl-10 pr-12"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setShowSecretKey(!showSecretKey)}
                    className="absolute right-1 top-1 h-8 w-8"
                  >
                    {showSecretKey ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              <Button
                type="submit"
                className="w-full h-10 font-medium cursor-pointer"
                disabled={isLoading}
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2"></div>
                    Connecting...
                  </div>
                ) : (
                  'Login'
                )}
              </Button>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}