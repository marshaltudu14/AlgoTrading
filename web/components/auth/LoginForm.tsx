'use client';

import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { useAuth } from '@/stores/authStore';

// Form validation schema
const loginSchema = z.object({
  fy_id: z.string().min(1, 'Fyers ID is required'),
  pin: z.string().min(4, 'PIN must be at least 4 digits'),
  totp_secret: z.string().min(1, 'TOTP secret is required'),
});

type LoginFormData = z.infer<typeof loginSchema>;

export function LoginForm() {
  const [totpCountdown, setTotpCountdown] = useState<number>(30);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  const { login, isLoading, error, clearError } = useAuth();

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
    setValue,
  } = useForm<LoginFormData>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      fy_id: '',
      pin: '',
      totp_secret: '',
    },
  });

  // TOTP countdown timer
  useEffect(() => {
    const interval = setInterval(() => {
      setTotpCountdown((prev) => {
        if (prev <= 1) return 30;
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Clear error when form is submitted
  const onSubmit = async () => {
    try {
      clearError();
      await login();
    } catch (error) {
      // Error is handled by the auth store
      console.error('Login failed:', error);
    }
  };

  // Pre-fill with environment variables for development
  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      setValue('fy_id', 'XM22383');
      setValue('pin', '4628');
      setValue('totp_secret', 'EAQD6K4IUYOEGPJNVE6BMPTUSDCWIOHW');
    }
  }, [setValue]);

  const onSubmit = async () => {
    try {
      await login();
    } catch (error) {
      // Error is handled by the auth store
      console.error('Login failed:', error);
    }
  };

  const getTotpProgressColor = () => {
    if (totpCountdown > 20) return 'bg-green-500';
    if (totpCountdown > 10) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getTotpProgressWidth = () => {
    return `${(totpCountdown / 30) * 100}%`;
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Trading Platform
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Sign in to access your trading dashboard
          </p>
        </div>

        {/* TOTP Countdown */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">
              TOTP Refresh
            </span>
            <span className="text-sm font-medium text-gray-900 dark:text-white">
              {totpCountdown}s
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-1000 ${getTotpProgressColor()}`}
              style={{ width: getTotpProgressWidth() }}
            />
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
            <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
          </div>
        )}

        {/* Login Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {/* Fyers ID */}
          <div>
            <label
              htmlFor="fy_id"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
            >
              Fyers ID
            </label>
            <input
              {...register('fy_id')}
              type="text"
              id="fy_id"
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              placeholder="Enter your Fyers ID"
            />
            {errors.fy_id && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                {errors.fy_id.message}
              </p>
            )}
          </div>

          {/* PIN */}
          <div>
            <label
              htmlFor="pin"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
            >
              PIN
            </label>
            <input
              {...register('pin')}
              type="password"
              id="pin"
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
              placeholder="Enter your PIN"
            />
            {errors.pin && (
              <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                {errors.pin.message}
              </p>
            )}
          </div>

          {/* Advanced Settings Toggle */}
          <div>
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300"
            >
              {showAdvanced ? 'Hide' : 'Show'} Advanced Settings
            </button>
          </div>

          {/* TOTP Secret (Advanced) */}
          {showAdvanced && (
            <div>
              <label
                htmlFor="totp_secret"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                TOTP Secret
              </label>
              <input
                {...register('totp_secret')}
                type="password"
                id="totp_secret"
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                placeholder="Enter your TOTP secret"
              />
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Only needed if credentials are not pre-configured
              </p>
              {errors.totp_secret && (
                <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                  {errors.totp_secret.message}
                </p>
              )}
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isLoading || !isValid}
            className="w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition duration-200 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Authenticating...
              </div>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Authentication is handled securely via Fyers API
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
            TOTP codes refresh every 30 seconds
          </p>
        </div>
      </div>
    </div>
  );
}