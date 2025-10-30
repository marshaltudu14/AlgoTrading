'use client';

import React, { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { CandlestickChart } from '@/components/charts/CandlestickChart';
import { useAuth } from '@/stores/authStore';
import { useMarketData } from '@/stores/marketStore';
import { MarketDataService, Timeframe } from '@/lib/marketData';

export default function DashboardPage() {
  const router = useRouter();
  const { isAuthenticated, user, isLoading: authLoading, accessToken, logout } = useAuth();
  const {
    symbol,
    setSymbol,
    fetchHistoricalData,
    refreshData,
    isLoading: marketLoading,
    error: marketError,
    hasData
  } = useMarketData();

  const [selectedSymbol, setSelectedSymbol] = useState(MarketDataService.SYMBOLS.NIFTY);
  const [selectedTimeframe, setSelectedTimeframe] = useState<Timeframe>('15');
  const [hasMounted, setHasMounted] = useState(false);

  // Track if component has mounted - using setTimeout to avoid synchronous setState
  useEffect(() => {
    const timer = setTimeout(() => setHasMounted(true), 0);
    return () => clearTimeout(timer);
  }, []);

  // Available symbols
  const symbolOptions = [
    { value: MarketDataService.SYMBOLS.NIFTY, label: 'NIFTY 50' },
    { value: MarketDataService.SYMBOLS.BANKNIFTY, label: 'NIFTY BANK' },
    { value: MarketDataService.SYMBOLS.SENSEX, label: 'SENSEX' },
    { value: MarketDataService.SYMBOLS.RELIANCE, label: 'RELIANCE' },
    { value: MarketDataService.SYMBOLS.TCS, label: 'TCS' },
    { value: MarketDataService.SYMBOLS.HDFCBANK, label: 'HDFC BANK' },
  ];

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // Load initial data
  useEffect(() => {
    if (isAuthenticated && accessToken && hasMounted) {
      fetchHistoricalData(accessToken, selectedSymbol, selectedTimeframe);
    }
  }, [isAuthenticated, accessToken, selectedSymbol, selectedTimeframe, hasMounted, fetchHistoricalData]);

  // Handle symbol change
  const handleSymbolChange = (newSymbol: string) => {
    setSelectedSymbol(newSymbol);
    setSymbol(newSymbol);
  };

  // Handle timeframe change
  const handleTimeframeChange = (newTimeframe: Timeframe) => {
    setSelectedTimeframe(newTimeframe);
  };

  // Handle refresh
  const handleRefresh = () => {
    if (accessToken) {
      refreshData(accessToken);
    }
  };

  // Auto-refresh every 30 seconds
  useEffect(() => {
    if (!isAuthenticated || !accessToken) return;

    const interval = setInterval(() => {
      refreshData(accessToken);
    }, 30000);

    return () => clearInterval(interval);
  }, [isAuthenticated, accessToken, refreshData]);

  // Loading state
  if (authLoading || !hasMounted) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return null; // Will redirect to login
  }

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-bold text-white">Trading Dashboard</h1>
            </div>

            <div className="flex items-center space-x-4">
              {/* Symbol Selector */}
              <select
                value={selectedSymbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                className="bg-gray-700 text-white border border-gray-600 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                {symbolOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>

              {/* Refresh Button */}
              <button
                onClick={handleRefresh}
                disabled={marketLoading}
                className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-white p-2 rounded-lg transition-colors"
                title="Refresh data"
              >
                <svg
                  className={`w-5 h-5 ${marketLoading ? 'animate-spin' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
              </button>

              {/* User Menu */}
              <div className="flex items-center space-x-3">
                <div className="text-right">
                  <p className="text-sm font-medium text-white">{user?.name}</p>
                  <p className="text-xs text-gray-400">
                    Balance: ₹{user?.available_balance?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '0'}
                  </p>
                </div>
                <button
                  onClick={logout}
                  className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm transition-colors"
                >
                  Logout
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error Display */}
        {marketError && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <p className="text-red-400 text-sm">{marketError}</p>
            </div>
          </div>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center">
              <div className="p-2 bg-blue-600 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">Current Symbol</p>
                <p className="text-lg font-semibold text-white">
                  {symbolOptions.find(s => s.value === symbol)?.label || 'NIFTY 50'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center">
              <div className="p-2 bg-green-600 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">Available Balance</p>
                <p className="text-lg font-semibold text-white">
                  ₹{user?.available_balance?.toLocaleString('en-IN', { maximumFractionDigits: 0 }) || '0'}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center">
              <div className="p-2 bg-purple-600 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">Timeframe</p>
                <p className="text-lg font-semibold text-white">
                  {selectedTimeframe === 'D' ? '1D' : `${selectedTimeframe}m`}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center">
              <div className="p-2 bg-yellow-600 rounded-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-400">Last Updated</p>
                <p className="text-lg font-semibold text-white">Just now</p>
              </div>
            </div>
          </div>
        </div>

        {/* Chart */}
        <div className="bg-gray-800 rounded-lg p-6">
          <CandlestickChart
            onTimeframeChange={handleTimeframeChange}
            showVolume={true}
            showToolbar={true}
          />
        </div>

        {/* Market Status */}
        <div className="mt-8 bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold text-white mb-4">Market Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Market Status</p>
              <p className="text-lg font-medium text-green-400">Open</p>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Data Points</p>
              <p className="text-lg font-medium text-white">
                {hasData ? 'Loaded' : 'Loading...'}
              </p>
            </div>
            <div className="bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Auto Refresh</p>
              <p className="text-lg font-medium text-blue-400">Every 30s</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}