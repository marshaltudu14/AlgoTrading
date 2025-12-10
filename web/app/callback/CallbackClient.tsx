"use client";

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Loader2 } from 'lucide-react';

export default function CallbackClient() {
  const router = useRouter();

  useEffect(() => {
    // Check immediately if token is available
    const checkToken = () => {
      const hasToken = document.cookie.includes('fyers_access_token');
      if (hasToken) {
        router.push('/dashboard');
        return true;
      }
      return false;
    };

    // If token is already available, redirect immediately
    if (checkToken()) return;

    // Otherwise, poll for the token
    const interval = setInterval(() => {
      if (checkToken()) {
        clearInterval(interval);
      }
    }, 100);

    // Cleanup
    return () => clearInterval(interval);
  }, [router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
        <p className="text-gray-600">Processing authentication...</p>
      </div>
    </div>
  );
}