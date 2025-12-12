"use client";

import { useEffect } from 'react';
import { Loader2 } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export default function CallbackClient() {
  const { checkAuthStatus } = useAuthStore();

  useEffect(() => {
    // Check auth status immediately
    checkAuthStatus();
  }, [checkAuthStatus]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center">
        <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
        <p className="text-gray-600">Processing authentication...</p>
      </div>
    </div>
  );
}