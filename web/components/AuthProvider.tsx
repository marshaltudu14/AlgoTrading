"use client";

import { useEffect, useState } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useRouter, usePathname } from 'next/navigation';

interface AuthProviderProps {
  children: React.ReactNode;
}

export default function AuthProvider({ children }: AuthProviderProps) {
  const { isAuthenticated, checkAuthStatus } = useAuthStore();
  const router = useRouter();
  const pathname = usePathname();
  const [hasCheckedAuth, setHasCheckedAuth] = useState(false);

  useEffect(() => {
    // Check auth status on mount
    const checkAuth = async () => {
      await checkAuthStatus();
      setHasCheckedAuth(true);
    };
    checkAuth();
  }, [checkAuthStatus]); // Only run once on mount

  useEffect(() => {
    // Only handle redirects after auth check is complete
    if (!hasCheckedAuth) return;

    // If user is on login page but already authenticated, redirect to dashboard
    if (pathname === '/' && isAuthenticated) {
      router.replace('/dashboard');
      return;
    }

    // If user is on dashboard routes but not authenticated, redirect to login
    if (pathname.startsWith('/dashboard') && !isAuthenticated) {
      // Check for temp credentials (OAuth in progress)
      const tempAppId = document.cookie.includes('fyers_temp_app_id');
      if (!tempAppId) {
        router.replace('/');
      }
    }
  }, [isAuthenticated, pathname, hasCheckedAuth, router]);

  return <>{children}</>;
}