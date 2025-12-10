"use client";

import { useEffect } from 'react';
import { useAuthStore } from '@/stores/authStore';
import { useRouter, usePathname } from 'next/navigation';

interface AuthProviderProps {
  children: React.ReactNode;
}

export default function AuthProvider({ children }: AuthProviderProps) {
  const { isAuthenticated, checkAuthStatus } = useAuthStore();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    // Check auth status on mount
    checkAuthStatus();

    // If user is on login page but already authenticated, redirect to dashboard
    if (pathname === '/' && isAuthenticated) {
      router.replace('/dashboard');
      return;
    }

    // If user is on dashboard routes but not authenticated, redirect to login
    if (pathname.startsWith('/dashboard') && !isAuthenticated) {
      // Check for auth code in cookies to handle OAuth callback
      const authCodeCookie = document.cookie.includes('fyers_auth_code');
      if (!authCodeCookie) {
        router.replace('/');
      }
    }
  }, [isAuthenticated, pathname, router, checkAuthStatus]);

  return <>{children}</>;
}