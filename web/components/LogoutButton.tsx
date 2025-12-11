"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { LogOut } from 'lucide-react';

export default function LogoutButton() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleLogout = async () => {
    setIsLoading(true);

    try {
      // Get auth token from localStorage for the API call
      const accessToken = localStorage.getItem('access_token');

      // Call Next.js API route (which will call backend)
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(accessToken && { 'Authorization': `Bearer ${accessToken}` }),
        },
      });

      if (!response.ok) {
        throw new Error('Logout failed');
      }
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      // Clear any stored authentication tokens
      localStorage.removeItem('access_token');
      localStorage.removeItem('user_profile');

      // Redirect to login page
      router.push('/');
    }
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0 cursor-pointer"
      onClick={handleLogout}
      disabled={isLoading}
    >
      <LogOut className="h-4 w-4" />
    </Button>
  );
}