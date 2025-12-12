"use client";

import { Button } from '@/components/ui/button';
import { LogOut } from 'lucide-react';
import { useAuthStore } from '@/stores/authStore';

export default function LogoutButton() {
  const { logout, isLoading } = useAuthStore();

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-8 w-8 p-0 cursor-pointer"
      onClick={logout}
      disabled={isLoading}
    >
      <LogOut className="h-4 w-4" />
    </Button>
  );
}