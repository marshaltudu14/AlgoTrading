/**
 * Authentication hooks for user profile and session management
 */

import { useEffect, useState } from 'react'
import { apiClient } from '@/lib/api'

export interface UserProfile {
  user_id: string;
  name: string;
  capital: number;
}

export function useAuth() {
  const [user, setUser] = useState<UserProfile | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const profile = await apiClient.getProfile()
        setUser(profile)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch user profile')
        setUser(null)
      } finally {
        setIsLoading(false)
      }
    }

    fetchProfile()
  }, [])

  return {
    user,
    userId: user?.user_id || null,
    isLoading,
    error,
    refreshProfile: async () => {
      try {
        const profile = await apiClient.getProfile()
        setUser(profile)
        setError(null)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to refresh profile')
      }
    }
  }
}