import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { FyersUser } from '@/types/fyers';

interface AuthState {
  // State
  isAuthenticated: boolean;
  isLoading: boolean;
  user: FyersUser | null;
  accessToken: string | null;
  refreshToken: string | null;
  error: string | null;
  lastLoginTime: number | null;

  // Actions
  login: () => Promise<void>;
  logout: () => void;
  validateToken: () => Promise<boolean>;
  clearError: () => void;
  setLoading: (loading: boolean) => void;
  setTokens: (accessToken: string, refreshToken?: string) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // Initial state
      isAuthenticated: false,
      isLoading: false,
      user: null,
      accessToken: null,
      refreshToken: null,
      error: null,
      lastLoginTime: null,

      // Login action
      login: async () => {
        try {
          set({ isLoading: true, error: null });

          const response = await fetch('/api/auth/fyers', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
          });

          const data = await response.json();

          if (!response.ok || !data.success) {
            throw new Error(data.error || 'Authentication failed');
          }

          set({
            isAuthenticated: true,
            user: data.data.user,
            accessToken: data.data.access_token,
            refreshToken: data.data.refresh_token || null,
            lastLoginTime: Date.now(),
            error: null,
          });

        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Authentication failed';
          set({
            error: errorMessage,
            isAuthenticated: false,
            user: null,
            accessToken: null,
            refreshToken: null,
          });
          throw error;
        } finally {
          set({ isLoading: false });
        }
      },

      // Logout action
      logout: () => {
        set({
          isAuthenticated: false,
          user: null,
          accessToken: null,
          refreshToken: null,
          error: null,
          lastLoginTime: null,
        });

        // Clear the HTTP-only cookie by setting it to expire
        document.cookie = 'fyers_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
      },

      // Validate token action
      validateToken: async () => {
        const { accessToken } = get();

        if (!accessToken) {
          return false;
        }

        try {
          const response = await fetch('/api/auth/validate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ accessToken }),
          });

          const data = await response.json();

          if (!response.ok || !data.success) {
            // Token is invalid, clear auth state
            get().logout();
            return false;
          }

          return data.data.valid;

        } catch (error) {
          console.error('Token validation error:', error);
          get().logout();
          return false;
        }
      },

      // Clear error action
      clearError: () => {
        set({ error: null });
      },

      // Set loading state
      setLoading: (loading: boolean) => {
        set({ isLoading: loading });
      },

      // Set tokens manually (useful for refresh scenarios)
      setTokens: (accessToken: string, refreshToken?: string) => {
        set({
          accessToken,
          refreshToken: refreshToken || null,
        });
      },
    }),
    {
      name: 'fyers-auth-storage',
      partialize: (state) => ({
        isAuthenticated: state.isAuthenticated,
        user: state.user,
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        lastLoginTime: state.lastLoginTime,
      }),
    }
  )
);

// Selectors for commonly used state combinations
export const useAuth = () => {
  const store = useAuthStore();

  return {
    // Basic auth state
    isAuthenticated: store.isAuthenticated,
    isLoading: store.isLoading,
    user: store.user,
    accessToken: store.accessToken,
    refreshToken: store.refreshToken,
    error: store.error,

    // Actions
    login: store.login,
    logout: store.logout,
    validateToken: store.validateToken,
    clearError: store.clearError,
    setTokens: store.setTokens,

    // Computed values
    isLoggedIn: store.isAuthenticated && !!store.accessToken,
    userName: store.user?.name || '',
    userEmail: store.user?.email || '',
    userBalance: store.user?.available_balance || 0,
  };
};