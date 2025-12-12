import { create } from 'zustand';

interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: {
    name?: string;
    fy_id?: string;
    email_id?: string;
  } | null;
  appId: string | null;
  accessToken: string | null;
  error: string | null;
  // Actions
  login: (appId: string, secretKey: string) => Promise<void>;
  validateAuth: (appId: string, secretKey: string) => Promise<void>;
  logout: () => void;
  checkAuthStatus: () => void;
  setError: (error: string | null) => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  isAuthenticated: false,
  isLoading: false,
  user: null,
  appId: null,
  accessToken: null,
  error: null,

  login: async (appId: string, secretKey: string) => {
    set({ isLoading: true, error: null });

    try {
      // Store credentials in server-side session
      const response = await fetch('/api/auth/session', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          appId,
          secretKey,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create session');
      }

      // Create redirect URL dynamically
      const redirectUrl = `${process.env.NEXT_PUBLIC_SITE_URL || window.location.origin}/api/auth/fyers/callback`;

      // Call the initiate API
      const initiateResponse = await fetch('/api/auth/fyers/initiate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          appId,
          redirectUrl,
        }),
      });

      const data = await initiateResponse.json();

      if (initiateResponse.ok) {
        // Redirect to Fyers for authentication
        window.location.href = data.authUrl;
      } else {
        set({ error: data.error || 'Failed to initiate authentication', isLoading: false });
      }
    } catch (error) {
      console.error('Login error:', error);
      set({ error: error instanceof Error ? error.message : 'Network error. Please try again.', isLoading: false });
    }
  },

  validateAuth: async (appId: string, secretKey: string) => {
    set({ isLoading: true, error: null });

    try {
      const response = await fetch('/api/auth/fyers/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({
          appId,
          secretKey,
        }),
      });

      const data = await response.json();

      if (response.ok && data.success) {
        // Clear sessionStorage
        sessionStorage.removeItem('fyers_app_id');
        sessionStorage.removeItem('fyers_secret_key');

        set({
          isAuthenticated: true,
          isLoading: false,
          user: data.profile,
          appId: data.appId,
          accessToken: data.access_token,
          error: null,
        });
      } else {
        set({ error: data.error || 'Authentication failed', isLoading: false });
      }
    } catch (error) {
      console.error('Auth validation error:', error);
      set({ error: 'Network error. Please try again.', isLoading: false });
    }
  },

  logout: async () => {
    try {
      // Delete server-side session
      await fetch('/api/auth/session', {
        method: 'DELETE',
        credentials: 'include',
      });

      // Clear local state
      set({
        isAuthenticated: false,
        isLoading: false,
        user: null,
        appId: null,
        accessToken: null,
        error: null,
      });

      // Redirect to login
      window.location.href = '/';
    } catch (error) {
      console.error('Logout error:', error);
      // Still clear local state and redirect even if server fails
      set({
        isAuthenticated: false,
        isLoading: false,
        user: null,
        appId: null,
        accessToken: null,
        error: null,
      });
      window.location.href = '/';
    }
  },

  checkAuthStatus: async () => {
    try {
      // Check auth status via session API
      const response = await fetch('/api/auth/session', {
        method: 'GET',
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        const session = data.session;

        if (session && session.isAuthenticated) {
          set({
            isAuthenticated: true,
            user: session.profile,
            appId: session.appId,
            accessToken: session.accessToken,
          });
        } else {
          set({
            isAuthenticated: false,
            user: null,
            appId: null,
            accessToken: null,
          });
        }
      } else {
        // Not authenticated
        set({
          isAuthenticated: false,
          user: null,
          appId: null,
          accessToken: null,
        });
      }
    } catch (error) {
      console.error('Error checking auth status:', error);
      set({
        isAuthenticated: false,
        user: null,
        appId: null,
        accessToken: null,
      });
    }
  },


  setError: (error: string | null) => {
    set({ error });
  },
}));