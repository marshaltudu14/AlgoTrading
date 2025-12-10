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

export const useAuthStore = create<AuthState>((set, get) => ({
  isAuthenticated: false,
  isLoading: false,
  user: null,
  appId: null,
  accessToken: null,
  error: null,

  login: async (appId: string, secretKey: string) => {
    set({ isLoading: true, error: null });

    try {
      // Store credentials for callback
      sessionStorage.setItem('fyers_app_id', appId);
      sessionStorage.setItem('fyers_secret_key', secretKey);

      // Create redirect URL dynamically
      const redirectUrl = `${process.env.NEXT_PUBLIC_SITE_URL || window.location.origin}/api/auth/fyers/callback`;

      // Call the initiate API
      const response = await fetch('/api/auth/fyers/initiate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          appId,
          redirectUrl,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // Redirect to Fyers for authentication
        window.location.href = data.authUrl;
      } else {
        set({ error: data.error || 'Failed to initiate authentication', isLoading: false });
      }
    } catch (error) {
      console.error('Login error:', error);
      set({ error: 'Network error. Please try again.', isLoading: false });
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

  logout: () => {
    // Clear cookies by setting them to expire
    document.cookie = 'fyers_access_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'fyers_refresh_token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'fyers_app_id=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
    document.cookie = 'fyers_user_profile=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';

    // Clear sessionStorage
    sessionStorage.clear();

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
  },

  checkAuthStatus: () => {
    // Check if we have stored tokens
    const hasToken = document.cookie.includes('fyers_access_token');

    if (hasToken) {
      set({
        isAuthenticated: true,
        appId: document.cookie.split('fyers_app_id=')[1]?.split(';')[0] || null,
      });
    } else {
      // Check if we have an auth_code (from OAuth callback)
      const authCodeCookie = document.cookie.includes('fyers_auth_code');

      if (authCodeCookie) {
        const storedAppId = sessionStorage.getItem('fyers_app_id');
        const storedSecretKey = sessionStorage.getItem('fyers_secret_key');

        if (storedAppId && storedSecretKey) {
          // Auto-validate with stored credentials
          get().validateAuth(storedAppId, storedSecretKey);
        }
      }
    }
  },

  
  setError: (error: string | null) => {
    set({ error });
  },
}));