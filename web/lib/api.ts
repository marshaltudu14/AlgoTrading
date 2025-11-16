// Centralized API configuration and utilities

const API_BASE_URL = 'http://localhost:8000';

export class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: unknown
  ) {
    super(message);
    this.name = 'APIError';
  }
}

/**
 * Make API calls to backend with consistent error handling
 */
export async function apiCall<T = unknown>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const defaultHeaders = {
    'Content-Type': 'application/json',
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    const data = await response.json();

    if (!response.ok) {
      throw new APIError(
        data.error || `HTTP error! status: ${response.status}`,
        response.status,
        data
      );
    }

    return data;
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }

    // Handle network errors, JSON parsing errors, etc.
    throw new APIError(
      error instanceof Error ? error.message : 'Unknown error occurred',
      500,
      error
    );
  }
}

/**
 * Get authentication headers for API calls
 */
export function getAuthHeaders(accessToken?: string): Record<string, string> {
  const headers: Record<string, string> = {};

  if (accessToken) {
    headers['Authorization'] = `Bearer ${accessToken}`;
  }

  return headers;
}

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  AUTH: {
    FYERS_LOGIN: '/auth/fyers/login',
    LOGOUT: '/auth/logout',
  },
} as const;