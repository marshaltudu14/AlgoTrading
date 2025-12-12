export interface SessionData {
  appId: string;
  secretKey: string;
  accessToken?: string;
  refreshToken?: string;
  profile?: Record<string, unknown>;
  authState?: string;
  isAuthenticated: boolean;
}