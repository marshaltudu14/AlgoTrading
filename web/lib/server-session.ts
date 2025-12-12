// Server-side session management using Next.js encrypted cookies
// All sensitive data stays server-side

import { cookies } from 'next/headers';
import { SignJWT, jwtVerify, JWTPayload } from 'jose';

// Use a secret key for JWT signing (in production, use environment variable)
const JWT_SECRET = new TextEncoder().encode(
  process.env.SESSION_SECRET || 'your-secret-key-change-in-production'
);

// Define session data interface
export interface SessionData extends JWTPayload {
  appId: string;
  secretKey: string;
  accessToken?: string;
  refreshToken?: string;
  profile?: Record<string, unknown>;
  authState?: string;
  isAuthenticated: boolean;
}

// Create an encrypted session cookie
export async function createSession(data: Omit<SessionData, 'isAuthenticated'>): Promise<void> {
  try {
    const cookieStore = await cookies();

    // Create JWT token with session data
    const token = await new SignJWT({
      ...data,
      isAuthenticated: false
    })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime('7d') // Session expires in 7 days
      .sign(JWT_SECRET);

    // Set encrypted session cookie
    cookieStore.set('auth_session', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    });
  } catch (error) {
    throw error;
  }
}

// Get and decrypt session from cookie
export async function getSession(): Promise<SessionData | null> {
  try {
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get('auth_session')?.value;

    if (!sessionToken) {
      return null;
    }

    // Verify and decrypt JWT
    const { payload } = await jwtVerify(sessionToken, JWT_SECRET);

    // Ensure required fields exist
    if (!payload.appId || !payload.secretKey) {
      throw new Error('Invalid session payload');
    }

    const sessionData = {
      appId: payload.appId as string,
      secretKey: payload.secretKey as string,
      accessToken: payload.accessToken as string | undefined,
      refreshToken: payload.refreshToken as string | undefined,
      profile: payload.profile as Record<string, unknown> | undefined,
      authState: payload.authState as string | undefined,
      isAuthenticated: payload.isAuthenticated as boolean
    };

    return sessionData;
  } catch {
    // Clear invalid session
    const cookieStore = await cookies();
    cookieStore.delete('auth_session');
    return null;
  }
}

// Update existing session
export async function updateSession(updates: Partial<SessionData>): Promise<boolean> {
  try {
    const currentSession = await getSession();
    if (!currentSession) {
      return false;
    }

    // Merge updates with current session
    const updatedSession: SessionData = {
      ...currentSession,
      ...updates
    };

    // Mark as authenticated if we have tokens
    if (updates.accessToken || updates.refreshToken) {
      updatedSession.isAuthenticated = true;
    }

    const cookieStore = await cookies();

    // Create new JWT token with updated data
    const token = await new SignJWT(updatedSession)
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime('7d')
      .sign(JWT_SECRET);

    // Update cookie
    cookieStore.set('auth_session', token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60 * 24 * 7, // 7 days
    });

    return true;
  } catch {
    return false;
  }
}

// Delete session (logout)
export async function deleteSession(): Promise<void> {
  try {
    const cookieStore = await cookies();
    cookieStore.delete('auth_session');
  } catch (error) {
    console.error('Failed to delete session:', error);
  }
}