import { NextRequest, NextResponse } from 'next/server';
import { FYERS_ENDPOINTS } from '@/lib/constants';

export async function POST(request: NextRequest) {
  try {
    const { appId, redirectUrl } = await request.json();

    if (!appId || !redirectUrl) {
      return NextResponse.json(
        { error: 'App ID and Redirect URL are required' },
        { status: 400 }
      );
    }

    // Generate a random state parameter for security
    const state = Math.random().toString(36).substring(2, 15) +
                  Math.random().toString(36).substring(2, 15);

    // Update session with auth state
    const { updateSession } = await import('@/lib/server-session');
    const sessionUpdated = await updateSession({ authState: state });

    if (!sessionUpdated) {
      return NextResponse.json(
        { error: 'No active session found' },
        { status: 401 }
      );
    }

    // Return auth URL
    const authUrl = `${FYERS_ENDPOINTS.GENERATE_AUTHCODE}?client_id=${appId}&redirect_uri=${encodeURIComponent(redirectUrl)}&response_type=code&state=${state}`;

    return NextResponse.json({ authUrl });
  } catch (error) {
    console.error('Error initiating auth:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}