import { NextRequest, NextResponse } from 'next/server';
import { FYERS_ENDPOINTS, COOKIE_NAMES } from '@/lib/constants';

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

    // Store state in session/cookie for verification later
    const response = NextResponse.json({
      authUrl: `${FYERS_ENDPOINTS.GENERATE_AUTHCODE}?client_id=${appId}&redirect_uri=${encodeURIComponent(redirectUrl)}&response_type=code&state=${state}`
    });

    // Set state in a secure cookie
    response.cookies.set(COOKIE_NAMES.AUTH_STATE, state, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 10 // 10 minutes
    });

    return response;
  } catch (error) {
    console.error('Error initiating auth:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}