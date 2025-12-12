import { NextRequest, NextResponse } from 'next/server';
import { validateAuthCode } from '@/lib/auth-utils';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const auth_code = searchParams.get('auth_code');
    searchParams.get('state'); // state parameter available but not used

    if (!auth_code) {
      // Handle error or user denial
      const error = searchParams.get('error') || 'Authorization code not found';
      return NextResponse.redirect(
        new URL(`/?error=${encodeURIComponent(error)}`, process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    // Get session data
    const { getSession, updateSession } = await import('@/lib/server-session');
    const sessionData = await getSession();

    if (!sessionData) {
      return NextResponse.redirect(
        new URL('/?error=Session expired. Please login again.', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    const { appId, secretKey } = sessionData;

    if (!appId || !secretKey) {
      return NextResponse.redirect(
        new URL('/?error=Missing authentication credentials', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    // Validate the authorization code
    const result = await validateAuthCode(auth_code, appId, secretKey);

    if (!result.success) {
      return NextResponse.redirect(
        new URL(`/?error=${encodeURIComponent(result.error || 'Authentication failed')}`, process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    // Update session with tokens
    await updateSession({
      accessToken: result.access_token,
      refreshToken: result.refresh_token,
      profile: result.profile,
    });

    // Create response - redirect to the callback page which will handle the final redirect
    return NextResponse.redirect(
      new URL('/callback', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
    );
  } catch (error) {
    console.error('Callback error:', error);
    return NextResponse.redirect(
      new URL('/?error=Internal server error', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
    );
  }
}