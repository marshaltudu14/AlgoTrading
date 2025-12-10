import { NextRequest, NextResponse } from 'next/server';
import { validateAuthCode } from '@/lib/auth-utils';
import { COOKIE_NAMES } from '@/lib/constants';

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

    // Get temporary credentials from cookies
    const cookieStore = request.cookies;
    const tempAppId = cookieStore.get('fyers_temp_app_id')?.value;
    const tempSecret = cookieStore.get('fyers_temp_secret')?.value;

    if (!tempAppId || !tempSecret) {
      return NextResponse.redirect(
        new URL('/?error=Missing authentication credentials', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    // Validate the authorization code
    const result = await validateAuthCode(auth_code, tempAppId, tempSecret);

    if (!result.success) {
      return NextResponse.redirect(
        new URL(`/?error=${encodeURIComponent(result.error || 'Authentication failed')}`, process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
      );
    }

    // Create response - redirect to the callback page which will handle the final redirect
    const response = NextResponse.redirect(
      new URL('/callback', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
    );

    // Set secure cookies with authentication data
    response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN, result.access_token, {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, result.refresh_token, {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    response.cookies.set(COOKIE_NAMES.APP_ID, result.appId || '', {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    response.cookies.set(COOKIE_NAMES.USER_PROFILE, JSON.stringify(result.profile), {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 30, // 30 days
    });

    // Clear temporary cookies
    response.cookies.set('fyers_temp_app_id', '', {
      expires: new Date(0),
    });
    response.cookies.set('fyers_temp_secret', '', {
      expires: new Date(0),
    });

    return response;
  } catch (error) {
    console.error('Callback error:', error);
    return NextResponse.redirect(
      new URL('/?error=Internal server error', process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3001')
    );
  }
}