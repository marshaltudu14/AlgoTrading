import { NextRequest, NextResponse } from 'next/server';
import { COOKIE_NAMES } from '@/lib/constants';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const auth_code = searchParams.get('auth_code');
    const state = searchParams.get('state');
    const s = searchParams.get('s');
    const message = searchParams.get('message');

    // Check for errors in the response
    if (s === 'error' || !auth_code) {
      const errorUrl = new URL('/', request.url);
      errorUrl.searchParams.set('error', message || 'Authentication failed');
      return NextResponse.redirect(errorUrl);
    }

    // Get stored state from cookie
    const storedState = request.cookies.get(COOKIE_NAMES.AUTH_STATE)?.value;

    // Verify state parameter to prevent CSRF attacks
    if (!storedState || state !== storedState) {
      const errorUrl = new URL('/', request.url);
      errorUrl.searchParams.set('error', 'Invalid state parameter');
      return NextResponse.redirect(errorUrl);
    }

    // Clear the state cookie and redirect to dashboard
    const response = NextResponse.redirect(new URL('/dashboard', request.url));
    response.cookies.delete(COOKIE_NAMES.AUTH_STATE);

    // Store auth_code temporarily for the next step
    response.cookies.set(COOKIE_NAMES.AUTH_CODE, auth_code, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 5 // 5 minutes
    });

    return response;
  } catch (error) {
    console.error('Error in callback:', error);
    const errorUrl = new URL('/', request.url);
    errorUrl.searchParams.set('error', 'Callback processing failed');
    return NextResponse.redirect(errorUrl);
  }
}