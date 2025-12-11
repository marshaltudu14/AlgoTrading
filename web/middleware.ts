import { NextRequest, NextResponse } from 'next/server';

async function validateAccessToken(accessToken: string, appId: string): Promise<boolean> {
  try {
    // Quick token validation by calling Fyers API
    const response = await fetch('https://api-t1.fyers.in/api/v3/profile', {
      headers: {
        'Authorization': `${appId}:${accessToken}`,
      },
    });

    return response.ok;
  } catch (error) {
    console.error('[MIDDLEWARE] Token validation error:', error);
    return false;
  }
}

export async function middleware(request: NextRequest) {
  // Get the pathname of the request (e.g. /, /dashboard)
  const path = request.nextUrl.pathname;

  // Get authentication cookies
  const accessToken = request.cookies.get('fyers_access_token')?.value;
  const appId = request.cookies.get('fyers_app_id')?.value;

  // Define public paths that don't require authentication
  const publicPaths = ['/', '/callback'];

  // Check if the path is public or an API route
  const isPublicPath = publicPaths.includes(path);
  const isApiPath = path.startsWith('/api/');
  const isDashboardPath = path.startsWith('/dashboard');

  // Allow access to public paths and API routes (all APIs need auth tokens which will be checked in the route handlers)
  if (isPublicPath || isApiPath) {
    return NextResponse.next();
  }

  // If user is not authenticated and trying to access protected routes (dashboard group)
  if (!accessToken || !appId) {
    // Only redirect if not already on the home page
    if (path !== '/') {
      // Create response that clears cookies and redirects
      const response = NextResponse.redirect(new URL('/', request.url));
      response.cookies.delete('fyers_access_token');
      response.cookies.delete('fyers_refresh_token');
      response.cookies.delete('fyers_app_id');
      response.cookies.delete('fyers_user_profile');
      response.cookies.delete('fyers_auth_state');
      response.cookies.delete('fyers_auth_code');

      // Add header to indicate this is an auth redirect
      response.headers.set('x-redirect-reason', 'unauthenticated');

      return response;
    } else {
      // Already on home page, just clear cookies and continue
      const response = NextResponse.next();
      response.cookies.delete('fyers_access_token');
      response.cookies.delete('fyers_refresh_token');
      response.cookies.delete('fyers_app_id');
      response.cookies.delete('fyers_user_profile');
      response.cookies.delete('fyers_auth_state');
      response.cookies.delete('fyers_auth_code');
      return response;
    }
  }

  // For dashboard routes, validate the tokens
  if (isDashboardPath) {
    // Check if we have tokens
    if (!accessToken || !appId) {
      return NextResponse.redirect(new URL('/', request.url));
    }

    // Validate tokens are still valid (with timeout to avoid blocking)
    try {
      const isValid = await Promise.race([
        validateAccessToken(accessToken, appId),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('Timeout')), 3000)
        )
      ]);

      if (!isValid) {
        // Clear cookies and redirect
        const response = NextResponse.redirect(new URL('/', request.url));
        response.cookies.delete('fyers_access_token');
        response.cookies.delete('fyers_refresh_token');
        response.cookies.delete('fyers_app_id');
        response.cookies.delete('fyers_user_profile');
        response.cookies.delete('fyers_auth_state');
        response.cookies.delete('fyers_auth_code');
        return response;
      }

      return NextResponse.next();
    } catch {
      // On validation error, redirect to login
      return NextResponse.redirect(new URL('/', request.url));
    }
  }

  // Allow all other requests
  return NextResponse.next();
}

// Configure the middleware to run on specific paths
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
  ],
};