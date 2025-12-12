import { NextRequest, NextResponse } from 'next/server';

export async function middleware(request: NextRequest) {
  // Get the pathname of the request (e.g. /, /dashboard)
  const path = request.nextUrl.pathname;

  // Define public paths that don't require authentication
  const publicPaths = ['/', '/callback'];

  // Check if the path is public or an API route
  const isPublicPath = publicPaths.includes(path);
  const isApiPath = path.startsWith('/api/');
  const isDashboardPath = path.startsWith('/dashboard');

  // Allow access to public paths
  if (isPublicPath) {
    return NextResponse.next();
  }

  // For API routes, let them handle their own auth logic
  if (isApiPath) {
    return NextResponse.next();
  }

  // For dashboard routes, we need to check if the user is authenticated
  // We'll let the dashboard page itself handle auth checking via the AuthProvider
  // The middleware just ensures no direct access without going through auth flow

  // Check if there's an auth session cookie
  const hasAuthCookie = request.cookies.get('auth_session')?.value;

  if (!hasAuthCookie && isDashboardPath) {
    // No auth cookie and trying to access dashboard - redirect to login
    return NextResponse.redirect(new URL('/', request.url));
  }

  // Allow all other requests (dashboard pages will handle their own auth checks)
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