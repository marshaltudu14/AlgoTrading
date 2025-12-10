import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  // Get the pathname of the request (e.g. /, /dashboard)
  const path = request.nextUrl.pathname;

  // Get the access token from cookies
  const accessToken = request.cookies.get('fyers_access_token')?.value;

  // Define public paths that don't require authentication
  const publicPaths = ['/'];
  const apiPaths = ['/api/auth/fyers/initiate', '/api/auth/fyers/callback', '/api/auth/fyers/validate', '/api/auth/fyers/refresh'];

  // Check if the path is public or an API route
  const isPublicPath = publicPaths.includes(path);
  const isApiPath = apiPaths.some(apiPath => path.startsWith(apiPath));
  const isDashboardPath = path.startsWith('/dashboard');

  // If the path is public or an API route, allow access
  if (isPublicPath || isApiPath) {
    return NextResponse.next();
  }

  // If user is not authenticated and trying to access protected routes (dashboard group), redirect to login
  if (!accessToken && isDashboardPath) {
    const loginUrl = new URL('/', request.url);
    return NextResponse.redirect(loginUrl);
  }

  // If user is authenticated, allow access
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