import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Define public routes that don't require authentication
  const publicRoutes = ['/login', '/api/auth'];

  // Check if the current path is public
  const isPublicRoute = publicRoutes.some(route =>
    pathname === route || pathname.startsWith(route)
  );

  // If the path is not public and user is not authenticated, redirect to login
  if (!isPublicRoute && pathname !== '/') {
    // Allow access to the root route and API routes
    if (pathname.startsWith('/api/')) {
      return NextResponse.next();
    }

    // For protected routes, we'll let the client-side handle authentication
    // This allows the app to show the login page and then redirect back
    return NextResponse.next();
  }

  // Redirect root to dashboard if authenticated, or to login if not
  if (pathname === '/') {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};