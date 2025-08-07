import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export async function middleware(request: NextRequest) {
  // Get the pathname of the request (e.g. /, /dashboard, /backtest)
  const { pathname } = request.nextUrl

  // Define protected routes
  const protectedRoutes = ['/dashboard', '/dashboard/backtest']
  
  // Check if the current path is protected
  const isProtectedRoute = protectedRoutes.some(route => 
    pathname.startsWith(route)
  )

  // If it's a protected route, check for authentication
  if (isProtectedRoute) {
    try {
      // Make an internal API call to validate the session
      // The 'credentials: "include"' is crucial for sending HTTP-only cookies
      const apiResponse = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/profile`, {
        headers: {
          'Cookie': request.headers.get('Cookie') || ''
        }
      });

      if (apiResponse.status === 401) {
        // If the API returns 401, redirect to login
        const loginUrl = new URL('/login', request.url);
        return NextResponse.redirect(loginUrl);
      } else if (!apiResponse.ok) {
        // Handle other API errors (e.g., 500)
        console.error('Middleware API error:', apiResponse.status, apiResponse.statusText);
        const loginUrl = new URL('/login', request.url); // Redirect to login on any API error for safety
        return NextResponse.redirect(loginUrl);
      }
      // If API response is OK, continue to the requested page
    } catch (error) {
      console.error('Middleware fetch error:', error);
      const loginUrl = new URL('/login', request.url); // Redirect to login on network errors
      return NextResponse.redirect(loginUrl);
    }
  }

  // Allow the request to continue
  return NextResponse.next()
}

// Configure which paths the middleware should run on
export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - login (login page)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|login).*)',
  ],
}
