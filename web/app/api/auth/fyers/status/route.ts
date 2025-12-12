import { NextResponse } from 'next/server';

export async function GET() {
  try {
    const { getSession } = await import('@/lib/server-session');
    const session = await getSession();

    if (session && session.isAuthenticated) {
      // Return session data
      return NextResponse.json({
        authenticated: true,
        appId: session.appId,
        access_token: session.accessToken,
        profile: session.profile,
      });
    }

    return NextResponse.json({
      authenticated: false,
    });
  } catch (error) {
    console.error('Error checking auth status:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}