import { NextRequest, NextResponse } from 'next/server';
import { COOKIE_NAMES } from '@/lib/constants';

export async function GET(request: NextRequest) {
  try {
    const cookieStore = request.cookies;
    const accessToken = cookieStore.get(COOKIE_NAMES.ACCESS_TOKEN)?.value;
    const appId = cookieStore.get(COOKIE_NAMES.APP_ID)?.value;
    const userProfile = cookieStore.get(COOKIE_NAMES.USER_PROFILE)?.value;

    if (accessToken && appId) {
      let profile = null;
      try {
        profile = userProfile ? JSON.parse(userProfile) : null;
      } catch (e) {
        console.error('Error parsing user profile:', e);
      }

      return NextResponse.json({
        authenticated: true,
        appId,
        access_token: accessToken,
        profile,
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