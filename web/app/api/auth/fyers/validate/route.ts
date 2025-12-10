import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { FYERS_ENDPOINTS, COOKIE_NAMES } from '@/lib/constants';

export async function POST(request: NextRequest) {
  try {
    const { appId, secretKey } = await request.json();

    // Get auth_code from cookie
    const auth_code = request.cookies.get(COOKIE_NAMES.AUTH_CODE)?.value;

    if (!auth_code) {
      return NextResponse.json(
        { error: 'Authorization code not found or expired' },
        { status: 400 }
      );
    }

    if (!appId || !secretKey) {
      return NextResponse.json(
        { error: 'App ID and Secret Key are required' },
        { status: 400 }
      );
    }

    // Create SHA-256 hash of app_id:app_secret
    const appIdHash = crypto
      .createHash('sha256')
      .update(`${appId}:${secretKey}`)
      .digest('hex');

    // Exchange auth_code for access_token
    const validateResponse = await fetch(FYERS_ENDPOINTS.VALIDATE_AUTHCODE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'authorization_code',
        appIdHash: appIdHash,
        code: auth_code,
      }),
    });

    const data = await validateResponse.json();

    if (data.s !== 'ok') {
      return NextResponse.json(
        { error: data.message || 'Failed to validate authorization code' },
        { status: 400 }
      );
    }

    // Get user profile information
    const profileResponse = await fetch(FYERS_ENDPOINTS.PROFILE, {
      headers: {
        'Authorization': `${appId}:${data.access_token}`,
      },
    });

    let profileData = null;
    if (profileResponse.ok) {
      const profileResult = await profileResponse.json();
      if (profileResult.s === 'ok') {
        profileData = profileResult.data;
      }
    }

    // Create response with tokens and profile
    const response = NextResponse.json({
      success: true,
      access_token: data.access_token,
      refresh_token: data.refresh_token,
      profile: profileData,
      appId: appId,
    });

    // Clear the auth_code cookie
    response.cookies.delete(COOKIE_NAMES.AUTH_CODE);

    // Set secure HTTP-only cookies for tokens
    if (data.access_token) {
      response.cookies.set(COOKIE_NAMES.ACCESS_TOKEN, data.access_token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 60 * 60 * 24, // 24 hours
        path: '/'
      });
    }

    if (data.refresh_token) {
      response.cookies.set(COOKIE_NAMES.REFRESH_TOKEN, data.refresh_token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 60 * 60 * 24 * 15, // 15 days
        path: '/'
      });
    }

    // Set app_id in a regular cookie (client needs this for API calls)
    response.cookies.set(COOKIE_NAMES.APP_ID, appId, {
      httpOnly: false,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 60 * 60 * 24 * 30, // 30 days
      path: '/'
    });

    // Store user profile in cookie for easy access
    if (profileData) {
      response.cookies.set(COOKIE_NAMES.USER_PROFILE, JSON.stringify(profileData), {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 60 * 60 * 24, // 24 hours
        path: '/'
      });
    }

    return response;
  } catch (error) {
    console.error('Error validating auth code:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}